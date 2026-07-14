import fcntl
import json
import multiprocessing
import os
import stat
import time
from pathlib import Path

import pytest

import vllm_router.services.structured_output.capture as capture_module
from vllm_router.services.structured_output.capture import (
    CapturePolicy,
    SecureJSONLCaptureSink,
)
from vllm_router.services.structured_output.transform import RepairTelemetry

_NOW = 1_720_000_000.0
_DAY = "20240703"


def _secure_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "captures"
    directory.mkdir(mode=0o700)
    directory.chmod(0o700)
    return directory


def _capture_once_in_process(directory: Path, results) -> None:
    sink = SecureJSONLCaptureSink(
        directory,
        CapturePolicy(sample_rate=1.0),
        random_value=lambda: 0.0,
        wall_time=lambda: _NOW,
    )
    started = time.monotonic()
    captured = sink.capture(
        model="m",
        output="secret",
        telemetry=RepairTelemetry("unknown", "none", 0),
    )
    results.put((captured, time.monotonic() - started))


def _capture_twice_in_process(directory: Path, results) -> None:
    sink = SecureJSONLCaptureSink(
        directory,
        CapturePolicy(sample_rate=1.0),
        random_value=lambda: 0.0,
        wall_time=lambda: _NOW,
    )
    event = RepairTelemetry("unknown", "none", 0)
    started = time.monotonic()
    captures = [
        sink.capture(model="m", output="secret", telemetry=event),
        sink.capture(model="m", output="secret", telemetry=event),
    ]
    results.put((captures, time.monotonic() - started))


def _concurrent_capture_writer(directory: Path, start, results, writer: int) -> None:
    sink = SecureJSONLCaptureSink(
        directory,
        CapturePolicy(sample_rate=1.0),
        random_value=lambda: 0.0,
        wall_time=lambda: _NOW,
    )
    event = RepairTelemetry("unknown", "none", 0)
    captured = 0
    deadline = time.monotonic() + 2.0
    start.wait()
    while captured < 100 and time.monotonic() < deadline:
        if sink.capture(model=f"m{writer}", output=f"output {writer}", telemetry=event):
            captured += 1
        else:
            time.sleep(0.0001)
    results.put(captured)


def _finish_process_promptly(process, timeout: float) -> bool:
    process.join(timeout)
    hung = process.is_alive()
    if hung:
        process.terminate()
        process.join(1.0)
    return not hung


def test_capture_is_sampled_redacted_bounded_and_private(tmp_path):
    directory = _secure_dir(tmp_path)
    sink = SecureJSONLCaptureSink(
        directory,
        CapturePolicy(sample_rate=1.0, max_bytes=64, retention_days=7),
        random_value=lambda: 0.0,
        wall_time=lambda: _NOW,
    )
    raw = '{{"secret":"customer@example.com","long":"' + "x" * 500 + '"}'
    assert sink.capture(
        model="m",
        output=raw,
        telemetry=RepairTelemetry("unknown", "none", 0),
    )

    files = list(directory.glob("structured-output-*.jsonl"))
    assert len(files) == 1
    assert directory.stat().st_mode & 0o777 == 0o700
    assert files[0].stat().st_mode & 0o777 == 0o600
    assert (
        directory / ".structured-output-capture.lock"
    ).stat().st_mode & 0o777 == 0o600
    stored_bytes = files[0].read_bytes()
    assert len(stored_bytes) <= 64 + capture_module._RECORD_OVERHEAD_BYTES
    assert b"customer@example.com" not in stored_bytes
    assert b"x" * 20 not in stored_bytes
    record = json.loads(stored_bytes)
    assert len(record["redacted_output"].encode()) <= 64
    assert record["output_bytes"] == len(raw.encode())
    assert set(record) == {
        "captured_at",
        "model",
        "status",
        "mode",
        "garbage_prefix_bytes",
        "output_bytes",
        "redacted_output",
    }


def test_sampled_out_record_creates_no_file(tmp_path):
    directory = _secure_dir(tmp_path)
    sink = SecureJSONLCaptureSink(
        directory,
        CapturePolicy(sample_rate=0.01),
        random_value=lambda: 0.5,
    )
    assert not sink.capture(
        model="m",
        output="secret",
        telemetry=RepairTelemetry("ambiguous", "none", 0),
    )
    assert list(directory.iterdir()) == []


def test_rejects_insecure_directory(tmp_path):
    directory = tmp_path / "captures"
    directory.mkdir(mode=0o755)
    directory.chmod(0o755)
    with pytest.raises(ValueError, match="0700"):
        SecureJSONLCaptureSink(directory)


def test_missing_directory_raises_actionable_value_error(tmp_path):
    requirement = (
        "--structured-output-repair-capture-dir must exist, be owned by the "
        "router's uid, and have mode 0700"
    )
    with pytest.raises(ValueError, match=requirement):
        SecureJSONLCaptureSink(tmp_path / "missing")


def test_purges_expired_capture_files(tmp_path):
    directory = _secure_dir(tmp_path)
    expired = directory / "structured-output-20200101.jsonl"
    expired.write_text("{}\n")
    expired.chmod(0o600)
    os.utime(expired, (1.0, 1.0))
    sink = SecureJSONLCaptureSink(
        directory,
        CapturePolicy(sample_rate=1.0, retention_days=7),
        random_value=lambda: 0.0,
        wall_time=lambda: _NOW,
    )
    assert sink.capture(
        model="m",
        output="{{secret}",
        telemetry=RepairTelemetry("unknown", "none", 0),
    )
    assert not expired.exists()


def test_aggregate_byte_cap_silently_drops_record_across_days(tmp_path, monkeypatch):
    directory = _secure_dir(tmp_path)
    wall_time = [_NOW - 86_400]
    sink = SecureJSONLCaptureSink(
        directory,
        CapturePolicy(sample_rate=1.0),
        random_value=lambda: 0.0,
        wall_time=lambda: wall_time[0],
    )
    event = RepairTelemetry("ambiguous", "none", 0)

    assert sink.capture(model="m", output="secret", telemetry=event)
    first_day = directory / "structured-output-20240702.jsonl"
    record_bytes = first_day.stat().st_size
    monkeypatch.setattr(capture_module, "_MAX_TOTAL_BYTES", record_bytes * 3)

    wall_time[0] = _NOW
    assert sink.capture(model="m", output="secret", telemetry=event)
    assert sink.capture(model="m", output="secret", telemetry=event)
    second_day = directory / f"structured-output-{_DAY}.jsonl"
    size_at_cap = second_day.stat().st_size

    assert first_day.stat().st_size + size_at_cap == record_bytes * 3
    assert not sink.capture(model="m", output="secret", telemetry=event)
    assert second_day.stat().st_size == size_at_cap


def test_write_error_is_silent(tmp_path):
    directory = _secure_dir(tmp_path)
    capture_path = directory / f"structured-output-{_DAY}.jsonl"
    capture_path.write_text("unchanged\n")
    capture_path.chmod(0o400)
    sink = SecureJSONLCaptureSink(
        directory,
        CapturePolicy(sample_rate=1.0),
        random_value=lambda: 0.0,
        wall_time=lambda: _NOW,
    )

    assert not sink.capture(
        model="m",
        output="secret",
        telemetry=RepairTelemetry("unknown", "none", 0),
    )
    assert capture_path.read_text() == "unchanged\n"


def test_directory_permission_change_is_silent(tmp_path):
    directory = _secure_dir(tmp_path)
    sink = SecureJSONLCaptureSink(
        directory,
        CapturePolicy(sample_rate=1.0),
        random_value=lambda: 0.0,
    )
    directory.chmod(0o755)
    try:
        assert not sink.capture(
            model="m",
            output="secret",
            telemetry=RepairTelemetry("unknown", "none", 0),
        )
    finally:
        directory.chmod(0o700)
    assert not (directory / capture_module._LOCK_NAME).exists()


def test_capture_file_symlink_is_purged_via_public_capture(tmp_path):
    directory = _secure_dir(tmp_path)
    target = tmp_path / "served-file"
    target.write_text("unchanged")
    capture_path = directory / "structured-output-20240703.jsonl"
    capture_path.symlink_to(target)
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    process = context.Process(
        target=_capture_twice_in_process,
        args=(directory, results),
    )

    process.start()
    completed = _finish_process_promptly(process, 1.5)

    assert completed, "capture() blocked while purging a capture symlink"
    captures, elapsed = results.get(timeout=0.5)
    assert captures == [True, True]
    assert elapsed < 0.9
    assert target.read_text() == "unchanged"
    assert not capture_path.is_symlink()
    assert capture_path.is_file()


def test_untrusted_status_and_mode_cannot_select_a_filename(tmp_path):
    directory = _secure_dir(tmp_path)
    sink = SecureJSONLCaptureSink(
        directory,
        CapturePolicy(sample_rate=1.0),
        random_value=lambda: 0.0,
        wall_time=lambda: _NOW,
    )
    assert not sink.capture(
        model="m",
        output="secret",
        telemetry=RepairTelemetry("../../escape", "../../escape", 0),
    )
    assert sink.capture(
        model="m",
        output="secret",
        telemetry=RepairTelemetry("unknown", "../../escape", 0),
    )

    capture_file = next(directory.glob("structured-output-*.jsonl"))
    assert json.loads(capture_file.read_text())["mode"] == "other"
    assert not (tmp_path / "escape").exists()


def test_held_lock_returns_false_promptly(tmp_path):
    directory = _secure_dir(tmp_path)
    lock_path = directory / capture_module._LOCK_NAME
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    process = context.Process(
        target=_capture_once_in_process,
        args=(directory, results),
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        process.start()
        completed = _finish_process_promptly(process, 1.5)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    assert completed, "capture() blocked on a held flock"
    captured, elapsed = results.get(timeout=0.5)
    assert captured is False
    assert elapsed < 0.9


def test_fifo_capture_path_is_purged_via_public_capture(tmp_path):
    directory = _secure_dir(tmp_path)
    capture_path = directory / f"structured-output-{_DAY}.jsonl"
    os.mkfifo(capture_path, mode=0o600)
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    process = context.Process(
        target=_capture_twice_in_process,
        args=(directory, results),
    )

    process.start()
    completed = _finish_process_promptly(process, 1.5)

    assert completed, "capture() blocked while purging a capture FIFO"
    captures, elapsed = results.get(timeout=0.5)
    assert captures == [True, True]
    assert elapsed < 0.9
    assert capture_path.is_file()
    assert stat.S_ISREG(capture_path.lstat().st_mode)


def test_fifo_lock_path_returns_false_promptly_and_is_purged(tmp_path):
    directory = _secure_dir(tmp_path)
    lock_path = directory / capture_module._LOCK_NAME
    os.mkfifo(lock_path, mode=0o600)
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    process = context.Process(
        target=_capture_twice_in_process,
        args=(directory, results),
    )

    process.start()
    completed = _finish_process_promptly(process, 1.5)

    assert completed, "capture() blocked while opening a lock FIFO"
    captures, elapsed = results.get(timeout=0.5)
    assert captures == [False, True]
    assert elapsed < 0.9
    assert lock_path.is_file()
    assert stat.S_ISREG(lock_path.lstat().st_mode)


def test_large_output_redacts_only_a_bounded_prefix_and_returns_promptly(
    tmp_path, monkeypatch
):
    directory = _secure_dir(tmp_path)
    policy = CapturePolicy(sample_rate=1.0)
    sink = SecureJSONLCaptureSink(
        directory,
        policy,
        random_value=lambda: 0.0,
        wall_time=lambda: _NOW,
    )
    original_redactor = capture_module._REDACT_RUN
    redacted_lengths = []

    class ObservedRedactor:
        def sub(self, replacement, value):
            redacted_lengths.append(len(value))
            return original_redactor.sub(replacement, value)

    monkeypatch.setattr(capture_module, "_REDACT_RUN", ObservedRedactor())
    output = "s" * (8 * 1024 * 1024)

    started = time.monotonic()
    assert sink.capture(
        model="m",
        output=output,
        telemetry=RepairTelemetry("unknown", "none", 0),
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.9
    assert redacted_lengths == [
        policy.max_bytes * capture_module._REDACTION_PREFIX_FACTOR
    ]
    capture_file = next(directory.glob("structured-output-*.jsonl"))
    assert capture_file.stat().st_size <= (
        policy.max_bytes + capture_module._RECORD_OVERHEAD_BYTES
    )


@pytest.mark.parametrize("whitespace", ["\n", "\t", "\u3000"])
def test_whitespace_heavy_output_is_captured_within_serialized_cap(
    tmp_path, whitespace
):
    directory = _secure_dir(tmp_path)
    policy = CapturePolicy(sample_rate=1.0)
    sink = SecureJSONLCaptureSink(
        directory,
        policy,
        random_value=lambda: 0.0,
        wall_time=lambda: _NOW,
    )

    assert sink.capture(
        model="m",
        output=whitespace * 10_000,
        telemetry=RepairTelemetry("unknown", "none", 0),
    )

    capture_file = next(directory.glob("structured-output-*.jsonl"))
    stored = capture_file.read_bytes()
    assert len(stored) <= policy.max_bytes + capture_module._RECORD_OVERHEAD_BYTES
    assert json.loads(stored)["redacted_output"]


def test_concurrent_writers_never_produce_torn_jsonl(tmp_path):
    directory = _secure_dir(tmp_path)
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_capture_writer,
            args=(directory, start, results, writer),
        )
        for writer in range(2)
    ]
    for process in processes:
        process.start()
    start.set()

    completed = [_finish_process_promptly(process, 3.0) for process in processes]
    assert all(completed), "concurrent capture writers exceeded the hard timeout"
    counts = [results.get(timeout=0.5) for _ in processes]
    assert all(count > 0 for count in counts)

    stored = next(directory.glob("structured-output-*.jsonl")).read_bytes()
    assert stored.endswith(b"\n")
    lines = stored.splitlines()
    assert len(lines) == sum(counts)
    assert all(json.loads(line)["model"] in {"m0", "m1"} for line in lines)
