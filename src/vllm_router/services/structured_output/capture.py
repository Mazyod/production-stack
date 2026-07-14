"""Secure, bounded diagnostic captures for refused repair outcomes."""

from __future__ import annotations

import fcntl
import json
import os
import random
import re
import stat
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from vllm_router.services.structured_output.transform import RepairTelemetry

_CAPTURE_NAME = re.compile(r"structured-output-\d{8}\.jsonl\Z")
_REDACT_RUN = re.compile(r"[^{}\[\]:,`\s]+")
_CAPTURE_STATUSES = frozenset({"ambiguous", "unknown"})
_CAPTURE_MODES = frozenset({"none", "code_fence", "extra_brace", "dup_prefix", "other"})
_LOCK_NAME = ".structured-output-capture.lock"
_MAX_MODEL_BYTES = 256
_MAX_TOTAL_BYTES = 64 * 1024 * 1024
_RECORD_OVERHEAD_BYTES = 2_048
_REDACTION_PREFIX_FACTOR = 4


@dataclass(frozen=True)
class CapturePolicy:
    sample_rate: float = 0.01
    max_bytes: int = 4096
    retention_days: int = 7

    def __post_init__(self) -> None:
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError("sample_rate must be between 0 and 1")
        if self.max_bytes <= 0 or self.retention_days <= 0:
            raise ValueError("capture limits must be positive")


def _bound_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")[:max_bytes]
    return encoded.decode("utf-8", errors="ignore")


def _redact_and_bound(output: str, max_bytes: int) -> str:
    prefix = output[: max_bytes * _REDACTION_PREFIX_FACTOR]
    redacted = _REDACT_RUN.sub("<redacted>", prefix)
    return _bound_utf8(redacted, max_bytes)


def _serialize_record(
    record: dict[str, object], redacted_output: str, max_bytes: int
) -> bytes | None:
    """Serialize a record containing only the bounded, redacted output excerpt."""

    def encode(excerpt: str) -> bytes:
        candidate = {
            **record,
            "redacted_output": excerpt,
        }
        return json.dumps(candidate, separators=(",", ":")).encode("utf-8") + b"\n"

    encoded = encode(redacted_output)
    if len(encoded) <= max_bytes:
        return encoded

    excerpt_bytes = redacted_output.encode("utf-8")
    low = 0
    high = len(excerpt_bytes)
    best: bytes | None = None
    while low <= high:
        midpoint = (low + high) // 2
        excerpt = excerpt_bytes[:midpoint].decode("utf-8", errors="ignore")
        candidate = encode(excerpt)
        if len(candidate) <= max_bytes:
            best = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best


class SecureJSONLCaptureSink:
    def __init__(
        self,
        directory: str | Path,
        policy: CapturePolicy | None = None,
        *,
        random_value: Callable[[], float] = random.random,
        wall_time: Callable[[], float] = time.time,
    ) -> None:
        self._directory = Path(directory)
        self._policy = policy or CapturePolicy()
        self._random_value = random_value
        self._wall_time = wall_time
        self._validate_directory()

    def _validate_directory(self) -> None:
        requirement = (
            "--structured-output-repair-capture-dir must exist, be owned by the "
            "router's uid, and have mode 0700"
        )
        try:
            stat_result = self._directory.lstat()
        except FileNotFoundError as exc:
            raise ValueError(requirement) from exc
        if (
            not stat.S_ISDIR(stat_result.st_mode)
            or stat_result.st_uid != os.geteuid()
            or stat_result.st_mode & 0o777 != 0o700
        ):
            raise ValueError(requirement)

    def _capture_paths(self) -> Iterator[Path]:
        for path in self._directory.iterdir():
            if _CAPTURE_NAME.fullmatch(path.name):
                yield path

    def _purge_expired(self, now: float) -> None:
        cutoff = now - self._policy.retention_days * 86_400
        for path in self._capture_paths():
            try:
                stat_result = path.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(stat_result.st_mode):
                path.unlink()
            elif stat_result.st_mtime < cutoff:
                path.unlink()

        self._unlink_non_regular_lock()

    def _unlink_non_regular_lock(self) -> None:
        lock_path = self._directory / _LOCK_NAME
        try:
            stat_result = lock_path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISREG(stat_result.st_mode):
            lock_path.unlink()

    def _within_byte_limit(self, record_bytes: int) -> bool:
        total = 0
        for path in self._capture_paths():
            try:
                stat_result = os.stat(path, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISREG(stat_result.st_mode):
                total += stat_result.st_size
        return total + record_bytes <= _MAX_TOTAL_BYTES

    @contextmanager
    def _locked(self) -> Iterator[bool]:
        flags = os.O_CREAT | os.O_RDWR | os.O_NONBLOCK
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor: int | None = None
        try:
            descriptor = os.open(self._directory / _LOCK_NAME, flags, 0o600)
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
                raise OSError("capture lock is not a private regular file")
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            if descriptor is not None:
                os.close(descriptor)
            self._unlink_non_regular_lock()
            yield False
            return

        try:
            yield True
        finally:
            os.close(descriptor)

    def _write_record(self, path: Path, encoded: bytes) -> None:
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | os.O_NONBLOCK
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
                raise OSError("capture target is not a private regular file")
            os.fchmod(descriptor, 0o600)
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("capture write made no progress")
                view = view[written:]
        finally:
            os.close(descriptor)

    def capture(
        self,
        *,
        model: str,
        output: str,
        telemetry: RepairTelemetry,
    ) -> bool:
        if telemetry.status not in _CAPTURE_STATUSES:
            return False

        try:
            if self._random_value() >= self._policy.sample_rate:
                return False

            now = self._wall_time()
            output_bytes = len(output.encode("utf-8"))
            safe_telemetry = RepairTelemetry(
                telemetry.status,
                telemetry.mode if telemetry.mode in _CAPTURE_MODES else "other",
                max(0, int(telemetry.garbage_prefix_bytes)),
            )
            record = {
                "captured_at": datetime.fromtimestamp(now, UTC).isoformat(),
                "model": _bound_utf8(str(model), _MAX_MODEL_BYTES),
                **asdict(safe_telemetry),
                "output_bytes": output_bytes,
            }
            encoded = _serialize_record(
                record,
                _redact_and_bound(output, self._policy.max_bytes),
                self._policy.max_bytes + _RECORD_OVERHEAD_BYTES,
            )
            if encoded is None:
                return False

            day = datetime.fromtimestamp(now, UTC).strftime("%Y%m%d")
            path = self._directory / f"structured-output-{day}.jsonl"
            self._validate_directory()
            with self._locked() as locked:
                if not locked:
                    return False
                self._purge_expired(now)
                if not self._within_byte_limit(len(encoded)):
                    return False
                self._write_record(path, encoded)
            return True
        except Exception:  # noqa: BLE001 - diagnostics never affect output
            return False
