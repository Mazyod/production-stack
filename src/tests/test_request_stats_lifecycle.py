import threading

import pytest

from vllm_router.stats.request_stats import (
    RequestHandle,
    RequestStatsMonitor,
    SingletonMeta,
)


ENGINE = "http://engine"


@pytest.fixture
def monitor():
    SingletonMeta._instances.pop(RequestStatsMonitor, None)
    instance = RequestStatsMonitor(60.0)
    yield instance
    SingletonMeta._instances.pop(RequestStatsMonitor, None)


def _stats(monitor):
    return monitor.get_request_stats(10.0)[ENGINE]


def _assert_no_per_request_state(monitor):
    assert monitor.request_start_time == {}
    assert monitor.first_token_time == {}
    assert monitor._active_requests == {}


def test_aborting_prefill_does_not_steal_concurrent_decoding_count(monitor):
    prefill_handle = monitor.on_new_request(ENGINE, "request-a", 1.0)
    decoding_handle = monitor.on_new_request(ENGINE, "request-b", 1.0)
    monitor.on_request_response(decoding_handle, 2.0)

    monitor.on_request_abort(prefill_handle, 3.0)

    stats = _stats(monitor)
    assert stats.in_prefill_requests == 0
    assert stats.in_decoding_requests == 1
    monitor.on_request_complete(decoding_handle, 4.0)
    assert _stats(monitor).in_decoding_requests == 0


def test_complete_directly_from_prefill(monitor):
    handle = monitor.on_new_request(ENGINE, "empty-response", 1.0)

    monitor.on_request_complete(handle, 2.0)

    stats = _stats(monitor)
    assert stats.in_prefill_requests == 0
    assert stats.in_decoding_requests == 0
    assert stats.finished_requests == 1
    _assert_no_per_request_state(monitor)


def test_duplicate_response_is_idempotent(monitor):
    handle = monitor.on_new_request(ENGINE, "request", 1.0)

    monitor.on_request_response(handle, 2.0)
    monitor.on_request_response(handle, 5.0)

    stats = _stats(monitor)
    assert stats.in_prefill_requests == 0
    assert stats.in_decoding_requests == 1
    assert monitor.first_token_time == {handle: 2.0}
    assert monitor.ttft_monitors[ENGINE].values == pytest.approx([1.0])


def test_duplicate_and_unknown_finalization_are_idempotent(monitor):
    handle = monitor.on_new_request(ENGINE, "request", 1.0)
    monitor.on_request_response(handle, 2.0)

    monitor.on_request_complete(handle, 3.0)
    monitor.on_request_complete(handle, 4.0)
    monitor.on_request_complete(RequestHandle(999_999), 4.0)

    stats = _stats(monitor)
    assert stats.in_prefill_requests == 0
    assert stats.in_decoding_requests == 0
    assert stats.finished_requests == 1
    _assert_no_per_request_state(monitor)


@pytest.mark.parametrize(
    ("terminal_method", "expected_finished"),
    [
        ("on_request_complete", 1),
        ("on_request_fail", 0),
        ("on_request_abort", 0),
    ],
)
def test_all_terminal_outcomes_pop_per_request_state(
    monitor, terminal_method, expected_finished
):
    handle = monitor.on_new_request(ENGINE, "request", 1.0)
    monitor.on_request_response(handle, 2.0)

    getattr(monitor, terminal_method)(handle, 3.0)

    stats = _stats(monitor)
    assert stats.in_prefill_requests == 0
    assert stats.in_decoding_requests == 0
    assert stats.finished_requests == expected_finished
    _assert_no_per_request_state(monitor)


def test_same_external_request_id_gets_independent_handles(monitor):
    first = monitor.on_new_request(ENGINE, "shared-id", 1.0)
    second = monitor.on_new_request(ENGINE, "shared-id", 1.5)

    assert first != second
    monitor.on_request_response(first, 2.0)
    monitor.on_request_response(second, 2.5)
    monitor.on_request_complete(first, 3.0)

    stats = _stats(monitor)
    assert stats.in_decoding_requests == 1
    assert stats.finished_requests == 1

    monitor.on_request_complete(second, 4.0)
    stats = _stats(monitor)
    assert stats.in_decoding_requests == 0
    assert stats.finished_requests == 2
    _assert_no_per_request_state(monitor)


def test_monitor_uses_reentrant_thread_lock_for_hooks_and_snapshots(monitor):
    assert isinstance(monitor._lock, type(threading.RLock()))

    handle = monitor.on_new_request(ENGINE, "request", 1.0)
    with monitor._lock:
        monitor.on_request_response(handle, 2.0)
        assert monitor.get_request_stats(2.0)[ENGINE].in_decoding_requests == 1
        monitor.on_request_complete(handle, 3.0)

    _assert_no_per_request_state(monitor)
