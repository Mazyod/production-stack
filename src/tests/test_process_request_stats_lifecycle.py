import asyncio
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from fastapi import HTTPException

from vllm_router.aiohttp_client import DEFAULT_BACKEND_CLIENT_TIMEOUT
from vllm_router.services.request_service.request import process_request
from vllm_router.stats.request_stats import (
    RequestStatsMonitor,
    SingletonMeta,
)

ENGINE = "http://engine"
REQUEST_BODY = b'{"model":"test-model","stream":true}'


@pytest.fixture
def monitor():
    SingletonMeta._instances.pop(RequestStatsMonitor, None)
    instance = RequestStatsMonitor(60.0)
    yield instance
    SingletonMeta._instances.pop(RequestStatsMonitor, None)


def _request(monitor, iter_any=None, *, enter_error=None):
    state = MagicMock()
    state.request_stats_monitor = monitor
    state.otel_enabled = False
    state.semantic_cache_available = False
    state.callbacks = None

    request = MagicMock()
    request.app.state = state
    request.method = "POST"
    request.headers = {"content-type": "application/json"}

    response = MagicMock()
    response.status = 200
    response.headers = {"content-type": "text/event-stream"}
    response.content.iter_any = iter_any

    context = MagicMock()
    if enter_error is None:
        context.__aenter__ = AsyncMock(return_value=response)
    else:
        context.__aenter__ = AsyncMock(side_effect=enter_error)
    context.__aexit__ = AsyncMock(return_value=False)

    client = MagicMock()
    client.request.return_value = context
    state.aiohttp_client_wrapper = MagicMock(return_value=client)
    return request


def _assert_drained(monitor, engine=ENGINE, *, finished):
    stats = monitor.get_request_stats(10.0)[engine]
    assert stats.in_prefill_requests == 0
    assert stats.in_decoding_requests == 0
    assert stats.finished_requests == finished
    assert monitor.request_start_time == {}
    assert monitor.first_token_time == {}
    assert monitor._active_requests == {}


def _generator(request, *, engine=ENGINE, request_id="request-id"):
    return process_request(
        request,
        REQUEST_BODY,
        engine,
        request_id,
        "/v1/chat/completions",
        MagicMock(),
    )


@pytest.mark.asyncio
async def test_normal_full_exhaustion_drains_all_request_state(monitor):
    async def iter_any():
        yield b"first"
        yield b"second"

    generator = _generator(_request(monitor, iter_any))

    assert [item async for item in generator][1:] == [b"first", b"second"]
    _assert_drained(monitor, finished=1)


@pytest.mark.asyncio
async def test_aclose_after_headers_aborts_prefill(monitor):
    async def iter_any():
        yield b"unread"

    generator = _generator(_request(monitor, iter_any))
    assert (await anext(generator))[1] == 200

    await generator.aclose()

    _assert_drained(monitor, finished=0)


@pytest.mark.asyncio
async def test_aclose_after_first_body_chunk_aborts_decoding(monitor):
    async def iter_any():
        yield b"first"
        yield b"unread"

    generator = _generator(_request(monitor, iter_any))
    await anext(generator)
    assert await anext(generator) == b"first"

    await generator.aclose()

    _assert_drained(monitor, finished=0)


@pytest.mark.asyncio
async def test_cancellation_while_waiting_for_first_chunk_aborts_prefill(monitor):
    waiting = asyncio.Event()
    blocker = asyncio.Event()

    async def iter_any():
        waiting.set()
        await blocker.wait()
        yield b"unreachable"

    async def consume():
        async for _ in _generator(_request(monitor, iter_any)):
            pass

    task = asyncio.create_task(consume())
    await waiting.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    _assert_drained(monitor, finished=0)


@pytest.mark.asyncio
async def test_cancellation_while_waiting_for_next_chunk_aborts_decoding(monitor):
    waiting = asyncio.Event()
    blocker = asyncio.Event()

    async def iter_any():
        yield b"first"
        waiting.set()
        await blocker.wait()
        yield b"unreachable"

    async def consume():
        async for _ in _generator(_request(monitor, iter_any)):
            pass

    task = asyncio.create_task(consume())
    await waiting.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    _assert_drained(monitor, finished=0)


@pytest.mark.asyncio
async def test_backend_connect_failure_drains_prefill(monitor):
    generator = _generator(
        _request(monitor, enter_error=ConnectionError("connection refused"))
    )

    with pytest.raises(ConnectionError, match="connection refused"):
        await anext(generator)

    _assert_drained(monitor, finished=0)


@pytest.mark.asyncio
async def test_backend_body_iteration_failure_drains_decoding(monitor):
    async def iter_any():
        yield b"first"
        raise RuntimeError("body failed")

    generator = _generator(_request(monitor, iter_any))
    await anext(generator)
    assert await anext(generator) == b"first"

    with pytest.raises(RuntimeError, match="body failed"):
        await anext(generator)

    _assert_drained(monitor, finished=0)


@pytest.mark.asyncio
async def test_empty_backend_body_completes_from_prefill(monitor):
    async def iter_any():
        if False:
            yield b""

    generator = _generator(_request(monitor, iter_any))

    assert [item async for item in generator][0][1] == 200
    _assert_drained(monitor, finished=1)


@pytest.mark.asyncio
async def test_socket_read_timeout_mid_body_drains_decoding_as_failure(monitor):
    async def iter_any():
        yield b"first"
        raise aiohttp.SocketTimeoutError("no bytes from backend")

    generator = _generator(_request(monitor, iter_any))
    await anext(generator)
    assert await anext(generator) == b"first"

    with pytest.raises(aiohttp.SocketTimeoutError):
        await anext(generator)

    _assert_drained(monitor, finished=0)


@pytest.mark.asyncio
async def test_entry_deadline_expiry_retires_as_failure_not_abort(monitor):
    """A connect/upload/header-wait stall breaching the entry deadline is a
    backend failure: it must surface as TimeoutError (so the failover loop
    retries it), and the attempt must retire through on_request_fail even
    though the deadline is delivered as a CancelledError."""

    async def hang_forever():
        await asyncio.Event().wait()

    request = _request(monitor)
    context = request.app.state.aiohttp_client_wrapper.return_value.request.return_value
    context.__aenter__ = AsyncMock(side_effect=hang_forever)
    request.app.state.backend_client_timeout = aiohttp.ClientTimeout(
        total=None, sock_connect=0.05, sock_read=0.05
    )

    outcomes = []
    original_fail = monitor.on_request_fail
    original_abort = monitor.on_request_abort
    monitor.on_request_fail = lambda *a, **kw: (
        outcomes.append("fail"),
        original_fail(*a, **kw),
    )
    monitor.on_request_abort = lambda *a, **kw: (
        outcomes.append("abort"),
        original_abort(*a, **kw),
    )

    generator = _generator(request)
    with pytest.raises(TimeoutError):
        await anext(generator)

    assert outcomes == ["fail"]
    _assert_drained(monitor, finished=0)


@pytest.mark.asyncio
async def test_backend_request_uses_the_wired_client_timeout(monitor):
    async def iter_any():
        yield b"only"

    configured = aiohttp.ClientTimeout(total=None, sock_connect=1.0, sock_read=2.0)
    request = _request(monitor, iter_any)
    request.app.state.backend_client_timeout = configured

    async for _ in _generator(request):
        pass

    client = request.app.state.aiohttp_client_wrapper.return_value
    assert client.request.call_args.kwargs["timeout"] is configured


@pytest.mark.asyncio
async def test_backend_request_defaults_socket_timeouts_when_unwired(monitor):
    async def iter_any():
        yield b"only"

    request = _request(monitor, iter_any)
    del request.app.state.backend_client_timeout

    async for _ in _generator(request):
        pass

    client = request.app.state.aiohttp_client_wrapper.return_value
    assert client.request.call_args.kwargs["timeout"] is DEFAULT_BACKEND_CLIENT_TIMEOUT


@pytest.mark.asyncio
async def test_body_validation_failure_does_not_register_request(monitor):
    request = _request(monitor)
    generator = process_request(
        request,
        b"not-json",
        ENGINE,
        "request-id",
        "/v1/chat/completions",
        MagicMock(),
    )

    with pytest.raises(HTTPException):
        await anext(generator)

    assert monitor.get_request_stats(10.0) == {}
    assert monitor.request_start_time == {}
