import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

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
