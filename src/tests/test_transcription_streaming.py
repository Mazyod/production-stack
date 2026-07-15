"""
Tests for transcription streaming path in proxy_multipart_request:

1. stream=True returns a StreamingResponse that proxies SSE chunks.
2. stream=False (default) returns a JSONResponse (backward compatible).
3. Stats hooks are called correctly for streaming transcription.
4. Upstream response closed after normal completion.
5. Upstream response closed when consumer aborts mid-stream (aclose).
6. Connection failure before headers updates request stats.
"""

import asyncio
import gc
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aiohttp import FormData

from vllm_router.routers.routing_logic import RoundRobinRouter
from vllm_router.stats.request_stats import (
    RequestStatsMonitor,
)
from vllm_router.stats.request_stats import SingletonMeta as RequestStatsSingletonMeta
from vllm_router.utils import SingletonABCMeta


class EndpointInfo:
    def __init__(self, url, model_names=None, sleep=False, Id=None):
        self.url = url
        self.model_names = model_names or ["whisper-model"]
        self.sleep = sleep
        self.Id = Id


ENDPOINTS = [EndpointInfo(url="http://whisper-engine")]


@pytest.fixture(autouse=True)
def cleanup_singletons():
    RequestStatsSingletonMeta._instances.pop(RequestStatsMonitor, None)
    yield
    RequestStatsSingletonMeta._instances.pop(RequestStatsMonitor, None)
    for cls in list(SingletonABCMeta._instances.keys()):
        del SingletonABCMeta._instances[cls]


@pytest.fixture
def setup_mocks():
    sd = MagicMock()
    sd.get_endpoint_info.return_value = ENDPOINTS
    sd.aliases = None

    with patch(
        "vllm_router.services.request_service.request.get_service_discovery",
        return_value=sd,
    ):
        yield sd


def _make_mock_request():
    router = RoundRobinRouter()
    router.max_instance_failover_reroute_attempts = 0

    state = MagicMock()
    state.router = router
    state.engine_stats_scraper.get_engine_stats.return_value = {}
    state.request_stats_monitor = RequestStatsMonitor(60.0)
    state.otel_enabled = False
    state.semantic_cache_available = False
    state.callbacks = None

    req = MagicMock()
    req.headers = {
        "content-type": "multipart/form-data",
        "authorization": "Bearer test-key",
    }
    req.query_params = {}
    req.method = "POST"
    req.url = "http://router/v1/audio/transcriptions"
    req.app.state = state

    return req


def _assert_stats(req, *, prefill=0, decoding=0, finished=0):
    monitor = req.app.state.request_stats_monitor
    stats = monitor.get_request_stats(1.0)["http://whisper-engine"]
    assert stats.in_prefill_requests == prefill
    assert stats.in_decoding_requests == decoding
    assert stats.finished_requests == finished
    if prefill == 0 and decoding == 0:
        assert monitor.request_start_time == {}
        assert monitor.first_token_time == {}
        assert monitor._active_requests == {}


async def _run_asyncgen_finalizers():
    gc.collect()
    for _ in range(3):
        await asyncio.sleep(0)


def _make_backend_response(chunks, content_type="text/event-stream", status=200):
    async def iter_any():
        for c in chunks:
            yield c

    content = MagicMock()
    content.iter_any = iter_any

    resp = MagicMock()
    resp.status = status
    resp.headers = {"content-type": content_type, "x-request-id": "test-123"}
    resp.content = content
    resp.close = MagicMock()
    return resp


@pytest.mark.asyncio
async def test_transcription_streaming_returns_streaming_response(setup_mocks):
    req = _make_mock_request()

    mock_backend_response = _make_backend_response(
        [b'data: {"text": "Hello"}\n\n', b'data: {"text": " World"}\n\n']
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_backend_response)
    req.app.state.aiohttp_client_wrapper = MagicMock(return_value=mock_client)

    from vllm_router.services.request_service.request import proxy_multipart_request

    form_data = FormData()
    form_data.add_field(
        "file", b"fake-audio", filename="test.wav", content_type="audio/wav"
    )
    form_data.add_field("model", "whisper-model")
    form_data.add_field("stream", "true")

    resp = await proxy_multipart_request(
        form_data, "whisper-model", "/v1/audio/transcriptions", req, stream=True
    )

    from fastapi.responses import StreamingResponse

    assert isinstance(resp, StreamingResponse)
    assert resp.status_code == 200
    assert resp.media_type == "text/event-stream"
    _assert_stats(req, prefill=1)
    await anext(resp.body_iterator)
    await resp.body_iterator.aclose()
    _assert_stats(req)


@pytest.mark.asyncio
async def test_transcription_non_streaming_returns_json_response(setup_mocks):
    req = _make_mock_request()

    mock_backend_response = MagicMock()
    mock_backend_response.status = 200
    mock_backend_response.headers = {
        "content-type": "application/json",
        "x-request-id": "test-123",
    }
    mock_backend_response.json = AsyncMock(
        return_value={"text": "Hello world transcription"}
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_backend_response)
    req.app.state.aiohttp_client_wrapper = MagicMock(return_value=mock_client)

    from vllm_router.services.request_service.request import proxy_multipart_request

    form_data = FormData()
    form_data.add_field(
        "file", b"fake-audio", filename="test.wav", content_type="audio/wav"
    )
    form_data.add_field("model", "whisper-model")

    resp = await proxy_multipart_request(
        form_data, "whisper-model", "/v1/audio/transcriptions", req, stream=False
    )

    from fastapi.responses import JSONResponse

    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 200
    assert resp.body == b'{"text":"Hello world transcription"}'
    _assert_stats(req, finished=1)
    mock_backend_response.close.assert_called_once()


@pytest.mark.asyncio
async def test_non_streaming_non_json_response_returns_502_and_completes(setup_mocks):
    req = _make_mock_request()
    parse_error = aiohttp.ContentTypeError(
        MagicMock(), (), message="unexpected content type"
    )
    backend_response = MagicMock()
    backend_response.status = 200
    backend_response.headers = {"content-type": "text/html"}
    backend_response.json = AsyncMock(side_effect=parse_error)
    backend_response.text = AsyncMock(return_value="<html>backend error</html>")
    backend_response.close = MagicMock()
    client = MagicMock()
    client.post = AsyncMock(return_value=backend_response)
    req.app.state.aiohttp_client_wrapper = MagicMock(return_value=client)

    from vllm_router.services.request_service.request import proxy_multipart_request

    response = await proxy_multipart_request(
        FormData(),
        "whisper-model",
        "/v1/audio/transcriptions",
        req,
        stream=False,
    )

    assert response is not None
    assert response.status_code == 502
    assert b"<html>backend error</html>" in response.body
    _assert_stats(req, finished=1)
    backend_response.close.assert_called_once()


@pytest.mark.asyncio
async def test_non_streaming_non_json_text_read_failure_returns_502_and_completes(
    setup_mocks,
):
    req = _make_mock_request()
    parse_error = aiohttp.ContentTypeError(
        MagicMock(), (), message="unexpected content type"
    )
    backend_response = MagicMock()
    backend_response.status = 200
    backend_response.headers = {"content-type": "text/html"}
    backend_response.json = AsyncMock(side_effect=parse_error)
    backend_response.text = AsyncMock(side_effect=aiohttp.ClientError("read failed"))
    backend_response.close = MagicMock()
    client = MagicMock()
    client.post = AsyncMock(return_value=backend_response)
    req.app.state.aiohttp_client_wrapper = MagicMock(return_value=client)

    from vllm_router.services.request_service.request import proxy_multipart_request

    response = await proxy_multipart_request(
        FormData(),
        "whisper-model",
        "/v1/audio/transcriptions",
        req,
        stream=False,
    )

    assert response is not None
    assert response.status_code == 502
    _assert_stats(req, finished=1)
    backend_response.close.assert_called_once()


@pytest.mark.asyncio
async def test_streaming_calls_stats_hooks(setup_mocks):
    req = _make_mock_request()

    mock_backend_response = _make_backend_response(
        [b'data: {"text": "Hello"}\n\n', b'data: {"text": " World"}\n\n']
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_backend_response)
    req.app.state.aiohttp_client_wrapper = MagicMock(return_value=mock_client)

    from vllm_router.services.request_service.request import proxy_multipart_request

    form_data = FormData()
    form_data.add_field(
        "file", b"fake-audio", filename="test.wav", content_type="audio/wav"
    )
    form_data.add_field("model", "whisper-model")
    form_data.add_field("stream", "true")

    resp = await proxy_multipart_request(
        form_data, "whisper-model", "/v1/audio/transcriptions", req, stream=True
    )

    _assert_stats(req, prefill=1)

    chunks = []
    async for chunk in resp.body_iterator:
        chunks.append(chunk)

    _assert_stats(req, finished=1)
    mock_backend_response.close.assert_called_once()


@pytest.mark.asyncio
async def test_streaming_preserves_content_type(setup_mocks):
    req = _make_mock_request()

    mock_backend_response = _make_backend_response(
        [b'data: {"text": "Hello"}\n\n'], content_type="application/x-ndjson"
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_backend_response)
    req.app.state.aiohttp_client_wrapper = MagicMock(return_value=mock_client)

    from vllm_router.services.request_service.request import proxy_multipart_request

    form_data = FormData()
    form_data.add_field(
        "file", b"fake-audio", filename="test.wav", content_type="audio/wav"
    )
    form_data.add_field("model", "whisper-model")
    form_data.add_field("stream", "true")

    resp = await proxy_multipart_request(
        form_data, "whisper-model", "/v1/audio/transcriptions", req, stream=True
    )

    assert resp.media_type == "application/x-ndjson"
    await anext(resp.body_iterator)
    await resp.body_iterator.aclose()


@pytest.mark.asyncio
async def test_streaming_uses_aiohttp_multipart_boundary(setup_mocks):
    req = _make_mock_request()

    mock_backend_response = _make_backend_response([b'data: {"text": "Hello"}\n\n'])

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_backend_response)
    req.app.state.aiohttp_client_wrapper = MagicMock(return_value=mock_client)

    from vllm_router.services.request_service.request import proxy_multipart_request

    form_data = FormData()
    form_data.add_field(
        "file", b"fake-audio", filename="test.wav", content_type="audio/wav"
    )
    form_data.add_field("model", "whisper-model")
    form_data.add_field("stream", "true")

    response = await proxy_multipart_request(
        form_data, "whisper-model", "/v1/audio/transcriptions", req, stream=True
    )

    call_args = mock_client.post.call_args
    assert call_args is not None
    kwargs = call_args[1] if call_args[1] else {}
    # Headers are forwarded (auth, request-id, etc.) but content-type must be absent
    # so aiohttp can generate the multipart/form-data boundary automatically.
    passed_headers = kwargs.get("headers") or {}
    assert "content-type" not in {k.lower() for k in passed_headers}
    await anext(response.body_iterator)
    await response.body_iterator.aclose()


@pytest.mark.asyncio
async def test_streaming_closes_upstream_on_full_consumption(setup_mocks):
    req = _make_mock_request()

    mock_backend_response = _make_backend_response(
        [b'data: {"text": "Hello"}\n\n', b'data: {"text": " World"}\n\n']
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_backend_response)
    req.app.state.aiohttp_client_wrapper = MagicMock(return_value=mock_client)

    from vllm_router.services.request_service.request import proxy_multipart_request

    form_data = FormData()
    form_data.add_field(
        "file", b"fake-audio", filename="test.wav", content_type="audio/wav"
    )
    form_data.add_field("model", "whisper-model")
    form_data.add_field("stream", "true")

    resp = await proxy_multipart_request(
        form_data, "whisper-model", "/v1/audio/transcriptions", req, stream=True
    )

    chunks = []
    async for chunk in resp.body_iterator:
        chunks.append(chunk)

    assert len(chunks) == 2
    mock_backend_response.close.assert_called_once()
    _assert_stats(req, finished=1)


@pytest.mark.asyncio
async def test_streaming_closes_upstream_on_consumer_abort(setup_mocks):
    """Downstream client disconnects mid-stream: body_iterator.aclose() must
    run the finally block and close the upstream response."""
    req = _make_mock_request()

    mock_backend_response = _make_backend_response(
        [
            b'data: {"text": "chunk1"}\n\n',
            b'data: {"text": "chunk2"}\n\n',
            b'data: {"text": "chunk3"}\n\n',
        ]
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_backend_response)
    req.app.state.aiohttp_client_wrapper = MagicMock(return_value=mock_client)

    from vllm_router.services.request_service.request import proxy_multipart_request

    form_data = FormData()
    form_data.add_field(
        "file", b"fake-audio", filename="test.wav", content_type="audio/wav"
    )
    form_data.add_field("model", "whisper-model")
    form_data.add_field("stream", "true")

    resp = await proxy_multipart_request(
        form_data, "whisper-model", "/v1/audio/transcriptions", req, stream=True
    )

    body_iter = resp.body_iterator
    first = await body_iter.__anext__()
    assert first == b'data: {"text": "chunk1"}\n\n'

    await body_iter.aclose()

    mock_backend_response.close.assert_called_once()
    _assert_stats(req)


@pytest.mark.asyncio
async def test_streaming_abandon_without_aclose_retires_decoding(setup_mocks):
    req = _make_mock_request()
    backend_response = _make_backend_response([b"first", b"unread"])
    client = MagicMock()
    client.post = AsyncMock(return_value=backend_response)
    req.app.state.aiohttp_client_wrapper = MagicMock(return_value=client)

    from vllm_router.services.request_service.request import proxy_multipart_request

    response = await proxy_multipart_request(
        FormData(), "whisper-model", "/v1/audio/transcriptions", req, stream=True
    )

    async def disconnect_after_first_chunk():
        async for chunk in response.body_iterator:
            assert chunk == b"first"
            raise OSError("client disconnected")

    with pytest.raises(OSError, match="client disconnected"):
        await disconnect_after_first_chunk()
    _assert_stats(req, decoding=1)

    response = None
    await _run_asyncgen_finalizers()

    _assert_stats(req)
    backend_response.close.assert_called_once()


@pytest.mark.asyncio
async def test_streaming_abandon_before_first_chunk_without_aclose_retires_prefill(
    setup_mocks,
):
    req = _make_mock_request()
    empty_chunk_read = asyncio.Event()

    async def iter_any():
        empty_chunk_read.set()
        yield b""
        await asyncio.Event().wait()

    backend_response = _make_backend_response([])
    backend_response.content.iter_any = iter_any
    client = MagicMock()
    client.post = AsyncMock(return_value=backend_response)
    req.app.state.aiohttp_client_wrapper = MagicMock(return_value=client)

    from vllm_router.services.request_service.request import proxy_multipart_request

    response = await proxy_multipart_request(
        FormData(), "whisper-model", "/v1/audio/transcriptions", req, stream=True
    )
    _assert_stats(req, prefill=1)

    async def disconnect_before_first_chunk():
        async for _ in response.body_iterator:
            raise OSError("client disconnected")

    consumer = asyncio.create_task(disconnect_before_first_chunk())
    await empty_chunk_read.wait()
    await asyncio.sleep(0)
    _assert_stats(req, prefill=1)

    if consumer.done():
        with pytest.raises(OSError, match="client disconnected"):
            await consumer
    else:
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer

    consumer = None
    response = None
    await _run_asyncgen_finalizers()

    _assert_stats(req)
    backend_response.close.assert_called_once()


@pytest.mark.asyncio
async def test_non_streaming_cancellation_during_json_read_aborts(setup_mocks):
    req = _make_mock_request()
    reading = asyncio.Event()

    async def read_json():
        reading.set()
        await asyncio.Event().wait()

    backend_response = MagicMock()
    backend_response.status = 200
    backend_response.headers = {"content-type": "application/json"}
    backend_response.json = read_json
    backend_response.close = MagicMock()
    client = MagicMock()
    client.post = AsyncMock(return_value=backend_response)
    req.app.state.aiohttp_client_wrapper = MagicMock(return_value=client)

    from vllm_router.services.request_service.request import proxy_multipart_request

    request_task = asyncio.create_task(
        proxy_multipart_request(
            FormData(),
            "whisper-model",
            "/v1/audio/transcriptions",
            req,
            stream=False,
        )
    )
    await reading.wait()
    _assert_stats(req, decoding=1)

    request_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request_task

    _assert_stats(req)
    backend_response.close.assert_called_once()


@pytest.mark.asyncio
async def test_streaming_stats_on_connection_failure(setup_mocks):
    req = _make_mock_request()

    import aiohttp

    err = aiohttp.ClientConnectorError(
        connection_key=MagicMock(), os_error=OSError("connection refused")
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(side_effect=err)
    req.app.state.aiohttp_client_wrapper = MagicMock(return_value=mock_client)

    from vllm_router.services.request_service.request import proxy_multipart_request

    form_data = FormData()
    form_data.add_field(
        "file", b"fake-audio", filename="test.wav", content_type="audio/wav"
    )
    form_data.add_field("model", "whisper-model")
    form_data.add_field("stream", "true")

    resp = await proxy_multipart_request(
        form_data, "whisper-model", "/v1/audio/transcriptions", req, stream=True
    )

    from fastapi.responses import JSONResponse

    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 503
    _assert_stats(req)


@pytest.mark.asyncio
async def test_streaming_close_after_first_chunk_aborts_decoding(setup_mocks):
    req = _make_mock_request()
    backend_response = _make_backend_response([b"unread"])
    client = MagicMock()
    client.post = AsyncMock(return_value=backend_response)
    req.app.state.aiohttp_client_wrapper = MagicMock(return_value=client)

    from vllm_router.services.request_service.request import proxy_multipart_request

    response = await proxy_multipart_request(
        FormData(), "whisper-model", "/v1/audio/transcriptions", req, stream=True
    )
    _assert_stats(req, prefill=1)

    assert await anext(response.body_iterator) == b"unread"
    _assert_stats(req, decoding=1)
    await response.body_iterator.aclose()

    _assert_stats(req)
    backend_response.close.assert_called_once()


@pytest.mark.asyncio
async def test_streaming_empty_body_completes_from_prefill(setup_mocks):
    req = _make_mock_request()
    backend_response = _make_backend_response([])
    client = MagicMock()
    client.post = AsyncMock(return_value=backend_response)
    req.app.state.aiohttp_client_wrapper = MagicMock(return_value=client)

    from vllm_router.services.request_service.request import proxy_multipart_request

    response = await proxy_multipart_request(
        FormData(), "whisper-model", "/v1/audio/transcriptions", req, stream=True
    )

    assert [chunk async for chunk in response.body_iterator] == []
    _assert_stats(req, finished=1)


@pytest.mark.asyncio
async def test_multipart_cancellation_during_connect_aborts_prefill(setup_mocks):
    req = _make_mock_request()
    waiting = asyncio.Event()

    async def post(*args, **kwargs):
        waiting.set()
        await asyncio.Event().wait()

    client = MagicMock()
    client.post = post
    req.app.state.aiohttp_client_wrapper = MagicMock(return_value=client)

    from vllm_router.services.request_service.request import proxy_multipart_request

    task = asyncio.create_task(
        proxy_multipart_request(
            FormData(),
            "whisper-model",
            "/v1/audio/transcriptions",
            req,
            stream=True,
        )
    )
    await waiting.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    _assert_stats(req)
