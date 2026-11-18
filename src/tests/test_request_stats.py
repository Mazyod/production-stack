import json
import time
import types

import pytest

from vllm_router.services.request_service import request as request_module
from vllm_router.services.request_service.request import (
    process_request,
    proxy_multipart_request,
)
from vllm_router.stats.request_stats import RequestStatsMonitor, SingletonMeta

URL = "http://engine-1:8000"


@pytest.fixture
def monitor():
    SingletonMeta._instances.pop(RequestStatsMonitor, None)
    m = RequestStatsMonitor(sliding_window_size=10)
    yield m
    SingletonMeta._instances.pop(RequestStatsMonitor, None)


class _FakeContent:
    def __init__(self, chunks):
        self._chunks = chunks

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class _OkRequestCM:
    async def __aenter__(self):
        return types.SimpleNamespace(
            status=200,
            headers={"content-type": "application/json"},
            content=_FakeContent([b'{"usage": {}}']),
        )

    async def __aexit__(self, *exc):
        return False


def _fake_request(monitor, cm_factory):
    state = types.SimpleNamespace(
        otel_enabled=False,
        semantic_cache_available=False,
        request_stats_monitor=monitor,
        aiohttp_client_wrapper=lambda: types.SimpleNamespace(
            request=lambda **kw: cm_factory()
        ),
    )
    return types.SimpleNamespace(
        method="POST",
        headers={"content-type": "application/json"},
        app=types.SimpleNamespace(state=state),
    )


@pytest.mark.asyncio
async def test_process_request_counts_overlapping_requests_sharing_an_id(monitor):
    # Clients can reuse an X-Request-Id; two overlapping requests that share it
    # must each release their own slot.
    request = _fake_request(monitor, _OkRequestCM)
    body = json.dumps({"model": "test-model", "stream": False}).encode()
    gens = [
        process_request(request, body, URL, "req-1", "/v1/chat/completions", None)
        for _ in range(2)
    ]
    for _ in range(2):  # both dispatched, then both past their first chunk
        for gen in gens:
            await anext(gen)
    for gen in gens:
        async for _ in gen:
            pass

    stats = monitor.get_request_stats(time.time())
    assert stats[URL].in_prefill_requests == 0
    assert stats[URL].in_decoding_requests == 0
    assert stats[URL].finished_requests == 2


class _StreamingBackendResponse:
    status = 200
    headers = {"content-type": "text/event-stream"}

    def __init__(self):
        self.content = _FakeContent([b"data: 1", b"data: 2"])

    def close(self):
        pass


@pytest.mark.asyncio
async def test_multipart_request_counts_overlapping_requests_sharing_an_id(
    monitor, monkeypatch
):
    # Same as above for the multipart (audio/image) proxy.
    endpoint = types.SimpleNamespace(url=URL, model_names=["test-model"], sleep=False)
    monkeypatch.setattr(
        request_module,
        "get_service_discovery",
        lambda: types.SimpleNamespace(get_endpoint_info=lambda: [endpoint]),
    )

    async def post(*args, **kwargs):
        return _StreamingBackendResponse()

    state = types.SimpleNamespace(
        router=types.SimpleNamespace(route_request=lambda *args: URL),
        engine_stats_scraper=types.SimpleNamespace(get_engine_stats=lambda: {}),
        request_stats_monitor=monitor,
        aiohttp_client_wrapper=lambda: types.SimpleNamespace(post=post),
    )
    request = types.SimpleNamespace(
        headers={"X-Request-Id": "req-1"}, app=types.SimpleNamespace(state=state)
    )
    responses = [
        await proxy_multipart_request(
            b"form", "test-model", "/v1/audio/transcriptions", request, stream=True
        )
        for _ in range(2)
    ]
    bodies = [response.body_iterator for response in responses]
    for body in bodies:  # both past their first chunk
        await anext(body)
    for body in bodies:
        async for _ in body:
            pass

    stats = monitor.get_request_stats(time.time())
    assert stats[URL].in_prefill_requests == 0
    assert stats[URL].in_decoding_requests == 0
    assert stats[URL].finished_requests == 2
