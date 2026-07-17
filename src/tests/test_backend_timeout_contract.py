"""Structured 504/502 contract for backend timeouts (CR-router-504-timeout-contract).

Pre-headers: read/entry timeouts short-circuit to a structured 504 without
burning a backend-retry attempt; connect-phase failures rotate backends and
exhaust into a structured 502. Mid-stream on SSE: an in-band terminal error
event + ``data: [DONE]`` instead of a bare connection abort.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from vllm_router.aiohttp_client import build_backend_client_timeout
from vllm_router.routers.routing_logic import RoundRobinRouter
from vllm_router.stats.request_stats import RequestStatsMonitor
from vllm_router.stats.request_stats import SingletonMeta as RequestStatsSingletonMeta
from vllm_router.utils import SingletonABCMeta


class EndpointInfo:
    def __init__(self, url, model_names=None):
        self.url = url
        self.model_names = model_names or ["test-model"]
        self.sleep = False
        self.Id = None


@pytest.fixture(autouse=True)
def cleanup_singletons():
    RequestStatsSingletonMeta._instances.pop(RequestStatsMonitor, None)
    yield
    RequestStatsSingletonMeta._instances.pop(RequestStatsMonitor, None)
    for cls in list(SingletonABCMeta._instances.keys()):
        del SingletonABCMeta._instances[cls]


ENDPOINTS = [EndpointInfo(url="http://engine1"), EndpointInfo(url="http://engine2")]


@pytest.fixture
def setup():
    """Yield a request wired like production: two engines, one failover retry,
    and a real backend ClientTimeout (connect=2, read=3 -> entry deadline 5)."""
    router = RoundRobinRouter()
    router.max_instance_failover_reroute_attempts = 1

    sd = MagicMock()
    sd.get_endpoint_info.return_value = ENDPOINTS
    sd.aliases = None
    sd.has_ever_seen_model.return_value = True

    state = MagicMock()
    state.router = router
    state.engine_stats_scraper.get_engine_stats.return_value = {}
    state.request_stats_monitor = RequestStatsMonitor(60.0)
    state.otel_enabled = False
    state.semantic_cache_available = False
    state.structured_output_repair_enabled = False
    state.callbacks = None
    state.external_provider_registry = None
    state.backend_client_timeout = build_backend_client_timeout(2.0, 3.0)

    req = MagicMock()
    req.headers = {
        "content-type": "application/json",
        "X-Request-Id": "req-contract-1",
    }
    req.query_params = {}
    req.method = "POST"
    req.url = "http://router/v1/chat/completions"
    req.app.state = state

    async def body():
        return json.dumps({"model": "test-model", "stream": False}).encode()

    req.body = body

    patches = [
        patch(
            "vllm_router.services.request_service.request.get_service_discovery",
            return_value=sd,
        ),
        patch(
            "vllm_router.services.request_service.request.is_request_rewriter_initialized",
            return_value=False,
        ),
    ]
    for p in patches:
        p.start()
    yield req
    for p in patches:
        p.stop()


def _client_raising(req, error_factory):
    """Install an aiohttp-client mock whose request context raises on enter;
    returns the list of attempted backend URLs."""
    attempted = []

    def backend_request(*, url, **kwargs):
        attempted.append(url.removesuffix("/v1/chat/completions"))
        context = MagicMock()
        context.__aenter__ = AsyncMock(side_effect=error_factory(url))
        context.__aexit__ = AsyncMock(return_value=False)
        return context

    client = MagicMock()
    client.request.side_effect = backend_request
    req.app.state.aiohttp_client_wrapper = MagicMock(return_value=client)
    return attempted


def _error_body(response):
    return json.loads(response.body)["error"]


async def _route(req):
    from vllm_router.services.request_service.request import route_general_request

    return await route_general_request(req, "/v1/chat/completions", MagicMock())


def _assert_contract_headers(response):
    assert response.headers["x-request-id"] == "req-contract-1"
    assert response.headers["retry-after"] == "1"
    assert response.headers["content-type"] == "application/json"


READ_TIMEOUT_ENVELOPE = {
    "error": {
        "message": (
            "Upstream LLM backend timed out after 3s with no data received. "
            "The request was not completed and may be safely retried."
        ),
        "type": "gateway_timeout",
        "code": "backend_read_timeout",
        "param": None,
    }
}

ENTRY_TIMEOUT_ENVELOPE = {
    "error": {
        "message": (
            "Upstream LLM backend timed out after 5s before returning "
            "response headers. The request was not completed and may be "
            "safely retried."
        ),
        "type": "gateway_timeout",
        "code": "backend_entry_timeout",
        "param": None,
    }
}

CONNECT_ERROR_ENVELOPE = {
    "error": {
        "message": (
            "Could not establish a connection to the upstream LLM backend "
            "within the permitted failover attempts. The request was not "
            "completed and may be safely retried."
        ),
        "type": "bad_gateway",
        "code": "backend_connect_error",
        "param": None,
    }
}


@pytest.mark.asyncio
async def test_read_timeout_returns_504_without_burning_a_retry(setup):
    req = setup
    attempted = _client_raising(
        req, lambda url: aiohttp.SocketTimeoutError("read gap exceeded")
    )

    response = await _route(req)

    assert response.status_code == 504
    assert attempted == ["http://engine1"], "read timeout must not rotate backends"
    assert json.loads(response.body) == READ_TIMEOUT_ENVELOPE
    _assert_contract_headers(response)


@pytest.mark.asyncio
async def test_entry_timeout_returns_504_with_entry_code(setup):
    req = setup
    attempted = _client_raising(req, lambda url: TimeoutError("entry deadline"))

    response = await _route(req)

    assert response.status_code == 504
    assert attempted == ["http://engine1"]
    assert json.loads(response.body) == ENTRY_TIMEOUT_ENVELOPE
    _assert_contract_headers(response)


@pytest.mark.asyncio
async def test_connect_timeout_rotates_backends_then_returns_502(setup):
    req = setup
    attempted = _client_raising(
        req, lambda url: aiohttp.ConnectionTimeoutError(f"connect to {url}")
    )

    response = await _route(req)

    assert response.status_code == 502
    assert sorted(attempted) == ["http://engine1", "http://engine2"], (
        "connect failures must rotate before surfacing"
    )
    assert json.loads(response.body) == CONNECT_ERROR_ENVELOPE
    _assert_contract_headers(response)


@pytest.mark.asyncio
async def test_connect_refused_rotates_backends_then_returns_502(setup):
    req = setup

    def refused(url):
        return aiohttp.ClientConnectorError(
            MagicMock(), OSError(61, "Connection refused")
        )

    attempted = _client_raising(req, refused)

    response = await _route(req)

    assert response.status_code == 502
    assert sorted(attempted) == ["http://engine1", "http://engine2"]
    assert json.loads(response.body) == CONNECT_ERROR_ENVELOPE
    _assert_contract_headers(response)


@pytest.mark.asyncio
async def test_connect_failures_log_one_enriched_warning_per_attempt(setup):
    """One structured WARNING per attempt, each carrying the contract fields;
    exhaustion must not add a second line for the same timeout."""
    import vllm_router.services.request_service.request as request_module

    req = setup
    _client_raising(req, lambda url: aiohttp.ConnectionTimeoutError("connect"))

    with patch.object(request_module, "logger") as mock_logger:
        response = await _route(req)

    assert response.status_code == 502
    warnings = [str(call) for call in mock_logger.warning.call_args_list]
    assert len(warnings) == 2, warnings
    for line in warnings:
        assert "bound=connect" in line
        assert "req-contract-1" in line
        assert "test-model" in line
        assert "/v1/chat/completions" in line


@pytest.mark.asyncio
async def test_timeout_responses_retire_request_stats(setup):
    req = setup
    _client_raising(req, lambda url: aiohttp.SocketTimeoutError("read gap"))

    await _route(req)

    monitor = req.app.state.request_stats_monitor
    stats = monitor.get_request_stats(10.0)
    assert stats["http://engine1"].in_prefill_requests == 0
    assert stats["http://engine1"].finished_requests == 0
    assert monitor.request_start_time == {}
    assert monitor._active_requests == {}


@pytest.mark.asyncio
async def test_generic_exceptions_keep_the_existing_raise_path(setup):
    req = setup
    attempted = _client_raising(req, lambda url: ConnectionError("reset by peer"))

    with pytest.raises(ConnectionError):
        await _route(req)

    assert sorted(attempted) == ["http://engine1", "http://engine2"]
