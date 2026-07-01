"""
Tests for the /pooling proxy route.

Jina Embeddings v4 multi-vector (ColBERT) output is served by vLLM only on
/pooling. This route must be registered and must delegate to the shared
forwarder (route_general_request) with the "/pooling" endpoint, so backend
selection by the request body's "model" field behaves exactly like the other
proxied routes (/v1/embeddings, /score, /rerank).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm_router.routers.main_router import main_router, route_pooling


def test_pooling_route_is_registered():
    """The router app must expose POST /pooling."""
    routes = {
        (route.path, method)
        for route in main_router.routes
        for method in getattr(route, "methods", set())
    }
    assert ("/pooling", "POST") in routes


@pytest.mark.asyncio
async def test_pooling_delegates_to_forwarder_with_endpoint():
    """route_pooling must forward to route_general_request with "/pooling"."""
    request = MagicMock()
    background_tasks = MagicMock()
    sentinel = object()

    with patch(
        "vllm_router.routers.main_router.route_general_request",
        new=AsyncMock(return_value=sentinel),
    ) as forwarder:
        result = await route_pooling(request, background_tasks)

    forwarder.assert_awaited_once_with(request, "/pooling", background_tasks)
    assert result is sentinel
