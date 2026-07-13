import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

STRICT = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}


def _request_body(*, engaged, stream=False):
    body = {"model": "m", "messages": [], "stream": stream}
    if engaged:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "s", "schema": STRICT},
        }
    return body


def _make_request(body):
    router = MagicMock()
    router.max_instance_failover_reroute_attempts = 0
    router.route_request.return_value = "http://engine"
    router.extract_session_id.return_value = None

    service_discovery = MagicMock()
    endpoint = SimpleNamespace(
        url="http://engine", model_names=["m"], sleep=False, Id="engine"
    )
    service_discovery.get_endpoint_info.return_value = [endpoint]
    service_discovery.aliases = None
    service_discovery.has_ever_seen_model.return_value = True

    state = SimpleNamespace(
        router=router,
        engine_stats_scraper=MagicMock(),
        request_stats_monitor=MagicMock(),
        otel_enabled=False,
        semantic_cache_available=False,
        callbacks=None,
        external_provider_registry=None,
        structured_output_repair_enabled=True,
        structured_output_repair_max_bytes=1_048_576,
        structured_output_repair_max_seconds=30.0,
    )
    state.engine_stats_scraper.get_engine_stats.return_value = {}
    state.request_stats_monitor.get_request_stats.return_value = {}

    request = MagicMock()
    request.headers = {"content-type": "application/json"}
    request.query_params = {}
    request.method = "POST"
    request.url = "http://router/v1/chat/completions"
    request.app.state = state
    encoded = json.dumps(body).encode()
    request.body = AsyncMock(return_value=encoded)
    request.json = AsyncMock(return_value=body)

    patches = [
        patch(
            "vllm_router.services.request_service.request.get_service_discovery",
            return_value=service_discovery,
        ),
        patch(
            "vllm_router.services.request_service.request.is_request_rewriter_initialized",
            return_value=False,
        ),
    ]
    return request, patches


@pytest.fixture
def setup_engaged():
    request, patches = _make_request(_request_body(engaged=True))
    for active_patch in patches:
        active_patch.start()
    try:
        yield request, MagicMock()
    finally:
        for active_patch in reversed(patches):
            active_patch.stop()


@pytest.fixture
def setup_unengaged():
    request, patches = _make_request(_request_body(engaged=False))
    for active_patch in patches:
        active_patch.start()
    try:
        yield request, MagicMock()
    finally:
        for active_patch in reversed(patches):
            active_patch.stop()


def _engaged_request(*, stream=False):
    """A request usable by route_chat_completion cache-boundary tests."""
    # Kept for Task 9's semantic-cache boundary tests.
    body = _request_body(engaged=True, stream=stream)
    request = MagicMock()
    request.app.state = SimpleNamespace(
        structured_output_repair_enabled=True,
        semantic_cache_available=True,
    )
    request.json = AsyncMock(return_value=body)
    return request


def _non_streaming_body(content):
    return json.dumps(
        {
            "id": "x",
            "choices": [
                {
                    "index": 0,
                    "message": {"content": content},
                    "finish_reason": "stop",
                }
            ],
        }
    ).encode()


def _chunk(delta, finish_reason=None, index=0):
    payload = {
        "id": "x",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": index,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


async def _collect(response):
    if hasattr(response, "body"):
        return response.body
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    return b"".join(chunks)


@pytest.mark.asyncio
async def test_route_chat_completion_passes_body_to_downstream_without_preread():
    """The cache boundary must leave the body for general routing to consume."""
    import vllm_router.routers.main_router as main_router_module

    request = _engaged_request()
    encoded = json.dumps(_request_body(engaged=True)).encode()
    request.body = AsyncMock(return_value=encoded)
    background_tasks = MagicMock()

    async def downstream(downstream_request, endpoint, tasks):
        assert endpoint == "/v1/chat/completions"
        assert tasks is background_tasks
        return await downstream_request.body()

    with patch.object(
        main_router_module,
        "route_general_request",
        new_callable=AsyncMock,
        side_effect=downstream,
    ) as route:
        response = await main_router_module.route_chat_completion(
            request, background_tasks
        )

    assert response == encoded
    route.assert_awaited_once_with(request, "/v1/chat/completions", background_tasks)
    request.body.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_semantic_cache_lookup_and_storage_skipped_post_rewrite(
    setup_unengaged,
):
    """A rewriter-added schema is authoritative for both cache decisions."""
    import vllm_router.services.request_service.request as request_module

    req, _ = setup_unengaged
    req.app.state.semantic_cache_available = True
    rewritten = json.dumps(_request_body(engaged=True)).encode()
    rewriter = MagicMock()
    rewriter.rewrite_request.return_value = rewritten

    async def backend(*args, **kwargs):
        yield {"content-type": "application/json"}, 200
        yield _non_streaming_body('{{"summary": "x"}')

    with (
        patch.object(
            request_module, "check_semantic_cache", new_callable=AsyncMock, create=True
        ) as cache,
        patch.object(request_module, "semantic_cache_available", True),
        patch.object(
            request_module, "is_request_rewriter_initialized", return_value=True
        ),
        patch.object(request_module, "get_request_rewriter", return_value=rewriter),
        patch.object(request_module, "process_request", side_effect=backend) as process,
    ):
        response = await request_module.route_general_request(
            req, "/v1/chat/completions", MagicMock()
        )
        await _collect(response)

    cache.assert_not_awaited()
    assert process.call_args.kwargs["skip_semantic_cache"] is True


@pytest.mark.asyncio
async def test_unengaged_cache_lookup_uses_post_rewrite_json(setup_unengaged):
    import vllm_router.services.request_service.request as request_module

    req, _ = setup_unengaged
    req.app.state.semantic_cache_available = True
    cached = MagicMock()
    with (
        patch.object(
            request_module, "check_semantic_cache", new_callable=AsyncMock, create=True
        ) as cache,
        patch.object(request_module, "semantic_cache_available", True),
        patch.object(request_module, "process_request") as process,
    ):
        cache.return_value = cached
        response = await request_module.route_general_request(
            req, "/v1/chat/completions", MagicMock()
        )

    assert response is cached
    cache.assert_awaited_once_with(
        request=req,
        request_json=_request_body(engaged=False),
    )
    process.assert_not_called()


@pytest.mark.asyncio
async def test_feature_flag_off_preserves_engaged_cache_lookup(setup_engaged):
    """Repair disabled means an otherwise engaged request still uses the cache."""
    import vllm_router.services.request_service.request as request_module

    req, _ = setup_engaged
    req.app.state.structured_output_repair_enabled = False
    req.app.state.semantic_cache_available = True
    cached = MagicMock()

    with (
        patch.object(
            request_module, "check_semantic_cache", new_callable=AsyncMock, create=True
        ) as cache,
        patch.object(request_module, "semantic_cache_available", True),
        patch.object(request_module, "process_request") as process,
    ):
        cache.return_value = cached
        response = await request_module.route_general_request(
            req, "/v1/chat/completions", MagicMock()
        )

    assert response is cached
    cache.assert_awaited_once_with(
        request=req,
        request_json=_request_body(engaged=True),
    )
    process.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("skip_semantic_cache", "expected_store_count"),
    [(True, 0), (False, 1)],
)
async def test_process_request_semantic_cache_storage_guard(
    setup_engaged, skip_semantic_cache, expected_store_count
):
    """The process boundary suppresses only explicitly skipped cache stores."""
    import vllm_router.services.request_service.request as request_module

    req, _ = setup_engaged
    req.app.state.semantic_cache_available = True
    content = MagicMock()

    async def iter_any():
        yield _non_streaming_body('{{"summary": "x"}')

    content.iter_any = iter_any
    backend_response = MagicMock(status=200, headers={})
    backend_response.content = content
    client_request = MagicMock()
    client_request.__aenter__ = AsyncMock(return_value=backend_response)
    client_request.__aexit__ = AsyncMock(return_value=False)
    client = MagicMock()
    client.request.return_value = client_request
    req.app.state.aiohttp_client_wrapper = MagicMock(return_value=client)

    with patch.object(
        request_module,
        "store_in_semantic_cache",
        new_callable=AsyncMock,
        create=True,
    ) as store:
        chunks = [
            item
            async for item in request_module.process_request(
                req,
                json.dumps(_request_body(engaged=True)).encode(),
                "http://engine",
                "request-id",
                "/v1/chat/completions",
                MagicMock(),
                skip_semantic_cache=skip_semantic_cache,
            )
        ]

    assert len(chunks) == 2
    assert store.await_count == expected_store_count


@pytest.mark.asyncio
async def test_semantic_cache_optional_extra_disabled_is_safe(setup_unengaged):
    """An unavailable semantic-cache import never touches its missing hooks."""
    import vllm_router.services.request_service.request as request_module

    req, _ = setup_unengaged
    original = _non_streaming_body('{{"summary": "x"}')

    async def backend(*args, **kwargs):
        yield {"content-type": "application/json"}, 200
        yield original

    with (
        patch.object(request_module, "semantic_cache_available", False),
        patch.object(
            request_module,
            "check_semantic_cache",
            side_effect=AssertionError("optional cache hook must not be used"),
            create=True,
        ) as cache,
        patch.object(request_module, "process_request", side_effect=backend),
    ):
        response = await request_module.route_general_request(
            req, "/v1/chat/completions", MagicMock()
        )

    assert await _collect(response) == original
    cache.assert_not_called()


@pytest.mark.asyncio
async def test_non_streaming_repairs_and_sets_content_length(setup_engaged):
    req, _ = setup_engaged

    async def backend(*a, **kw):
        yield {"content-type": "application/json"}, 200
        yield _non_streaming_body('{{"summary": "x"}')

    with patch(
        "vllm_router.services.request_service.request.process_request",
        side_effect=backend,
    ):
        from vllm_router.services.request_service.request import route_general_request

        resp = await route_general_request(req, "/v1/chat/completions", MagicMock())

    body = await _collect(resp)
    assert json.loads(body)["choices"][0]["message"]["content"] == '{"summary": "x"}'
    assert resp.headers["content-length"] == str(len(body))


@pytest.mark.asyncio
async def test_unengaged_request_is_byte_identical(setup_unengaged):
    """No response_format -> the new machinery must not touch a single byte."""
    req, _ = setup_unengaged
    original = _non_streaming_body('{{"summary": "x"}')

    async def backend(*a, **kw):
        yield {"content-type": "application/json"}, 200
        yield original

    with patch(
        "vllm_router.services.request_service.request.process_request",
        side_effect=backend,
    ):
        from vllm_router.services.request_service.request import route_general_request

        resp = await route_general_request(req, "/v1/chat/completions", MagicMock())

    assert await _collect(resp) == original


@pytest.mark.asyncio
async def test_non_streaming_non_2xx_is_byte_identical(setup_engaged):
    req, _ = setup_engaged
    original = _non_streaming_body('{{"summary": "x"}')

    async def backend(*args, **kwargs):
        yield {"content-type": "application/json"}, 400
        yield original

    with patch(
        "vllm_router.services.request_service.request.process_request",
        side_effect=backend,
    ):
        from vllm_router.services.request_service.request import route_general_request

        response = await route_general_request(req, "/v1/chat/completions", MagicMock())

    assert response.status_code == 400
    assert await _collect(response) == original


@pytest.mark.asyncio
async def test_streaming_non_2xx_is_byte_identical():
    req, patches = _make_request(_request_body(engaged=True, stream=True))
    original = b'{"error":"bad {{ output"}'

    async def backend(*args, **kwargs):
        yield {"content-type": "application/json"}, 503
        yield original

    for active_patch in patches:
        active_patch.start()
    try:
        with patch(
            "vllm_router.services.request_service.request.process_request",
            side_effect=backend,
        ):
            from vllm_router.services.request_service.request import (
                route_general_request,
            )

            response = await route_general_request(
                req, "/v1/chat/completions", MagicMock()
            )
        assert response.status_code == 503
        assert await _collect(response) == original
    finally:
        for active_patch in reversed(patches):
            active_patch.stop()


@pytest.mark.asyncio
async def test_non_streaming_transport_error_replays_then_raises(setup_engaged):
    req, _ = setup_engaged
    first = b'{"partial":'

    async def backend(*args, **kwargs):
        yield {"content-type": "application/json"}, 200
        yield first
        raise ConnectionError("backend disconnected")

    with patch(
        "vllm_router.services.request_service.request.process_request",
        side_effect=backend,
    ):
        from vllm_router.services.request_service.request import route_general_request

        response = await route_general_request(req, "/v1/chat/completions", MagicMock())

    iterator = response.body_iterator.__aiter__()
    assert await iterator.__anext__() == first
    with pytest.raises(ConnectionError, match="backend disconnected"):
        await iterator.__anext__()


@pytest.mark.asyncio
async def test_stream_watchdog_replays_without_cancelling_backend_read():
    req, patches = _make_request(_request_body(engaged=True, stream=True))
    req.app.state.structured_output_repair_max_seconds = 0.01
    release = asyncio.Event()
    cancelled = False
    first = _chunk({"content": '{{"summary": '})
    second = _chunk({"content": '"x"}'}, finish_reason="stop")

    async def backend(*args, **kwargs):
        nonlocal cancelled
        yield {"content-type": "text/event-stream"}, 200
        yield first
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled = True
            raise
        yield second

    for active_patch in patches:
        active_patch.start()
    try:
        original_ensure_future = asyncio.ensure_future
        with (
            patch(
                "vllm_router.services.request_service.request.process_request",
                side_effect=backend,
            ),
            patch(
                "vllm_router.services.request_service.request.structured_output_repairs_total.labels"
            ) as metric_labels,
            patch(
                "vllm_router.services.request_service.request.asyncio.ensure_future",
                wraps=original_ensure_future,
            ) as ensure_future,
        ):
            from vllm_router.services.request_service.request import (
                route_general_request,
            )

            response = await route_general_request(
                req, "/v1/chat/completions", MagicMock()
            )
            iterator = response.body_iterator.__aiter__()
            received = [await iterator.__anext__()]
            assert received == [first]
            assert cancelled is False
            release.set()
            received.append(await iterator.__anext__())
            with pytest.raises(StopAsyncIteration):
                await iterator.__anext__()

            assert b"".join(received) == first + second
            metric_labels.assert_any_call(model="m", status="timeout", mode="other")
            assert ensure_future.call_count == 1
    finally:
        release.set()
        for active_patch in reversed(patches):
            active_patch.stop()


@pytest.mark.asyncio
async def test_feature_flag_off_bypasses_transform_byte_identically():
    req, patches = _make_request(_request_body(engaged=True))
    req.app.state.structured_output_repair_enabled = False
    original = _non_streaming_body('{{"summary": "x"}')

    async def backend(*args, **kwargs):
        yield {"content-type": "application/json"}, 200
        yield original

    for active_patch in patches:
        active_patch.start()
    try:
        with (
            patch(
                "vllm_router.services.request_service.request.process_request",
                side_effect=backend,
            ),
            patch(
                "vllm_router.services.request_service.request.transform_response_body"
            ) as transform,
            patch(
                "vllm_router.services.request_service.request.extract_output_contract"
            ) as extract_contract,
            patch(
                "vllm_router.services.request_service.request.OutputContract"
            ) as output_contract,
            patch(
                "vllm_router.services.request_service.request._record_repair_metrics"
            ) as record_metrics,
        ):
            from vllm_router.services.request_service.request import (
                route_general_request,
            )

            response = await route_general_request(
                req, "/v1/chat/completions", MagicMock()
            )

        assert await _collect(response) == original
        transform.assert_not_called()
        extract_contract.assert_not_called()
        output_contract.assert_called_once_with()
        record_metrics.assert_not_called()
    finally:
        for active_patch in reversed(patches):
            active_patch.stop()


@pytest.mark.asyncio
async def test_contract_is_extracted_from_post_rewrite_body():
    req, patches = _make_request(_request_body(engaged=False))
    rewritten_body = json.dumps(_request_body(engaged=True)).encode()
    rewriter = MagicMock()
    rewriter.rewrite_request.return_value = rewritten_body

    async def backend(*args, **kwargs):
        yield {"content-type": "application/json"}, 200
        yield _non_streaming_body('{{"summary": "x"}')

    patches[0].start()
    try:
        with (
            patch(
                "vllm_router.services.request_service.request.is_request_rewriter_initialized",
                return_value=True,
            ),
            patch(
                "vllm_router.services.request_service.request.get_request_rewriter",
                return_value=rewriter,
            ),
            patch(
                "vllm_router.services.request_service.request.process_request",
                side_effect=backend,
            ),
        ):
            from vllm_router.services.request_service.request import (
                route_general_request,
            )

            response = await route_general_request(
                req, "/v1/chat/completions", MagicMock()
            )

        body = await _collect(response)
        assert json.loads(body)["choices"][0]["message"]["content"] == (
            '{"summary": "x"}'
        )
        rewriter.rewrite_request.assert_called_once()
    finally:
        patches[0].stop()


@pytest.mark.asyncio
async def test_metric_and_metric_warning_failures_do_not_break_response(setup_engaged):
    req, _ = setup_engaged

    async def backend(*args, **kwargs):
        yield {"content-type": "application/json"}, 200
        yield _non_streaming_body('{{"summary": "x"}')

    with (
        patch(
            "vllm_router.services.request_service.request.process_request",
            side_effect=backend,
        ),
        patch(
            "vllm_router.services.request_service.request.structured_output_repairs_total.labels",
            side_effect=RuntimeError("metrics unavailable"),
        ),
        patch(
            "vllm_router.services.request_service.request.logger.warning",
            side_effect=RuntimeError("logging unavailable"),
        ),
    ):
        from vllm_router.services.request_service.request import route_general_request

        response = await route_general_request(req, "/v1/chat/completions", MagicMock())

    body = await _collect(response)
    assert json.loads(body)["choices"][0]["message"]["content"] == '{"summary": "x"}'


@pytest.mark.asyncio
async def test_streaming_transport_error_replays_then_raises():
    req, patches = _make_request(_request_body(engaged=True, stream=True))
    first = _chunk({"content": '{{"summary": '})

    async def backend(*args, **kwargs):
        yield {"content-type": "text/event-stream"}, 200
        yield first
        raise ConnectionError("backend disconnected")

    for active_patch in patches:
        active_patch.start()
    try:
        with (
            patch(
                "vllm_router.services.request_service.request.process_request",
                side_effect=backend,
            ),
            patch(
                "vllm_router.services.request_service.request.structured_output_repairs_total.labels"
            ) as metric_labels,
        ):
            from vllm_router.services.request_service.request import (
                route_general_request,
            )

            response = await route_general_request(
                req, "/v1/chat/completions", MagicMock()
            )

            iterator = response.body_iterator.__aiter__()
            received = await iterator.__anext__()
            assert received == first
            with pytest.raises(ConnectionError, match="backend disconnected"):
                await iterator.__anext__()
            metric_labels.assert_any_call(model="m", status="error", mode="other")
    finally:
        for active_patch in reversed(patches):
            active_patch.stop()


@pytest.mark.asyncio
async def test_stream_cancellation_aborts_retained_bytes():
    from vllm_router.services.structured_output.transform import StreamRepairer

    req, patches = _make_request(_request_body(engaged=True, stream=True))
    backend_waiting = asyncio.Event()
    never_release = asyncio.Event()
    abort_calls = 0
    first = _chunk({"content": '{{"summary": '})
    messages = []

    class TrackingRepairer(StreamRepairer):
        def abort(self):
            nonlocal abort_calls
            abort_calls += 1
            return super().abort()

    async def backend(*args, **kwargs):
        yield {"content-type": "text/event-stream"}, 200
        yield first
        backend_waiting.set()
        await never_release.wait()

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        messages.append(message)

    for active_patch in patches:
        active_patch.start()
    try:
        with (
            patch(
                "vllm_router.services.request_service.request.process_request",
                side_effect=backend,
            ),
            patch(
                "vllm_router.services.request_service.request.StreamRepairer",
                TrackingRepairer,
            ),
        ):
            from vllm_router.services.request_service.request import (
                route_general_request,
            )

            response = await route_general_request(
                req, "/v1/chat/completions", MagicMock()
            )
            response_task = asyncio.create_task(
                response(
                    {
                        "type": "http",
                        "method": "POST",
                        "path": "/v1/chat/completions",
                        "headers": [],
                        "asgi": {"version": "3.0", "spec_version": "2.4"},
                    },
                    receive,
                    send,
                )
            )
            await backend_waiting.wait()
            response_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await response_task

        assert abort_calls == 1
        bodies = [
            message["body"]
            for message in messages
            if message["type"] == "http.response.body"
        ]
        assert bodies == [first]
    finally:
        never_release.set()
        for active_patch in reversed(patches):
            active_patch.stop()


@pytest.mark.asyncio
async def test_flag_off_server_cancellation_ends_span_as_client_closed():
    req, patches = _make_request(_request_body(engaged=True, stream=True))
    req.app.state.structured_output_repair_enabled = False
    req.app.state.otel_enabled = True
    parent_span = MagicMock()
    backend_waiting = asyncio.Event()

    async def backend(*args, **kwargs):
        yield {"content-type": "text/event-stream"}, 200
        backend_waiting.set()
        await asyncio.Event().wait()

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        pass

    for active_patch in patches:
        active_patch.start()
    try:
        with (
            patch(
                "vllm_router.services.request_service.request.process_request",
                side_effect=backend,
            ),
            patch("vllm_router.services.request_service.request.otel_available", True),
            patch(
                "vllm_router.services.request_service.request.extract_context",
                return_value=MagicMock(),
            ),
            patch(
                "vllm_router.services.request_service.request.start_span",
                return_value=(parent_span, MagicMock()),
            ),
            patch("vllm_router.services.request_service.request.end_span") as end_span,
        ):
            from vllm_router.services.request_service.request import (
                route_general_request,
            )

            response = await route_general_request(
                req, "/v1/chat/completions", MagicMock()
            )
            response_task = asyncio.create_task(
                response(
                    {
                        "type": "http",
                        "method": "POST",
                        "path": "/v1/chat/completions",
                        "headers": [],
                        "asgi": {"version": "3.0", "spec_version": "2.4"},
                    },
                    receive,
                    send,
                )
            )
            await backend_waiting.wait()
            response_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await response_task

        end_span.assert_called_once_with(parent_span, status_code=499)
    finally:
        for active_patch in reversed(patches):
            active_patch.stop()


@pytest.mark.asyncio
async def test_repaired_response_ends_parent_span_once(setup_engaged):
    req, _ = setup_engaged
    req.app.state.otel_enabled = True
    parent_span = MagicMock()
    span_context = MagicMock()

    async def backend(*args, **kwargs):
        yield {"content-type": "application/json"}, 200
        yield _non_streaming_body('{{"summary": "x"}')

    with (
        patch(
            "vllm_router.services.request_service.request.process_request",
            side_effect=backend,
        ),
        patch(
            "vllm_router.services.request_service.request.otel_available",
            True,
        ),
        patch(
            "vllm_router.services.request_service.request.extract_context",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_router.services.request_service.request.start_span",
            return_value=(parent_span, span_context),
        ),
        patch("vllm_router.services.request_service.request.end_span") as end_span,
    ):
        from vllm_router.services.request_service.request import route_general_request

        response = await route_general_request(req, "/v1/chat/completions", MagicMock())

    assert response.status_code == 200
    end_span.assert_called_once_with(parent_span, status_code=200)
