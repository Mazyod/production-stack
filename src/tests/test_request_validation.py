import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks, Request

from vllm_router.routers.main_router import route_completion, route_v1_rerank
from vllm_router.services.request_service.request import (
    route_general_transcriptions,
)

REQUEST_ID = "test-request-id"


# Build a minimal request with only the attributes needed for JSON validation.
def _json_request(body: bytes):
    return Request(
        {
            "type": "http",
            "headers": [(b"x-request-id", REQUEST_ID.encode())],
            "query_string": b"",
            "app": SimpleNamespace(
                state=SimpleNamespace(
                    router=object(),
                    otel_enabled=False,
                    callbacks=None,
                )
            ),
        },
        receive=AsyncMock(return_value={"type": "http.request", "body": body}),
    )


@pytest.mark.parametrize(
    "body",
    [
        b'{"model":"test-model"',
        b"\xff",
        b"[]",
        b"[" * 2_000 + b"]" * 2_000,
    ],
)
@pytest.mark.asyncio
@pytest.mark.parametrize("route", [route_completion, route_v1_rerank])
async def test_json_routes_reject_invalid_body(body, route):
    response = await route(_json_request(body), BackgroundTasks())

    assert response.status_code == 400
    assert json.loads(response.body)["error"]
    assert response.headers["X-Request-Id"] == REQUEST_ID


@pytest.mark.asyncio
async def test_transcription_rejects_non_numeric_temperature():
    request = SimpleNamespace(
        headers={"X-Request-Id": REQUEST_ID},
        form=AsyncMock(
            return_value={
                "file": object(),
                "model": "whisper-model",
                "temperature": "not-a-number",
            }
        ),
    )

    response = await route_general_transcriptions(
        request, "/v1/audio/transcriptions", BackgroundTasks()
    )

    assert response.status_code == 400
    assert json.loads(response.body) == {"error": "Invalid multipart/form-data request"}
    assert response.headers["X-Request-Id"] == REQUEST_ID
