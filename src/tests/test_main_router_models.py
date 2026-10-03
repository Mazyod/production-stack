import json
import logging
from unittest.mock import MagicMock

import pytest

from vllm_router.routers.main_router import show_models
from vllm_router.service_discovery import EndpointInfo, ModelInfo


def _build_request():
    request = MagicMock()
    request.app.state = MagicMock()
    request.app.state.external_provider_registry = None
    return request


def _build_endpoint(model_payload):
    model_id = model_payload["id"]
    return EndpointInfo(
        url="http://engine",
        model_names=[model_id],
        Id="engine-1",
        added_timestamp=0.0,
        model_label=model_id,
        sleep=False,
        model_info={model_id: ModelInfo.from_dict(model_payload)},
    )


def _mock_service_discovery(endpoints, aliases=None):
    """Build a service-discovery mock with explicit endpoints/aliases.

    ``aliases`` is set explicitly (default ``None``) so ``getattr`` does not
    resolve to an auto-generated MagicMock attribute.
    """
    return MagicMock(
        get_endpoint_info=lambda: endpoints,
        aliases=aliases,
    )


def test_model_info_preserves_extra_fields():
    payload = {
        "id": "test-model",
        "object": "model",
        "created": 123,
        "owned_by": "vllm",
        "root": "models/test-model",
        "parent": None,
        "max_model_len": 8192,
        "permission": [{"id": "perm-1", "allow_sampling": True}],
    }

    model_info = ModelInfo.from_dict(payload)

    assert model_info.extra_fields == {
        "max_model_len": 8192,
        "permission": [{"id": "perm-1", "allow_sampling": True}],
    }
    assert model_info.to_dict() == payload


@pytest.mark.asyncio
async def test_show_models_returns_full_backend_model_payload(monkeypatch):
    payload = {
        "id": "rich-model",
        "object": "model",
        "created": 1774275193,
        "owned_by": "vllm",
        "root": "models/rich-model",
        "parent": None,
        "max_model_len": 262144,
        "permission": [
            {
                "id": "modelperm-1",
                "object": "model_permission",
                "allow_sampling": True,
                "allow_logprobs": True,
            }
        ],
    }

    monkeypatch.setattr(
        "vllm_router.routers.main_router.get_service_discovery",
        lambda: _mock_service_discovery([_build_endpoint(payload)]),
    )

    response = await show_models(_build_request())

    assert response.status_code == 200
    body = json.loads(response.body)
    assert body["data"] == [payload]


@pytest.mark.asyncio
async def test_show_models_includes_static_aliases(monkeypatch):
    payload = {
        "id": "llama3",
        "object": "model",
        "created": 1774275193,
        "owned_by": "vllm",
        "root": None,
        "parent": None,
    }

    monkeypatch.setattr(
        "vllm_router.routers.main_router.get_service_discovery",
        lambda: _mock_service_discovery(
            [_build_endpoint(payload)],
            aliases={"my-model": "llama3", "gpt-4": "llama3"},
        ),
    )

    response = await show_models(_build_request())

    assert response.status_code == 200
    body = json.loads(response.body)
    ids = [card["id"] for card in body["data"]]
    assert ids == ["llama3", "my-model", "gpt-4"]

    alias_card = next(card for card in body["data"] if card["id"] == "my-model")
    assert alias_card["root"] == "llama3"
    assert alias_card["owned_by"] == "vllm"


@pytest.mark.asyncio
async def test_show_models_alias_shadowed_by_real_model_is_not_duplicated(monkeypatch):
    payload = {
        "id": "llama3",
        "object": "model",
        "created": 1774275193,
        "owned_by": "vllm",
        "root": None,
        "parent": None,
    }

    monkeypatch.setattr(
        "vllm_router.routers.main_router.get_service_discovery",
        # Alias key collides with an already-listed backend model id.
        lambda: _mock_service_discovery(
            [_build_endpoint(payload)],
            aliases={"llama3": "llama3"},
        ),
    )

    response = await show_models(_build_request())

    assert response.status_code == 200
    body = json.loads(response.body)
    assert [card["id"] for card in body["data"]] == ["llama3"]


@pytest.mark.asyncio
async def test_show_models_does_not_warn_for_preserved_backend_fields(
    monkeypatch, caplog
):
    payload = {
        "id": "rich-model",
        "object": "model",
        "created": 1774275193,
        "owned_by": "vllm",
        "root": "models/rich-model",
        "parent": None,
        "max_model_len": 262144,
        "permission": [
            {
                "id": "modelperm-1",
                "object": "model_permission",
                "allow_sampling": True,
                "allow_logprobs": True,
            }
        ],
    }

    monkeypatch.setattr(
        "vllm_router.routers.main_router.get_service_discovery",
        lambda: _mock_service_discovery([_build_endpoint(payload)]),
    )

    with caplog.at_level(logging.WARNING, logger="vllm_router.protocols"):
        response = await show_models(_build_request())

    assert response.status_code == 200
    assert "ignored" not in caplog.text
