"""Apply boundary repair to backend responses.

Every failure path returns the ORIGINAL body object unchanged -- identity is
asserted in the tests, so a caller can rely on `out is body` meaning "untouched".
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from vllm_router.services.structured_output.contract import OutputContract
from vllm_router.services.structured_output.repair import RepairResult, repair

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepairTelemetry:
    status: str
    mode: str
    garbage_prefix_bytes: int


def _telemetry(result: RepairResult) -> RepairTelemetry:
    return RepairTelemetry(
        status=result.status,
        mode=result.mode,
        garbage_prefix_bytes=len(result.garbage_prefix.encode("utf-8")),
    )


def _warn_transform_failure() -> None:
    try:
        logger.warning("structured-output repair failed; passing response through")
    except Exception:  # noqa: BLE001 - logging must not break response pass-through
        pass


def _repair_choice(choice: dict, contract: OutputContract) -> RepairResult | None:
    """Repair one choice's content. None means "nothing to look at here"."""
    message = choice.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, str) or not content:
        return None

    finish_reason = choice.get("finish_reason")
    if finish_reason is None:
        return None

    return repair(
        content,
        contract.content_schema,
        finish_reason=finish_reason,
    )


def transform_response_body(
    body: bytes, contract: OutputContract
) -> tuple[bytes, list[RepairTelemetry]]:
    """Contain every transform exception, including RecursionError."""
    try:
        return _transform_response_body(body, contract)
    except Exception:  # noqa: BLE001 - the transform is an availability boundary
        _warn_transform_failure()
        return body, []


def _transform_response_body(
    body: bytes, contract: OutputContract
) -> tuple[bytes, list[RepairTelemetry]]:
    if not contract.engaged:
        return body, []

    try:
        payload = json.loads(body)
        choices = payload["choices"]
        if not isinstance(choices, list):
            return body, []
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError):
        return body, []

    telemetry: list[RepairTelemetry] = []
    changed = False

    try:
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            result = _repair_choice(choice, contract)
            if result is None:
                continue
            telemetry.append(_telemetry(result))
            if result.status == "repaired":
                choice["message"]["content"] = result.text
                changed = True
    except Exception:  # noqa: BLE001 - a repair failure must never break a response
        _warn_transform_failure()
        return body, []

    if not changed:
        return body, telemetry

    try:
        return json.dumps(payload, ensure_ascii=False).encode("utf-8"), telemetry
    except (TypeError, ValueError):
        return body, []
