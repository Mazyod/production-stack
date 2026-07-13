"""Decide whether a request is eligible for boundary repair, and extract its schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OutputContract:
    content_schema: dict[str, Any] | None = None
    rejected_non_discriminating: bool = False

    @property
    def engaged(self) -> bool:
        return self.content_schema is not None


def _schema_disposition(schema: Any) -> str:
    """A schema we can safely use as an oracle.

    OBJECT ROOTS ONLY. Two independent reasons, and each alone is sufficient:

    1. Scalar roots are unrecoverable: `G="-"` + `J="1"` yields `"-1"`, which is
       itself valid and schema-valid. There is nothing to repair.
    2. Array roots have no safe oracle. The discriminating keywords below
       (`required`, `additionalProperties`) are OBJECT keywords -- they say nothing
       about an array, so an array schema carrying `additionalProperties: false`
       would engage while remaining unable to reject a nested fragment. Arrays are
       also where the ambiguity gate is weakest, since an array has no key/colon
       structure to make garbage syntactically illegal (`[[1, 2]` is a perfectly
       valid incomplete document). See spec §5.1 and §7.

    Beyond the root type, the schema must be *discriminating* enough to reject a
    nested fragment -- otherwise a malformed-but-not-truncated body whose nested
    object validates would be "repaired" into that fragment.
    """
    if not isinstance(schema, dict):
        return "absent"
    if schema.get("type") != "object":
        return "unsupported"
    if schema.get("additionalProperties") is False:
        return "repairable"
    required = schema.get("required")
    if isinstance(required, list) and len(required) > 0:
        return "repairable"
    return "non_discriminating"


def _content_schema(request_json: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    candidates: list[Any] = []
    response_format = request_json.get("response_format")
    if (
        isinstance(response_format, dict)
        and response_format.get("type") == "json_schema"
    ):
        json_schema = response_format.get("json_schema")
        if isinstance(json_schema, dict):
            schema = json_schema.get("schema")
            if schema is not None:
                candidates.append(schema)

    structured_outputs = request_json.get("structured_outputs")
    if isinstance(structured_outputs, dict):
        schema = structured_outputs.get("json")
        if schema is not None:
            candidates.append(schema)

    if len(candidates) > 1 and any(
        schema != candidates[0] for schema in candidates[1:]
    ):
        return None, False

    rejected = False
    for schema in candidates:
        disposition = _schema_disposition(schema)
        if disposition == "repairable":
            return schema, rejected
        rejected = rejected or disposition == "non_discriminating"
    return None, rejected


def extract_output_contract(request_json: dict[str, Any]) -> OutputContract:
    """Never raises. A request we cannot understand is simply not engaged."""
    try:
        if not isinstance(request_json, dict):
            return OutputContract()

        # The corruption lives in the sampled tokens, so logprobs describe `G + J`.
        # Repairing only the content would leave them contradicting each other.
        if (
            "logprobs" in request_json
            and request_json["logprobs"] is not None
            and request_json["logprobs"] is not False
        ):
            return OutputContract()

        content_schema, rejected = _content_schema(request_json)
        return OutputContract(content_schema, rejected)
    except Exception:  # noqa: BLE001 - engagement must never break a request
        return OutputContract()
