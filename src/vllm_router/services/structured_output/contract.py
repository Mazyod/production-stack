"""Decide whether a request is eligible for boundary repair, and extract its schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_REGEX_KEYWORDS = frozenset({"pattern", "patternProperties"})
_MAX_SCHEMA_NODES = 10_000
_MAX_SCHEMA_DEPTH = 128


@dataclass(frozen=True)
class OutputContract:
    content_schema: dict[str, Any] | None = None
    rejection_reason: str | None = None

    @property
    def engaged(self) -> bool:
        return self.content_schema is not None

    @property
    def rejected_non_discriminating(self) -> bool:
        return self.rejection_reason == "non_discriminating"


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


def schema_has_unsafe_regex(schema: Any) -> bool:
    """Return whether a schema has regex semantics or cannot be inspected safely.

    Repeated containers are rejected as well as over-deep/over-large schemas. JSON
    schemas arriving over HTTP are trees, so repetition by identity indicates an
    in-process cyclic or otherwise non-JSON input. Failing closed keeps this public
    boundary total without recursive traversal.
    """
    stack = [(schema, 0)]
    seen_containers: set[int] = set()
    nodes = 0

    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_SCHEMA_NODES or depth > _MAX_SCHEMA_DEPTH:
            return True

        if isinstance(value, dict):
            identity = id(value)
            if identity in seen_containers:
                return True
            seen_containers.add(identity)
            if any(keyword in value for keyword in _REGEX_KEYWORDS):
                return True
            stack.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            identity = id(value)
            if identity in seen_containers:
                return True
            seen_containers.add(identity)
            stack.extend((child, depth + 1) for child in value)

    return False


def _content_schema(
    request_json: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
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

    if any(schema_has_unsafe_regex(schema) for schema in candidates):
        return None, "unsafe_regex"

    if len(candidates) > 1 and any(
        schema != candidates[0] for schema in candidates[1:]
    ):
        return None, "conflicting_carriers"

    rejection_reason = None
    for schema in candidates:
        disposition = _schema_disposition(schema)
        if disposition == "repairable":
            return schema, rejection_reason
        if disposition == "non_discriminating":
            rejection_reason = "non_discriminating"
    return None, rejection_reason


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

        content_schema, rejection_reason = _content_schema(request_json)
        return OutputContract(content_schema, rejection_reason)
    except Exception:  # noqa: BLE001 - engagement must never break a request
        return OutputContract()
