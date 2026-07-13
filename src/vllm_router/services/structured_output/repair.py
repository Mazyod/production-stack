"""Recover a grammar-produced JSON value from a short corrupt prefix."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jsonschema
from referencing import Registry
from referencing.exceptions import NoSuchResource

from vllm_router.services.structured_output.json_prefix import is_valid_json_prefix


@dataclass(frozen=True)
class RepairResult:
    status: str
    text: str | None
    value: object | None
    garbage_prefix: str
    trailing: str
    mode: str
    candidates_tried: int


def repair_tool_arguments(
    arguments: str,
    parameters_schema: dict[str, Any] | None,
    *,
    finish_reason: str | None,
) -> RepairResult:
    """Repair arguments using the matching tool's parameters schema."""
    return repair(arguments, parameters_schema, finish_reason=finish_reason)


def repair(
    content: str,
    schema: dict[str, Any] | None,
    *,
    finish_reason: str | None,
    max_prefix_bytes: int = 256,
) -> RepairResult:
    """Find a complete, schema-valid object or array after corrupt prefix bytes.

    Every failure inside the transform is contained. The required
    ``finish_reason`` keyword deliberately remains outside that containment: a
    caller which omits it gets Python's normal ``TypeError`` before this
    function runs.
    """
    try:
        if finish_reason == "length":
            return _empty_result("incomplete")
        return _repair(content, schema, max_prefix_bytes=max_prefix_bytes)
    except Exception:
        return _empty_result("unknown")


def _repair(
    content: str,
    schema: dict[str, Any] | None,
    *,
    max_prefix_bytes: int,
) -> RepairResult:
    validator = _validator(schema)

    try:
        value = json.loads(content, parse_constant=_reject_non_json_constant)
    except (json.JSONDecodeError, UnicodeError, ValueError):
        pass
    else:
        if validator is None or validator.is_valid(value):
            return RepairResult(
                status="clean",
                text=content,
                value=value,
                garbage_prefix="",
                trailing="",
                mode="none",
                candidates_tried=0,
            )

    openers = _root_openers(schema)
    if not openers:
        return _empty_result("unknown")

    if is_valid_json_prefix(content):
        return _empty_result("ambiguous")

    decoder = json.JSONDecoder(parse_constant=_reject_non_json_constant)
    candidate_offsets = _candidate_offsets(
        content,
        openers,
        max_prefix_bytes=max_prefix_bytes,
    )
    for candidates_tried, offset in enumerate(candidate_offsets, start=1):
        prefix = content[:offset]
        try:
            value, end = decoder.raw_decode(content, offset)
        except (json.JSONDecodeError, ValueError):
            # A later opener inside this valid-but-incomplete value could be a
            # nested fragment. The lexical-inconsistency regression reaches
            # this path even though the whole content is not a JSON prefix.
            if is_valid_json_prefix(content[offset:]):
                return _empty_result("unknown", candidates_tried=candidates_tried)
            continue

        if not validator.is_valid(value):
            continue

        suffix = content[end:]
        if not _is_allowed_trailing(suffix, prefix):
            continue

        return RepairResult(
            status="repaired",
            text=content[offset:end],
            value=value,
            garbage_prefix=prefix,
            trailing=suffix if suffix.strip() else "",
            mode=_repair_mode(prefix),
            candidates_tried=candidates_tried,
        )

    return _empty_result("unknown", candidates_tried=len(candidate_offsets))


def _empty_result(status: str, *, candidates_tried: int = 0) -> RepairResult:
    return RepairResult(
        status=status,
        text=None,
        value=None,
        garbage_prefix="",
        trailing="",
        mode="other",
        candidates_tried=candidates_tried,
    )


def _validator(schema: dict[str, Any] | None):
    if schema is None:
        return None
    schema_key = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return _compiled_validator(schema_key)


@lru_cache(maxsize=128)
def _compiled_validator(schema_key: str):
    schema = json.loads(schema_key)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema, registry=Registry(retrieve=_reject_remote_reference))


def _reject_remote_reference(uri: str):
    raise NoSuchResource(ref=uri)


def _reject_non_json_constant(constant: str):
    raise ValueError(f"{constant} is not legal JSON")


def _root_openers(schema: dict[str, Any] | None) -> frozenset[str]:
    if not isinstance(schema, dict):
        return frozenset()
    root_type = schema.get("type")
    if root_type == "object":
        return frozenset("{")
    if root_type == "array":
        return frozenset("[")
    return frozenset()


def _candidate_offsets(
    content: str,
    openers: frozenset[str],
    *,
    max_prefix_bytes: int,
) -> list[int]:
    if max_prefix_bytes <= 0:
        return []

    offsets: list[int] = []
    byte_offset = 0
    for character_offset, character in enumerate(content):
        if byte_offset >= max_prefix_bytes:
            break
        if character in openers:
            offsets.append(character_offset)
        byte_offset += len(character.encode("utf-8"))
    return offsets


def _is_allowed_trailing(trailing: str, prefix: str) -> bool:
    stripped = trailing.strip()
    if not stripped:
        return True
    return stripped == "```" and "```" in prefix


def _repair_mode(prefix: str) -> str:
    if "```" in prefix:
        return "code_fence"

    stripped = prefix.strip()
    if stripped in {"{", "["}:
        return "extra_brace"
    if stripped.startswith(('{"', '["')):
        return "dup_prefix"
    return "other"
