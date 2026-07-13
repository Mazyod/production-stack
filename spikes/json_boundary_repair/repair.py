"""Recover a grammar-produced JSON value from a short corrupt prefix."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jsonschema
from referencing import Registry
from referencing.exceptions import NoSuchResource


@dataclass(frozen=True)
class RepairResult:
    status: str
    text: str | None
    value: object | None
    garbage_prefix: str
    trailing: str
    mode: str
    candidates_tried: int


@dataclass
class _Container:
    kind: str
    state: str


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


def is_valid_json_prefix(content: str) -> bool:
    """Return whether some legal JSON document starts with ``content``.

    This incremental recognizer accepts incomplete tokens and structures when
    more bytes can make them legal. It rejects a token as soon as no extension
    can repair it.
    """
    containers: list[_Container] = []
    root_started = False
    token_kind: str | None = None
    token_state = ""
    token_target = ""
    literal = ""
    literal_offset = 0
    index = 0

    def expectation() -> str:
        if containers:
            return containers[-1].state
        return "done" if root_started else "value"

    def begin_value() -> None:
        nonlocal root_started
        if containers:
            containers[-1].state = "comma_or_end"
        else:
            root_started = True

    while index < len(content):
        character = content[index]

        if token_kind == "string":
            if token_state == "escape":
                if character == "u":
                    token_state = "unicode_1"
                elif character in '"\\/bfnrt':
                    token_state = "body"
                else:
                    return False
                index += 1
                continue

            if token_state.startswith("unicode_"):
                if character not in "0123456789abcdefABCDEF":
                    return False
                digit = int(token_state.removeprefix("unicode_"))
                token_state = "body" if digit == 4 else f"unicode_{digit + 1}"
                index += 1
                continue

            if character == '"':
                token_kind = None
                if token_target == "key":
                    containers[-1].state = "colon"
                index += 1
                continue
            if character == "\\":
                token_state = "escape"
                index += 1
                continue
            if ord(character) < 0x20:
                return False
            index += 1
            continue

        if token_kind == "literal":
            if character != literal[literal_offset]:
                return False
            literal_offset += 1
            index += 1
            if literal_offset == len(literal):
                token_kind = None
            continue

        if token_kind == "number":
            if token_state == "sign":
                if character == "0":
                    token_state = "zero"
                elif character in "123456789":
                    token_state = "integer"
                else:
                    return False
                index += 1
                continue
            if token_state == "zero":
                if character == ".":
                    token_state = "dot"
                    index += 1
                    continue
                if character in "eE":
                    token_state = "exponent"
                    index += 1
                    continue
                if character in "0123456789":
                    return False
            elif token_state == "integer":
                if character in "0123456789":
                    index += 1
                    continue
                if character == ".":
                    token_state = "dot"
                    index += 1
                    continue
                if character in "eE":
                    token_state = "exponent"
                    index += 1
                    continue
            elif token_state == "dot":
                if character in "0123456789":
                    token_state = "fraction"
                    index += 1
                    continue
                return False
            elif token_state == "fraction":
                if character in "0123456789":
                    index += 1
                    continue
                if character in "eE":
                    token_state = "exponent"
                    index += 1
                    continue
            elif token_state == "exponent":
                if character in "+-":
                    token_state = "exponent_sign"
                elif character in "0123456789":
                    token_state = "exponent_digits"
                else:
                    return False
                index += 1
                continue
            elif token_state == "exponent_sign":
                if character in "0123456789":
                    token_state = "exponent_digits"
                    index += 1
                    continue
                return False
            elif token_state == "exponent_digits":
                if character in "0123456789":
                    index += 1
                    continue

            token_kind = None
            continue

        wanted = expectation()
        if character in " \t\r\n":
            index += 1
            continue

        if wanted == "done":
            return False

        if wanted in {"key_or_end", "key"}:
            if wanted == "key_or_end" and character == "}":
                containers.pop()
                index += 1
                continue
            if character != '"':
                return False
            token_kind = "string"
            token_state = "body"
            token_target = "key"
            index += 1
            continue

        if wanted == "colon":
            if character != ":":
                return False
            containers[-1].state = "value"
            index += 1
            continue

        if wanted == "comma_or_end":
            container = containers[-1]
            closing = "}" if container.kind == "object" else "]"
            if character == closing:
                containers.pop()
                index += 1
                continue
            if character != ",":
                return False
            container.state = "key" if container.kind == "object" else "value"
            index += 1
            continue

        if wanted == "value_or_end" and character == "]":
            containers.pop()
            index += 1
            continue

        if wanted not in {"value", "value_or_end"}:
            return False

        begin_value()
        if character == "{":
            containers.append(_Container("object", "key_or_end"))
        elif character == "[":
            containers.append(_Container("array", "value_or_end"))
        elif character == '"':
            token_kind = "string"
            token_state = "body"
            token_target = "value"
        elif character == "-":
            token_kind = "number"
            token_state = "sign"
        elif character == "0":
            token_kind = "number"
            token_state = "zero"
        elif character in "123456789":
            token_kind = "number"
            token_state = "integer"
        elif character in "tfn":
            token_kind = "literal"
            literal = {"t": "true", "f": "false", "n": "null"}[character]
            literal_offset = 1
            if literal_offset == len(literal):
                token_kind = None
        else:
            return False
        index += 1

    return True


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
