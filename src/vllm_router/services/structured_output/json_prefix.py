"""Recognize prefixes that can still become legal JSON documents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Container:
    kind: str
    state: str


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
