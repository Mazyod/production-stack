import json

import jsonschema
import pytest

from vllm_router.services.structured_output.json_prefix import is_valid_json_prefix
from vllm_router.services.structured_output.repair import (
    _compiled_validator,
    repair,
    repair_tool_arguments,
)


OBJECT_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}

ARRAY_SCHEMA = {"type": "array", "items": {"type": "integer"}}

TOOL_PARAMETERS_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}

REAL_CORRUPTION_CASES = [
    ('{{"summary": "x"}', OBJECT_SCHEMA, '{"summary": "x"}', "extra_brace"),
    (
        '{"{"name": "x"}',
        {
            "type": "object",
            "properties": {"name": {"enum": ["x"]}},
            "required": ["name"],
            "additionalProperties": False,
        },
        '{"name": "x"}',
        "dup_prefix",
    ),
    (
        '```json\n{"a": 1}\n```',
        {
            "type": "object",
            "properties": {"a": {"enum": [1]}},
            "required": ["a"],
            "additionalProperties": False,
        },
        '{"a": 1}',
        "code_fence",
    ),
    (
        '{"{\n  "x": "ONLY_VALUE"\n}',
        {
            "type": "object",
            "properties": {"x": {"enum": ["ONLY_VALUE"]}},
            "required": ["x"],
            "additionalProperties": False,
        },
        '{\n  "x": "ONLY_VALUE"\n}',
        "dup_prefix",
    ),
]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("", True),
        (" \t\r\n", True),
        ("{", True),
        ('{"a":', True),
        ('{"a": {"x": 1}', True),
        ("[[1, 2]", True),
        ('"unterminated', True),
        ('"escaped\\', True),
        ('"unicode \\u12', True),
        ("-", True),
        ("1.", True),
        ("1e", True),
        ("1e+", True),
        ("t", True),
        ("tr", True),
        ("tru", True),
        ("true", True),
        ('{{"summary": "x"}', False),
        ('{"{"summary": "x"}', False),
        (' ```json{"summary": "x"} ', False),
        ('"garbage{"a":{"x":1}', False),
        ('{"a",', False),
        ('{"a":01', False),
        ("[1,]", False),
        ('"bad\\x', False),
        ('"bad\n', False),
        ("true false", False),
    ],
)
def test_is_valid_json_prefix(content, expected):
    assert is_valid_json_prefix(content) is expected


@pytest.mark.parametrize(
    ("content", "schema", "expected", "mode"),
    REAL_CORRUPTION_CASES,
)
@pytest.mark.parametrize("finish_reason", [None, "stop"])
def test_all_real_corruption_modes_still_repair(
    content, schema, expected, mode, finish_reason
):
    result = repair(content, schema, finish_reason=finish_reason)

    assert result.status == "repaired"
    assert result.text == expected
    assert result.mode == mode
    assert result.garbage_prefix + result.text + result.trailing == content
    jsonschema.validate(json.loads(result.text), schema)


@pytest.mark.parametrize(
    "content",
    [
        '```json{"summary": "x"}',
        '```{"summary": "x"}',
        '```json{"summary": "x"}```',
        '```JSON\n{"summary": "x"}\n```',
        '```json{{"summary": "x"}',
        '```json{"{"summary": "x"}',
    ],
)
def test_fused_and_combined_variants_still_repair(content):
    result = repair(content, OBJECT_SCHEMA, finish_reason="stop")

    assert result.status == "repaired"
    assert result.text == '{"summary": "x"}'
    assert result.mode == "code_fence"
    assert result.garbage_prefix + result.text + result.trailing == content


def test_scalar_root_not_repaired():
    result = repair("-1", {"type": "integer"}, finish_reason="stop")

    # Scalar schemas never engage recovery. This happens to be valid as a whole.
    assert result.status == "clean"
    assert result.text == "-1"


def test_invalid_scalar_root_is_unknown_without_candidate_search():
    result = repair('noise "value"', {"type": "string"}, finish_reason="stop")

    assert result.status == "unknown"
    assert result.candidates_tried == 0


def test_ambiguous_object_truncation():
    content = '{"a": {"x": 1}'

    result = repair(content, {"type": "object"}, finish_reason="stop")

    assert result.status == "ambiguous"
    assert result.text is None
    assert result.value is None


def test_ambiguous_array_truncation():
    result = repair("[[1, 2]", ARRAY_SCHEMA, finish_reason="stop")

    assert result.status == "ambiguous"
    assert result.text is None


def test_lexically_inconsistent_prefix():
    content = '"garbage{"a":{"x":1}'

    result = repair(content, {"type": "object"}, finish_reason="stop")

    # The whole text is impossible JSON, but the first opener begins a valid
    # incomplete object. Returning its nested {"x":1} would be unsafe, so the
    # candidate-level ambiguity check returns unknown and preserves the input.
    assert result.status == "unknown"
    assert result.text is None
    assert result.value is None


def test_many_openers_not_capped():
    content = "{{{{{{{{" + '{"summary": "x"}'

    result = repair(content, OBJECT_SCHEMA, finish_reason="stop")

    assert result.status == "repaired"
    assert result.text == '{"summary": "x"}'
    assert result.candidates_tried == 9


def test_complete_object_in_prefix():
    content = '{} note: {"summary": "x"}'

    result = repair(content, OBJECT_SCHEMA, finish_reason="stop")

    # The whole content is not a JSON prefix: text after the complete {} is
    # illegal. The first object fails the schema and the right-anchored second
    # object is therefore an unambiguous repair.
    assert result.status == "repaired"
    assert result.text == '{"summary": "x"}'
    assert result.candidates_tried == 2


def test_fence_without_opening_fence():
    result = repair("[[1]```", ARRAY_SCHEMA, finish_reason="stop")

    assert result.status == "unknown"
    assert result.text != "[1]"
    assert result.value is None


def test_closing_fence_is_allowed_with_matching_opening_fence():
    result = repair('```json\n{"summary": "x"}```', OBJECT_SCHEMA, finish_reason="stop")

    assert result.status == "repaired"
    assert result.trailing == "```"


@pytest.mark.parametrize(
    "original",
    [
        '{"summary": "x"}',
        ' \n\t {"summary": "x"} \r\n ',
    ],
)
def test_clean_passthrough_is_byte_identical(original):
    result = repair(original, OBJECT_SCHEMA, finish_reason=None)

    assert result.status == "clean"
    assert result.text == original
    assert result.value == {"summary": "x"}
    assert result.garbage_prefix == ""
    assert result.trailing == ""
    assert result.mode == "none"
    assert result.candidates_tried == 0


def test_finish_reason_cannot_be_omitted():
    with pytest.raises(TypeError):
        repair('{"summary": "x"}', OBJECT_SCHEMA)


def test_length_finish_reason_fails_closed_before_repair():
    result = repair('{{"summary": "x"}', OBJECT_SCHEMA, finish_reason="length")

    assert result.status == "incomplete"
    assert result.text is None
    assert result.candidates_tried == 0


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "https://example.invalid/hostile-schema.json", "type": "object"},
        {"type": "not-a-real-type"},
    ],
)
def test_hostile_schema_is_unknown_and_never_escapes(schema):
    result = repair('{"summary": "x"}', schema, finish_reason="stop")

    assert result.status == "unknown"
    assert result.text is None
    assert result.value is None


def test_recursion_error_from_hostile_schema_is_contained():
    schema = {"type": "object"}
    schema["properties"] = {"self": schema}

    result = repair("{}", schema, finish_reason="stop")

    assert result.status == "unknown"


def test_schema_less_content_does_not_recover_without_an_oracle():
    result = repair('```json{"a": 1}', None, finish_reason="stop")

    assert result.status == "unknown"
    assert result.candidates_tried == 0


@pytest.mark.parametrize(
    ("arguments", "expected_mode"),
    [
        ('```json{"query": "x"}', "code_fence"),
        ('{{"query": "x"}', "extra_brace"),
        ('{"{"query": "x"}', "dup_prefix"),
    ],
)
def test_repair_tool_arguments_uses_parameters_schema(arguments, expected_mode):
    result = repair_tool_arguments(
        arguments,
        TOOL_PARAMETERS_SCHEMA,
        finish_reason="tool_calls",
    )

    assert result.status == "repaired"
    assert result.text == '{"query": "x"}'
    assert result.mode == expected_mode


def test_repair_tool_arguments_requires_finish_reason():
    with pytest.raises(TypeError):
        repair_tool_arguments('{"query": "x"}', TOOL_PARAMETERS_SCHEMA)


def test_unknown_tool_arguments_are_not_repaired_without_schema():
    result = repair_tool_arguments(
        '```json["fallback", 1]',
        None,
        finish_reason="tool_calls",
    )

    assert result.status == "unknown"
    assert result.text is None


def test_parseable_schema_violation_is_ambiguous_not_repaired():
    result = repair('{"wrong": 1}', OBJECT_SCHEMA, finish_reason="stop")

    assert result.status == "ambiguous"
    assert result.text is None


def test_earliest_right_anchored_schema_valid_candidate_wins():
    content = 'garbage {"summary": "x", "nested": {}}'
    schema = {
        "type": "object",
        "required": ["summary"],
        "properties": {"summary": {"type": "string"}},
    }

    result = repair(content, schema, finish_reason="stop")

    assert result.status == "repaired"
    assert result.text == '{"summary": "x", "nested": {}}'
    assert result.candidates_tried == 1


def test_substantive_trailing_content_rejects_candidate():
    result = repair('x {"summary": "x"} trailing', OBJECT_SCHEMA, finish_reason="stop")

    assert result.status == "unknown"
    assert result.text is None


def test_max_prefix_bytes_is_measured_in_utf8_bytes():
    content = "éé{" + '"summary": "x"}'

    outside = repair(content, OBJECT_SCHEMA, finish_reason="stop", max_prefix_bytes=4)
    inside = repair(content, OBJECT_SCHEMA, finish_reason="stop", max_prefix_bytes=5)

    assert outside.status == "unknown"
    assert inside.status == "repaired"


def test_max_prefix_bytes_does_not_limit_candidate_count():
    content = "{" * 20 + '"summary": "x"}'

    result = repair(content, OBJECT_SCHEMA, finish_reason="stop", max_prefix_bytes=256)

    assert result.status == "repaired"
    assert result.candidates_tried == 20


def test_validator_cache_is_retained():
    _compiled_validator.cache_clear()

    repair('{"summary": "x"}', OBJECT_SCHEMA, finish_reason="stop")
    repair('{"summary": "y"}', OBJECT_SCHEMA, finish_reason="stop")

    assert _compiled_validator.cache_info().hits == 1
    assert _compiled_validator.cache_info().misses == 1
