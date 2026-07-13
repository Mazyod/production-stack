import json

import jsonschema

import vllm_router.services.structured_output.transform as transform_module
from vllm_router.services.structured_output.contract import OutputContract
from vllm_router.services.structured_output.transform import (
    RepairTelemetry,
    transform_response_body,
)

STRICT = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}
CONTRACT = OutputContract(content_schema=STRICT)


def _body(content, finish_reason="stop"):
    return json.dumps(
        {
            "id": "x",
            "choices": [
                {
                    "index": 0,
                    "message": {"content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
    ).encode()


def test_repairs_doubled_brace_with_stop_finish_reason():
    out, telemetry = transform_response_body(_body('{{"summary": "x"}'), CONTRACT)
    assert json.loads(out)["choices"][0]["message"]["content"] == '{"summary": "x"}'
    assert telemetry == [RepairTelemetry("repaired", "extra_brace", 1)]


def test_missing_finish_reason_passes_through_unchanged():
    payload = json.loads(_body('{{"summary": "x"}'))
    del payload["choices"][0]["finish_reason"]
    body = json.dumps(payload).encode()

    out, telemetry = transform_response_body(body, CONTRACT)

    assert out is body
    assert json.loads(out)["choices"][0]["message"]["content"] == '{{"summary": "x"}'
    assert telemetry == []


def test_null_finish_reason_passes_through_unchanged():
    body = _body('{{"summary": "x"}', finish_reason=None)

    out, telemetry = transform_response_body(body, CONTRACT)

    assert out is body
    assert json.loads(out)["choices"][0]["message"]["content"] == '{{"summary": "x"}'
    assert telemetry == []


def test_clean_body_is_byte_identical():
    body = _body('{"summary": "x"}')
    out, telemetry = transform_response_body(body, CONTRACT)
    assert out is body
    assert telemetry == [RepairTelemetry("clean", "none", 0)]


def test_discarded_repair_does_not_report_telemetry_on_encoding_failure():
    filler = "a" * 300
    content = "{" + '{"summary": "' + filler + chr(0xD800) + 'b"}'
    body = _body(content)

    out, telemetry = transform_response_body(body, CONTRACT)

    assert out is body
    assert telemetry == []


def test_partially_completed_repair_pass_does_not_report_telemetry(monkeypatch):
    body = json.dumps(
        {
            "choices": [
                {
                    "message": {"content": '{{"summary": "first"}'},
                    "finish_reason": "stop",
                },
                {
                    "message": {"content": '{{"summary": "second"}'},
                    "finish_reason": "stop",
                },
            ]
        }
    ).encode()
    original_repair = transform_module.repair
    calls = 0

    def raise_on_second_choice(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second choice failed")
        return original_repair(*args, **kwargs)

    monkeypatch.setattr(transform_module, "repair", raise_on_second_choice)

    out, telemetry = transform_response_body(body, CONTRACT)

    assert calls == 2
    assert out is body
    assert telemetry == []


def test_length_finish_reason_passes_through_unchanged():
    body = _body('{{"summary": "x"}', finish_reason="length")
    out, telemetry = transform_response_body(body, CONTRACT)
    assert out is body
    assert telemetry == [RepairTelemetry("incomplete", "other", 0)]


def test_repair_preserves_raw_non_ascii_text():
    text = "مرحبا 🎉"
    body = _body('{{"summary": "مرحبا 🎉"}')

    out, telemetry = transform_response_body(body, CONTRACT)

    assert text.encode("utf-8") in out
    assert b"\\u" not in out
    repaired_content = json.loads(out)["choices"][0]["message"]["content"]
    repaired_document = json.loads(repaired_content)
    jsonschema.validate(repaired_document, STRICT)
    assert repaired_document == {"summary": text}
    assert telemetry == [RepairTelemetry("repaired", "extra_brace", 1)]


def test_usage_is_preserved():
    out, _ = transform_response_body(_body('{{"summary": "x"}'), CONTRACT)
    assert json.loads(out)["usage"] == {"prompt_tokens": 1, "completion_tokens": 2}


def test_unparseable_body_passes_through():
    body = b"not json at all"
    out, telemetry = transform_response_body(body, CONTRACT)
    assert out is body
    assert telemetry == []


def test_not_engaged_passes_through():
    body = _body('{{"summary": "x"}')
    out, telemetry = transform_response_body(body, OutputContract())
    assert out is body
    assert telemetry == []


def test_recursion_error_is_contained(monkeypatch):
    body = _body('{{"summary": "x"}')

    def explode(*args, **kwargs):
        raise RecursionError("malicious nesting")

    monkeypatch.setattr(json, "loads", explode)
    out, telemetry = transform_response_body(body, CONTRACT)
    assert out is body
    assert telemetry == []


def test_logger_failure_is_contained(monkeypatch):
    body = _body('{{"summary": "x"}')

    def transform_explode(*args, **kwargs):
        raise RuntimeError("transform failed")

    def warning_explode(*args, **kwargs):
        raise RuntimeError("logger failed")

    monkeypatch.setattr(transform_module, "_transform_response_body", transform_explode)
    monkeypatch.setattr(transform_module.logger, "warning", warning_explode)

    out, telemetry = transform_response_body(body, CONTRACT)

    assert out is body
    assert telemetry == []


def test_partial_repair_with_multiple_choices_is_deliberate():
    repairable = '{{"summary": "repair me"}'
    unrepairable = "not JSON"
    body = json.dumps(
        {
            "id": "x",
            "choices": [
                {
                    "index": 0,
                    "message": {"content": repairable},
                    "finish_reason": "stop",
                },
                {
                    "index": 1,
                    "message": {"content": unrepairable},
                    "finish_reason": "stop",
                },
            ],
        }
    ).encode()

    out, telemetry = transform_response_body(body, CONTRACT)

    # Repair what we can; never worsen what we can't.
    choices = json.loads(out)["choices"]
    repaired_document = json.loads(choices[0]["message"]["content"])
    jsonschema.validate(repaired_document, STRICT)
    assert repaired_document == {"summary": "repair me"}
    assert choices[1]["message"]["content"] == unrepairable
    assert [entry.status for entry in telemetry] == ["repaired", "unknown"]
