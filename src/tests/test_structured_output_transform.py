import json

import jsonschema

import vllm_router.services.structured_output.transform as transform_module
from vllm_router.services.structured_output.contract import OutputContract
from vllm_router.services.structured_output.transform import (
    RepairTelemetry,
    StreamRepairer,
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


def _chunk(delta, finish_reason=None, index=0):
    payload = {
        "id": "x",
        "object": "chat.completion.chunk",
        "choices": [{"index": index, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


def _contents(out: bytes):
    """Concatenate every delta.content in an SSE byte stream."""
    parts = []
    for frame in out.split(b"\n\n"):
        if not frame.startswith(b"data: {"):
            continue
        payload = json.loads(frame[len(b"data: ") :])
        for choice in payload.get("choices", []):
            content = choice.get("delta", {}).get("content")
            if content:
                parts.append(content)
    return "".join(parts)


def test_reasoning_is_forwarded_live():
    r = StreamRepairer(CONTRACT)
    out = r.feed(_chunk({"reasoning": "thinking..."}))
    assert b"thinking..." in out  # not withheld


def test_content_is_buffered_then_repaired():
    r = StreamRepairer(CONTRACT)
    out = r.feed(_chunk({"reasoning": "t"}))
    out += r.feed(_chunk({"content": '{{"summary": '}))
    assert b"summary" not in out  # withheld while buffering
    out += r.feed(_chunk({"content": '"x"}'}, finish_reason="stop"))
    out += r.feed(b"data: [DONE]\n\n")
    out += r.flush()
    assert _contents(out) == '{"summary": "x"}'
    assert r.telemetry == [RepairTelemetry("repaired", "extra_brace", 1)]


def test_clean_stream_content_is_unchanged():
    r = StreamRepairer(CONTRACT)
    out = r.feed(_chunk({"content": '{"summary": "x"}'}, finish_reason="stop"))
    out += r.feed(b"data: [DONE]\n\n") + r.flush()
    assert _contents(out) == '{"summary": "x"}'
    assert r.telemetry == [RepairTelemetry("clean", "none", 0)]


def test_truncated_stream_passes_through():
    r = StreamRepairer(CONTRACT)
    out = r.feed(_chunk({"content": '{{"summary": "x'}, finish_reason="length"))
    out += r.flush()
    assert _contents(out) == '{{"summary": "x'  # original, untouched
    assert r.telemetry == [RepairTelemetry("incomplete", "other", 0)]


def test_abort_replays_withheld_frames_byte_for_byte():
    r = StreamRepairer(CONTRACT)
    first = _chunk({"content": '{{"summary": '})
    r.feed(first)
    replayed = r.abort()
    assert replayed == first  # the client is never left with less than today


def test_cap_breach_replays_and_disables():
    r = StreamRepairer(CONTRACT, max_buffered_bytes=32)
    original = _chunk({"content": "x" * 200})
    out = r.feed(original)
    assert out == original
    assert r.telemetry == [RepairTelemetry("capped", "other", 0)]
    out += r.feed(_chunk({"content": "more"}, finish_reason="stop"))
    assert b"more" in out  # repair disabled, straight passthrough


def test_cap_counts_large_unterminated_parser_tail():
    r = StreamRepairer(CONTRACT, max_buffered_bytes=32)
    original = b"data: " + b"x" * 200
    assert r.feed(original) == original
    assert r.telemetry == [RepairTelemetry("capped", "other", 0)]
    assert r.feed(b"after") == b"after"


def test_done_without_finish_reason_replays():
    r = StreamRepairer(CONTRACT)
    frame = _chunk({"content": '{{"summary": "x"}'})
    done = b"data: [DONE]\n\n"
    r.feed(frame)
    out = r.feed(done) + r.flush()
    assert out == frame + done
    assert _contents(out) == '{{"summary": "x"}'  # replayed unrepaired, not dropped
    assert r.telemetry == [RepairTelemetry("no_terminal", "other", 0)]


def test_heartbeat_containing_done_does_not_terminate():
    r = StreamRepairer(CONTRACT)
    first = _chunk({"content": '{{"summary": '})
    assert r.feed(first) == b""
    assert r.feed(b": heartbeat [DONE]\n\n") == b""
    out = r.feed(_chunk({"content": '"x"}'}, finish_reason="stop"))
    assert out == b""
    out += r.flush()
    assert _contents(out) == '{"summary": "x"}'


def test_unengaged_contract_is_passthrough():
    r = StreamRepairer(OutputContract())
    frame = _chunk({"content": '{{"summary": "x"}'}, finish_reason="stop")
    assert r.feed(frame) == frame


def test_malformed_choices_and_delta_are_total_and_byte_exact():
    frames = [
        b'data: {"choices": null}\n\n',
        b'data: {"choices": [null, {"delta": null}]}\n\n',
        b'data: {"choices": [{"index": [], "delta": {"content": "x"}}]}\n\n',
    ]
    r = StreamRepairer(CONTRACT)
    out = b"".join(r.feed(frame) for frame in frames) + r.flush()
    assert out == b"".join(frames)


def test_interleaved_finish_for_other_index_never_creates_hybrid():
    """Three-frame regression for terminal detection plus per-cycle state reset."""
    frames = [
        _chunk({"content": '{{"summary": '}, index=0),
        _chunk({}, finish_reason="stop", index=1),
        _chunk({"content": '"x"}'}, finish_reason="stop", index=0),
    ]
    original = b"".join(frames)
    r = StreamRepairer(CONTRACT)
    out = b"".join(r.feed(frame) for frame in frames) + r.flush()

    assert out != original
    assert _contents(out) == '{"summary": "x"}'
    assert json.loads(_contents(out)) == {"summary": "x"}


def test_feed_deadline_replays_and_disables():
    now = [0.0]
    r = StreamRepairer(CONTRACT, max_buffer_seconds=30.0, clock=lambda: now[0])
    first = _chunk({"content": '{{"summary": '})
    second = _chunk({"content": '"x"}'}, finish_reason="stop")
    assert r.feed(first) == b""
    now[0] = 31.0
    assert r.feed(second) == first + second
    assert r.telemetry == [RepairTelemetry("timeout", "other", 0)]
    assert r.feed(b"after") == b"after"


def test_multiline_data_content_mid_buffer_poisons_and_replays_byte_exact():
    frames = [
        _chunk({"content": '{{"summary": '}),
        (
            b'data: {"choices":[{"index":0,\n'
            b'data: "delta":{"content":"leaked"}}]}\n\n'
        ),
        _chunk({"content": '"x"}'}, finish_reason="stop"),
    ]
    original = b"".join(frames)
    r = StreamRepairer(CONTRACT)

    out = b"".join(r.feed(frame) for frame in frames) + r.flush()

    assert out == original


def test_sse_event_line_with_data_content_poisons_and_replays_byte_exact():
    frames = [
        _chunk({"content": '{{"summary": '}),
        b'event: message\ndata: {"choices":[{"index":0,"delta":{"content":"leaked"}}]}\n\n',
        _chunk({"content": '"x"}'}, finish_reason="stop"),
    ]
    original = b"".join(frames)
    r = StreamRepairer(CONTRACT)

    out = b"".join(r.feed(frame) for frame in frames) + r.flush()

    assert out == original


def test_sse_id_line_with_data_content_poisons_and_replays_byte_exact():
    frames = [
        _chunk({"content": '{{"summary": '}),
        b'id: 7\ndata: {"choices":[{"index":0,"delta":{"content":"leaked"}}]}\n\n',
        _chunk({"content": '"x"}'}, finish_reason="stop"),
    ]
    original = b"".join(frames)
    r = StreamRepairer(CONTRACT)

    out = b"".join(r.feed(frame) for frame in frames) + r.flush()

    assert out == original


def test_string_choice_index_with_content_poisons_and_replays_byte_exact():
    invalid = b'data: {"choices":[{"index":"0","delta":{"content":"leaked"}}]}\n\n'
    frames = [
        _chunk({"content": '{{"summary": '}),
        invalid,
        _chunk({"content": '"x"}'}, finish_reason="stop"),
    ]
    original = b"".join(frames)
    r = StreamRepairer(CONTRACT)

    out = b"".join(r.feed(frame) for frame in frames) + r.flush()

    assert out == original
    assert r.telemetry == [RepairTelemetry("poisoned", "other", 0)]


def test_null_delta_poisons_and_replays_byte_exact_without_attribute_error():
    frames = [
        _chunk({"content": '{{"summary": '}),
        b'data: {"choices":[{"index":0,"delta":null}]}\n\n',
        _chunk({"content": '"x"}'}, finish_reason="stop"),
    ]
    original = b"".join(frames)
    r = StreamRepairer(CONTRACT)

    out = b"".join(r.feed(frame) for frame in frames) + r.flush()

    assert out == original


def test_non_json_data_mid_buffer_poisons_and_replays_byte_exact():
    frames = [
        _chunk({"content": '{{"summary": '}),
        b"data: {not JSON}\n\n",
        _chunk({"content": '"x"}'}, finish_reason="stop"),
    ]
    original = b"".join(frames)
    r = StreamRepairer(CONTRACT)

    out = b"".join(r.feed(frame) for frame in frames) + r.flush()

    assert out == original


def test_comment_heartbeat_mid_buffer_is_accounted_and_reemitted_once():
    heartbeat = b": heartbeat with no data field\r\n\r\n"
    frames = [
        _chunk({"content": '{{"summary": '}),
        heartbeat,
        _chunk({"content": '"x"}'}, finish_reason="stop"),
    ]
    r = StreamRepairer(CONTRACT)

    out = b"".join(r.feed(frame) for frame in frames) + r.flush()

    assert out != b"".join(frames)
    assert _contents(out) == '{"summary": "x"}'
    assert out.count(heartbeat) == 1


def test_usage_only_chunk_mid_buffer_is_accounted_and_reemitted_once():
    usage = b'data: {"usage":{"prompt_tokens":1,"completion_tokens":2}}\n\n'
    frames = [
        _chunk({"content": '{{"summary": '}),
        usage,
        _chunk({"content": '"x"}'}, finish_reason="stop"),
    ]
    r = StreamRepairer(CONTRACT)

    out = b"".join(r.feed(frame) for frame in frames) + r.flush()

    assert out != b"".join(frames)
    assert _contents(out) == '{"summary": "x"}'
    assert out.count(usage) == 1


def test_length_then_stop_for_one_of_two_choices_declines_and_replays_byte_exact():
    frames = [
        _chunk({"content": '{{"summary": "x"}'}, index=0),
        _chunk({"content": '{"summary": "y"}'}, index=1),
        _chunk({}, finish_reason="length", index=0),
        _chunk({}, finish_reason="stop", index=0),
        _chunk({}, finish_reason="stop", index=1),
    ]
    original = b"".join(frames)
    r = StreamRepairer(CONTRACT)

    out = b"".join(r.feed(frame) for frame in frames) + r.flush()

    assert out == original
    assert r.telemetry == [RepairTelemetry("incomplete", "other", 0)]


def test_content_after_finish_for_same_index_poisons_and_replays_byte_exact():
    frames = [
        _chunk({"content": '{{"summary": '}, index=0),
        _chunk({"content": '{"summary": "other"}'}, index=1),
        _chunk({}, finish_reason="stop", index=0),
        _chunk({"content": '"leaked"}'}, index=0),
        _chunk({}, finish_reason="stop", index=1),
    ]
    original = b"".join(frames)
    r = StreamRepairer(CONTRACT)

    out = b"".join(r.feed(frame) for frame in frames) + r.flush()

    assert out == original


def test_clock_failure_at_buffering_start_does_not_drop_triggering_frame():
    frame = _chunk({"content": '{{"summary": "x"}'}, finish_reason="stop")

    def failing_clock():
        raise RuntimeError("clock failed")

    r = StreamRepairer(CONTRACT, clock=failing_clock)

    out = r.feed(frame) + r.flush()

    assert out == frame


def test_failing_logger_on_exception_path_still_replays_every_frame(monkeypatch):
    first = _chunk({"content": '{{"summary": '})
    second = _chunk({"content": '"x"}'}, finish_reason="stop")
    r = StreamRepairer(CONTRACT)
    assert r.feed(first) == b""

    def parse_failure(*args, **kwargs):
        raise RecursionError("malicious nesting")

    def logging_failure(*args, **kwargs):
        raise RuntimeError("logger failed")

    monkeypatch.setattr(transform_module.json, "loads", parse_failure)
    monkeypatch.setattr(transform_module.logger, "warning", logging_failure)

    out = r.feed(second) + r.flush()

    assert out == first + second
    assert r.telemetry == [RepairTelemetry("error", "other", 0)]


def test_flush_repair_exception_replays_byte_exact_with_error_telemetry(monkeypatch):
    original = _chunk(
        {"content": '{{"summary": "x"}'},
        finish_reason="stop",
    )
    r = StreamRepairer(CONTRACT)
    assert r.feed(original) == b""

    def repair_failure(*args, **kwargs):
        raise RuntimeError("repair failed")

    monkeypatch.setattr(transform_module, "repair", repair_failure)

    assert r.flush() == original
    assert r.telemetry == [RepairTelemetry("error", "other", 0)]


def test_flush_unterminated_tail_replays_byte_exact_with_error_telemetry():
    first = _chunk({"content": '{{"summary": '})
    tail = b'data: {"choices": ['
    r = StreamRepairer(CONTRACT)

    assert r.feed(first) == b""
    assert r.feed(tail) == b""

    assert r.flush() == first + tail
    assert r.telemetry == [RepairTelemetry("error", "other", 0)]


def test_replayed_multi_choice_stream_publishes_only_decline_telemetry():
    frames = [
        _chunk({"content": '{{"summary": "repairable"}'}, index=0),
        _chunk({"content": "not JSON"}, index=1),
        _chunk({}, finish_reason="stop", index=0),
        _chunk({}, finish_reason="stop", index=1),
    ]
    original = b"".join(frames)
    r = StreamRepairer(CONTRACT)

    out = b"".join(r.feed(frame) for frame in frames) + r.flush()

    assert out == original
    assert r.telemetry == [RepairTelemetry("unknown", "other", 0)]


def test_rewritten_content_frame_preserves_crlf_terminator():
    frames = [
        _chunk({"content": '{{"summary": '}).replace(b"\n", b"\r\n"),
        _chunk({"content": '"x"}'}, finish_reason="stop").replace(b"\n", b"\r\n"),
    ]
    r = StreamRepairer(CONTRACT)

    out = b"".join(r.feed(frame) for frame in frames) + r.flush()

    assert _contents(out.replace(b"\r\n", b"\n")) == '{"summary": "x"}'
    assert b"\n\n" not in out
    assert out.count(b"\r\n\r\n") == 2


def test_rewritten_content_frame_preserves_raw_non_ascii_utf8():
    text = "café ☕"
    payload = {
        "choices": [
            {
                "index": 0,
                "delta": {"content": '{{"summary": "café ☕"}'},
                "finish_reason": "stop",
            }
        ]
    }
    frame = ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode(
        "utf-8"
    )
    r = StreamRepairer(CONTRACT)

    out = r.feed(frame) + r.flush()

    assert out != frame
    assert text.encode("utf-8") in out
    assert b"\\u00e9" not in out
    assert b"\\u2615" not in out
    assert json.loads(_contents(out)) == {"summary": text}


def test_two_terminal_sequences_in_one_stream_never_commit_twice():
    frames = [
        _chunk({"content": '{{"summary": "x"}'}, finish_reason="stop"),
        _chunk({"content": '{{"summary": "y"}'}, finish_reason="stop"),
    ]
    original = b"".join(frames)
    r = StreamRepairer(CONTRACT)

    out = b"".join(r.feed(frame) for frame in frames) + r.flush()

    assert out == original
    assert _contents(out) == '{{"summary": "x"}{{"summary": "y"}'
    assert r.telemetry == [RepairTelemetry("poisoned", "other", 0)]


def test_null_content_in_preamble_does_not_disable_repair():
    preamble = _chunk({"content": None, "reasoning_content": "thinking"})
    content = _chunk({"content": '{{"summary": "x"}'}, finish_reason="stop")
    done = b"data: [DONE]\n\n"
    r = StreamRepairer(CONTRACT)

    out = r.feed(preamble)
    assert out == preamble
    out += r.feed(content) + r.feed(done) + r.flush()

    assert _contents(out) == '{"summary": "x"}'
    assert r.telemetry == [RepairTelemetry("repaired", "extra_brace", 1)]


def test_null_content_on_final_chunk_does_not_disable_repair():
    frames = [
        _chunk({"content": '{{"summary": "x"}'}),
        _chunk({"content": None}, finish_reason="stop"),
        b"data: [DONE]\n\n",
    ]
    r = StreamRepairer(CONTRACT)

    out = b"".join(r.feed(frame) for frame in frames) + r.flush()

    assert _contents(out) == '{"summary": "x"}'
    assert r.telemetry == [RepairTelemetry("repaired", "extra_brace", 1)]


def test_missing_choice_index_defaults_to_zero_and_repairs():
    payload = {
        "choices": [
            {
                "delta": {"content": '{{"summary": "x"}'},
                "finish_reason": "stop",
            }
        ]
    }
    frame = f"data: {json.dumps(payload)}\n\n".encode()
    r = StreamRepairer(CONTRACT)

    out = r.feed(frame) + r.flush()

    assert _contents(out) == '{"summary": "x"}'
    assert r.telemetry == [RepairTelemetry("repaired", "extra_brace", 1)]


def test_done_commits_repaired_frames_before_emitting_done_once():
    content = _chunk({"content": '{{"summary": "x"}'}, finish_reason="stop")
    done = b"data: [DONE]\n\n"
    r = StreamRepairer(CONTRACT)

    assert r.feed(content) == b""
    out = r.feed(done)

    assert _contents(out) == '{"summary": "x"}'
    assert out.endswith(done)
    assert out.count(done) == 1
    assert out.index(b'"content"') < out.index(done)
    assert r.flush() == b""


def test_flush_commits_terminal_stream_without_done():
    content = _chunk({"content": '{{"summary": "x"}'}, finish_reason="stop")
    r = StreamRepairer(CONTRACT)

    assert r.feed(content) == b""
    out = r.flush()

    assert _contents(out) == '{"summary": "x"}'
    assert r.telemetry == [RepairTelemetry("repaired", "extra_brace", 1)]


def test_later_choice_can_start_after_first_choice_finishes():
    frames = [
        _chunk(
            {"content": '{{"summary": "first"}'},
            finish_reason="stop",
            index=0,
        ),
        _chunk(
            {"content": '{{"summary": "second"}'},
            finish_reason="stop",
            index=1,
        ),
        b"data: [DONE]\n\n",
    ]
    r = StreamRepairer(CONTRACT)

    out = b"".join(r.feed(frame) for frame in frames) + r.flush()

    contents_by_index = {}
    for frame in out.split(b"\n\n"):
        if not frame.startswith(b"data: {"):
            continue
        payload = json.loads(frame[len(b"data: ") :])
        for choice in payload.get("choices", []):
            content = choice.get("delta", {}).get("content")
            if content:
                contents_by_index.setdefault(choice.get("index", 0), "")
                contents_by_index[choice.get("index", 0)] += content

    assert json.loads(contents_by_index[0]) == {"summary": "first"}
    assert json.loads(contents_by_index[1]) == {"summary": "second"}
    assert [item.status for item in r.telemetry] == ["repaired", "repaired"]


def test_repair_increments_counter():
    from vllm_router.services.metrics_service import structured_output_repairs_total

    def _value():
        for metric in structured_output_repairs_total.collect():
            for sample in metric.samples:
                if sample.labels == {
                    "model": "m",
                    "status": "repaired",
                    "mode": "extra_brace",
                }:
                    return sample.value
        return 0.0

    before = _value()
    structured_output_repairs_total.labels(
        model="m", status="repaired", mode="extra_brace"
    ).inc()
    assert _value() == before + 1


def test_repair_counter_accepts_every_telemetry_status():
    from vllm_router.services.metrics_service import structured_output_repairs_total

    statuses = {
        "clean",
        "repaired",
        "incomplete",
        "ambiguous",
        "unknown",
        "poisoned",
        "no_terminal",
        "capped",
        "timeout",
        "error",
    }

    for status in statuses:
        structured_output_repairs_total.labels(
            model="all-statuses", status=status, mode="other"
        ).inc()

    observed = {
        sample.labels["status"]
        for metric in structured_output_repairs_total.collect()
        for sample in metric.samples
        if sample.name == "vllm:structured_output_repairs_total"
        and sample.labels.get("model") == "all-statuses"
        and sample.labels.get("mode") == "other"
    }
    assert observed == statuses


def test_non_discriminating_rejection_counter_has_bounded_labels():
    from vllm_router.services.metrics_service import (
        structured_output_schema_rejections_total,
    )

    labels = {"model": "m", "reason": "non_discriminating"}

    def _value():
        for metric in structured_output_schema_rejections_total.collect():
            for sample in metric.samples:
                if sample.labels == labels:
                    return sample.value
        return 0.0

    before = _value()
    structured_output_schema_rejections_total.labels(**labels).inc()
    assert _value() == before + 1


def test_non_streaming_ambiguous_is_offered_to_capture_callback():
    content = '{"summary": {"summary": "x"}'
    body = _body(content)
    captured = []
    out, telemetry = transform_response_body(
        body,
        CONTRACT,
        capture_callback=lambda raw, event: captured.append((raw, event)),
    )
    assert out is body
    assert telemetry[0].status == "ambiguous"
    assert captured == [(content, telemetry[0])]


def test_raising_capture_callback_is_fail_safe():
    body = _body('{"summary": {"summary": "x"}')

    def explode(raw, event):
        raise RuntimeError("sink unavailable")

    out, telemetry = transform_response_body(body, CONTRACT, capture_callback=explode)
    assert out is body
    assert telemetry[0].status == "ambiguous"


def test_streaming_unknown_is_offered_to_capture_callback():
    captured = []
    repairer = StreamRepairer(
        CONTRACT,
        capture_callback=lambda raw, event: captured.append((raw, event)),
    )
    frame = _chunk({"content": "not json"}, finish_reason="stop")
    assert repairer.feed(frame) + repairer.flush() == frame
    assert len(captured) == 1
    assert captured[0][0] == "not json"
    assert captured[0][1].status == "unknown"
