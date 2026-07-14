# Structured-Output Boundary Repair — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair vLLM's grammar-produced JSON that is corrupted at the reasoning→answer boundary, centrally in the router, without changing caller-visible model semantics.

**Architecture:** A pure repair core (`repair()`) locates the true JSON document `J` inside a corrupted `G + J` response, using the caller's own JSON Schema as an oracle and an "is this a valid JSON prefix?" gate to refuse anything that could be a truncation. It is wired into `route_general_request` behind a feature flag: engaged requests take a **buffered** non-streaming path or a **frame-retaining** streaming path; everything else is untouched.

**Tech Stack:** Python 3.13, FastAPI, aiohttp, `jsonschema`, `prometheus_client`, pytest (`asyncio_mode = "auto"`).

**Spec:** `docs/superpowers/specs/2026-07-13-structured-output-boundary-repair-design.md` — read §3 and §5 before starting.

**Validated spike:** `spikes/json_boundary_repair/` — 69 passing tests. Tasks 1–2 promote this code; **do not rewrite the algorithm.**

## Global Constraints

- **Fail safe, always.** Every failure path (`incomplete`, `ambiguous`, `unknown`, any exception, any cap breach, any transport error) emits the **original bytes unchanged**. No exception may escape the transform. The worst case of enabling this feature is *no change*.
- **`finish_reason` is a REQUIRED keyword** on `repair()`. No default. A default fails open.
- **Non-structured traffic must be byte-for-byte unchanged**, via an early bypass *before* any buffering.
- **Never engage** when: `logprobs` is requested; the root schema is scalar-typed; the root schema is not discriminating (no non-empty `required` and not `additionalProperties: false`); `response_format` is `json_object` (no schema); the backend returned non-2xx.
- **Tool-call repair is out of scope for v1.** Engagement requires a discriminating content schema; `tools` alone never engages repair.
- **Both buffering caps are real:** `max_buffered_bytes` counts retained frames plus the `SSEParser` tail (default 1 MiB), and `max_buffer_seconds` is enforced both in `StreamRepairer.feed()` and by a non-cancelling `repaired_stream()` watchdog (default 30 seconds).
- **Never transform a non-2xx response.** The contract is replaced with an empty `OutputContract` unless `200 <= status < 300`; streaming and non-streaming error bodies remain byte-for-byte unchanged.
- **Remote `$ref` retrieval must be disabled** in the schema validator. An untrusted schema must never cause the router to fetch a URL.
- **Never put raw model output in a metric label or an ordinary log line.** It can contain personal data.
- Feature flag default **off**.
- Out of scope: the disaggregated-prefill response paths (`request.py:880`, `1019`, `1303`).
- Lint gate: `uv run pre-commit run --all-files`. Tests: `uv run pytest src/tests/`.

---

## File Structure

**Create:**

- `src/vllm_router/services/structured_output/__init__.py` — public exports
- `src/vllm_router/services/structured_output/json_prefix.py` — `is_valid_json_prefix()` (the ambiguity gate)
- `src/vllm_router/services/structured_output/repair.py` — `repair()`, `RepairResult` (the pure core)
- `src/vllm_router/services/structured_output/contract.py` — `OutputContract`, `extract_output_contract()` (the engagement decision)
- `src/vllm_router/services/structured_output/sse.py` — byte-preserving incremental SSE parser
- `src/vllm_router/services/structured_output/transform.py` — non-streaming transform + streaming state machine
- `src/vllm_router/services/structured_output/capture.py` — bounded, sampled, redacted diagnostic capture sink
- `src/tests/test_structured_output_repair.py`
- `src/tests/test_structured_output_contract.py`
- `src/tests/test_structured_output_sse.py`
- `src/tests/test_structured_output_transform.py`
- `src/tests/test_structured_output_integration.py`
- `src/tests/test_structured_output_capture.py`

**Modify:**

- `pyproject.toml:14-34` — add `jsonschema`
- `src/vllm_router/requirements.txt` — add `jsonschema` (must stay in sync with pyproject)
- `src/vllm_router/services/metrics_service/__init__.py` — add metrics
- `src/vllm_router/parsers/parser.py` — add flags
- `src/vllm_router/app.py:161` — wire config into `app.state`
- `src/vllm_router/services/request_service/request.py:625-676` — engagement + buffered/streaming branches
- `src/vllm_router/routers/main_router.py:51-60` — skip semantic cache for engaged requests
- `src/vllm_router/experimental/semantic_cache_integration.py:181-205` — accept post-rewrite JSON for lookup
- `README.md` — feature flags and secure-capture operations

---

### Task 1: Promote the repair core

**Files:**

- Create: `src/vllm_router/services/structured_output/__init__.py`, `json_prefix.py`, `repair.py`
- Modify: `pyproject.toml`, `src/vllm_router/requirements.txt`
- Test: `src/tests/test_structured_output_repair.py`

**Interfaces:**

- Consumes: nothing.
- Produces: `is_valid_json_prefix(content: str) -> bool`; `repair(content: str, schema: dict | None, *, finish_reason: str | None, max_prefix_bytes: int = 256) -> RepairResult`; `RepairResult(status, text, value, garbage_prefix, trailing, mode, candidates_tried)` where `status` ∈ `{"clean","repaired","incomplete","ambiguous","unknown"}`.

- [ ] **Step 1: Add the dependency to both files**

`pyproject.toml` — add to the `dependencies` list (the file comment says it must stay in sync with `src/vllm_router/requirements.txt`):

```toml
    "jsonschema>=4.23,<5",
```

`src/vllm_router/requirements.txt` — add the same line:

```text
jsonschema>=4.23,<5
```

- [ ] **Step 2: Install and verify**

Run: `uv sync && uv run python -c "import jsonschema; print(jsonschema.__version__)"`
Expected: prints `4.x` (it was previously NOT installed).

- [ ] **Step 3: Copy the validated spike code**

Copy `spikes/json_boundary_repair/repair.py` into the new package, splitting it:

- `json_prefix.py` gets `is_valid_json_prefix()` and its helpers.
- `repair.py` gets `RepairResult`, `repair()`, the validator cache, and imports `is_valid_json_prefix` from `json_prefix`.

**Do not change the algorithm.** It is validated by 69 tests and a property check over 95,854 prefixes.

`src/vllm_router/services/structured_output/__init__.py`:

```python
from vllm_router.services.structured_output.repair import RepairResult, repair

__all__ = ["RepairResult", "repair"]
```

- [ ] **Step 4: Port the spike's tests**

Copy `spikes/json_boundary_repair/test_repair.py` to `src/tests/test_structured_output_repair.py`, replacing its local import with this complete import list (the tool-argument seam remains in the core even though router wiring does not call it in v1):

```python
from vllm_router.services.structured_output.json_prefix import is_valid_json_prefix
from vllm_router.services.structured_output.repair import (
    _compiled_validator,
    repair,
    repair_tool_arguments,
)
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest src/tests/test_structured_output_repair.py -q`
Expected: 69 passed.

- [ ] **Step 6: Delete the spike**

```bash
git rm -r spikes/json_boundary_repair
```

The code now lives in the package; leaving a copy invites drift.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/vllm_router/requirements.txt src/vllm_router/services/structured_output src/tests/test_structured_output_repair.py
git commit -m "feat(structured-output): add boundary JSON repair core"
```

---

### Task 2: Output contract extraction (the engagement decision)

**Files:**

- Create: `src/vllm_router/services/structured_output/contract.py`
- Test: `src/tests/test_structured_output_contract.py`

**Interfaces:**

- Consumes: nothing from Task 1.
- Produces: `OutputContract(content_schema: dict | None, rejected_non_discriminating: bool, engaged: bool)` and `extract_output_contract(request_json: dict) -> OutputContract`.

Tool schemas are deliberately absent. In v1, a request engages only through a discriminating **content** schema. `repair_tool_arguments()` remains a core seam for a later, corpus-backed design.

This is where every "never engage" rule from the Global Constraints is enforced. It is the blast-radius boundary.

- [ ] **Step 1: Write the failing tests**

`src/tests/test_structured_output_contract.py`:

```python
import pytest

from vllm_router.services.structured_output.contract import extract_output_contract

STRICT = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}


def _req(**kw):
    base = {"model": "m", "messages": []}
    base.update(kw)
    return base


def _rf(schema):
    return {"type": "json_schema", "json_schema": {"name": "s", "schema": schema}}


def test_engages_on_strict_object_schema():
    c = extract_output_contract(_req(response_format=_rf(STRICT)))
    assert c.engaged is True
    assert c.content_schema == STRICT


def test_not_engaged_without_response_format():
    assert extract_output_contract(_req()).engaged is False


def test_not_engaged_on_json_object_mode():
    c = extract_output_contract(_req(response_format={"type": "json_object"}))
    assert c.engaged is False


def test_not_engaged_when_logprobs_requested():
    c = extract_output_contract(_req(response_format=_rf(STRICT), logprobs=True))
    assert c.engaged is False


def test_not_engaged_on_scalar_root():
    c = extract_output_contract(_req(response_format=_rf({"type": "integer"})))
    assert c.engaged is False


def test_not_engaged_on_array_root_even_if_it_looks_discriminating():
    # `additionalProperties` is an OBJECT keyword -- it says nothing about an array,
    # so it cannot reject a nested fragment here. Arrays are also where the ambiguity
    # gate is weakest (`[[1, 2]` is a valid incomplete document). Object roots only.
    # See spec §3.3 and §5.1.
    schema = {
        "type": "array",
        "items": {"type": "integer"},
        "additionalProperties": False,
    }
    c = extract_output_contract(_req(response_format=_rf(schema)))
    assert c.engaged is False


def test_not_engaged_on_non_discriminating_schema():
    # No `required`, and additionalProperties is not False -> the oracle cannot
    # reject a nested fragment. See spec §5.1.
    c = extract_output_contract(_req(response_format=_rf({"type": "object"})))
    assert c.engaged is False
    assert c.rejected_non_discriminating is True


def test_engages_on_additional_properties_false_without_required():
    schema = {"type": "object", "additionalProperties": False}
    c = extract_output_contract(_req(response_format=_rf(schema)))
    assert c.engaged is True


def test_tools_alone_do_not_engage_v1():
    c = extract_output_contract(
        _req(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": STRICT,
                    },
                }
            ]
        )
    )
    assert c.engaged is False


def test_structured_outputs_json_engages():
    c = extract_output_contract(_req(structured_outputs={"json": STRICT}))
    assert c.engaged is True
    assert c.content_schema == STRICT


def test_malformed_request_json_never_raises():
    assert extract_output_contract({"response_format": "nonsense"}).engaged is False
    assert extract_output_contract({"structured_outputs": {"json": None}}).engaged is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest src/tests/test_structured_output_contract.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vllm_router.services.structured_output.contract'`

- [ ] **Step 3: Implement**

`src/vllm_router/services/structured_output/contract.py`:

```python
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
    if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
        json_schema = response_format.get("json_schema")
        if isinstance(json_schema, dict):
            candidates.append(json_schema.get("schema"))

    structured_outputs = request_json.get("structured_outputs")
    if isinstance(structured_outputs, dict):
        candidates.append(structured_outputs.get("json"))

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
        if request_json.get("logprobs"):
            return OutputContract()

        content_schema, rejected = _content_schema(request_json)
        return OutputContract(content_schema, rejected)
    except Exception:  # noqa: BLE001 - engagement must never break a request
        return OutputContract()
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest src/tests/test_structured_output_contract.py -q`
Expected: 11 passed (including `tools`-only passthrough and the array-root rejection; there are no tool-schema extraction tests — tool calls are deferred, see "Deferred").

- [ ] **Step 5: Commit**

```bash
git add src/vllm_router/services/structured_output/contract.py src/tests/test_structured_output_contract.py
git commit -m "feat(structured-output): add output contract extraction and engagement rules"
```

---

### Task 3: Byte-preserving incremental SSE parser

**Files:**

- Create: `src/vllm_router/services/structured_output/sse.py`
- Test: `src/tests/test_structured_output_sse.py`

**Interfaces:**

- Produces: `SSEEvent(raw: bytes, data: str | None, is_done: bool)` and `SSEParser` with `feed(chunk: bytes) -> list[SSEEvent]`, `flush() -> bytes`, and `buffered_bytes: int`.

`raw` is the frame's **exact original bytes including its terminator** — this is what makes byte-for-byte replay possible. `data` is the decoded payload of a single-`data:` frame, or `None` for anything else (comments, heartbeats, `[DONE]`, multi-line/unknown frames), which must be passed through verbatim.

Splitting on byte terminators and decoding only *complete* frames means split UTF-8 code points are handled for free.

- [ ] **Step 1: Write the failing tests**

`src/tests/test_structured_output_sse.py`:

```python
from vllm_router.services.structured_output.sse import SSEParser


def test_single_frame():
    p = SSEParser()
    events = p.feed(b'data: {"a": 1}\n\n')
    assert len(events) == 1
    assert events[0].data == '{"a": 1}'
    assert events[0].raw == b'data: {"a": 1}\n\n'


def test_frame_split_across_chunks():
    p = SSEParser()
    assert p.feed(b'data: {"a') == []
    events = p.feed(b'": 1}\n\n')
    assert len(events) == 1
    assert events[0].data == '{"a": 1}'


def test_split_utf8_codepoint():
    payload = 'data: {"a": "é"}\n\n'.encode()
    p = SSEParser()
    # split in the middle of the 2-byte e-acute
    cut = payload.index(b"\xc3") + 1
    assert p.feed(payload[:cut]) == []
    events = p.feed(payload[cut:])
    assert len(events) == 1
    assert events[0].data == '{"a": "é"}'


def test_crlf_terminator():
    p = SSEParser()
    events = p.feed(b'data: {"a": 1}\r\n\r\n')
    assert len(events) == 1
    assert events[0].data == '{"a": 1}'
    assert events[0].raw == b'data: {"a": 1}\r\n\r\n'


def test_done_sentinel_has_no_data_payload():
    p = SSEParser()
    events = p.feed(b"data: [DONE]\n\n")
    assert events[0].data is None
    assert events[0].is_done is True
    assert events[0].raw == b"data: [DONE]\n\n"


def test_done_text_in_heartbeat_is_not_the_sentinel():
    event = SSEParser().feed(b": heartbeat [DONE]\n\n")[0]
    assert event.data is None
    assert event.is_done is False


def test_comment_and_multiline_are_opaque():
    p = SSEParser()
    events = p.feed(b": heartbeat\n\n" + b"data: a\ndata: b\n\n")
    assert [e.data for e in events] == [None, None]
    assert events[0].raw == b": heartbeat\n\n"


def test_flush_returns_unterminated_tail():
    p = SSEParser()
    p.feed(b"data: partial")
    assert p.buffered_bytes == len(b"data: partial")
    assert p.flush() == b"data: partial"
    assert p.buffered_bytes == 0


def test_raw_bytes_roundtrip_exactly():
    stream = b'data: {"a": 1}\n\n: hb\n\ndata: [DONE]\n\n'
    p = SSEParser()
    events = p.feed(stream)
    assert b"".join(e.raw for e in events) + p.flush() == stream
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest src/tests/test_structured_output_sse.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

`src/vllm_router/services/structured_output/sse.py`:

```python
"""Byte-preserving incremental SSE parser.

Frames are split on byte terminators and decoded only once complete, so a UTF-8
code point split across TCP chunks is handled for free. `raw` is the frame's exact
original bytes -- that is what makes byte-for-byte replay possible.
"""

from __future__ import annotations

from dataclasses import dataclass

_TERMINATORS = (b"\r\n\r\n", b"\n\n")
_DONE = "[DONE]"


@dataclass(frozen=True)
class SSEEvent:
    raw: bytes
    data: str | None
    is_done: bool


def _classify(frame: bytes) -> tuple[str | None, bool]:
    """Decode a frame that is exactly one `data:` line carrying JSON, else None.

    Anything else -- comments, heartbeats, [DONE], multi-line data, unknown fields --
    returns None and must be forwarded verbatim.
    """
    try:
        text = frame.decode("utf-8")
    except UnicodeDecodeError:
        return None, False

    lines = [line for line in text.replace("\r\n", "\n").split("\n") if line]
    if len(lines) != 1:
        return None, False

    line = lines[0]
    if not line.startswith("data:"):
        return None, False

    payload = line[len("data:") :]
    if payload.startswith(" "):
        payload = payload[1:]
    if payload == _DONE:
        return None, True
    if not payload:
        return None, False
    if not payload.startswith("{"):
        return None, False
    return payload, False


class SSEParser:
    def __init__(self) -> None:
        self._buffer = bytearray()

    @property
    def buffered_bytes(self) -> int:
        return len(self._buffer)

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        self._buffer.extend(chunk)
        events: list[SSEEvent] = []

        while True:
            cut = -1
            width = 0
            for terminator in _TERMINATORS:
                found = self._buffer.find(terminator)
                if found != -1 and (cut == -1 or found < cut):
                    cut = found
                    width = len(terminator)
            if cut == -1:
                break

            raw = bytes(self._buffer[: cut + width])
            del self._buffer[: cut + width]
            data, is_done = _classify(raw[: -width or None])
            events.append(SSEEvent(raw=raw, data=data, is_done=is_done))

        return events

    def flush(self) -> bytes:
        tail = bytes(self._buffer)
        self._buffer.clear()
        return tail
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest src/tests/test_structured_output_sse.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vllm_router/services/structured_output/sse.py src/tests/test_structured_output_sse.py
git commit -m "feat(structured-output): add byte-preserving incremental SSE parser"
```

---

### Task 4: Non-streaming transform

**Files:**

- Create: `src/vllm_router/services/structured_output/transform.py`
- Test: `src/tests/test_structured_output_transform.py`

**Interfaces:**

- Consumes: `repair`, `RepairResult` (Task 1); `OutputContract` (Task 2).
- Produces: `RepairTelemetry(status, mode, garbage_prefix_bytes)` and `transform_response_body(body: bytes, contract: OutputContract) -> tuple[bytes, list[RepairTelemetry]]`. **On any failure it returns the original `body` object unchanged.**

- [ ] **Step 1: Write the failing tests**

`src/tests/test_structured_output_transform.py`:

```python
import json

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
                {"index": 0, "message": {"content": content}, "finish_reason": finish_reason}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
    ).encode()


def test_repairs_doubled_brace():
    out, telemetry = transform_response_body(_body('{{"summary": "x"}'), CONTRACT)
    assert json.loads(out)["choices"][0]["message"]["content"] == '{"summary": "x"}'
    assert telemetry == [RepairTelemetry("repaired", "extra_brace", 1)]


def test_clean_body_is_byte_identical():
    body = _body('{"summary": "x"}')
    out, telemetry = transform_response_body(body, CONTRACT)
    assert out is body
    assert telemetry == [RepairTelemetry("clean", "none", 0)]


def test_truncated_passes_through_unchanged():
    body = _body('{"summary": "x', finish_reason="length")
    out, telemetry = transform_response_body(body, CONTRACT)
    assert out is body
    assert telemetry == [RepairTelemetry("incomplete", "other", 0)]


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
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest src/tests/test_structured_output_transform.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

`src/vllm_router/services/structured_output/transform.py`:

```python
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


def _repair_choice(choice: dict, contract: OutputContract) -> RepairResult | None:
    """Repair one choice's content. None means "nothing to look at here"."""
    message = choice.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, str) or not content:
        return None

    return repair(
        content,
        contract.content_schema,
        finish_reason=choice.get("finish_reason"),
    )


def transform_response_body(
    body: bytes, contract: OutputContract
) -> tuple[bytes, list[RepairTelemetry]]:
    """Contain every transform exception, including RecursionError."""
    try:
        return _transform_response_body(body, contract)
    except Exception:  # noqa: BLE001 - the transform is an availability boundary
        logger.warning("structured-output repair failed; passing response through")
        return body, []


def _transform_response_body(
    body: bytes, contract: OutputContract
) -> tuple[bytes, list[RepairTelemetry]]:
    if not contract.engaged or contract.content_schema is None:
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
        logger.warning("structured-output repair failed; passing response through")
        return body, telemetry

    if not changed:
        return body, telemetry

    try:
        return json.dumps(payload).encode("utf-8"), telemetry
    except (TypeError, ValueError):
        return body, telemetry
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest src/tests/test_structured_output_transform.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vllm_router/services/structured_output/transform.py src/tests/test_structured_output_transform.py
git commit -m "feat(structured-output): add non-streaming response transform"
```

---

### Task 5: Streaming state machine (frame retention + raw replay)

**Files:**

- Modify: `src/vllm_router/services/structured_output/transform.py`
- Test: `src/tests/test_structured_output_transform.py`

**Interfaces:**

- Consumes: `SSEParser`, `SSEEvent` (Task 3); `repair` (Task 1); `OutputContract` (Task 2).
- Produces: `StreamRepairer(contract, max_buffered_bytes=1_048_576, max_buffer_seconds=30.0)` with `feed(chunk: bytes) -> bytes`, `flush() -> bytes`, `abort() -> bytes`, `buffering: bool`, `seconds_remaining: float | None`, and `telemetry: list[RepairTelemetry]`.

**The rule that makes this safe:** forward everything until the first frame carrying `delta.content`. From that point, forward **nothing** and retain every raw frame. At the end, either rewrite the retained frames in place (success) or **replay them byte-for-byte** (any doubt). `abort()` exists so a mid-stream transport error replays what was withheld — without it the client would get *less* than it does today, which is the worst bug in the v1 design.

Rewriting *in place* (rather than synthesizing new frames) preserves ids, `model`, `usage`, `finish_reason`, `tool_calls` and any unknown extension fields for free.

> **n > 1 note:** buffering starts globally at the first content frame, so with multiple choices one choice's content start delays another choice's reasoning. Structured output with `n > 1` is rare; this is a deliberate simplification. Document it in the module docstring.

- [ ] **Step 1: Write the failing tests**

Append to `src/tests/test_structured_output_transform.py`:

```python
from vllm_router.services.structured_output.transform import StreamRepairer


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
    out = r.feed(_chunk({"content": "x" * 200}))
    assert b"x" * 200 in out  # flushed on breach rather than withheld
    out += r.feed(_chunk({"content": "more"}, finish_reason="stop"))
    assert b"more" in out  # repair disabled, straight passthrough


def test_cap_counts_large_unterminated_parser_tail():
    r = StreamRepairer(CONTRACT, max_buffered_bytes=32)
    original = b"data: " + b"x" * 200
    assert r.feed(original) == original
    assert r.feed(b"after") == b"after"


def test_done_without_finish_reason_replays():
    r = StreamRepairer(CONTRACT)
    frame = _chunk({"content": '{{"summary": "x"}'})
    r.feed(frame)
    out = r.feed(b"data: [DONE]\n\n") + r.flush()
    assert _contents(out) == '{{"summary": "x"}'  # replayed unrepaired, not dropped


def test_heartbeat_containing_done_does_not_terminate():
    r = StreamRepairer(CONTRACT)
    first = _chunk({"content": '{{"summary": '})
    assert r.feed(first) == b""
    assert r.feed(b": heartbeat [DONE]\n\n") == b""
    out = r.feed(_chunk({"content": '"x"}'}, finish_reason="stop"))
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

    if out == original:
        assert out == original
    else:
        assert json.loads(_contents(out)) == {"summary": "x"}


def test_feed_deadline_replays_and_disables():
    now = [0.0]
    r = StreamRepairer(CONTRACT, max_buffer_seconds=30.0, clock=lambda: now[0])
    first = _chunk({"content": '{{"summary": '})
    second = _chunk({"content": '"x"}'}, finish_reason="stop")
    assert r.feed(first) == b""
    now[0] = 31.0
    assert r.feed(second) == first + second
    assert r.feed(b"after") == b"after"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest src/tests/test_structured_output_transform.py -q -k Stream`
Expected: FAIL — `ImportError: cannot import name 'StreamRepairer'`

- [ ] **Step 3: Implement**

Append to `src/vllm_router/services/structured_output/transform.py`:

```python
import time
from collections.abc import Callable

from vllm_router.services.structured_output.sse import SSEEvent, SSEParser

_MAX_BUFFERED_BYTES = 1_048_576
_MAX_BUFFER_SECONDS = 30.0


class StreamRepairer:
    """Forward reasoning live; withhold content frames; repair or replay at the end.

    Buffering begins globally at the first frame carrying `delta.content`. With n > 1
    this delays other choices' reasoning -- a deliberate simplification, since
    structured output with multiple choices is rare.
    """

    def __init__(
        self,
        contract: OutputContract,
        max_buffered_bytes: int = _MAX_BUFFERED_BYTES,
        max_buffer_seconds: float = _MAX_BUFFER_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._contract = contract
        self._max_buffered_bytes = max_buffered_bytes
        self._max_buffer_seconds = max_buffer_seconds
        self._clock = clock
        self._parser = SSEParser()
        self._buffering = False
        self._disabled = not (contract.engaged and contract.content_schema is not None)
        self._retained: list[SSEEvent] = []
        self._retained_bytes = 0
        self._content: dict[int, list[str]] = {}
        self._finish: dict[int, str | None] = {}
        self._buffer_started_at: float | None = None
        self.telemetry: list[RepairTelemetry] = []

    # -- helpers ---------------------------------------------------------------

    def _reset(self) -> None:
        """Clear every per-buffering-cycle field at every commit/replay boundary."""
        self._retained = []
        self._retained_bytes = 0
        self._content = {}
        self._finish = {}
        self._buffering = False
        self._buffer_started_at = None

    def _replay(self) -> bytes:
        out = b"".join(event.raw for event in self._retained)
        self._reset()
        return out

    def _disable_and_replay(self, extra: bytes = b"") -> bytes:
        out = self._replay() + extra + self._parser.flush()
        self._disabled = True
        return out

    @staticmethod
    def _has_content(payload: object) -> bool:
        if not isinstance(payload, dict):
            return False
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return False
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str) and delta["content"]:
                return True
        return False

    def _absorb(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            index = choice.get("index", 0)
            if not isinstance(index, int):
                continue
            delta = choice.get("delta")
            content = delta.get("content") if isinstance(delta, dict) else None
            if isinstance(content, str) and content:
                self._content.setdefault(index, []).append(content)
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str):
                self._finish[index] = finish_reason

    def _commit(self) -> bytes:
        """Rewrite the retained frames in place, or replay them untouched."""
        repaired: dict[int, str] = {}
        for index, parts in self._content.items():
            if index not in self._finish:
                return self._replay()  # no terminal for this choice -> do not guess
            result = repair(
                "".join(parts),
                self._contract.content_schema,
                finish_reason=self._finish[index],
            )
            self.telemetry.append(_telemetry(result))
            if result.status == "repaired":
                repaired[index] = result.text
            elif result.status != "clean":
                return self._replay()

        if not repaired:
            return self._replay()

        out = bytearray()
        emitted: set[int] = set()
        for event in self._retained:
            if event.data is None:
                out.extend(event.raw)
                continue
            payload = json.loads(event.data)
            touched = False
            for choice in payload.get("choices", []):
                if not isinstance(choice, dict):
                    continue
                index = choice.get("index", 0)
                if index not in repaired or not choice.get("delta", {}).get("content"):
                    continue
                # First content frame for this choice carries the whole repaired
                # document; later ones are emptied. Everything else is preserved.
                if index in emitted:
                    choice["delta"]["content"] = ""
                else:
                    choice["delta"]["content"] = repaired[index]
                    emitted.add(index)
                touched = True
            out.extend(
                f"data: {json.dumps(payload)}\n\n".encode() if touched else event.raw
            )

        self._reset()
        return bytes(out)

    # -- public ----------------------------------------------------------------

    @property
    def buffering(self) -> bool:
        return self._buffering and not self._disabled

    @property
    def seconds_remaining(self) -> float | None:
        if not self.buffering or self._buffer_started_at is None:
            return None
        elapsed = self._clock() - self._buffer_started_at
        return max(0.0, self._max_buffer_seconds - elapsed)

    def feed(self, chunk: bytes) -> bytes:
        if self._disabled:
            return chunk

        out = bytearray()
        events: list[SSEEvent] = []
        position = -1
        try:
            if self.buffering and self.seconds_remaining == 0.0:
                return self._disable_and_replay() + chunk
        except Exception:  # noqa: BLE001 - preserve the not-yet-fed chunk
            logger.warning("structured-output deadline check failed; replaying raw")
            return self._disable_and_replay() + chunk

        try:
            events = self._parser.feed(chunk)
            for position, event in enumerate(events):
                payload = None
                if event.data is not None:
                    try:
                        payload = json.loads(event.data)
                    except (json.JSONDecodeError, ValueError):
                        payload = None

                if not self._buffering:
                    if payload is not None and self._has_content(payload):
                        self._buffering = True
                        self._buffer_started_at = self._clock()
                    else:
                        # Forward the current raw event before the loop can continue.
                        out.extend(event.raw)
                        continue

                # Retain the current raw event before _absorb(), cap checks, or commit.
                self._retained.append(event)
                self._retained_bytes += len(event.raw)
                if payload is not None:
                    self._absorb(payload)

                buffered_bytes = self._retained_bytes + self._parser.buffered_bytes
                if buffered_bytes > self._max_buffered_bytes or self.seconds_remaining == 0.0:
                    unprocessed = b"".join(e.raw for e in events[position + 1 :])
                    out.extend(self._disable_and_replay(unprocessed))
                    break

                terminal = bool(self._content) and all(
                    i in self._finish for i in self._content
                )
                if event.is_done or terminal:
                    out.extend(self._commit())
        except Exception:  # noqa: BLE001 - never break the stream
            logger.warning("structured-output stream repair failed; replaying raw")
            unprocessed = b"".join(e.raw for e in events[position + 1 :])
            out.extend(self._disable_and_replay(unprocessed))

        if not self._disabled and (
            self._retained_bytes + self._parser.buffered_bytes
            > self._max_buffered_bytes
        ):
            out.extend(self._disable_and_replay())

        return bytes(out)

    def flush(self) -> bytes:
        out = bytearray()
        if self._retained:
            out.extend(self._replay())
        out.extend(self._parser.flush())
        self._disabled = True
        return bytes(out)

    def abort(self) -> bytes:
        """A transport error occurred. Give the client back what we withheld."""
        return self._disable_and_replay()
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest src/tests/test_structured_output_transform.py -q`
Expected: 20 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vllm_router/services/structured_output/transform.py src/tests/test_structured_output_transform.py
git commit -m "feat(structured-output): add streaming repair with frame retention and raw replay"
```

---

### Task 6: Metrics

**Files:**

- Modify: `src/vllm_router/services/metrics_service/__init__.py`
- Test: `src/tests/test_structured_output_transform.py` (extend)

**Interfaces:**

- Produces: `structured_output_repairs_total` (Counter, labels `["model", "status", "mode"]`), `structured_output_garbage_prefix_bytes` (Histogram, labels `["model"]`), and `structured_output_schema_rejections_total` (Counter, labels `["model", "reason"]`).

Follow the existing convention exactly: module-level globals, `vllm:` name prefix, default registry (see `input_tokens_total` at `metrics_service/__init__.py:55-57`). **Never put raw model output in a label** — `mode` is a bounded enum, not free text.

- [ ] **Step 1: Add the metrics**

Append to `src/vllm_router/services/metrics_service/__init__.py`:

```python
structured_output_repairs_total = Counter(
    "vllm:structured_output_repairs_total",
    "Structured-output boundary repair outcomes",
    ["model", "status", "mode"],
)

structured_output_garbage_prefix_bytes = Histogram(
    "vllm:structured_output_garbage_prefix_bytes",
    "Size of the corrupt prefix stripped from a structured output",
    ["model"],
    buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
)

structured_output_schema_rejections_total = Counter(
    "vllm:structured_output_schema_rejections_total",
    "Structured-output schemas rejected at the engagement boundary",
    ["model", "reason"],
)
```

- [ ] **Step 2: Write a test asserting the counter moves**

```python
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
```

Task 8 records these metrics from each `RepairTelemetry`: `status` and `mode` go to the counter, and `garbage_prefix_bytes` is observed on the histogram for every `repaired` result. Metric recording is surrounded by one fail-safe `try/except`; Prometheus failures never affect a response. The rejection counter is incremented once when `OutputContract.rejected_non_discriminating` is true. Raw output is never a label.

- [ ] **Step 3: Run**

Run: `uv run pytest src/tests/test_structured_output_transform.py -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/vllm_router/services/metrics_service/__init__.py src/tests/test_structured_output_transform.py
git commit -m "feat(structured-output): add repair telemetry"
```

---

### Task 7: CLI flag and app wiring

**Files:**

- Modify: `src/vllm_router/parsers/parser.py`, `src/vllm_router/app.py`
- Test: `src/tests/test_parser.py`

**Interfaces:**

- Produces: `args.enable_structured_output_repair` (bool), `args.structured_output_repair_max_bytes` (int), `args.structured_output_repair_max_seconds` (float), and matching `app.state` values.

- [ ] **Step 1: Add the flags**

In `src/vllm_router/parsers/parser.py`, alongside the other feature flags (match the `--enable-batch-api` shape at parser.py:298-302):

```python
    parser.add_argument(
        "--enable-structured-output-repair",
        action="store_true",
        help="Repair grammar-produced JSON corrupted at the reasoning/answer boundary (see docs/superpowers/specs/2026-07-13-structured-output-boundary-repair-design.md).",
    )
    parser.add_argument(
        "--structured-output-repair-max-bytes",
        type=int,
        default=1048576,
        help="Max bytes buffered per request while repairing structured output. On breach, the response is passed through unchanged.",
    )
    parser.add_argument(
        "--structured-output-repair-max-seconds",
        type=float,
        default=30.0,
        help="Max seconds content may be withheld while repairing structured output. On breach, retained bytes are replayed and repair is disabled.",
    )
```

- [ ] **Step 2: Wire into app state**

In `src/vllm_router/app.py`, inside `initialize_all()` next to the other `app.state` assignments (app.py:361-365):

```python
    app.state.structured_output_repair_enabled = args.enable_structured_output_repair
    app.state.structured_output_repair_max_bytes = args.structured_output_repair_max_bytes
    app.state.structured_output_repair_max_seconds = args.structured_output_repair_max_seconds
```

- [ ] **Step 3: Add parser coverage**

Append to `src/tests/test_parser.py`:

```python
def test_structured_output_repair_flag_and_caps(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "router",
            "--port",
            "8080",
            "--service-discovery",
            "static",
            "--static-backends",
            "http://a",
            "--static-models",
            "m",
            "--routing-logic",
            "roundrobin",
            "--enable-structured-output-repair",
            "--structured-output-repair-max-bytes",
            "2048",
            "--structured-output-repair-max-seconds",
            "12.5",
        ],
    )
    args = parser.parse_args()
    assert args.enable_structured_output_repair is True
    assert args.structured_output_repair_max_bytes == 2048
    assert args.structured_output_repair_max_seconds == 12.5
```

- [ ] **Step 4: Run the parser test and verify defaults**

Run: `uv run pytest src/tests/test_parser.py -q -k structured_output_repair`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/vllm_router/parsers/parser.py src/vllm_router/app.py src/tests/test_parser.py
git commit -m "feat(structured-output): add --enable-structured-output-repair flag"
```

---

### Task 8: Wire into the response path

**Files:**

- Modify: `src/vllm_router/services/request_service/request.py:625-676`
- Test: `src/tests/test_structured_output_integration.py`

**Interfaces:**

- Consumes: everything from Tasks 1–7.
- Produces: no new public API — this is the integration.

**Where the hook goes.** `traced_stream()` (`request.py:662`) is the single point every chunk passes through on its way to the client. The branch goes in `route_general_request` right before the `StreamingResponse` is constructed at `request.py:671`.

**The contract must be extracted from the body *after* the request rewriter runs** (`request.py:462-477`), because rewriting can mutate the body after the initial parse at `request.py:417`.

- [ ] **Step 1: Write the failing integration tests**

Create `src/tests/test_structured_output_integration.py` with these complete, collecting fixtures and helpers. Response headers from the backend are plain dictionaries, so `.get()` and `.items()` have real behavior.

```python
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

STRICT = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}


def _request_body(*, engaged, stream=False):
    body = {"model": "m", "messages": [], "stream": stream}
    if engaged:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "s", "schema": STRICT},
        }
    return body


def _make_request(body):
    router = MagicMock()
    router.max_instance_failover_reroute_attempts = 0
    router.route_request.return_value = "http://engine"
    router.extract_session_id.return_value = None

    service_discovery = MagicMock()
    endpoint = SimpleNamespace(
        url="http://engine", model_names=["m"], sleep=False, Id="engine"
    )
    service_discovery.get_endpoint_info.return_value = [endpoint]
    service_discovery.aliases = None
    service_discovery.has_ever_seen_model.return_value = True

    state = SimpleNamespace(
        router=router,
        engine_stats_scraper=MagicMock(),
        request_stats_monitor=MagicMock(),
        otel_enabled=False,
        semantic_cache_available=False,
        callbacks=None,
        external_provider_registry=None,
        structured_output_repair_enabled=True,
        structured_output_repair_max_bytes=1_048_576,
        structured_output_repair_max_seconds=30.0,
    )
    state.engine_stats_scraper.get_engine_stats.return_value = {}
    state.request_stats_monitor.get_request_stats.return_value = {}

    request = MagicMock()
    request.headers = {"content-type": "application/json"}
    request.query_params = {}
    request.method = "POST"
    request.url = "http://router/v1/chat/completions"
    request.app.state = state
    encoded = json.dumps(body).encode()
    request.body = AsyncMock(return_value=encoded)
    request.json = AsyncMock(return_value=body)

    patches = [
        patch(
            "vllm_router.services.request_service.request.get_service_discovery",
            return_value=service_discovery,
        ),
        patch(
            "vllm_router.services.request_service.request.is_request_rewriter_initialized",
            return_value=False,
        ),
    ]
    return request, patches


@pytest.fixture
def setup_engaged():
    request, patches = _make_request(_request_body(engaged=True))
    for active_patch in patches:
        active_patch.start()
    try:
        yield request, MagicMock()
    finally:
        for active_patch in reversed(patches):
            active_patch.stop()


@pytest.fixture
def setup_unengaged():
    request, patches = _make_request(_request_body(engaged=False))
    for active_patch in patches:
        active_patch.start()
    try:
        yield request, MagicMock()
    finally:
        for active_patch in reversed(patches):
            active_patch.stop()


def _engaged_request(*, stream=False):
    """A request usable by route_chat_completion cache-boundary tests."""
    body = _request_body(engaged=True, stream=stream)
    request = MagicMock()
    request.app.state = SimpleNamespace(
        structured_output_repair_enabled=True,
        semantic_cache_available=True,
    )
    request.json = AsyncMock(return_value=body)
    return request


def _non_streaming_body(content):
    return json.dumps(
        {"id": "x", "choices": [{"index": 0, "message": {"content": content}, "finish_reason": "stop"}]}
    ).encode()


def _chunk(delta, finish_reason=None, index=0):
    payload = {
        "id": "x",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": index,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


async def _collect(response):
    if hasattr(response, "body"):
        return response.body
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    return b"".join(chunks)


@pytest.mark.asyncio
async def test_non_streaming_repairs_and_sets_content_length(setup_engaged):
    req, _ = setup_engaged

    async def backend(*a, **kw):
        yield {"content-type": "application/json"}, 200
        yield _non_streaming_body('{{"summary": "x"}')

    with patch(
        "vllm_router.services.request_service.request.process_request", side_effect=backend
    ):
        from vllm_router.services.request_service.request import route_general_request

        resp = await route_general_request(req, "/v1/chat/completions", MagicMock())

    body = await _collect(resp)
    assert json.loads(body)["choices"][0]["message"]["content"] == '{"summary": "x"}'
    assert resp.headers["content-length"] == str(len(body))


@pytest.mark.asyncio
async def test_unengaged_request_is_byte_identical(setup_unengaged):
    """No response_format -> the new machinery must not touch a single byte."""
    req, _ = setup_unengaged
    original = _non_streaming_body('{{"summary": "x"}')

    async def backend(*a, **kw):
        yield {"content-type": "application/json"}, 200
        yield original

    with patch(
        "vllm_router.services.request_service.request.process_request", side_effect=backend
    ):
        from vllm_router.services.request_service.request import route_general_request

        resp = await route_general_request(req, "/v1/chat/completions", MagicMock())

    assert await _collect(resp) == original


@pytest.mark.asyncio
async def test_non_streaming_non_2xx_is_byte_identical(setup_engaged):
    req, _ = setup_engaged
    original = _non_streaming_body('{{"summary": "x"}')

    async def backend(*args, **kwargs):
        yield {"content-type": "application/json"}, 400
        yield original

    with patch(
        "vllm_router.services.request_service.request.process_request",
        side_effect=backend,
    ):
        from vllm_router.services.request_service.request import route_general_request

        response = await route_general_request(
            req, "/v1/chat/completions", MagicMock()
        )

    assert response.status_code == 400
    assert await _collect(response) == original


@pytest.mark.asyncio
async def test_streaming_non_2xx_is_byte_identical():
    req, patches = _make_request(_request_body(engaged=True, stream=True))
    original = b'{"error":"bad {{ output"}'

    async def backend(*args, **kwargs):
        yield {"content-type": "application/json"}, 503
        yield original

    for active_patch in patches:
        active_patch.start()
    try:
        with patch(
            "vllm_router.services.request_service.request.process_request",
            side_effect=backend,
        ):
            from vllm_router.services.request_service.request import route_general_request

            response = await route_general_request(
                req, "/v1/chat/completions", MagicMock()
            )
        assert response.status_code == 503
        assert await _collect(response) == original
    finally:
        for active_patch in reversed(patches):
            active_patch.stop()


@pytest.mark.asyncio
async def test_non_streaming_transport_error_replays_then_raises(setup_engaged):
    req, _ = setup_engaged
    first = b'{"partial":'

    async def backend(*args, **kwargs):
        yield {"content-type": "application/json"}, 200
        yield first
        raise ConnectionError("backend disconnected")

    with patch(
        "vllm_router.services.request_service.request.process_request",
        side_effect=backend,
    ):
        from vllm_router.services.request_service.request import route_general_request

        response = await route_general_request(
            req, "/v1/chat/completions", MagicMock()
        )

    iterator = response.body_iterator.__aiter__()
    assert await iterator.__anext__() == first
    with pytest.raises(ConnectionError, match="backend disconnected"):
        await iterator.__anext__()


@pytest.mark.asyncio
async def test_stream_watchdog_replays_without_cancelling_backend_read():
    req, patches = _make_request(_request_body(engaged=True, stream=True))
    req.app.state.structured_output_repair_max_seconds = 0.01
    release = asyncio.Event()
    cancelled = False
    first = _chunk({"content": '{{"summary": '})
    second = _chunk({"content": '"x"}'}, finish_reason="stop")

    async def backend(*args, **kwargs):
        nonlocal cancelled
        yield {"content-type": "text/event-stream"}, 200
        yield first
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled = True
            raise
        yield second

    for active_patch in patches:
        active_patch.start()
    try:
        with patch(
            "vllm_router.services.request_service.request.process_request",
            side_effect=backend,
        ):
            from vllm_router.services.request_service.request import route_general_request

            response = await route_general_request(
                req, "/v1/chat/completions", MagicMock()
            )
        iterator = response.body_iterator.__aiter__()
        assert await iterator.__anext__() == first
        assert cancelled is False
        release.set()
        assert await iterator.__anext__() == second
    finally:
        release.set()
        for active_patch in reversed(patches):
            active_patch.stop()
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest src/tests/test_structured_output_integration.py -q`
Expected: FAIL — repair is not wired in and the watchdog/replay branches do not exist.

- [ ] **Step 3: Implement the branch**

Replace `request.py:671-676` (the single `return StreamingResponse(...)`) with:

```python
    candidate_contract = (
        extract_output_contract(request_json)
        if getattr(request.app.state, "structured_output_repair_enabled", False)
        else OutputContract()
    )
    contract = (
        candidate_contract if 200 <= status < 300 else OutputContract()
    )

    if not contract.engaged:
        _record_repair_metrics(
            [],
            requested_model,
            rejected_non_discriminating=(
                200 <= status < 300
                and candidate_contract.rejected_non_discriminating
            ),
        )
        return StreamingResponse(
            traced_stream(),
            status_code=status,
            headers=headers_dict,
            media_type=media_type,
        )

    if not request_json.get("stream", False):
        body = bytearray()
        try:
            async for chunk in traced_stream():
                body.extend(chunk)
        except Exception as exc:

            async def replay_then_raise(
                retained=bytes(body), error=exc
            ):
                if retained:
                    yield retained
                raise error

            return StreamingResponse(
                replay_then_raise(),
                status_code=status,
                headers=headers_dict,
                media_type=media_type,
            )

        new_body, telemetry = transform_response_body(bytes(body), contract)
        _record_repair_metrics(telemetry, requested_model)
        # Response computes content-length itself; headers_dict already had it stripped.
        return Response(
            content=new_body,
            status_code=status,
            headers=headers_dict,
            media_type=media_type,
        )

    max_bytes = getattr(
        request.app.state, "structured_output_repair_max_bytes", 1_048_576
    )
    max_seconds = getattr(
        request.app.state, "structured_output_repair_max_seconds", 30.0
    )

    async def repaired_stream():
        repairer = StreamRepairer(
            contract,
            max_buffered_bytes=max_bytes,
            max_buffer_seconds=max_seconds,
        )
        iterator = traced_stream().__aiter__()
        try:
            while True:
                read = asyncio.ensure_future(iterator.__anext__())
                try:
                    chunk = await asyncio.wait_for(
                        asyncio.shield(read),
                        timeout=repairer.seconds_remaining,
                    )
                except asyncio.TimeoutError:
                    # shield() leaves the backend read alive. Replay what was
                    # withheld, disable repair, then await that same read.
                    tail = repairer.abort()
                    if tail:
                        yield tail
                    try:
                        chunk = await read
                    except StopAsyncIteration:
                        break
                except StopAsyncIteration:
                    break

                out = repairer.feed(chunk)
                if out:
                    yield out
            tail = repairer.flush()
            if tail:
                yield tail
        except Exception:
            # Give the client back whatever we withheld, then let the error surface.
            tail = repairer.abort()
            if tail:
                yield tail
            raise
        finally:
            _record_repair_metrics(repairer.telemetry, requested_model)

    return StreamingResponse(
        repaired_stream(),
        status_code=status,
        headers=headers_dict,
        media_type=media_type,
    )
```

Add near the top of `request.py`:

```python
import asyncio

from fastapi.responses import Response

from vllm_router.services.metrics_service import (
    structured_output_garbage_prefix_bytes,
    structured_output_repairs_total,
    structured_output_schema_rejections_total,
)
from vllm_router.services.structured_output.contract import (
    OutputContract,
    extract_output_contract,
)
from vllm_router.services.structured_output.transform import (
    StreamRepairer,
    transform_response_body,
)


def _record_repair_metrics(
    telemetry,
    requested_model,
    *,
    rejected_non_discriminating=False,
) -> None:
    """Metrics are fail-safe and receive no raw model output."""
    try:
        if rejected_non_discriminating:
            structured_output_schema_rejections_total.labels(
                model=requested_model,
                reason="non_discriminating",
            ).inc()
        for event in telemetry:
            structured_output_repairs_total.labels(
                model=requested_model,
                status=event.status,
                mode=event.mode,
            ).inc()
            if event.status == "repaired":
                structured_output_garbage_prefix_bytes.labels(
                    model=requested_model
                ).observe(event.garbage_prefix_bytes)
    except Exception:  # noqa: BLE001 - telemetry must never affect a response
        logger.warning("failed to record structured-output repair metrics")
```

`request_json` here is the post-rewrite parse updated at `request.py:462-477`. Both metric call sites use the existing `requested_model` variable; there is no `model_name` in `route_general_request`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest src/tests/test_structured_output_integration.py -q`
Expected: 6 passed.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest src/tests/ -q`
Expected: all pass — in particular `test_instance_failover.py` and `test_request_auth_headers.py`, which prove existing behaviour is unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/vllm_router/services/request_service/request.py src/tests/test_structured_output_integration.py
git commit -m "feat(structured-output): wire boundary repair into the response path"
```

---

### Task 9: Skip semantic-cache lookup and storage for engaged requests

**Files:**

- Modify: `src/vllm_router/routers/main_router.py:35-65`, `src/vllm_router/services/request_service/request.py:275-370,417-490`, `src/vllm_router/experimental/semantic_cache_integration.py:181-205`
- Test: `src/tests/test_structured_output_integration.py` (extend)

**Why:** the cache key ignores the content schema. Engagement must be decided from the post-rewrite body, lookup must be skipped, and `process_request` must not store the raw corrupted response for a later schema-blind hit.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_semantic_cache_lookup_and_storage_skipped_post_rewrite(
    setup_unengaged,
):
    """A rewriter-added schema is authoritative for both cache decisions."""
    import vllm_router.services.request_service.request as request_module

    req, _ = setup_unengaged
    req.app.state.semantic_cache_available = True
    rewritten = json.dumps(_request_body(engaged=True)).encode()
    rewriter = MagicMock()
    rewriter.rewrite_request.return_value = rewritten

    async def backend(*args, **kwargs):
        yield {"content-type": "application/json"}, 200
        yield _non_streaming_body('{{"summary": "x"}')

    with patch.object(
        request_module, "check_semantic_cache", new_callable=AsyncMock, create=True
    ) as cache, patch.object(
        request_module, "semantic_cache_available", True
    ), patch.object(
        request_module, "is_request_rewriter_initialized", return_value=True
    ), patch.object(
        request_module, "get_request_rewriter", return_value=rewriter
    ), patch.object(
        request_module, "process_request", side_effect=backend
    ) as process:
        response = await request_module.route_general_request(
            req, "/v1/chat/completions", MagicMock()
        )
        await _collect(response)

    cache.assert_not_awaited()
    assert process.call_args.kwargs["skip_semantic_cache"] is True


@pytest.mark.asyncio
async def test_unengaged_cache_lookup_uses_post_rewrite_json(setup_unengaged):
    import vllm_router.services.request_service.request as request_module

    req, _ = setup_unengaged
    req.app.state.semantic_cache_available = True
    cached = MagicMock()
    with patch.object(
        request_module, "check_semantic_cache", new_callable=AsyncMock, create=True
    ) as cache, patch.object(
        request_module, "semantic_cache_available", True
    ), patch.object(request_module, "process_request") as process:
        cache.return_value = cached
        response = await request_module.route_general_request(
            req, "/v1/chat/completions", MagicMock()
        )

    assert response is cached
    cache.assert_awaited_once_with(
        request=req,
        request_json=_request_body(engaged=False),
    )
    process.assert_not_called()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest src/tests/test_structured_output_integration.py -q -k semantic`
Expected: FAIL — lookup still occurs in `main_router`, and `process_request` has no storage guard.

- [ ] **Step 3: Implement**

Delete the optional semantic-cache import and the entire lookup block from `route_chat_completion()` in `main_router.py`; it becomes:

```python
@main_router.post("/v1/chat/completions")
async def route_chat_completion(request: Request, background_tasks: BackgroundTasks):
    return await route_general_request(
        request, "/v1/chat/completions", background_tasks
    )
```

In the existing optional semantic-cache import in `request.py`, import both functions:

```python
from vllm_router.experimental.semantic_cache_integration import (
    check_semantic_cache,
    store_in_semantic_cache,
)
```

Immediately after request rewriting and its successful `request_json = json.loads(request_body)`, extract the feature-gated post-rewrite contract and do lookup only for an unengaged chat request:

```python
    candidate_contract = (
        extract_output_contract(request_json)
        if getattr(request.app.state, "structured_output_repair_enabled", False)
        else OutputContract()
    )
    skip_semantic_cache = candidate_contract.engaged

    if (
        endpoint == "/v1/chat/completions"
        and semantic_cache_available
        and request.app.state.semantic_cache_available
        and not skip_semantic_cache
    ):
        cache_response = await check_semantic_cache(
            request=request,
            request_json=request_json,
        )
        if cache_response:
            return cache_response
```

Make the lookup consume the same post-rewrite JSON instead of re-reading the original `Request` body. Change its signature and body selection in `semantic_cache_integration.py`:

```python
async def check_semantic_cache(
    request: Request,
    *,
    request_json: dict | None = None,
) -> Optional[JSONResponse]:
    """Return a cache hit using post-rewrite JSON when the router supplies it."""
    if not is_semantic_cache_enabled():
        logger.debug("Semantic cache is not enabled, skipping cache check")
        return None

    body = request_json if request_json is not None else await request.json()
    logger.info("Checking semantic cache for potential cache hit")
```

Keep the remainder of the existing function unchanged. This avoids deciding engagement from one body while performing the cache lookup with another.

Keep this `candidate_contract` and reuse it in Task 8's response-status gate; do not extract the contract a second time. Extend `process_request()` with a keyword-only storage decision:

```python
async def process_request(
    request: Request,
    body: bytes,
    backend_url: str,
    request_id: str,
    endpoint: str,
    background_tasks: BackgroundTasks,
    parent_span_context=None,
    *,
    skip_semantic_cache: bool = False,
):
```

Pass it from `route_general_request()`:

```python
            stream_generator = process_request(
                request,
                request_body,
                server_url,
                request_id,
                endpoint,
                background_tasks,
                parent_span_context=span_context,
                skip_semantic_cache=skip_semantic_cache,
            )
```

Finally, replace the storage condition at `request.py:359`:

```python
        if request.app.state.semantic_cache_available and not skip_semantic_cache:
            cache_chunk = bytes(full_response) if not is_streaming else chunk
            await store_in_semantic_cache(
                endpoint=endpoint,
                method=request.method,
                body=body,
                chunk=cache_chunk,
            )
```

The test uses `patch.object(..., create=True)` because the optional import may not define `check_semantic_cache` in the test environment, and explicitly forces module-level `semantic_cache_available=True`; without both, it can pass without exercising the lookup branch.

- [ ] **Step 4: Run**

Run: `uv run pytest src/tests/test_structured_output_integration.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/vllm_router/routers/main_router.py src/vllm_router/services/request_service/request.py src/vllm_router/experimental/semantic_cache_integration.py src/tests/test_structured_output_integration.py
git commit -m "fix(structured-output): skip semantic cache for engaged requests"
```

---

### Task 10: Golden-corpus regression

**Files:**

- Create: `src/tests/test_structured_output_corpus.py`

**Why:** the platform team holds `matrix_results.json` — **1,536 real requests with raw outputs**. That is the only test set drawn from production reality rather than our imagination. Every known-corrupt sample must repair to schema-valid; every clean sample must pass through **byte-identical**.

The file lives in the NLU platform repo, not here. The test reads it from `STRUCTURED_OUTPUT_CORPUS` and skips when unset, so CI stays green while the corpus stays out of this repo.

- [ ] **Step 1: Read the corpus and write down the field mapping (hard prerequisite)**

Run against the real file before writing the test:

```bash
test -n "$STRUCTURED_OUTPUT_CORPUS"
test -f "$STRUCTURED_OUTPUT_CORPUS"
jq '.[0] | paths(scalars)' "$STRUCTURED_OUTPUT_CORPUS"
jq 'group_by(.classification) | map({classification: .[0].classification, count: length})' "$STRUCTURED_OUTPUT_CORPUS"
```

Write the verified paths for `content`, `schema`, `finish_reason`, and the clean/corrupt classification in the `CORPUS FIELD MAPPING` comment before adding the test. Do not continue until all 1,536 records can be classified. The code below records the expected flat mapping. If the command disproves that mapping, stop Task 10 and revise this plan with the observed paths before writing code; there is no “adapt later” step.

- [ ] **Step 2: Write the corpus test**

`src/tests/test_structured_output_corpus.py`:

```python
import json
import os

import jsonschema
import pytest

from vllm_router.services.structured_output.repair import repair

CORPUS = os.environ.get("STRUCTURED_OUTPUT_CORPUS")

pytestmark = pytest.mark.skipif(
    not CORPUS or not os.path.exists(CORPUS),
    reason="set STRUCTURED_OUTPUT_CORPUS to matrix_results.json to run",
)

# CORPUS FIELD MAPPING (verified before commit):
# content -> record["content"]
# schema -> record["schema"]
# finish_reason -> record["finish_reason"] (defaults to "stop" only when absent)
# classification -> record["classification"] in {"corrupt", "clean"}

def _samples():
    with open(CORPUS) as handle:
        records = json.load(handle)
    assert len(records) == 1536
    yield from records


def test_corpus_classifications_are_exhaustive_and_correct():
    for record in _samples():
        classification = record["classification"]
        assert classification in {"corrupt", "clean"}, record
        assert isinstance(record["content"], str), record
        assert isinstance(record["schema"], dict), record

        result = repair(
            record["content"],
            record["schema"],
            finish_reason=record.get("finish_reason", "stop"),
        )
        if classification == "corrupt":
            assert result.status == "repaired", record
            assert result.text is not None
            assert result.text in record["content"]
            jsonschema.validate(result.value, record["schema"])
        else:
            assert result.status == "clean", record
            assert result.text == record["content"]
            assert result.text.encode("utf-8") == record["content"].encode("utf-8")
```

- [ ] **Step 3: Run (skips without the corpus)**

Run: `uv run pytest src/tests/test_structured_output_corpus.py -q`
Expected: `1 skipped`.

- [ ] **Step 4: Run against the real corpus**

Run: `uv run pytest src/tests/test_structured_output_corpus.py -q`
Expected: PASS. **If it fails, stop and report — a wrong answer on real production data invalidates the design, not the test.**

- [ ] **Step 5: Commit**

```bash
git add src/tests/test_structured_output_corpus.py
git commit -m "test(structured-output): add golden-corpus regression harness"
```

---

### Task 11: Secure diagnostic capture for ambiguous/unknown outcomes

**Files:**

- Create: `src/vllm_router/services/structured_output/capture.py`, `src/tests/test_structured_output_capture.py`
- Modify: `src/vllm_router/services/structured_output/transform.py`, `src/vllm_router/services/request_service/request.py`, `src/vllm_router/parsers/parser.py`, `src/vllm_router/app.py`

**Policy (all parts are load-bearing):** disabled unless a directory is configured; 1% sampling by default; 4 KiB maximum **after redaction**; seven-day retention; directory owned by the router uid with mode `0700`; files mode `0600`. The built-in redactor preserves only JSON/boundary punctuation and whitespace, replacing every other run with `<redacted>`. The sink stores status, bounded mode, byte counts, timestamp, model, and a SHA-256 digest for deduplication. It never stores an unredacted body or schema.

Raw output must never enter a Prometheus label or ordinary log. Sink failures and dropped samples are silent from the response's perspective; only aggregate drop counters may be added later.

- [ ] **Step 1: Write the failing sink tests**

Create `src/tests/test_structured_output_capture.py`:

```python
import json
import os
from pathlib import Path

import pytest

from vllm_router.services.structured_output.capture import (
    CapturePolicy,
    SecureJSONLCaptureSink,
)
from vllm_router.services.structured_output.transform import RepairTelemetry


def _secure_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "captures"
    directory.mkdir(mode=0o700)
    directory.chmod(0o700)
    return directory


def test_capture_is_sampled_redacted_bounded_and_private(tmp_path):
    directory = _secure_dir(tmp_path)
    sink = SecureJSONLCaptureSink(
        directory,
        CapturePolicy(sample_rate=1.0, max_bytes=64, retention_days=7),
        random_value=lambda: 0.0,
        wall_time=lambda: 1_720_000_000.0,
    )
    raw = '{{"secret":"customer@example.com","long":"' + "x" * 500 + '"}'
    assert sink.capture(
        model="m",
        output=raw,
        telemetry=RepairTelemetry("unknown", "none", 0),
    )

    files = list(directory.glob("structured-output-*.jsonl"))
    assert len(files) == 1
    assert files[0].stat().st_mode & 0o777 == 0o600
    stored_bytes = files[0].read_bytes()
    assert b"customer@example.com" not in stored_bytes
    record = json.loads(stored_bytes)
    assert len(record["redacted_output"].encode()) <= 64
    assert record["output_bytes"] == len(raw.encode())
    assert len(record["output_sha256"]) == 64


def test_sampled_out_record_creates_no_file(tmp_path):
    directory = _secure_dir(tmp_path)
    sink = SecureJSONLCaptureSink(
        directory,
        CapturePolicy(sample_rate=0.01),
        random_value=lambda: 0.5,
    )
    assert not sink.capture(
        model="m",
        output="secret",
        telemetry=RepairTelemetry("ambiguous", "none", 0),
    )
    assert list(directory.iterdir()) == []


def test_rejects_insecure_directory(tmp_path):
    directory = tmp_path / "captures"
    directory.mkdir(mode=0o755)
    directory.chmod(0o755)
    with pytest.raises(ValueError, match="0700"):
        SecureJSONLCaptureSink(directory)


def test_purges_expired_capture_files(tmp_path):
    directory = _secure_dir(tmp_path)
    expired = directory / "structured-output-20200101.jsonl"
    expired.write_text("{}\n")
    expired.chmod(0o600)
    os.utime(expired, (1.0, 1.0))
    sink = SecureJSONLCaptureSink(
        directory,
        CapturePolicy(sample_rate=1.0, retention_days=7),
        random_value=lambda: 0.0,
        wall_time=lambda: 1_720_000_000.0,
    )
    sink.capture(
        model="m",
        output="{{secret}",
        telemetry=RepairTelemetry("unknown", "none", 0),
    )
    assert not expired.exists()
```

- [ ] **Step 2: Implement the sink**

Create `src/vllm_router/services/structured_output/capture.py`:

```python
"""Secure, bounded diagnostic captures for refused repair outcomes."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from vllm_router.services.structured_output.transform import RepairTelemetry

_CAPTURE_NAME = re.compile(r"structured-output-\d{8}\.jsonl\Z")
_REDACT_RUN = re.compile(r"[^{}\[\]:,`\s]+")


@dataclass(frozen=True)
class CapturePolicy:
    sample_rate: float = 0.01
    max_bytes: int = 4096
    retention_days: int = 7

    def __post_init__(self) -> None:
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError("sample_rate must be between 0 and 1")
        if self.max_bytes <= 0 or self.retention_days <= 0:
            raise ValueError("capture limits must be positive")


def _redact_and_bound(output: str, max_bytes: int) -> str:
    redacted = _REDACT_RUN.sub("<redacted>", output)
    encoded = redacted.encode("utf-8")[:max_bytes]
    return encoded.decode("utf-8", errors="ignore")


class SecureJSONLCaptureSink:
    def __init__(
        self,
        directory: str | Path,
        policy: CapturePolicy | None = None,
        *,
        random_value: Callable[[], float] = random.random,
        wall_time: Callable[[], float] = time.time,
    ) -> None:
        self._directory = Path(directory)
        self._policy = policy or CapturePolicy()
        self._random_value = random_value
        self._wall_time = wall_time
        stat = self._directory.stat()
        if stat.st_uid != os.geteuid() or stat.st_mode & 0o777 != 0o700:
            raise ValueError("capture directory must be owned by the router uid and mode 0700")

    def _purge_expired(self, now: float) -> None:
        cutoff = now - self._policy.retention_days * 86_400
        for path in self._directory.iterdir():
            if _CAPTURE_NAME.fullmatch(path.name) and path.stat().st_mtime < cutoff:
                path.unlink()

    def capture(
        self,
        *,
        model: str,
        output: str,
        telemetry: RepairTelemetry,
    ) -> bool:
        if telemetry.status not in {"ambiguous", "unknown"}:
            return False
        if self._random_value() >= self._policy.sample_rate:
            return False

        now = self._wall_time()
        self._purge_expired(now)
        raw = output.encode("utf-8")
        record = {
            "captured_at": datetime.fromtimestamp(now, UTC).isoformat(),
            "model": model,
            **asdict(telemetry),
            "output_bytes": len(raw),
            "output_sha256": hashlib.sha256(raw).hexdigest(),
            "redacted_output": _redact_and_bound(output, self._policy.max_bytes),
        }
        day = datetime.fromtimestamp(now, UTC).strftime("%Y%m%d")
        path = self._directory / f"structured-output-{day}.jsonl"
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, json.dumps(record, separators=(",", ":")).encode() + b"\n")
        finally:
            os.close(descriptor)
        path.chmod(0o600)
        return True
```

- [ ] **Step 3: Add a fail-safe capture callback seam to both transforms**

In `transform.py`, add `CaptureCallback = Callable[[str, RepairTelemetry], None]` and this helper:

```python
def _capture_refusal(
    callback: CaptureCallback | None,
    content: str,
    result: RepairResult,
) -> None:
    if callback is None or result.status not in {"ambiguous", "unknown"}:
        return
    try:
        callback(content, _telemetry(result))
    except Exception:  # noqa: BLE001 - diagnostics never affect output
        logger.warning("structured-output diagnostic capture failed")
```

Add keyword-only `capture_callback: CaptureCallback | None = None` to `transform_response_body()`, `_transform_response_body()`, and `StreamRepairer.__init__()`. Forward it through the wrapper, save it on the repairer, and call `_capture_refusal(...)` immediately after each `repair()` result in `_transform_response_body()` and `_commit()`, before any reset. Append these tests to `test_structured_output_transform.py`:

```python
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

    out, telemetry = transform_response_body(
        body, CONTRACT, capture_callback=explode
    )
    assert out is body
    assert telemetry[0].status == "ambiguous"


def test_streaming_unknown_is_offered_to_capture_callback():
    captured = []
    repairer = StreamRepairer(
        CONTRACT,
        capture_callback=lambda raw, event: captured.append((raw, event)),
    )
    frame = _chunk({"content": "not json"}, finish_reason="stop")
    assert repairer.feed(frame) == frame
    assert captured[0][0] == "not json"
    assert captured[0][1].status == "unknown"
```

- [ ] **Step 4: Configure and wire the sink**

Add these parser flags beside Task 7's repair flags:

```python
    parser.add_argument(
        "--structured-output-repair-capture-dir",
        type=str,
        default=None,
        help="Router-owned 0700 directory for sampled, redacted repair diagnostics.",
    )
    parser.add_argument(
        "--structured-output-repair-capture-sample-rate",
        type=float,
        default=0.01,
        help="Fraction of ambiguous/unknown repair outcomes captured.",
    )
    parser.add_argument(
        "--structured-output-repair-capture-max-bytes",
        type=int,
        default=4096,
        help="Maximum UTF-8 bytes in each redacted output excerpt.",
    )
    parser.add_argument(
        "--structured-output-repair-capture-retention-days",
        type=int,
        default=7,
        help="Days to retain structured-output diagnostic capture files.",
    )
```

In `initialize_all()`, construct the sink only when configured; an insecure directory must fail startup:

```python
    capture_directory = args.structured_output_repair_capture_dir
    if capture_directory is None:
        app.state.structured_output_repair_capture_sink = None
    else:
        app.state.structured_output_repair_capture_sink = SecureJSONLCaptureSink(
            capture_directory,
            CapturePolicy(
                sample_rate=args.structured_output_repair_capture_sample_rate,
                max_bytes=args.structured_output_repair_capture_max_bytes,
                retention_days=args.structured_output_repair_capture_retention_days,
            ),
        )
```

Add the corresponding imports from `capture.py` at the top of `app.py`. Extend Task 7's parser test with:

```python
    assert args.structured_output_repair_capture_dir is None
    assert args.structured_output_repair_capture_sample_rate == 0.01
    assert args.structured_output_repair_capture_max_bytes == 4096
    assert args.structured_output_repair_capture_retention_days == 7
```

Set `app.state.structured_output_repair_capture_sink = None` when disabled. In `route_general_request()`, create this closure and pass it to both transforms:

```python
    capture_sink = getattr(
        request.app.state, "structured_output_repair_capture_sink", None
    )

    def capture_callback(content, event):
        if capture_sink is not None:
            capture_sink.capture(
                model=requested_model,
                output=content,
                telemetry=event,
            )
```

Pass `capture_callback=capture_callback` to `transform_response_body()` and `StreamRepairer()`. This is synchronous and sampled; if production measurements show file latency is material, replace the sink with a bounded worker queue without changing the callback contract.

- [ ] **Step 5: Run and commit**

Run: `uv run pytest src/tests/test_structured_output_capture.py src/tests/test_structured_output_transform.py src/tests/test_structured_output_integration.py src/tests/test_parser.py -q`
Expected: all pass.

```bash
git add src/vllm_router/services/structured_output/capture.py src/vllm_router/services/structured_output/transform.py src/vllm_router/services/request_service/request.py src/vllm_router/parsers/parser.py src/vllm_router/app.py src/tests/test_structured_output_capture.py src/tests/test_structured_output_transform.py src/tests/test_structured_output_integration.py src/tests/test_parser.py
git commit -m "feat(structured-output): add secure refusal capture sink"
```

---

### Task 12: Lint, full suite, and docs

- [ ] **Step 1: Lint**

Run: `uv run pre-commit run --all-files`
Expected: all hooks pass (black, isort, ruff on `src/tests/`, codespell, markdownlint).

- [ ] **Step 2: Full suite**

Run: `uv run pytest src/tests/ -q`
Expected: all pass.

- [ ] **Step 3: Document the flags in `README.md`**

Add this section verbatim:

```markdown
### Structured-output boundary repair

`--enable-structured-output-repair` enables router-side repair for content generated under a discriminating **object-rooted** JSON Schema. It is off by default. Requests using `logprobs`, schema-less `json_object` mode, **array- or scalar-rooted** schemas, non-discriminating schemas, tool schemas alone, and non-2xx backend responses remain on the existing byte-preserving path.

Buffering defaults to 1 MiB and 30 seconds; configure it with `--structured-output-repair-max-bytes` and `--structured-output-repair-max-seconds`. Any cap, timeout, ambiguity, exception, or transport failure replays the original bytes.

Diagnostic captures for `ambiguous` and `unknown` outcomes are disabled by default. Enable them with `--structured-output-repair-capture-dir` only after creating a router-owned `0700` directory. Captures are sampled, structurally redacted, capped at 4 KiB, retained for seven days, and written as `0600` files. Raw model output is never placed in metric labels or ordinary logs.

See the [structured-output boundary-repair design](docs/superpowers/specs/2026-07-13-structured-output-boundary-repair-design.md) for safety properties and limits.
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(structured-output): document the boundary repair flag"
```

---

## Verification checklist (before opening a PR)

- [ ] A request with **no** `response_format` is byte-for-byte unchanged (`test_unengaged_request_is_byte_identical`).
- [ ] The flag is **off** by default; with it off, nothing changes.
- [ ] Tool schemas alone do not engage v1; `OutputContract` has no `tool_schemas` field.
- [ ] Streaming and non-streaming non-2xx responses are byte-for-byte unchanged.
- [ ] A mid-stream error **replays** the withheld frames (`test_abort_replays_withheld_frames_byte_for_byte`).
- [ ] A non-streaming transport error replays all retained body bytes before re-raising.
- [ ] The three interleaved-index frames produce either the exact original bytes or schema-valid repaired JSON, never a hybrid.
- [ ] Every commit/replay boundary clears retained frames, retained byte count, content, finish reasons, and buffering state.
- [ ] Terminal detection is exactly `bool(self._content) and all(i in self._finish for i in self._content)`.
- [ ] `: heartbeat [DONE]` is not terminal; only an exact single `data:` payload of `[DONE]` sets `is_done`.
- [ ] `_has_content()` and `_absorb()` are total over malformed JSON values, and the current frame is forwarded or retained before any fallible work.
- [ ] The byte cap includes `SSEParser.buffered_bytes`; one large unterminated frame replays and disables repair.
- [ ] The time cap fires both in `feed()` and during a backend stall; the watchdog shields the outstanding `__anext__()` and does not cancel it.
- [ ] `finish_reason="length"` never repairs.
- [ ] `logprobs` requests are not engaged.
- [ ] Non-discriminating schemas are not engaged.
- [ ] Non-discriminating rejections increment the bounded-label §5.1 counter exactly once.
- [ ] Engagement is computed from the post-rewrite body; engaged requests skip both semantic-cache lookup and storage.
- [ ] Both integration metric call sites use `requested_model`, not undefined `model_name`.
- [ ] `RepairTelemetry` carries real `status`, `mode`, and UTF-8 garbage-prefix bytes; the histogram observes repaired prefix lengths.
- [ ] Metric recording is fail-safe and tests filter the full label set.
- [ ] No exception, including `RecursionError`, can escape `repair()`, `transform_response_body()`, or `StreamRepairer.feed()`.
- [ ] Secure captures are sampled, redacted, bounded, retained for seven days, and protected by `0700`/`0600`; raw output appears in no metric label or ordinary log line.
- [ ] Golden-corpus corrupt records all return `repaired` and schema-valid values; clean records all return `clean` byte-identically; no record is unclassified.
- [ ] The golden corpus passes (Task 10, Step 4).

## Deferred

Tool-call repair is intentionally deferred from v1. There are no real examples of corrupted `tool_calls`, so defining streaming assembly now would be speculative. The promoted repair core keeps the tested `repair_tool_arguments()` seam. Once a representative corpus exists, write a separate design for tool selection, per-index streaming argument assembly, finish semantics, ambiguity handling, and cache engagement before wiring that seam into the router.
