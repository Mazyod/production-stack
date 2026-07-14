# Structured-Output Boundary Repair — Design (v2)

**Date:** 2026-07-13
**Status:** revised after adversarial review; pending user review
**Scope:** `vllm_router` — recover grammar-produced JSON corrupted at the reasoning→answer boundary.

> **v2 changelog.** An adversarial review broke three v1 claims: (a) the "worst case is no change"
> safety property was false — a lexically-inconsistent prefix could yield a silently-wrong *fragment*;
> (b) "non-streaming is already buffered" was false — the router yields every chunk to the client
> immediately; (c) "exact, not heuristic" was an overclaim — some `G + J` outputs are byte-identical to
> a truncated document. v2 replaces the ad-hoc descent guard with a single principled discriminator
> (§3.2), rebuilds the response path around **frame retention with raw replay** (§5), and states the
> ambiguity honestly (§7).

---

## 1. Problem

Reasoning models behind vLLM with JSON-Schema structured outputs (xgrammar). When **speculative
decoding**, **thinking**, and **a grammar** are all active, the JSON is corrupted at its opening:

| mode | example |
|---|---|
| brace fused to fence / language tag | ` ```json{"summary": … ` |
| doubled brace | `{{"summary": …` |
| brace-quote-brace | `{"{"summary": …` |

Each call site strips this by hand today. That is tedious, and impossible for call sites we do not own.

### Root cause (measured — `docs/reports/2026-07-04-gemma4-specdecode-json-verification/`, 1,536 requests)

Three-way interaction: **speculative decoding × thinking × a grammar**. Remove any leg → **exactly
zero** corruption (0/768 with spec-decode off). Requires **concurrency** (0% at 1–2, 100% at 16). Does
**not** scale with `k` (k=1 corrupts as much as k=5). It is **in the sampled tokens** (27/27).

Mechanism (`vllm/v1/structured_output/__init__.py`): when the end-of-thinking marker lands inside a
speculative draft window, those drafts carry no bitmask, and vLLM sets `advance_grammar = False` — **the
FSM never consumes what was just emitted**. The grammar then restarts **from state 0** and emits a
complete, correct document. The client receives two producers' output concatenated.

### The invariant

> **`output = G + J`** — `G` a short unconstrained garbage prefix; `J` a **complete, schema-conforming
> JSON document** produced by the grammar from state 0.

In 15/17 corrupt samples the body is perfectly valid JSON once `G` is stripped. Never mid-body, never
tail.

---

## 2. Constraints (hard)

- **Speculative decoding stays on.** ~3× decode.
- **Thinking stays on** where callers ask for it.
- **Caller-visible semantics must not change silently.** Injecting `enable_thinking: false` would hand a
  caller who asked for a reasoning model a non-reasoning one. **Rejected.**
- **`enable_in_reasoning: true` is forbidden** — it zeroes the corruption metric by *destroying the
  reasoning phase* (0/24 thinking blocks vs 24/24). It looks like a fix on every dashboard.
- **Callers cannot be modified.**

### Why not an in-engine logits processor

The corruption *is* at sampling, so an LP can see it — but it **cannot repair it**. The garbage token is
*already emitted and accepted*, and the FSM refuses to consume it. Repair requires **deleting
already-emitted text**; an LP only masks *future* logits and cannot retract a token. Not the wrong
build — an **incapable** one.

*(True upstream fixes — `advance_grammar = True`, plus the unexplained batch-slot dependence — should be
filed. Neither helps on a pinned build.)*

---

## 3. The repair core (pure function — no I/O, no router imports)

```python
def repair(
    content: str,
    schema: dict | None,
    *,
    finish_reason: str | None,      # REQUIRED keyword, no default
    max_prefix_bytes: int = 256,    # cap on |G|; NOT a candidate count
) -> RepairResult

@dataclass(frozen=True)
class RepairResult:
    status: str            # "clean" | "repaired" | "incomplete" | "ambiguous" | "unknown"
    text: str | None       # exact substring of input; None unless repaired/clean
    value: object | None
    garbage_prefix: str
    trailing: str
    mode: str              # telemetry only; never affects selection
    candidates_tried: int
```

### 3.1 Order of operations

1. **`finish_reason == "length"` → `incomplete`.** Truncation is a real failure, not this bug. Never
   repaired.
2. **Fast path.** Whole content parses **and** validates → **`clean`**, returned **byte-identical**.
3. **Ambiguity gate (§3.2).** If the content could be a truncated document → **`ambiguous`**. Stop.
4. **Candidate search.** Scan **every** opener (`{`/`[`, per the schema root) within `max_prefix_bytes`.
   `json.JSONDecoder().raw_decode(content, o)` parses exactly one value and returns its end. Accept the
   first candidate that **parses**, **validates against the schema**, and **consumes the remainder**
   (trailing is whitespace, or a closing fence *only if a matching opening fence appears in `G`*).
5. Otherwise → **`unknown`**.

### 3.2 The ambiguity gate — the discriminator that makes this safe

> **Repair only when `content` is NOT a valid prefix of any JSON document** — i.e. when it is
> *syntactically impossible* as a truncation.

A small incremental JSON **prefix validator** answers this: does a legal JSON document exist that begins
with these bytes? If **yes**, the content may genuinely be truncated, and repairing it could return a
nested **fragment**. We refuse and pass through.

| content | valid JSON prefix? | verdict |
|---|---|---|
| `{{"summary": "x"}` | **no** — after `{`, a `{` is illegal | unambiguous garbage → **repair** |
| `{"{"summary": "x"}` | **no** — after key `"{"`, `s` is illegal | **repair** |
| ` ```json{…} ` | **no** — a backtick is not JSON | **repair** |
| `{"a": {"x": 1}` | **yes** — a valid incomplete object | could be truncation → **`ambiguous`** |
| `[[1, 2]` | **yes** | → **`ambiguous`** |

This **replaces** v1's structural-descent guard, which was both unsound (an unmatched quote in `G` put
its lexer into string state and hid a real colon, yielding a fragment) and over-eager (a complete `{}`
in `G` poisoned every later candidate).

### 3.3 What this is, honestly

**Not** "exact." It is a **conservative, schema-rooted repair that fails safe**:

- Any `G` that is *syntactically impossible* as a document prefix is repaired — this covers every
  observed mode, and every fused/combined variant, **with no artifact-specific matching**.
- Any `G` that is *syntactically valid* as a document prefix (e.g. `{"a":`) is **irreducibly
  ambiguous** with truncation. We **refuse and pass through** — a false negative, never a wrong answer.

The only artifact-specific element that remains is the trailing **closing-fence** allowance, and it is
now gated on a matching **opening** fence in `G` (previously it could convert a truncated array into a
valid inner array). `mode` labels are pattern-derived but are **telemetry only** and never affect
selection.

**Root types — object roots only.** Two independent reasons, either sufficient on its own:

- **Scalar roots are unrecoverable.** For `{"type":"integer"}`, `G="-"` + `J="1"` yields `"-1"` — itself
  valid and schema-valid. There is nothing to recover, and none is attempted.
- **Array roots have no safe oracle.** The discriminating keywords (§5.1) — `required`,
  `additionalProperties` — are **object** keywords. An array schema carrying `additionalProperties:
  false` would satisfy the check while remaining unable to reject a nested fragment. Arrays are also
  where the ambiguity gate is weakest: an array has no key/colon structure to make garbage
  syntactically *illegal* (`[[1, 2]` is a perfectly valid incomplete document), so almost every
  array corruption lands in `ambiguous` anyway.

Production schemas are object-rooted, so this costs nothing in practice.

---

## 4. Safety property (restated correctly)

> Repair commits **only** when the content is syntactically impossible as a truncation, a candidate
> parses, it validates against the caller's own schema, and it consumes the remainder. In **every**
> other case — `incomplete`, `ambiguous`, `unknown`, any exception, any cap breach — the **original
> bytes are emitted unchanged**.

Repair is a **substring selection**; it cannot fabricate. Every failure path is an exact passthrough,
including mid-stream errors (§5.3). **All exceptions inside the transform are caught and converted to
passthrough** — a schema error must never break a response the backend served successfully.

---

## 5. Router wiring

### 5.1 Engagement (blast radius)

Behind a feature flag, default **off**. Engage only when **all** hold:

- the request carries an **object-rooted** `response_format.json_schema` (or `structured_outputs`
  with an equivalent schema); `tools[]` alone do not engage v1;
- the root schema is **discriminating** (see below);
- **`logprobs` is not requested** (§5.4);
- the backend returned **2xx**.

#### Discriminating schemas — the precondition that closes the last wrong-answer class

One residual class survives the ambiguity gate: content that is **malformed but not truncated**, whose
*nested* object happens to validate the root schema.

```text
content = '{"a" {"x": 1}'     # malformed: after key "a", a `{` is illegal — so the gate lets it through
```

- With a **permissive** root (`{"type":"object"}`), the nested `{"x": 1}` validates → we return a
  **fragment**. Wrong answer.
- With a **discriminating** root (`required: ["summary"]`, `additionalProperties: false`), the oracle
  **rejects** the fragment → `unknown` → **passthrough**. Correct.

The oracle can only reject a nested fragment if the schema is strict enough to *distinguish* it. So a
discriminating root schema is a **precondition for engagement**, not a hope:

> Engage only when the root schema has a non-empty **`required`** list, or
> **`additionalProperties: false`**.

OpenAI `strict: true` json_schemas satisfy this by construction (which is what production sends). A
non-discriminating schema is **not engaged** — it takes the existing path unchanged, and increments a
counter so we can see if anyone is affected.

The schema is extracted **after** request rewriting (`request.py:462-477`), which can mutate the body
after the initial parse at `request.py:417`.

Everything else takes the existing code path **bit-for-bit unchanged**, via an **early bypass before any
buffering**. Non-structured traffic must not even enter the new machinery.

### 5.2 Non-streaming — must buffer *before* emitting

**v1 was wrong here.** `request.py:318-331` yields headers and then every chunk **immediately**;
`full_response` accumulates only for observation, after the bytes have already reached the client. There
is no buffered path today.

For engaged non-streaming requests: **fully consume the backend body, transform it, then respond** —
a real `Response` (not `StreamingResponse`) with `content-length` computed **after** serialization.
Repair each `choices[].message.content` (and `tool_calls[].function.arguments`) using **that choice's**
`finish_reason`.

### 5.3 Streaming — retain raw frames, replay on any doubt

A byte-preserving incremental SSE parser (see §5.5) feeding a **per-choice** state machine keyed on
`choice.index`:

- `delta.reasoning` / `delta.reasoning_content` → forwarded **live**. Thinking keeps its TTFT.
- On the first `delta.content` for a choice → **buffer**, and **retain the original raw frames**.
- On `finish_reason` for that choice → repair the accumulation. On **success**, emit the repaired
  content as a single delta. On **any** failure — `incomplete`, `ambiguous`, `unknown`, exception, cap
  breach, transport error, unknown terminal, `[DONE]` with no finish reason — **replay the retained raw
  frames byte-for-byte**, then continue passthrough.

> **This is the correction to v1's worst bug.** v1 would have *withheld buffered content forever* on a
> mid-stream error. Today a client at least receives what was emitted. Retention + replay restores that:
> the client is never left with less than it gets now.

Must also handle: frames carrying multiple choices; one choice buffering while another streams reasoning;
last content and `finish_reason` in the same frame; `usage` chunks with `choices: []`; `refusal` and
unknown extension fields (preserved verbatim); `delta.tool_calls[]` assembly by index.

### 5.4 logprobs

The corruption is in the **sampled tokens**, so `choices[].logprobs.content[]` describes `G + J`.
Repairing only `message.content` would leave them contradicting each other. v1 ignored this.

**v2: do not engage when `logprobs` is requested.** Passthrough. Token-aligned prefix removal from the
logprob structures is possible but is not worth the risk now; revisit with a real use case.

`usage` is always left as the backend reported it — those are real tokens the model generated.

### 5.5 SSE parsing

Byte-preserving and incremental: split UTF-8 code points across chunk boundaries; `\n\n` **and**
`\r\n\r\n` terminators; multi-line `data:` fields; comments/heartbeats; non-`data` fields; and non-SSE
error bodies returned to a `stream=true` request. **Only recognized `data:` JSON events on 2xx responses
are transformed. Every unrecognized byte is preserved verbatim.**

### 5.6 Caps and resource safety

Buffering is bounded, and a breach is a **passthrough**, not a failure:

- `max_buffered_bytes` per request (default 1 MiB) → breach: replay retained frames, disable repair.
- `max_buffer_seconds` → same.
- The schema validator is **compiled once per request** and cached; **remote `$ref` retrieval is
  disabled** (an untrusted schema must not make the router fetch a URL).

Buffering the JSON region is acceptable because a caller cannot act on half a JSON document — but it
does introduce a silent interval, so the caps above are load-bearing, not decorative.

### 5.7 Semantic cache and callbacks

- **Semantic cache is skipped for engaged requests.** Its lookup runs *before* `route_general_request`
  (`main_router.py:51-60`) and its key ignores schema/tools — so a structured request could otherwise be
  served a cached **corrupted** response, bypassing repair entirely.
- **`post_request` continues to receive the raw backend bytes** (unchanged semantics — it is a backend
  observer). Caller-visible repair is reported via telemetry (§5.8), not by mutating the callback
  contract.

### 5.8 Telemetry — repair must not suppress the signal

- counter by `mode` and by `status` (`repaired` / `incomplete` / `ambiguous` / `unknown`)
- histogram of garbage-prefix length
- **bounded, sampled, redacted capture** of `ambiguous`/`unknown` bodies — with a defined sink, size cap,
  retention limit and access control. Model output can contain personal data or secrets: **never put raw
  output in a metric label or an ordinary log line.**

The upstream defect stays **measured**. We will know its rate per deploy, which we do not today.

---

## 6. Testing

- **Unit (repair core):** every real corruption mode (bare, fused, combined); the ambiguity gate
  (`{"a": {"x":1}` → `ambiguous`); the lexical-inconsistency case (`'"garbage{"a":{"x":1}'`);
  `{{{{{{{{` + `J` (no candidate-count cap); `[[1]` + fence (fence gated on an opening fence);
  scalar roots (not engaged); byte-identical clean passthrough; `finish_reason` required (fail closed).
- **Golden corpus:** `matrix_results.json` (1,536 real requests). Known-corrupt → schema-valid; clean →
  **byte-identical**.
- **Router integration:** SSE frames split mid-UTF-8 and mid-frame; `n>1`; interleaved choices; usage
  chunk; `[DONE]` with no finish reason; **mid-stream transport error after buffering → retained frames
  replayed**; cap breach → replay; `logprobs` → not engaged; non-structured request → byte-identical.

Measured core cost: **~8 µs** clean path.

---

## 7. Limits (stated, not discovered later)

- **Syntactically-valid garbage prefixes are irreducibly ambiguous.** `{"a": {"x":1}` is byte-identical
  whether it is garbage+`J` or a truncated document. We refuse (pass through). If a future corpus shows
  the engine emits such prefixes, the *only* sound discriminators are engine-side, not byte-side.
- **Malformed-but-not-truncated content with a validating nested fragment** (`{"a" {"x":1}`) is the last
  wrong-answer class. It is closed by the **discriminating-schema precondition** (§5.1): a strict schema's
  oracle rejects the fragment. It would reopen if that precondition were relaxed. **Do not relax it
  without re-reading §5.1.** Rejecting this class outright is not an option — the same test would reject
  the real, observed `{{"summary":"x"}` corruption.
- **Array-rooted schemas are not engaged.** They are doubly exposed: no key/colon structure to make
  garbage syntactically illegal (so the ambiguity gate rarely fires), *and* the discriminating keywords
  (`required`, `additionalProperties`) are object keywords that say nothing about an array — so the
  oracle could not reject a nested fragment even when the gate let one through.
- **Scalar-rooted schemas** cannot be repaired at all, and are not engaged.
- **`json_object` mode has no schema** → no oracle. **Not engaged.**
- **Tool calls are out of scope for v1.** There are no real corrupted `tool_calls` examples yet, so
  streaming argument assembly would be speculative. The core retains the `repair_tool_arguments()`
  seam; tool-call engagement and repair will be specified only after a representative corpus exists.
  For avoidance of doubt, the tool-call parenthetical in §5.2 and assembly item in §5.3 describe that
  future seam, not v1 implementation requirements.
- **Mid-body/tail corruption** is not handled; it has never been observed and surfaces as `unknown`.
- **Disaggregated-prefill paths** (`request.py:880`, `1019`, `1303`) construct their own
  `StreamingResponse` and are **out of scope** for this change.

## 8. Non-goals

- Fixing the vLLM bug (file upstream: `advance_grammar`; and the unexplained batch-slot dependence).
- Enforcing schema compliance beyond what the grammar already produced.
- Any change to caller-visible model semantics (thinking, MTP).
