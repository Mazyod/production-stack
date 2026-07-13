# Boundary JSON repair spike (v2)

The engine failure has a narrow shape: `output = G + J`, where `G` is a short
unconstrained prefix and `J` is a complete JSON document emitted by the grammar
after its FSM restarts. Repair finds `J`'s boundary; it never synthesizes JSON.

`repair.py` applies these checks in order:

1. `finish_reason="length"` fails closed as `incomplete`.
2. A whole-input parse and schema validation returns the original text
   byte-for-byte as `clean`.
3. `is_valid_json_prefix()` asks whether any legal JSON document can start with
   the supplied text. A possible truncation returns `ambiguous` unchanged.
4. Only for an explicitly object- or array-rooted schema, every matching opener
   whose UTF-8 byte offset is below `max_prefix_bytes` is tried. A repair must
   parse, validate, and consume the remainder.
5. Every other outcome and every internal exception returns `unknown`.

The incremental prefix recognizer handles object/array parser states, strings,
escapes and partial Unicode escapes, strict JSON numbers (including completable
`-`, `1.`, and `1e+` prefixes), partial literals, nesting, and JSON whitespace.
The v1 structural-descent/colon guard and candidate-count cap are gone.

There is one extra use of the same ambiguity predicate during candidate search.
If an earlier opener begins a valid-but-incomplete JSON value, recovery will not
select a later opener nested inside it. This is what makes
`"garbage{"a":{"x":1}` return `unknown` rather than the silently wrong
`{"x":1}` fragment.

A trailing closing Markdown fence is allowed only when the candidate's garbage
prefix contains an opening fence. Thus `[[1]````, with no opening fence, cannot
repair to `[1]`. Mode labels remain telemetry only and never affect candidate
selection.

Schemas are compiled once and cached by canonical serialization. A registry
with a rejecting retrieval callback disables remote `$ref` loading. Malformed,
recursive, non-serializable, and remotely-referencing hostile schemas are
contained as `unknown`; no schema exception escapes. Schema-less and
scalar-rooted inputs can return `clean` on the whole-input fast path but never
enter recovery.

Tool-call arguments use `repair_tool_arguments()`. The caller must provide the
matching tool's `function.parameters`; an unknown tool passes `None` and is not
repaired because there is no validation oracle.

`finish_reason` is a required keyword for both public functions. Omitting it
raises `TypeError` at the call boundary. Passing `None` explicitly is allowed.

## Run

```console
python -m pip install -r spikes/json_boundary_repair/requirements.txt
pytest -q spikes/json_boundary_repair/test_repair.py
python spikes/json_boundary_repair/bench.py
```

## Observed results

Run on 2026-07-13 with Python 3.13.13 and cached `jsonschema` 4.26.0:

```text
.....................................................................    [100%]
69 passed in 0.10s
```

An additional generated-input check accepted all 95,854 prefixes of 2,000
legal JSON documents.

```text
median of 7 runs x 10,000 calls
clean             8.63 us/call
extra_brace      14.23 us/call
dup_prefix       15.11 us/call
code_fence       12.36 us/call
other            12.66 us/call
clean fast path is cheaper than the median corrupt path: 8.63 < 13.44 us/call
```

## Remaining ambiguity outside the engine invariant

The gate proves only that an input cannot be a legal truncation. It cannot prove
that the invalid leading bytes came from this particular engine defect rather
than some other malformed-output defect. For example, with the permissive
schema `{"type":"object"}`, the input `{"a" {"x":1}` repairs to `{"x":1}`.
That is correct under the scoped `G + J` invariant, but could be a wrong fragment
if the backend instead intended one malformed document.

No byte-only rule can distinguish that example from observed forms such as
`{{"summary":"x"}` without rejecting those repairs too. Engagement must
therefore remain limited to the measured structured-output failure mode. Within
the truncation threat model, the prefix gate is conservative: every legal JSON
prefix passes through as `ambiguous`.
