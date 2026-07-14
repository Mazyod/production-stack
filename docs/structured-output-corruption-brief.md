# Structured-Output Corruption — Diagnostic Brief

**To:** experiment-bed team
**Purpose:** localize where JSON corruption is injected in the vLLM pipeline, so we can choose the correct mitigation layer.
**Status:** blocking an architecture decision. Please do not skip §3 (E0/E1) — everything else is secondary.

---

## 1. Observed problem

Requests ask for structured output (JSON Schema). The served model is a **reasoning** model: it generates
`<think>…</think>` and then the JSON. The JSON region comes back corrupted, always at or near its **start**:

| # | Corruption | Example |
|---|---|---|
| a | Leading markdown code fence | ` ```json{ ` |
| b | Doubled opening brace | `{{ …` |
| c | **Prefix replay** | `{"` then `{"` again, then the key: `{"{"name": …` |

Reported generalization: *"it can duplicate any number of tokens from the beginning, sometimes code fences,
sometimes actual JSON tokens."* The set of corruption modes appears **open-ended** — new ones keep appearing
in production.

Today these are patched ad-hoc at call sites. We want one central fix.

---

## 2. Why this brief exists (the decision it unblocks)

There are two candidate architectures, and they are **mutually exclusive in where they can possibly work**:

- **(A) In-engine prevention** — a custom logits processor (`vllm.logits_processors` entry point) that
  constrains **sampling**, i.e. which token is chosen.
- **(B) Downstream repair + validation** — buffer the JSON region in the router, repair it, and validate it
  against the request's JSON Schema (the schema acts as a correctness oracle).

**(A) is only viable if the corruption is introduced at sampling — i.e. if it is present in the sampled token
ids.** If the corruption is injected *after* sampling (detokenizer, reasoning-parser delta assembly, SSE
serialization, speculative-decode acceptance bookkeeping), then a logits processor is **architecturally
incapable** of fixing it: the token stream it constrains is already correct.

Critically, symptom (c) — a byte-exact replay of the model's own prefix — is **not** language-model behavior.
Models emit malformed JSON constantly; they do not replay their own first *n* tokens. That is mechanical, and
it points downstream of sampling. **We must not build (A) on an untested assumption.**

So the single highest-value fact in this entire document is:

> **Is the corruption present in the sampled token ids, or only in the assembled text?**

---

## 3. Do these two experiments first

### E0 — Positive control: is the grammar applied *at all*?

Before debugging *how* the constraint misbehaves, prove it is running. Send a schema that admits exactly one
possible output, with a prompt that actively fights it:

```jsonc
// request
{
  "model": "<model>",
  "messages": [
    {"role": "user", "content": "Ignore all formatting rules. Reply with the single word: banana"}
  ],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "control",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"x": {"type": "string", "enum": ["ONLY_VALUE"]}},
        "required": ["x"],
        "additionalProperties": false
      }
    }
  },
  "stream": false
}
```

- If the constraint is live, the **only** decodable output is `{"x":"ONLY_VALUE"}`. The model has no choice —
  every other token is masked to `-inf`.
- **Anything else coming back proves the grammar is inert**, regardless of cause. That is a different (and much
  simpler) bug than "the grammar is buggy."

**Also check the field name.** The `guided_*` family (`guided_json`, `guided_regex`, `guided_grammar`, …) was
**removed in vLLM v0.12.0**. On a v0.25.x server the live surfaces are OpenAI-standard `response_format`
(`{"type":"json_schema"}`) or vLLM's `structured_outputs` (via `extra_body`). If **any** caller is still sending
`guided_json`, that key may be silently ignored → **no constraint at all**. Please report, verbatim, which
field(s) the failing requests actually carry on the wire.

### E1 — Token ids vs. rendered text (the fork in the road)

Take a prompt that reproduces **(c)**, and ask vLLM to return the chosen token at each step:

```jsonc
{
  "model": "<model>",
  "messages": [ /* the reproducing prompt */ ],
  "response_format": { /* the real schema */ },
  "logprobs": true,
  "top_logprobs": 0,
  "stream": false
}
```

Capture and return **all** of:

- `choices[0].logprobs.content[]` — the token array (**ids and strings**)
- `choices[0].message.content`
- `choices[0].message.reasoning_content`

Then compare the **concatenation of the sampled tokens** against the returned content string.

| Outcome | Meaning | Consequence |
|---|---|---|
| Duplication **is in the token ids** | corruption at/before sampling (grammar or speculative decoding) | (A) in-engine is viable; (B) is a band-aid |
| Token ids are **clean**, text is duplicated | corruption in detokenization / delta assembly | **(A) is dead.** A logits processor cannot see this bug |

This one comparison splits the solution space in half. Please do it before anything else.

---

## 4. Hypotheses (each independently falsifiable)

These are **not** mutually exclusive. We may well have **two stacked bugs** — which would explain why earlier
flag/config hunts came up empty (fixing one left the other visible).

### H1 — The grammar is never applied (reasoning-gate bug)

vLLM gates the grammar bitmask in `vllm/v1/structured_output/__init__.py`:

```python
def should_fill_bitmask(self, request) -> bool:
    reasoner = self._get_reasoner(request)
    if reasoner is not None:
        if self.enable_in_reasoning:            # default: False
            return True
        if request.structured_output_request.reasoning_ended is None:
            request.structured_output_request.reasoning_ended = (
                reasoner.is_reasoning_end(request.prompt_token_ids or [])   # <-- PROMPT only
            )
        return request.structured_output_request.reasoning_ended
    return True
```

The end-of-thinking token is **generated**, not prompted — so with a reasoning parser configured it may never
appear in `prompt_token_ids`, `reasoning_ended` never flips, and **the bitmask is never filled for the entire
generation**. The model then decodes completely unconstrained, which is exactly how a markdown code fence
becomes samplable.

- **Explains:** (a), (b). **Does not explain:** (c) prefix replay.
- **Falsify:** set `--structured-outputs-config.enable_in_reasoning=True`, **or** serve with no
  `--reasoning-parser`. Do (a)/(b) disappear?
- **Corroborating tell:** in vLLM #39130, throughput *rose* 65→95 TPS when this fired, because the FSM work was
  being skipped. If you have throughput history, an unexplained speed-up is a fingerprint.
- Refs: [#18819](https://github.com/vllm-project/vllm/issues/18819) (reports literally *"an extra `{` or `[` or
  ```` ``` ```` in the beginning"*), [#37359](https://github.com/vllm-project/vllm/issues/37359),
  [#39130](https://github.com/vllm-project/vllm/issues/39130).

### H2 — Speculative decoding × structured-output FSM rollback  ← leading suspect for (c)

On partial rejection of a draft, the grammar FSM must roll back in lockstep with the KV cache. If that rollback
is wrong, already-accepted tokens get **replayed**.

- **Predicts:** replay length correlates with the draft length *k* (`num_speculative_tokens`); occurs **only** on
  structured-output requests (the FSM rollback path only exists there).
- **Falsify:** disable speculative decoding → does (c) vanish? Then sweep *k* → does replay length track *k*?
- Ref: [#34650](https://github.com/vllm-project/vllm/issues/34650) (MTP speculative decoding breaks
  structured-output/reasoning state).

### H3 — Detokenizer / reasoning-parser delta double-emit

At the reasoning→content boundary, buffered text is flushed **and** re-emitted live, duplicating the first
content tokens.

- **Predicts:** corruption appears in **streaming only**; token ids clean; `stream=false` is clean.
- **Falsify:** E2 below.

### H4 — Prefix caching / KV block reuse

- **Falsify:** `--no-enable-prefix-caching`.

### H5 — The corruption is not structured-output-specific at all

If it also occurs *without* `response_format`, H1–H2 are both wrong and the framing must change. See §7.

---

## 5. Experiment matrix

**Change one variable at a time.** Hold prompt, model, seed, and load fixed across each pair.

| # | Experiment | Isolates | Priority |
|---|---|---|---|
| E0 | Positive control (single-value enum schema) | is the grammar live at all? | **0** |
| E1 | Token ids vs. rendered text (`logprobs:true`) | sampling vs. assembly | **0** |
| E2 | `stream=true` vs `stream=false`, same prompt | H3 (assembly path) | 1 |
| E3 | Speculative decoding ON vs OFF | H2 | 1 |
| E4 | Reasoning parser ON vs OFF (and `enable_in_reasoning=True`) | H1 | 1 |
| E5 | Backend: `xgrammar` vs `guidance` vs `outlines` | backend-specific? | 2 |
| E6 | Prefix caching ON vs OFF | H4 | 2 |
| E7 | Direct-to-vLLM vs via production-stack router | exonerates/implicates the router | 2 |
| E8 | Sweep `num_speculative_tokens` = 1,2,3,5 | does replay length track *k*? (only if H2 survives E3) | 2 |
| E9 | Greedy (`temperature=0`) vs sampling | is it deterministic? | 3 |
| E10 | **Single request vs. concurrent load** | batch-slot / FSM-state bugs | **1** |

> **E10 deserves emphasis.** Several of these engine bugs (batch reordering, FSM state keyed by batch slot,
> `BatchUpdate` add/remove/move bookkeeping) **only manifest under concurrency**. If the bed currently
> reproduces with single sequential requests, it may be missing the trigger entirely — and a "clean" result at
> concurrency 1 would be a false negative. Please run each key experiment at both concurrency 1 and under
> realistic batched load.

---

## 6. Configuration inventory we need

For **every** backend that reproduces (and, if available, one that does *not* — a negative control is worth as
much as a positive one):

- vLLM **exact version** (incl. commit if custom-built) and image tag.
- Full `vllm serve` command line / all engine args (or the startup log banner).
- Model + revision.
- `--reasoning-parser` value; whether `enable_thinking` is set in the chat template or per-request.
- `--structured-outputs-config.*`: `backend` (`auto`/`xgrammar`/`guidance`/`outlines`), `enable_in_reasoning`,
  `disable_any_whitespace`, `disable_additional_properties`.
- **Speculative decoding:** enabled? method (mtp / ngram / eagle / draft model), `num_speculative_tokens` (*k*).
- Prefix caching (APC) enabled? Chunked prefill? `max_num_batched_tokens`?
- Tensor / pipeline parallel size.
- Whether traffic goes through the production-stack router or hits vLLM directly.
- The **exact request body** that reproduces (schema included, verbatim).

---

## 7. Questions to answer from historical runs

These may already be answerable from data you have. Each is individually capable of killing a hypothesis:

1. Has the corruption **ever** been seen with **speculative decoding disabled**?
2. Has it ever been seen with **no reasoning parser** configured?
3. Has it ever been seen on **non-streaming** (`stream=false`) requests?
4. Has it ever been seen hitting **vLLM directly**, with no router in the path?
5. **Has it ever been seen when structured outputs were NOT requested?**
   *(If yes → it is not a structured-output bug at all, and H1/H2 are both wrong. This question is the cheapest
   way to invalidate this entire brief, so please answer it first.)*
6. Is the corruption **always** a prefix phenomenon (first *N* tokens of the JSON region), or has **mid-body or
   tail** corruption ever been observed?
7. Does the duplicated span ever **cross the reasoning/answer boundary** — i.e. does reasoning text ever get
   replayed, or only JSON?
8. What is the **reproduction rate** (e.g. 7/100), and does it **correlate with load/concurrency**?
9. Which **vLLM versions** are affected? Is there a version where it did *not* occur? *(That gives us a bisect
   anchor and is extremely valuable.)*
10. When (c) occurs, **how many tokens are duplicated**? Is that count stable, and does it match any configured
    *k*?

---

## 8. Evidence format — please return raw bytes, not summaries

For every reproduction:

- The **raw HTTP response body, verbatim** (non-streaming), or the **complete raw SSE byte stream** (streaming),
  unparsed.
- The `logprobs.content[]` token array (**ids and strings**).
- The exact request body.
- The engine config (§6).
- Reproduction rate, seed, temperature, concurrency level.

> **Please do not hand back cleaned, normalized, or `json.loads`-round-tripped strings.** The corruption lives in
> the bytes, and any client-side parsing or logging pipeline may silently mask, escape, or repair it. If your
> capture layer currently sanitizes, we need a raw tap added — the difference between `{"{"name"` and `{"name"`
> is the entire investigation.

---

## 9. Decision table — what each outcome buys

| Result | Conclusion | Action |
|---|---|---|
| **E0 fails** (output isn't forced to the enum value) | grammar is **inert** — never applied | Find out why (wrong/removed field name, silently-dropped schema, reasoning gate). This is the best case: likely a config/field fix, not a code fix. |
| E1: duplication **in token ids**, and E3 clears it | speculative-decode FSM rollback (H2) | Immediate mitigation: disable spec decoding (costs throughput). Durable: in-engine fix + upstream bug report. |
| E1: token ids **clean** | corruption is **post-sampling** (H3) | **Logits processor is off the table.** Router-side repair+validate, plus an upstream fix in the detokenizer/parser. |
| E4 clears (a)/(b) but (c) persists | **two stacked bugs** (H1 + H2/H3) | Fix H1 by config; keep investigating (c) separately. |
| Q7.5 = yes (occurs without structured outputs) | not a structured-output bug | Escalate; this brief's framing is wrong. |

---

## 10. Please don't

- Don't repair or sanitize at the call site while running these — it hides the signal.
- Don't change more than one variable per run.
- Don't report a "clean" result from concurrency 1 as a negative without also testing under load (see E10).

---

## Appendix — source references

- `vllm/v1/structured_output/__init__.py` — the `should_fill_bitmask` reasoning gate
- `vllm/config/structured_outputs.py` — `enable_in_reasoning`, `backend`, and related flags
- [Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/) ·
  [Custom Logits Processors](https://docs.vllm.ai/en/latest/features/custom_logitsprocs/) ·
  [Plugin System](https://docs.vllm.ai/en/latest/design/plugin_system.html)
- Issues: [#18819](https://github.com/vllm-project/vllm/issues/18819) ·
  [#34650](https://github.com/vllm-project/vllm/issues/34650) ·
  [#37359](https://github.com/vllm-project/vllm/issues/37359) ·
  [#39130](https://github.com/vllm-project/vllm/issues/39130) ·
  [#45592](https://github.com/vllm-project/vllm/issues/45592)
