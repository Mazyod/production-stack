# Dynamic-config tolerant watcher — Design

**Date:** 2026-07-18
**Status:** implemented (refined after Codex adversarial review)
**Scope:** `vllm_router` — let every fork-added operator flag (and any upstream flag) live in the `--dynamic-config-yaml` / `--dynamic-config-json` file without breaking the dynamic-config watcher.

---

## 1. Problem

The fork adds operator flags that are consumed **only at startup** (wired into `app.state` in `initialize_all`):

| Flag | Startup consumer |
|---|---|
| `--backend-connect-timeout` | `app.state.backend_client_timeout` |
| `--backend-read-timeout` | `app.state.backend_client_timeout` |

(`--timeout-keep-alive` was the first fork flag to hit this and is already handled — it is the precedent this design generalizes.)

Operators want a single source of truth: the dynamic-config YAML they already pass with `--dynamic-config-yaml`, not a long `command:` array in Docker Compose. There are two YAML code paths, and they disagree:

1. **Startup load** — `load_initial_config_from_config_file_if_required` does `parser.set_defaults(**yaml_config)` then re-parses. `set_defaults` writes **any** key into the namespace as an attribute; a key only takes effect if it is *also* a declared argparse destination that some code path reads. Every flag above is such a destination, so it takes effect at startup when placed in the YAML.
2. **The dynamic watcher** — `DynamicConfigWatcher` (started whenever `--dynamic-config-yaml`/`--dynamic-config-json` is set) re-reads the same file every 10 s into `DynamicRouterConfig(**config)`, a dataclass that **rejects any unknown key** with `TypeError`.

So moving a fork flag into the dynamic YAML makes the watcher thread raise `TypeError: __init__() got an unexpected keyword argument 'backend_connect_timeout'` every 10 s. It is caught by the watcher's `except Exception` and logged as a warning, but the consequence is that hot-reload of service discovery / routing / callbacks is **silently dead** for the life of the process. The flag still works at startup, which is exactly what makes the failure hard to notice.

## 2. Goal / non-goal

- **Goal:** every fork flag (and, as a free consequence, any upstream flag) can be declared in the dynamic-config file, honored at startup, without breaking the watcher.
- **Non-goal:** hot-reloading these flags. The operator does not rely on hot-reload. Startup-time load matching the `timeout_keep_alive` precedent is sufficient. Runtime edits to these keys require a router restart.

## 3. Decision

Make the watcher **tolerant** by *classifying* the keys in the loaded file instead of blanket-dropping unknown ones. `DynamicRouterConfig.from_yaml` / `from_json` route their parsed dict through `_from_config_dict`, which sorts every non-field key into one of two buckets before the strict `DynamicRouterConfig(**config)` construction:

- **Recognized startup-only flags** — any argparse destination that is not a reconfigurable field (the fork flags, plus upstream flags like `engine_stats_interval`). Honored already at startup; ignored by the watcher and logged at `debug`.
- **Unrecognized keys** — neither a field nor a known flag. Almost certainly a typo, so **raise** rather than drop (see the safety rationale below). The watcher's `except Exception` catches it, logs a warning, and keeps the running configuration.

```python
@lru_cache(maxsize=1)
def _recognized_arg_dests() -> frozenset[str]:
    from vllm_router.parsers.parser import build_parser
    return frozenset(a.dest for a in build_parser()._actions)

@classmethod
def _from_config_dict(cls, config: dict) -> "DynamicRouterConfig":
    known_fields = {f.name for f in fields(cls)}
    extra = [k for k in config if k not in known_fields]
    recognized = _recognized_arg_dests()
    startup_only = sorted(k for k in extra if k in recognized)
    unrecognized = sorted(k for k in extra if k not in recognized)
    if startup_only:
        logger.debug("... ignoring startup-only config keys ...: %s", ", ".join(startup_only))
    if unrecognized:
        raise ValueError(f"Unrecognized dynamic-config key(s) {unrecognized}: ...")
    return cls(**{k: v for k, v in config.items() if k in known_fields})
```

`_recognized_arg_dests()` introspects the *same* parser used at startup (a new `build_parser()` in `parsers/parser.py`, which `parse_args()` now calls), so the recognized set stays in sync with the flags automatically — zero per-flag maintenance.

### Why classify instead of blanket-drop — the typo-safety rationale

Blanket-dropping unknown keys has a sharp edge (surfaced by adversarial review). The reconfigurable fields a file *omits* fall back to their dataclass defaults. So renaming a live hot-reloadable key — e.g. `callbacks:` → `callback:` — would drop the typo'd `callback`, leave `callbacks` absent, and the resulting config would carry `callbacks=None`. That differs from `current_config`, fires `reconfigure_all`, and **silently disables callbacks** (the same pattern clears `session_key`, empties backend lists, etc.). The old strict parser incidentally protected against this by rejecting the whole snapshot on any unknown key. Classification preserves that protection *only* for truly-unrecognized keys, while still tolerating the legitimate startup-only flags — so a typo keeps the running config, and a fork flag does not.

### Why this over the alternatives

- **vs. enumerating the fork flags as `DynamicRouterConfig` fields** (the literal `timeout_keep_alive` precedent): tolerance covers *all* flags with zero per-flag maintenance, now and for any future flag. It is also safer on the spurious-reconfigure axis (below).
- **vs. a new static `--config` flag that skips the watcher:** larger change, and it forces operators to change their launch command. Not needed given tolerance keeps the existing command working.

### The fork flags are inert to the watcher — the key property

The watcher reconfigures on `config != self.current_config`, where both operands are `DynamicRouterConfig` instances. The fork flags never enter the dataclass on **either** side — neither `from_args(args)` (reads only dataclass fields) nor `from_yaml(path)` (classifies them as startup-only and skips them). Therefore editing a fork key at runtime does **not** fire a reconfigure (both operands filter it identically), and adding fork keys is neutral to the first-tick comparison.

`timeout_keep_alive` is the one exception: it remains a reconfigurable dataclass field (per the earlier precedent), so editing *it* at runtime still fires a disruptive no-op `reconfigure_all`. That is a pre-existing minor and is out of scope; the inertness claim is scoped to the fork flags.

### First-tick reconfigure — fixed here, not deferred

Adversarial review noted that reviving the watcher for fork-flag configs exposes them to a *pre-existing* guaranteed first-tick reconfigure: `from_args` copied argparse defaults for `k8s_port` (8000), `k8s_namespace` (`"default"`), `k8s_label_selector` (`""`) while the dataclass defaulted them to `None`, and `from_args` omitted `prefill_model_labels` / `decode_model_labels` / `static_model_labels` entirely — so an *unchanged* file already produced `from_yaml != current_config`, firing `reconfigure_all` on the first tick. Because the watcher's first iteration runs before its first sleep and is started before `initialize_routing_logic`, that reconfigure races startup and, for kvaware/prefixaware routing, rebuilds the router *without* its lmcache/threshold args.

This change fixes the root cause rather than deferring it: the dataclass defaults for `k8s_port` / `k8s_namespace` / `k8s_label_selector` are aligned to the argparse defaults, and the three label fields are copied in `from_args`. An unchanged file now yields `from_args(args) == from_yaml(path)`, so the first tick is a genuine no-op — verified by a test and in a live run (`Config changed, reconfiguring` count 0 at startup).

## 4. Scope of change

- **`src/vllm_router/parsers/parser.py`** — extract `build_parser()` (the argument construction) from `parse_args()`; `parse_args()` now calls it. Enables `_recognized_arg_dests()` to introspect the flag set.
- **`src/vllm_router/dynamic_config.py`** — add `_recognized_arg_dests()` and `_from_config_dict` (classify + tolerate + reject-typo), route `from_yaml`/`from_json` through it; align the three k8s dataclass defaults with argparse; copy the three label fields in `from_args`.
- **No** new fork-flag dataclass fields, and **no** `app.py` / `reconfigure_all` change.

## 5. Caveats

1. **Startup-only.** The fork flags are read once at boot. Editing them in the file while the router runs has no effect until restart (they are inert to the watcher, so the edit is a no-op — not even a reconfigure).
2. **Underscored keys.** No dash→underscore translation happens (`read_and_process_yaml_config_file` uses YAML keys verbatim). The key must be `backend_read_timeout`, not `backend-read-timeout`. A dashed key is *unrecognized* and now raises on the watcher tick (retaining the running config) — and matches no argparse dest at startup, so it also no-ops there.
3. **Unrecognized keys are rejected, not silently applied.** A key that is neither a field nor a known flag raises on the watcher tick; the watcher logs a warning and keeps the running configuration. This protects against a typo of a hot-reloadable key silently reverting that field to its default.

## 6. Testing

Unit tests in `src/tests/test_dynamic_config.py`:

1. `from_yaml` accepts all fork flags **plus** a sample upstream key (`engine_stats_interval`) in one file without crashing, and still parses the reconfigurable fields (`service_discovery`, `routing_logic`); `from_json` equivalent.
2. **Fork flags are inert:** two YAML files identical except that one carries all fork flags produce **equal** `DynamicRouterConfig` objects via `from_yaml`.
3. **Unrecognized key rejected:** a typo key (`callback`) makes `from_yaml` raise `ValueError`; a recognized startup-only key (`backend_read_timeout`) does not.
4. **No first-tick reconfigure:** with a representative file (static models + fork flags), `from_args(parse_args(--dynamic-config-yaml=f)) == from_yaml(f)`.
5. Existing `timeout_keep_alive` tests continue to pass (regression).

**Real-environment verification** (project `verify` skill), all confirmed against a live router + fake engine:

- Router boots via the dynamic-YAML path carrying fork flags; **no `TypeError` watcher warnings** across ≥ 3 ticks; the classification debug line lists the fork keys as startup-only.
- A low `backend_read_timeout` set **in the YAML** fires the structured **504** at ~2 s against a stalling engine — the value is *consumed*, not merely *accepted*.
- **No spurious first-tick reconfigure** (`Config changed` count 0 at startup).
- A runtime typo (`callback` appended to the live file) is **rejected with a warning, no reconfigure, config retained, router still serving**.

**Adversarial review:** Codex (GPT-5.6-Sol, read-only) over the diff; findings 1–5 addressed here.

## 7. Known limitations, out of scope

Deeper *pre-existing* watcher issues that only bite on an actual runtime config change (now that the first tick is a genuine no-op) — left for a separate change:

1. **`reconfigure_routing_logic` drops startup args.** It passes only `routing_logic` + `session_key`, so a real hot-reload of a kvaware/prefixaware deployment rebuilds the router without its lmcache ports / thresholds. Unrelated to the fork flags.
2. **`external-only` cannot be reconfigured.** `reconfigure_service_discovery` handles only `static` / `k8s` and raises `ValueError` otherwise, so any real edit to an `external-only` dynamic-config file warns every tick without applying. Unrelated to the fork flags.
3. **Watcher-start ordering.** The watcher thread is started before `initialize_routing_logic` completes. Harmless once the first tick is a no-op, but a defensive reorder (start the watcher after `initialize_all`) would remove the race entirely.
