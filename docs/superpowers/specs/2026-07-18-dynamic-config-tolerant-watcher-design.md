# Dynamic-config tolerant watcher — Design

**Date:** 2026-07-18
**Status:** approved (design), pending implementation
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

1. **Startup load** — `load_initial_config_from_config_file_if_required` does `parser.set_defaults(**yaml_config)` then re-parses. `set_defaults` accepts **any** key that matches an argparse dest, so every flag above already takes effect at startup when placed in the YAML.
2. **The dynamic watcher** — `DynamicConfigWatcher` (started whenever `--dynamic-config-yaml`/`--dynamic-config-json` is set) re-reads the same file every 10 s into `DynamicRouterConfig(**config)`, a dataclass that **rejects any unknown key** with `TypeError`.

So moving a fork flag into the dynamic YAML makes the watcher thread raise `TypeError: __init__() got an unexpected keyword argument 'backend_connect_timeout'` every 10 s. It is caught by the watcher's `except Exception` and logged as a warning, but the consequence is that hot-reload of service discovery / routing / callbacks is **silently dead** for the life of the process. The flag still works at startup, which is exactly what makes the failure hard to notice.

## 2. Goal / non-goal

- **Goal:** every fork flag (and, as a free consequence, any upstream flag) can be declared in the dynamic-config file, honored at startup, without breaking the watcher.
- **Non-goal:** hot-reloading these flags. The operator does not rely on hot-reload. Startup-time load matching the `timeout_keep_alive` precedent is sufficient. Runtime edits to these keys require a router restart.

## 3. Decision

Make the watcher **tolerant**: `DynamicRouterConfig.from_yaml` / `from_json` filter the loaded dict down to the dataclass's own fields before the strict `DynamicRouterConfig(**config)` construction. Keys that are not reconfigurable fields are dropped (honored already at startup, ignored by the watcher) and logged at `debug`.

```python
@classmethod
def _from_config_dict(cls, config: dict) -> "DynamicRouterConfig":
    known = {f.name for f in dataclasses.fields(cls)}
    dropped = sorted(k for k in config if k not in known)
    if dropped:
        logger.debug(
            "DynamicConfigWatcher: ignoring non-reconfigurable config keys "
            "(honored at startup, not hot-reloaded): %s",
            ", ".join(dropped),
        )
    return cls(**{k: v for k, v in config.items() if k in known})
```

`from_yaml` and `from_json` route their parsed dict through `_from_config_dict`.

### Why this over the alternatives

- **vs. enumerating the fork flags as `DynamicRouterConfig` fields** (the literal `timeout_keep_alive` precedent): tolerance covers *all* flags with zero per-flag maintenance, now and for any future flag. It is also strictly safer on the spurious-reconfigure axis (below).
- **vs. a new static `--config` flag that skips the watcher:** larger change, and it forces operators to change their launch command. Not needed given tolerance keeps the existing command working.

### Fork flags are inert to the watcher — the key property

The watcher decides to reconfigure with `if config != self.current_config`, where both operands are `DynamicRouterConfig` instances. Because the fork flags are dropped by the filter, they never enter the dataclass on **either** side of that comparison — neither `from_args(args)` (which only reads dataclass fields) nor `from_yaml(path)` (which now filters). Therefore:

- **Editing a startup-only fork key at runtime does not fire a reconfigure.** Both `current_config` (set from a prior `from_yaml`) and the new tick's `config` filter the key out identically, so the comparison is unchanged. (The edited value still needs a restart to take effect.)
- Adding fork flags to the file is **neutral** with respect to the first-tick comparison: they contribute nothing to it.

This is a concrete improvement over the enumerate-9-fields approach, where each fork flag *would* be a dataclass field and editing one at runtime *would* fire a disruptive, no-op `reconfigure_all`.

**Honest limit:** this change does **not** claim `from_args(args) == from_yaml(path)` in general. It is not true today: `from_args` copies argparse defaults for `k8s_port` (8000), `k8s_namespace` (`"default"`), and `k8s_label_selector` (`""`), while the dataclass defaults for those are `None`, so a YAML that omits them already differs from `current_config` on the first tick. That pre-existing first-tick reconfigure is orthogonal to this change (see §7); tolerance neither causes nor cures it.

## 4. Scope of change

- **Code:** `src/vllm_router/dynamic_config.py` only — add `_from_config_dict`, route `from_yaml`/`from_json` through it. No new dataclass fields. No `from_args` change. No `app.py`, parser, or `reconfigure_all` change.

## 5. Caveats (documented, not "fixed")

1. **Startup-only.** These keys are read once at boot. Editing them in the YAML while the router runs has no effect until restart — identical to `timeout_keep_alive` today.
2. **Underscored keys.** No dash→underscore translation happens (`read_and_process_yaml_config_file` uses YAML keys verbatim). The YAML key must be `backend_read_timeout`, not `backend-read-timeout`; a dashed key silently no-ops (it matches no argparse dest at startup and is dropped by the watcher).
3. **Typos in hot-reloadable keys are now silently ignored** instead of crashing the watcher. Net-better than today (one typo no longer kills all hot-reload), surfaced at `debug`. Example: `static_backend` (missing `s`) is dropped rather than raising `TypeError`.

## 6. Testing

Unit tests in `src/tests/test_dynamic_config.py`:

1. `from_yaml` accepts all fork flags **plus** a sample upstream key (e.g. `engine_stats_interval`) in one file without crashing, and still parses the reconfigurable fields (`service_discovery`, `routing_logic`).
2. `from_json` equivalent.
3. **Fork flags are inert:** two YAML files identical except that one carries all 9 fork flags produce **equal** `DynamicRouterConfig` objects via `from_yaml` — proving the fork flags cannot perturb the watcher's `config != current_config` comparison. (Deliberately not `from_args == from_yaml`; see §3 "Honest limit".)
4. Existing `timeout_keep_alive` tests continue to pass (regression).

**Real-environment verification** (project `verify` skill): boot the actual router with `--dynamic-config-yaml` carrying fork flags against a fake engine, and prove:

- Router starts and serves; **no `Error loading config file` / `TypeError` watcher warnings** across ≥ 2 watch intervals (~20 s).
- A low `backend_read_timeout` set **in the YAML** actually fires the structured **504** against a stalling fake engine — proving the value is *consumed*, not merely *accepted*.

**Adversarial review:** Codex (read-only) over the diff; address findings.

## 7. Noted, out of scope

Two **pre-existing** first-tick spurious-reconfigure triggers, both unrelated to the fork flags and left untouched here (flagged for a separate change and to Codex):

1. **`from_args` argparse-vs-dataclass default mismatch.** `from_args` copies `args.k8s_port` (8000), `args.k8s_namespace` (`"default"`), and `args.k8s_label_selector` (`""`), but the dataclass defaults for those fields are `None`. Any dynamic-config file that omits these keys therefore produces a `from_yaml` config that differs from the startup `current_config`, firing one `reconfigure_all` on the first watch tick. This affects essentially every dynamic-config deployment today.
2. **`from_args` omits label fields.** `prefill_model_labels`, `decode_model_labels`, and `static_model_labels` exist on the dataclass but are not copied from `args`. If an operator sets one of these *hot-reloadable* keys in the YAML, `from_args` yields `None` while `from_yaml` yields the value — another first-tick `reconfigure_all`.

Neither is caused or cured by the tolerance change. A clean fix (align `from_args` with the dataclass, or vice versa) belongs in its own commit.
