# Fork maintenance

This fork maintains production router fixes with a small patch series over
upstream. The [README patch index](../README.md#about-this-fork) describes shipped
operator behavior; [AGENTS.md](../AGENTS.md) contains the working agreements.

## Patch and release workflow

`origin` is `Mazyod/production-stack`; `upstream` is
`vllm-project/production-stack`. Keep topic patches self-contained and normally
append follow-up commits after publication. Prefer upstream behavior when a
workaround is no longer needed. Commit bodies explain the reason and report
actual verification, including failures or checks that were not run.

The [image workflow](../.github/workflows/build-router.yml) does the following:

1. Resolves an upstream `vllm-stack-*` tag, or uses a manually supplied tag.
2. Finds the merge base with upstream main and collects subsequent fork commits.
3. Checks out the upstream release tag and replays those commits in order.
   Patch deletions win modify/delete conflicts; other conflicts fail the build.
4. Runs `src/tests` on that reconstructed source before building and publishing
   the amd64/arm64 images to `openimage/production-stack-router`.
5. Requires HTTP 200 from the published image's `/health` before promoting
   `latest`. Versioned tags are already published at this point.

Triggers are a Monday schedule, manual dispatch, and pushes to `main` touching
`docker/Dockerfile`, `src/vllm_router/**`, `pyproject.toml`, or the workflow itself.
Docs-only and tests-only pushes do not trigger publication. Scheduled runs and
dispatches without a supplied tag skip an already-published upstream version;
an explicit tag forces a build. These existing trigger rules are intentional
release behavior, not a guarantee that every commit runs CI.

Images get `vllm-stack-X.Y.Z`, `vX.Y.Z`, and, after the smoke test, `latest` tags.
A qualifying push can overwrite the versioned tags with newer fork patches.
Record the source commit, upstream tag, workflow run, and image digest when
reporting a release. The image source is upstream-tag-plus-patches, not simply
the checkout's HEAD; passing local tests alone does not verify that replay.

GitHub CLI can resolve this checkout to upstream. Use explicit repository names:

```bash
gh run list -R Mazyod/production-stack --workflow build-router.yml
gh run watch RUN_ID -R Mazyod/production-stack
gh api repos/Mazyod/production-stack/actions/workflows/build-router.yml
```

After an authorized release, inspect the run and image digest before reporting
success. A workflow dispatch or successful build alone is not a completed release.
The audio image and its workflow moved to the engine fork in commit `5605dc8`;
this repository publishes only the router.

## Decisions to preserve

### Request-attempt lifecycle and backend timeouts — July 2026

An attempt owns an opaque handle. Caller-controlled `X-Request-Id` values can
collide and are metadata, not identity. Finalization pops the attempt's active
record and retires its recorded stage exactly once. Cancellation can raise
`BaseException`; `except Exception` alone does not clean up abandoned streams.
The statistics reader also runs on an OS thread, hence the shared `RLock`.

For the main `process_request` path, healthy streams have no total-duration
limit. Connect and read-silence bounds serve different purposes: `connect`
covers DNS/pool acquisition, `sock_connect` bounds socket establishment, and
`sock_read` bounds silence after upload. An entry deadline closes the upload
gap and is disabled once response headers arrive. Slow-client backpressure
suspends aiohttp's read watchdog. Preserve these distinctions when changing
timeouts; the multipart path has its own existing total timeout.

Connect failures may rotate backends and exhaust into 502. Read/entry timeouts
return 504 without rotation to avoid multiplying workload-shaped stalls.
Mid-stream SSE failures use an error event plus `[DONE]`; non-SSE failures keep
the abrupt close. Exact defaults, envelopes, and the accepted long-queue risk
are in the [operator contract](../README.md#backend-socket-timeouts).

Evidence: `38eb146`, `a193009`, `62a0289`, and the lifecycle, socket-timeout,
timeout-contract, and transcription tests under `src/tests/`.

### Dynamic configuration — July 2026

Startup and reload read the same file through different paths. Recognized
startup-only flags must not break reload, but unknown keys must reject the
reload snapshot to prevent a typo from resetting a live field to its default.
Parser and dataclass defaults must agree so an unchanged file does not trigger
a first-tick reconfiguration. Test that a YAML value reaches its consumer,
not merely that the parser accepts it.

The [implemented design](superpowers/specs/2026-07-18-dynamic-config-tolerant-watcher-design.md)
records the decisions and tests. Known limits remain: runtime reconfiguration
drops some routing startup arguments; `external-only` discovery cannot reload;
and watcher startup precedes completion of router initialization.
`timeout_keep_alive` is startup-only for uvicorn but remains a dataclass field:
editing it triggers reconfiguration without changing the server's keep-alive
timeout. Restart to change it. Backend timeout edits are inert until restart.

### Structured-output repair removed — 2026-07-19

The router-side corruption workaround was deliberately removed after an engine
fix resolved the reasoning-to-answer boundary issue. Its module, flags, tests,
metrics, and dedicated docs are no longer part of this fork. Associated cache
changes were also reverted: lookup happens before request rewriting in
`main_router.py`, and cache hits do not invoke `pre_request` callbacks.
Do not resurrect this workaround from an old report without new evidence.

The removal used a specifically authorized history rewrite. A local
`backup-main-orig` branch was retained at the time; its presence and other branch
cleanup state must be checked with Git, not inferred from historical notes.
Dropping the repair patch also dropped an `asyncio` import used by retained
timeout code, producing failures that looked like a test-suite hang. After
removing patches, run the full suite and compare failures against the previous
baseline to distinguish new regressions from existing problems.

Ignored `.superpowers/sdd/` reports describe that removed implementation and
obsolete execution-environment restrictions. Local dashboards may also still
reference its deleted metrics. They are historical artifacts, not current
requirements or evidence that the feature is available.

## Verification and maintaining context

Follow [CONTRIBUTING.md](../CONTRIBUTING.md#code-quality-and-validation) for setup,
tests, and lint, and the [verify skill](../.agents/skills/verify/SKILL.md) for live
HTTP scenarios. The ordinary router suite uses mocks and local sockets; it does
not require a GPU or a vLLM installation. Optional performance/engine tools have
separate requirements. Image health checks do not exercise inference.

When changing behavior, update the operator README, regression coverage, and
the relevant decision here together. Mark superseded decisions with the reason
and date. Link existing specifications rather than copying them. Record measured
results for each change; historical test counts are not a verification target.

This document was distilled on 2026-09-14 from the project's Claude memories
(`fork-patch-workflow`, `backend-socket-timeouts-decision`, `gh-cli-fork-repo`,
`structured-output-patch-removed`), available project prompt history, Git history,
and current code. Full session transcripts were unavailable. Personal model,
notification, permission, and agent-wrapper settings remain outside this repo.
