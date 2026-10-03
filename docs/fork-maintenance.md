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

Keep the fork as focused patches over upstream. Periodic upstream refreshes may
rebase and consolidate that series after reconciling overlapping work. Preserve
backup refs before rewriting history, compare the final tree with the verified
reconciliation, and push with an explicit force-with-lease against the observed
remote commit. Routine fixes still append to published history.

A requested sync of upstream main can include newer work in the checkout; the
released image still uses the selected tag. Choose a release containing the
upstream features needed in the image.

The release applies the reconciled fork tree's difference from its merge base
with upstream main. This includes conflict resolutions recorded in merge
commits and excludes upstream-only changes newer than the selected tag.
Historical per-commit replay was replaced in October 2026: it failed on the
NumPy patch after upstream changed adjacent dependencies, and could not retain
merge-only lifecycle reconciliation. The aggregate delta also works after a
rebase; release reconstruction does not depend on historical patch boundaries.

The [image workflow](../.github/workflows/build-router.yml) does the following:

1. Resolves an upstream `vllm-stack-*` tag, or uses a manually supplied tag.
2. Finds the merge base with upstream main and exports the binary fork delta.
3. Uses [prepare-router-release.sh](../.github/scripts/prepare-router-release.sh)
   to check out the tag, retain fork deletions, and apply the remaining delta
   with a three-way merge. Every apply failure stops the build, including errors
   that do not leave unmerged index entries. The source commit, upstream base,
   target commit, and resulting tree are recorded.
4. Runs `src/tests` on that reconstructed source before building and publishing
   the amd64/arm64 images to `openimage/production-stack-router`.
5. Requires HTTP 200 from the published image's `/health` before promoting
   `latest`. Versioned tags are already published at this point.

Triggers are a Monday schedule, manual dispatch, and pushes to `main` touching
`docker/Dockerfile`, `src/vllm_router/**`, `pyproject.toml`, the release-source
script, or the workflow itself.
Docs-only and tests-only pushes do not trigger publication. Scheduled runs and
dispatches without a supplied tag skip an already-published upstream version;
an explicit tag forces a build. These existing trigger rules are intentional
release behavior, not a guarantee that every commit runs CI.

Images get `vllm-stack-X.Y.Z`, `vX.Y.Z`, and, after the smoke test, `latest` tags.
A qualifying push can overwrite the versioned tags with newer fork patches.
Record the source commit, upstream tag, workflow run, and image digest when
reporting a release. The image source is upstream-tag-plus-patches, not simply
the checkout's HEAD; passing local tests alone does not verify reconstruction.
Run the release-source script in a disposable clean clone, compare the resulting
tree with the intended source, and run the suite there before publishing.

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

## Maintenance release — 2026-09-14

The confirmed baseline is `vllm-stack-0.1.12`
(`66b60661aa3052810859a417559e9e830772a091`). Release source
`5cca5e4414dd07192ff37f7e88da0e6513bb69d3` includes the shared project guidance,
upstream tag merge, and merge-aware patch replay. All were pushed to `main`.
[Release run 34873842221](https://github.com/Mazyod/production-stack/actions/runs/34873842221)
succeeded, publishing `openimage/production-stack-router` for linux/amd64 and
linux/arm64. Tags `v0.1.12`, `vllm-stack-0.1.12`, and `latest` were verified to
resolve to `sha256:7144e56c84abcb3e3f42a5eb37dcee16a46945f980a04d3d51e4134b51768733`.

Verification: 29 non-merge patches replayed onto the tag produced a tree
identical to the release source. All 239 router tests passed locally, in a fresh
release checkout, and in CI; standard pre-commit hooks passed. Live HTTP checks
covered aliases, pooling, backend Server-header stripping, YAML-driven 504s,
terminal SSE errors, healthy streams beyond the read bound, and counter cleanup.
Operator deployment tests passed (`go test ./internal/controller -run
'TestDeployment' -count=1`); this was not the full Kubernetes/envtest suite.
The published image passed the HTTP-200 health gate before `latest` promotion.
Full Helm dependency validation remained blocked by the local keychain;
`helm lint` without those dependencies is not full chart validation.

At closeout, upstream `main` was `ebbb8624010514c2d4fa7edf3de95a6440a755b2`:
28 commits beyond the selected release tag. All 28 were marked `+` by
`git cherry main upstream/main`, and source comparison confirmed real changes,
including load-aware/priority routing and FastAPI, aiohttp, and Kubernetes
dependency updates. GitHub's behind count therefore reflects intentionally
excluded newer upstream work, not merely equivalent patches with different
hashes. Evaluate it against the chosen release baseline; do not merge upstream
main or publish again solely to clear that indicator. Documentation-only
closeout commits after the release source do not change the published image.

## Upstream reconciliation — 2026-10-03

At the user's request, the checkout includes upstream main through
`014d070e6f7611978d321bdb05cbe9a934b614e7`, including `vllm-stack-0.1.13`
(`e8cb4959ebfa333714ef468a3aefe158a65a09e6`). The two post-tag commits add Helm
container command/args support and E2E cleanup waits. Those upstream-only
changes belong to the checkout, not a router image reconstructed on 0.1.13.
The initial reconciliation preserved published history without publishing an
image. The subsequent authorized patch-series refresh is recorded below.

Upstream #1072 now retires in-flight counts from the recorded stage and clears
request timestamps. It still keys attempts by caller-controlled request ID and
uses a single completion outcome. The fork retains opaque handles, collision
isolation, separate completion/failure/abort outcomes, locking, and all backend
timeout/error contracts. Its existing lifecycle suites cover the incoming
upstream statistics tests and replace that duplicate suite's obsolete API.

Upstream audio fixes now support translation multipart bodies, non-JSON audio
formats, and automatic language detection. Reconciliation also awaits priority
routing and supplies an empty prompt for prefix routing of multipart requests.
The reranker wrapper delegates malformed or non-object JSON to upstream's
shared validation while retaining its valid-request template bytes and callback
ordering. Valid text/SRT/VTT responses pass through unchanged; HTML backend
error pages and malformed backend JSON retain the fork's structured 502 behavior. Raw-body reads
use the same completion/cancellation cleanup as JSON reads.

Priority routing configuration works with the tolerant config watcher, including
custom fields and headers. Existing reload limitations described above remain;
`loadaware_beta` is also startup-only and is not retained by a later router
reconfiguration. No claim of full routing-parameter hot reload is made.

Dependency reconciliation keeps the fork's Python 3.13/NumPy compatibility,
adopts upstream FastAPI/aiohttp/Kubernetes constraints, and refreshes the lock
including patched Starlette. Lightweight tests explicitly depend on `httpx`.
Upstream Gatekeeper and gateway CI workflows remain disabled along with the
previously removed upstream workflows; the fork owns its router image workflow.

Verification of implementation source `8d177bc`: all 316 router/release tests
passed on Python 3.13 both in the checkout with freshly resolved dependencies
and in a clean 0.1.13 release reconstruction installed with `uv sync --locked`
and the test/lint groups. Reconstruction differs from the checkout in exactly
the five files belonging to the two post-tag upstream commits; router code,
tests, dependency files, and Dockerfile match. Release-source regression tests
cover merge-resolution preservation, future upstream changes, retained file
deletions, and fatal content/missing-file conflicts. Independent review found
no remaining blockers.

Helm dependencies and lint passed. With Helm 3.22.0 and helm-unittest 1.1.2,
143 of 145 chart tests passed. Two inherited assertions failed: the chat-template
multiline comparison and invalid JSONPath syntax for RayCluster annotations.
Direct `helm template` parsing verified the exact chat-template bytes and all
three RayCluster annotation values. Chart source and tests remain identical to
upstream main. Operator deployment checks passed (`go test ./internal/controller
-run TestDeployment -count=1`); full Kubernetes/envtest and GPU inference were
not run.

The locally built linux/amd64 image from the reconstructed source passed HTTP
checks against a fake engine: health, alias discovery, pooling, Qwen reranking,
malformed completion/rerank bodies, priority forwarding, multipart translations
and transcription auto-detection/text output, HTML 502s, failover exhaustion,
YAML-driven 504s, terminal SSE errors, healthy streams beyond the read bound,
duplicate request IDs with client disconnect, and counter cleanup. Watcher
checks covered unchanged files, typo rejection, priority reload, and inert
startup-only backend timeout edits. The image runs Python 3.13 and includes no
PyTorch, vLLM, or sentence-transformers. Standard pre-commit hooks passed.
ARM64 image execution was not tested. No image was published by this sync;
the September release provenance above remains the last verified publication.

## Patch-series refresh — 2026-10-03

The user subsequently authorized returning the final state to main and clarified
that this fork periodically rebases its patches to keep history clean. The
reconciled tree `01d7da4` was rebuilt as seven topic commits directly on upstream
`014d070`, retaining original commit references and author credits in the new
commit bodies. Before updating maintenance guidance, the rebuilt tree was
byte-for-byte identical to `01d7da4` (tree
`c586c29e75497067dab94c26d8c5428ff52d751d`). No router, test, dependency, chart,
operator, or release-workflow behavior changed during this history cleanup.

The original main `6970618` is retained in
`backup/main-before-upstream-2026-10-03`; the tested reconciliation remains on
`reconcile/upstream-2026-10-03`. These refs preserve the history referenced by
earlier verification and release records.
