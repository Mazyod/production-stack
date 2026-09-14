# Working in this fork

This is `Mazyod/production-stack`, a small router patch series over
`vllm-project/production-stack`. Keep changes focused and reuse upstream behavior
unless there is a demonstrated reason to diverge. Engine patches and the audio
image belong in `Mazyod/vllm`.

## Context and ownership

- [README](README.md#about-this-fork): operator-facing fork patch index and flags.
- [Fork maintenance](docs/fork-maintenance.md): patch/release workflow, decisions,
  verification, and known limitations. Read it before changing those areas.
- `src/vllm_router/app.py`: startup, shared components, and server configuration.
- `src/vllm_router/routers/main_router.py`: HTTP routes and model discovery.
- `src/vllm_router/services/request_service/request.py`: backend requests,
  streaming, failover, and multipart handling.
- `src/vllm_router/stats/request_stats.py`: per-attempt lifecycle accounting.
- `src/vllm_router/parsers/` and `dynamic_config.py`: startup and reload paths.
- `src/tests/`: router regression tests; `docker/` and `.github/workflows/`:
  image build and release. Helm, operator, and tutorials mostly follow upstream.

## Working agreements

- Trace callers and related request paths before fixing a bug. Preserve input
  validation, cancellation cleanup, and client-visible error contracts.
- Keep patches self-contained by topic. Use subjects such as
  `[Router] <topic>: <change>` or `[Docs]`, `[CI]`, `[Docker]`, `[Build]`, `[Chore]`.
  Explain why in the body and sign off commits as required by CONTRIBUTING.md.
- Normally append fixes to published history. The July 2026 structured-output
  patch removal was a specifically authorized exception, not a rewrite policy.
- Update the README patch index and operator instructions for observable changes.
  Cover CLI and YAML/JSON startup configuration when changing flags; distinguish
  startup-only settings from hot reload.
- Record durable decisions and superseded behavior in the maintenance document
  with evidence. Keep personal settings and temporary task reports out of it.
- Verify consequential fixes at the HTTP surface and seek independent review
  when warranted and available. Report what actually ran and any limits;
  historical test counts are not current verification.

## Checks and release awareness

Setup: `uv venv --python 3.13`, then
`uv pip install -e . --group test --group lint`.
Use dependency groups, not the heavyweight `test` extra or `--all-extras`.

- Tests: `.venv/bin/python -m pytest src/tests -q`.
- Lint: `.venv/bin/pre-commit run --all-files` (hooks may format files).
- HTTP verification: [verify skill](.agents/skills/verify/SKILL.md).
- GitHub CLI: use `-R Mazyod/production-stack` for repo-scoped commands;
  `gh api` needs an explicit `repos/Mazyod/production-stack/...` endpoint.
- A qualifying push to `main` publishes router images. Read the maintenance
  release procedure before pushing or dispatching a workflow; historical release
  requests do not authorize a new release.
