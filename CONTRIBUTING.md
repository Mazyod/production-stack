# Contributing

Thank you for your contribution to production-stack! As a potential contributor, your changes and ideas are welcome at any hour of the day or night, weekdays, weekends, and holidays. Please do not ever hesitate to ask a question or send a pull request.

For this fork, read [AGENTS.md](AGENTS.md) and the [fork maintenance guide](docs/fork-maintenance.md) for patch conventions, ownership, and releases. The proposal and upstream review guidance below applies when contributing back to `vllm-project/production-stack`.

## Submitting a Proposal

For **major changes, new features, or significant architectural modifications**, please **submit a proposal** under [proposals/](proposals/) folder using the [designated template](proposals/TEMPLATE.md) before contributing code. This ensures alignment with the project's goals, allows maintainers and contributors to provide feedback early, and helps prevent unnecessary rework.

Once submitted, your proposal will be reviewed by the maintainers, and discussions may take place before approval. We encourage open collaboration, so feel free to participate in the discussion and refine your proposal based on feedback.

*For small changes like bug fixes, documentation updates, minor optimizations, and simple features, feel free to directly create an issue or PR without the proposal.*

## Opening a Pull Request

Before submitting your pull request, please ensure it meets the following criteria. This helps maintain code quality and streamline the review process.

Follow the standard GitHub workflow:

1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with a detailed description.

<h2>PR Title and Classification</h2>
<p>Please try to classify PRs for easy understanding of the type of changes. The PR title is prefixed appropriately to indicate the type of change. Please use one of the following:</p>
<ul>
    <li><code>[Bugfix]</code> for bug fixes.</li>
    <li><code>[CI/Build]</code> for build or continuous integration improvements.</li>
    <li><code>[Doc]</code> for documentation fixes and improvements.</li>
    <li><code>[Feat]</code> for new features in the cluster (e.g., autoscaling, disaggregated prefill, etc.).</li>
    <li><code>[Router]</code> for changes to the <code>vllm_router</code> (e.g., routing algorithm, router observability, etc.).</li>
    <li><code>[Misc]</code> for PRs that do not fit the above categories. Please use this sparingly.</li>
</ul>
<p><strong>Note:</strong> If the PR spans more than one category, please include all relevant prefixes.</p>

## Code Quality and Validation

### Linter Checks

Install the router with the test and lint dependency groups, using Python 3.13 to match the image:

```bash
uv venv --python 3.13
uv pip install -e . --group test --group lint
.venv/bin/pre-commit install
```

The `test` dependency group runs the ordinary router suite without vLLM or a GPU. The optional `test` extra and `--all-extras` install heavyweight engine/cache dependencies; use them only for work that needs those features.

This uses the image's installation approach: resolve from `pyproject.toml`. The current lockfile still selects NumPy 1.26.4, which has no Python 3.13 Linux wheel, so a locally cached installation is not evidence that `uv sync --locked` will work on a fresh runner.

It will run automatically before every commit. You can also run it manually on
all files with:

```bash
.venv/bin/pre-commit run --all-files
```

There are a subset of hooks which require additional dependencies that you may
not have installed in your development environment (i.e. Docker and non-Python
packages). These are configured to only run in the `manual` `pre-commit` stage.
The fork does not run these manual hooks in CI. Locally they can be run with:

```bash
# Runs all hooks including manual stage hooks
.venv/bin/pre-commit run --all-files --hook-stage manual
# Runs only the manual stage hook shellcheck
.venv/bin/pre-commit run --all-files --hook-stage manual shellcheck
```

If any of these hooks are failing in CI but you cannot run them locally, you
can identify what needs changing by examining the GitHub Actions logs in your
pull request.

> You can read more about `pre-commit` at [https://pre-commit.com](https://pre-commit.com).

### Router Tests and Release Checks

Run the regression suite:

```bash
.venv/bin/python -m pytest src/tests -q
```

For request/configuration changes, also follow the [HTTP verification skill](.agents/skills/verify/SKILL.md) to exercise the live router against a fake engine. Report what ran and any gaps; a unit test or `/health` response alone does not verify inference.

The fork's `build-router.yml` runs the router suite on the upstream release tag plus replayed fork patches before publishing images. HTTP 200 from the published image's `/health` is required before promoting `latest`. The upstream PR, Helm, operator, and lint workflows were removed from this fork; run the relevant local checks before submitting changes. See the [release procedure](docs/fork-maintenance.md#patch-and-release-workflow) for triggers and tag behavior.

## Adding Examples and Tests

Please include sufficient examples in your PR. Unit tests and integrations are also welcome, and you’re encouraged to contribute them in future PRs.

<h2>DCO and Signed-off-by</h2>
<p>When contributing changes to this project, you must agree to the <a href="https://github.com/vllm-project/vllm/blob/main/DCO">DCO</a>. Commits must include a <code>Signed-off-by:</code> header which certifies agreement with the terms of the DCO.</p>
<p>Using <code>-s</code> with <code>git commit</code> will automatically add this header.</p>

<h2>What to Expect for the Reviews</h2>

We aim to address all PRs in a timely manner. If no one reviews your PR within 5 days, please @-mention one of ruizhang0101, YuhanLiu11, Shaoting-Feng or ApostaC.
