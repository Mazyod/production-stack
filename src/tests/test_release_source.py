import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / ".github/scripts/prepare-router-release.sh"
)


@pytest.mark.parametrize(
    "target", ["release", "future", "content-conflict", "missing-file"]
)
def test_release_preserves_reconciliation_and_rejects_conflicts(tmp_path, target):
    def git(*args, check=True):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=check
        )

    def commit(message):
        git("add", "-A")
        git("commit", "-qm", message)
        return git("rev-parse", "HEAD").stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Release test")
    git("config", "user.email", "release-test@example.invalid")
    git("config", "commit.gpgsign", "false")
    git("config", "core.hooksPath", "/dev/null")
    config = tmp_path / "router.txt"
    workflow = tmp_path / "upstream workflow.yml"
    config.write_text("base\n")
    workflow.write_text("upstream workflow\n")
    base = commit("base")

    git("switch", "-c", "fork")
    config.write_text("original fork patch\n")
    workflow.unlink()
    commit("fork patch")

    git("switch", "-c", "upstream", base)
    config.write_text("upstream improvement\n")
    commit("new upstream release")
    git("tag", "vllm-stack-test")
    (tmp_path / "unreleased.txt").write_text("main only\n")
    upstream = commit("upstream after release")
    git("update-ref", "refs/remotes/upstream/main", upstream)

    git("switch", "fork")
    assert git("merge", "--no-commit", "upstream", check=False).returncode == 1
    config.write_text("reconciled fork and upstream\n")
    fork = commit("preserve both contracts in merge resolution")

    git("checkout", "--detach", "vllm-stack-test")
    if target == "future":
        workflow.write_text("changed upstream workflow\n")
        (tmp_path / "new-upstream.txt").write_text("future improvement\n")
    elif target == "content-conflict":
        config.write_text("incompatible new upstream change\n")
    elif target == "missing-file":
        config.unlink()
    if target != "release":
        commit("target changes")
    git("tag", "target")
    git("checkout", "--detach", fork)

    result = subprocess.run(
        ["bash", str(SCRIPT), "target"], cwd=tmp_path, capture_output=True, text=True
    )
    if target in {"content-conflict", "missing-file"}:
        assert result.returncode != 0, result.stdout + result.stderr
        assert git("rev-parse", "HEAD").stdout == git("rev-parse", "target").stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert config.read_text() == "reconciled fork and upstream\n"
        assert not workflow.exists()
        assert not (tmp_path / "unreleased.txt").exists()
        if target == "future":
            assert (tmp_path / "new-upstream.txt").read_text() == "future improvement\n"
        assert git("status", "--porcelain").stdout == ""
