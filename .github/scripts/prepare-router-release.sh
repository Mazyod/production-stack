#!/usr/bin/env bash
# Run in a disposable, clean fork checkout; leaves the release source checked out.
set -euo pipefail

UPSTREAM_TAG=${1:?Usage: prepare-router-release.sh UPSTREAM_TAG}
FORK_SHA=$(git rev-parse HEAD)
UPSTREAM_BASE=$(git merge-base "$FORK_SHA" upstream/main)
TARGET_SHA=$(git rev-parse --verify --end-of-options "${UPSTREAM_TAG}^{commit}")
git diff --exit-code
git diff --cached --exit-code

PATCH_FILE=$(mktemp)
trap 'rm -f "$PATCH_FILE"' EXIT

# Apply the reconciled fork delta, including merge resolutions. Replaying old
# commits would reintroduce superseded behavior and conflicts resolved in merges.
git diff --binary --full-index --no-renames --diff-filter=d \
    "$UPSTREAM_BASE" "$FORK_SHA" > "$PATCH_FILE"
git checkout --detach "$TARGET_SHA"

# Fork deletions win even if the release modified those files. git apply cannot
# three-way merge deletions, so handle them separately from the content patch.
while IFS= read -r -d '' path; do
    git rm --ignore-unmatch -- "$path"
done < <(git diff --no-renames --name-only --diff-filter=D -z "$UPSTREAM_BASE" "$FORK_SHA")

if [[ -s "$PATCH_FILE" ]]; then
    # Every nonzero exit is fatal, including failures without unmerged paths.
    git apply --3way --index "$PATCH_FILE"
fi

git commit --allow-empty -m "[Build] apply reconciled fork to $UPSTREAM_TAG" \
    -m "Fork source: $FORK_SHA
Upstream base: $UPSTREAM_BASE
Upstream target: $TARGET_SHA"
echo "Source ready: $UPSTREAM_TAG + fork delta from $FORK_SHA (base $UPSTREAM_BASE)"
echo "Release tree: $(git rev-parse 'HEAD^{tree}')"
