#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
#
# Run utils/conv-perf-test.py at each commit in a git range, then compare.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: conv-perf-range.sh <git-range> [-- <perf-test-args>...]

Run utils/conv-perf-test.py at each commit in <git-range> and print a
comparison table. Results are kept in a mktemp dir for later re-compare.

Note: <git-range> follows git's A..B semantics (B included, A excluded).
Use A^..B to include A as the baseline commit.

Examples:
  conv-perf-range.sh HEAD~5..HEAD -- -f NV12,YUYV -t 2
  conv-perf-range.sh origin/main..HEAD -- --width 3840 --height 2160 \
      -f NV12,XRGB8888 --backends numpy
EOF
}

if [[ $# -eq 0 || "${1:-}" == '-h' || "${1:-}" == '--help' ]]; then
    usage
    exit 0
fi

RANGE="$1"
shift

# Optional '--' separator; everything after is forwarded to conv-perf-test.py.
if [[ $# -gt 0 && "$1" == '--' ]]; then
    shift
fi
PERF_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERF_TEST="$SCRIPT_DIR/conv-perf-test.py"
PERF_COMPARE="$SCRIPT_DIR/conv-perf-compare.py"

for f in "$PERF_TEST" "$PERF_COMPARE"; do
    if [[ ! -f "$f" ]]; then
        echo "error: missing $f" >&2
        exit 1
    fi
done

if [[ -n "$(git status --porcelain)" ]]; then
    echo 'error: working tree is dirty; commit or stash first' >&2
    exit 1
fi

# Record starting ref so we can restore it in the EXIT trap.
if START_REF="$(git symbolic-ref --short -q HEAD)"; then
    :
else
    START_REF="$(git rev-parse HEAD)"
fi

restore() {
    if ! git checkout --quiet "$START_REF" 2>/dev/null; then
        echo "warning: could not restore starting ref '$START_REF'" >&2
        echo "         you may need to run: git checkout $START_REF" >&2
    fi
}
trap restore EXIT

mapfile -t COMMITS < <(git rev-list --reverse "$RANGE")
if [[ ${#COMMITS[@]} -eq 0 ]]; then
    echo "error: no commits in range '$RANGE'" >&2
    exit 1
fi

OUTDIR="$(mktemp -d -t conv-perf-XXXXXX)"
echo "Results dir: $OUTDIR"
echo "Commits to test (${#COMMITS[@]}): ${COMMITS[*]}"
echo

i=0
for sha in "${COMMITS[@]}"; do
    i=$((i + 1))
    short="$(git rev-parse --short "$sha")"
    json_path="$OUTDIR/$(printf '%03d' "$i")-${short}.json"
    echo "=== [$i/${#COMMITS[@]}] $short ==="
    git checkout --quiet "$sha"
    python3 "$PERF_TEST" "${PERF_ARGS[@]}" -o "$json_path"
    echo
done

# Trap restores HEAD; run the compare after restore so imports resolve against
# the starting commit's source.
restore
trap - EXIT

echo '=== Comparison ==='
python3 "$PERF_COMPARE" --timeline "$OUTDIR"/*.json

echo
echo "Results kept in: $OUTDIR"
