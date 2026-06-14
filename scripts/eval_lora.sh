#!/usr/bin/env bash
# Wrapper around experiments/eval_lora.py.
#
# Usage:
#   experiments/eval_lora.sh <test_set> <lora_path> [extra args...]
#
# Examples:
#   experiments/eval_lora.sh math500  runs/math_caps
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <test_set> <lora_path> [extra args...]" >&2
    exit 1
fi

TEST_SET="$1"
LORA_PATH="$2"
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source .venv/bin/activate

python experiments/eval_lora.py \
    --test_set "$TEST_SET" \
    --lora_path "$LORA_PATH" \
    "$@"
