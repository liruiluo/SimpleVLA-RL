#!/bin/bash
set -euo pipefail
set -x

# 1-image launcher for LIBERO-Long VLA-Adapter token RL/eval.
# Placeholder: set `SFT_MODEL_PATH=/path/to/long_1img_ckpt` before running.
#
# Note: VLA-Adapter uses "LIBERO-Long" naming; in SimpleVLA-RL the task suites are typically `libero_10`/`libero_90`.
# If needed, override `DATASET_NAME` at runtime.

# Determine repo root.
if [ -z "${REPO_ROOT:-}" ]; then
    if [ -d "$PWD/verl" ] && [ -d "$PWD/examples" ]; then
        REPO_ROOT="$PWD"
    elif [ -d "$PWD/../verl" ] && [ -d "$PWD/../examples" ]; then
        REPO_ROOT="$(cd "$PWD/.." && pwd)"
    else
        REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    fi
fi

# TODO: replace with the actual long 1img ckpt path once available.
SFT_MODEL_PATH="${SFT_MODEL_PATH:-${REPO_ROOT}/models/token-1img/TODO-long-1img-ckpt}"
export SFT_MODEL_PATH

DATASET_NAME="${DATASET_NAME:-libero_10}"
export DATASET_NAME

EXPERIMENT_NAME="${EXPERIMENT_NAME:-libero_long_vla_adapter_token_rl_1img}"
export EXPERIMENT_NAME

if [ ! -d "$SFT_MODEL_PATH" ]; then
    echo "ERROR: long 1img checkpoint not found: $SFT_MODEL_PATH" >&2
    echo "Set it via: SFT_MODEL_PATH=/abs/path/to/long_1img_ckpt bash $0 ..." >&2
    echo "Optionally override suite: DATASET_NAME=libero_90 bash $0 ..." >&2
    exit 2
fi

bash "${REPO_ROOT}/examples/run_vla_adapter_token_rl_libero_1img.sh" "$@"

