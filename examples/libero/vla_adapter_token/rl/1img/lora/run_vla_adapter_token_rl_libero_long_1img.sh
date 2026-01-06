#!/bin/bash
set -euo pipefail
set -x

# 1-image launcher for LIBERO-Long (libero_10/libero_90) VLA-Adapter token RL/eval.
# Override the checkpoint via: `SFT_MODEL_PATH=/abs/path/to/long_1img_ckpt bash $0 ...`
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
        _script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        _probe="$_script_dir"
        while [ "$_probe" != "/" ]; do
            if [ -d "$_probe/verl" ] && [ -d "$_probe/examples" ]; then
                REPO_ROOT="$_probe"
                break
            fi
            _probe="$(dirname "$_probe")"
        done
        if [ -z "${REPO_ROOT:-}" ]; then
            echo "ERROR: cannot find repo root (missing verl/ and examples/)." >&2
            exit 2
        fi
    fi
fi

# Default to the included long(=libero_10) 1-image checkpoint (override via env var if needed).
SFT_MODEL_PATH="${SFT_MODEL_PATH:-${REPO_ROOT}/models/token-1img/configs+libero_10_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--token--long-4GPU--1img--libero_10_no_noops--2026-01-02_00-10-30--30000_chkpt}"
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

bash "${REPO_ROOT}/examples/libero/vla_adapter_token/rl/1img/run_vla_adapter_token_rl_libero_1img.sh" "$@"
