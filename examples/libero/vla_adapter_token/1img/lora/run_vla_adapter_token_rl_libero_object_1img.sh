#!/bin/bash
set -euo pipefail
set -x

# 1-image launcher for LIBERO-Object VLA-Adapter token RL/eval.
# Defaults to the 1img libero_object checkpoint; pass extra Hydra overrides as "$@".

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

SFT_MODEL_PATH="${SFT_MODEL_PATH:-${REPO_ROOT}/models/token-1img/configs+libero_object_no_noops+b64+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--token--1img--libero_object_no_noops--2025-12-25_00-20-20--2500_chkpt}"
export SFT_MODEL_PATH

DATASET_NAME="${DATASET_NAME:-libero_object}"
export DATASET_NAME

EXPERIMENT_NAME="${EXPERIMENT_NAME:-libero_object_vla_adapter_token_rl_1img}"
export EXPERIMENT_NAME

if [ ! -d "$SFT_MODEL_PATH" ]; then
    echo "ERROR: checkpoint not found: $SFT_MODEL_PATH" >&2
    exit 2
fi

bash "${REPO_ROOT}/examples/libero/vla_adapter_token/1img/run_vla_adapter_token_rl_libero_1img.sh" "$@"

