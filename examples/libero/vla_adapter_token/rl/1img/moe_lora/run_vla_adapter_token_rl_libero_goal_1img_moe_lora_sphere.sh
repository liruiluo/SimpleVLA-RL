#!/bin/bash
set -euo pipefail
set -x

# 1-image launcher for LIBERO-Goal VLA-Adapter token RL using MoE-LoRA + SPHERE.
#
# This wraps:
#   `examples/libero/vla_adapter_token/rl/1img/moe_lora/run_vla_adapter_token_rl_libero_1img_moe_lora_sphere.sh`
#
# Usage:
#   bash examples/libero/vla_adapter_token/rl/1img/moe_lora/run_vla_adapter_token_rl_libero_goal_1img_moe_lora_sphere.sh

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

SFT_MODEL_PATH="${SFT_MODEL_PATH:-${REPO_ROOT}/models/token-1img/configs+libero_goal_no_noops+b64+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--token--1img--libero_goal_no_noops--2025-12-29_13-38-52--25000_chkpt}"
export SFT_MODEL_PATH

DATASET_NAME="${DATASET_NAME:-libero_goal}"
export DATASET_NAME

EXPERIMENT_NAME="${EXPERIMENT_NAME:-libero_goal_vla_adapter_token_moe_lora_sphere_rl_1img}"
export EXPERIMENT_NAME

bash "${REPO_ROOT}/examples/libero/vla_adapter_token/rl/1img/moe_lora/run_vla_adapter_token_rl_libero_1img_moe_lora_sphere.sh" \
  "$@"

