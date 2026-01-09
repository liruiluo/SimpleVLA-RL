#!/bin/bash
set -euo pipefail
set -x

# 1-image launcher for LIBERO VLA-Adapter token RL using MoE-LoRA + SPHERE regularization.
#
# This enables:
# - MoE-LoRA: `actor_rollout_ref.model.use_moe_lora=True` (fresh adapter; no checkpoint loading)
# - SPHERE: paper-aligned weighted expert-feature Gram regularizer on MoE-LoRA (LoRA-A features)
#
# Usage examples:
#   # Default: libero_object 1img shipped checkpoint
#   bash examples/libero/vla_adapter_token/rl/1img/moe_lora/run_vla_adapter_token_rl_libero_1img_moe_lora_sphere.sh
#
#   # Run on another suite/checkpoint
#   DATASET_NAME=libero_goal \
#   SFT_MODEL_PATH=/abs/path/to/goal_1img_ckpt \
#   bash examples/libero/vla_adapter_token/rl/1img/moe_lora/run_vla_adapter_token_rl_libero_1img_moe_lora_sphere.sh
#
#   # Tune MoE/SPHERE
#   MOE_NUM_EXPERTS=4 MOE_TOP_K=2 LORA_RANK=32 \
#   SPHERE_MODE=grad_norm SPHERE_RHO=0.1 SPHERE_ENABLE_COEF=1.0 \
#   bash examples/libero/vla_adapter_token/rl/1img/moe_lora/run_vla_adapter_token_rl_libero_1img_moe_lora_sphere.sh

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

# Default to the shipped 1-image LIBERO-Object token-VLA checkpoint.
SFT_MODEL_PATH="${SFT_MODEL_PATH:-${REPO_ROOT}/models/token-1img/configs+libero_object_no_noops+b64+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--token--1img--libero_object_no_noops--2025-12-25_00-20-20--2500_chkpt}"
export SFT_MODEL_PATH

DATASET_NAME="${DATASET_NAME:-libero_object}"
export DATASET_NAME

EXPERIMENT_NAME="${EXPERIMENT_NAME:-${DATASET_NAME}_vla_adapter_token_moe_lora_sphere_rl_1img}"
export EXPERIMENT_NAME

# MoE-LoRA knobs.
MOE_NUM_EXPERTS="${MOE_NUM_EXPERTS:-3}"
MOE_TOP_K="${MOE_TOP_K:-2}"
LORA_RANK="${LORA_RANK:-64}"

# SPHERE knobs (paper-aligned):
# - SPHERE_ENABLE_COEF must be > 0 to enable computation (acts as an on/off gate).
# - `grad_norm` mode is more paper-like but can be memory-heavy (it needs extra autograd.grad calls).
SPHERE_ENABLE_COEF="${SPHERE_ENABLE_COEF:-0.05}"
SPHERE_MODE="${SPHERE_MODE:-fixed}"  # fixed | loss_ratio | grad_norm
SPHERE_RHO="${SPHERE_RHO:-0.1}"
SPHERE_TARGET_RATIO="${SPHERE_TARGET_RATIO:-0.0}"
SPHERE_TEMPERATURE="${SPHERE_TEMPERATURE:-1.0}"

# Ensure `prismatic.util.moe_lora` resolves to VLA-Adapter (not OpenVLA-OFT).
VLA_ADAPTER_REPO_PATH="${VLA_ADAPTER_REPO_PATH:-${REPO_ROOT}/../VLA-Adapter}"
export VLA_ADAPTER_REPO_PATH
export PYTHONPATH="${VLA_ADAPTER_REPO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

bash "${REPO_ROOT}/examples/libero/vla_adapter_token/rl/1img/run_vla_adapter_token_rl_libero_1img.sh" \
  actor_rollout_ref.model.vla=vla-adapter-token \
  actor_rollout_ref.model.vla_adapter_repo_path="${VLA_ADAPTER_REPO_PATH}" \
  actor_rollout_ref.model.use_moe_lora=True \
  actor_rollout_ref.model.moe_num_experts="${MOE_NUM_EXPERTS}" \
  actor_rollout_ref.model.moe_top_k="${MOE_TOP_K}" \
  actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
  actor_rollout_ref.model.lora_load_from_checkpoint=False \
  actor_rollout_ref.actor.sphere_coef="${SPHERE_ENABLE_COEF}" \
  actor_rollout_ref.actor.sphere_temperature="${SPHERE_TEMPERATURE}" \
  actor_rollout_ref.actor.sphere_mode="${SPHERE_MODE}" \
  actor_rollout_ref.actor.sphere_rho="${SPHERE_RHO}" \
  actor_rollout_ref.actor.sphere_target_ratio="${SPHERE_TARGET_RATIO}" \
  "$@"
