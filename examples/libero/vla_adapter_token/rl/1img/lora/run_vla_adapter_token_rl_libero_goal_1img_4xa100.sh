#!/bin/bash
set -euo pipefail
set -x

# 4xGPU single-node launcher for VLA-Adapter token RL on LIBERO (libero_goal, 1-image checkpoint).
# Thin wrapper around `examples/libero/vla_adapter_token/rl/1img/lora/run_vla_adapter_token_rl_libero_goal_1img.sh`.
#
# Usage:
#   bash examples/libero/vla_adapter_token/rl/1img/lora/run_vla_adapter_token_rl_libero_goal_1img_4xa100.sh
#   # optional: pass extra Hydra overrides at the end
#   bash examples/libero/vla_adapter_token/rl/1img/lora/run_vla_adapter_token_rl_libero_goal_1img_4xa100.sh trainer.total_steps=230

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

# Default to the included goal 1-image checkpoint (override via env var if needed).
SFT_MODEL_PATH="${SFT_MODEL_PATH:-${REPO_ROOT}/models/token-1img/configs+libero_goal_no_noops+b64+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--token--1img--libero_goal_no_noops--2025-12-29_13-38-52--25000_chkpt}"
export SFT_MODEL_PATH

# ---- GPU layout ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NUM_GPUS="${NUM_GPUS:-4}"
export NUM_NODES="${NUM_NODES:-1}"

# ---- Headless rendering (LIBERO / robosuite) ----
# If EGL is not available, try: `MUJOCO_GL=osmesa` (slower).
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Ray memory knobs (avoid extra background processes).
export VERL_RAY_DISABLE_DASHBOARD="${VERL_RAY_DISABLE_DASHBOARD:-1}"

bash "${REPO_ROOT}/examples/libero/vla_adapter_token/rl/1img/lora/run_vla_adapter_token_rl_libero_goal_1img.sh" \
  data.train_batch_size=32 \
  data.val_batch_size=64 \
  actor_rollout_ref.rollout.micro_batch_size=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
  actor_rollout_ref.actor.fsdp_config.grad_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  "$@"
