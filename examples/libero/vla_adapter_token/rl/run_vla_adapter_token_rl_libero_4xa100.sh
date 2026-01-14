#!/bin/bash
set -euo pipefail
set -x

# Enable NaN/Inf debug checks in rollout (set VERL_DEBUG_NAN=0 to disable).
export VERL_DEBUG_NAN="${VERL_DEBUG_NAN:-1}"

# 4xA100 single-node launcher for VLA-Adapter token RL (LoRA by default).
# This is a thin wrapper around `examples/libero/vla_adapter_token/rl/run_vla_adapter_token_rl_libero_lora.sh`.

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

# ---- GPU layout ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NUM_GPUS="${NUM_GPUS:-4}"
export NUM_NODES="${NUM_NODES:-1}"

# ---- Headless rendering (LIBERO / robosuite) ----
# If EGL is not available on your Lambda image, try: `MUJOCO_GL=osmesa` (slower but uses no GPU for rendering).
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Ray memory knobs (avoid OS OOM-killer incidents).
export VERL_RAY_DISABLE_DASHBOARD="${VERL_RAY_DISABLE_DASHBOARD:-1}"
# Optional: set this only if you know your node has enough RAM.
# export VERL_RAY_OBJECT_STORE_MEMORY_GB=20

# Avoid CPU oversubscription: LIBERO rollouts spawn many subprocesses, and BLAS/OpenMP defaults
# can create dozens of threads per process (slow + high host RAM).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

# Prefer not to offload on A100 for speed; override the base script's defaults via trailing Hydra args.
bash "${REPO_ROOT}/examples/libero/vla_adapter_token/rl/run_vla_adapter_token_rl_libero_lora.sh" \
  data.train_batch_size=32 \
  data.val_batch_size=32 \
  data.n_samples=4 \
  actor_rollout_ref.rollout.micro_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size=32 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
  actor_rollout_ref.actor.fsdp_config.grad_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  trainer.ckpt_layout=rl \
  "$@"
