#!/bin/bash
set -euo pipefail
set -x

# 4xA100 single-node launcher for VLA-Adapter token RL (LoRA by default).
# This is a thin wrapper around `examples/run_vla_adapter_token_rl_libero_lora.sh`.

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

# ---- GPU layout ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NUM_GPUS="${NUM_GPUS:-4}"
export NUM_NODES="${NUM_NODES:-1}"

# ---- Headless rendering (LIBERO / robosuite) ----
# If EGL is not available on your Lambda image, try: `MUJOCO_GL=osmesa` (slower but uses no GPU for rendering).
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# ---- Throughput knobs (tune for your CPU/RAM) ----
# Each rollout chunk spawns roughly: ROLLOUT_MICRO_BATCH_SIZE * data.n_samples env workers.
export ROLLOUT_MICRO_BATCH_SIZE="${ROLLOUT_MICRO_BATCH_SIZE:-4}"
export VAL_ROLLOUT_MICRO_BATCH_SIZE="${VAL_ROLLOUT_MICRO_BATCH_SIZE:-2}"
export NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-2}"
export USE_MINIVLM="${USE_MINIVLM:-True}"

# Portable relative-path defaults (override as needed):
#   SFT_MODEL_PATH=models/<your_checkpoint_dir>
#   VLA_ADAPTER_REPO_PATH=../VLA-Adapter

# Log-prob micro-batches are divided by world_size inside workers; keep >= NUM_GPUS.
export LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-64}"
export REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-64}"

# Token-action prediction microbatching (safe default).

# Speed-focused defaults (override as you like).
export TEST_FREQ="${TEST_FREQ:--1}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-1}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"

# Ray memory knobs (avoid OS OOM-killer incidents).
export VERL_RAY_DISABLE_DASHBOARD="${VERL_RAY_DISABLE_DASHBOARD:-1}"
# Optional: set this only if you know your node has enough RAM.
# export VERL_RAY_OBJECT_STORE_MEMORY_GB=20

# Prefer not to offload on A100 for speed; override the base script's defaults via trailing Hydra args.
bash "${REPO_ROOT}/examples/run_vla_adapter_token_rl_libero_lora.sh" \
  actor_rollout_ref.actor.fsdp_config.grad_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  "$@"
