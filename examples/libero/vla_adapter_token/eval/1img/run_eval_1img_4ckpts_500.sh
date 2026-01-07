#!/bin/bash
set -euo pipefail
set -x

# Run "untrained" (val-only) evaluation for the 4 shipped 1-image token-VLA checkpoints:
#   - libero_spatial
#   - libero_object
#   - libero_goal
#   - libero_long (defaults to libero_10; override via DATASET_NAME=libero_90)
#
# Default: ~500 rollouts per suite by setting:
#   num_trials_per_task=50 and evaluating 10 tasks (common LIBERO suites) => 500 episodes.
#
# Tunables (env):
#   VAL_BATCH_SIZE=50
#   MAX_VAL_BATCHES=10              # VAL_BATCH_SIZE * MAX_VAL_BATCHES rollouts
#   NUM_TRIALS_PER_TASK=50
#   DATASET_NAME=libero_90          # only affects the "long" launcher
#
# Usage:
#   bash examples/libero/vla_adapter_token/eval/1img/run_eval_1img_4ckpts_500.sh

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

VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-50}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-10}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"

common_args=(
  trainer.val_before_train=True
  trainer.val_only=True
  trainer.max_val_batches="${MAX_VAL_BATCHES}"
  data.val_batch_size="${VAL_BATCH_SIZE}"
  data.num_trials_per_task="${NUM_TRIALS_PER_TASK}"
  # Training settings are irrelevant in val-only mode but must be valid.
  data.train_batch_size=1
  data.n_samples=1
  trainer.save_freq=-1
  trainer.test_freq=-1
  trainer.logger="['console']"
  trainer.wandb_mode=disabled
)

echo "Eval config: VAL_BATCH_SIZE=${VAL_BATCH_SIZE} MAX_VAL_BATCHES=${MAX_VAL_BATCHES} NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}" >&2
echo "Expected rollouts per suite ~= VAL_BATCH_SIZE*MAX_VAL_BATCHES = $((VAL_BATCH_SIZE * MAX_VAL_BATCHES))" >&2

run_one() {
  local name="$1"
  local cmd="$2"
  shift 2
  local log_dir="${LOG_DIR:-${REPO_ROOT}/runs/eval_1img_untrained}"
  mkdir -p "${log_dir}"
  local log_file="${log_dir}/${name}.log"
  echo "=== ${name} ===" >&2
  # shellcheck disable=SC2086
  bash "$cmd" "${common_args[@]}" "$@" 2>&1 | tee "${log_file}"
  # Best-effort summary line (RayTrainer prints this when val_before_train=True and val_only=True)
  grep -E "Initial validation metrics:|val/test_score/all|test_score/all" "${log_file}" | tail -n 3 || true
}

run_one "libero_spatial" \
  "${REPO_ROOT}/examples/libero/vla_adapter_token/rl/1img/lora/run_vla_adapter_token_rl_libero_spatial_1img.sh"

run_one "libero_object" \
  "${REPO_ROOT}/examples/libero/vla_adapter_token/rl/1img/lora/run_vla_adapter_token_rl_libero_object_1img.sh"

run_one "libero_goal" \
  "${REPO_ROOT}/examples/libero/vla_adapter_token/rl/1img/lora/run_vla_adapter_token_rl_libero_goal_1img.sh"

# "Long" defaults to libero_10; allow overriding the suite via env var if desired.
long_args=()
if [ -n "${DATASET_NAME:-}" ]; then
  long_args+=(data.task_suite_name="${DATASET_NAME}" actor_rollout_ref.rollout.task_suite_name="${DATASET_NAME}")
fi
run_one "libero_long" \
  "${REPO_ROOT}/examples/libero/vla_adapter_token/rl/1img/lora/run_vla_adapter_token_rl_libero_long_1img.sh" \
  "${long_args[@]}"
