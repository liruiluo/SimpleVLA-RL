#!/bin/bash
set -x

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

# ---- Environment setup (Python / Conda) ----
# Prefer the repo-local venv if it exists; this avoids relying on interactive shell init (conda).
if [ -f "${REPO_ROOT}/env/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/env/bin/activate"
fi

# Prefer repo-local interpreter even if activation fails (e.g. conda not available in non-interactive shells).
if [ -z "${PYTHON:-}" ]; then
    if [ -x "${REPO_ROOT}/env/bin/python" ]; then
        PYTHON="${REPO_ROOT}/env/bin/python"
    else
        PYTHON="python"
    fi
fi

export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export ROBOT_PLATFORM=LIBERO
export VERL_QUIET_VIDEO_LOG="${VERL_QUIET_VIDEO_LOG:-1}"
export VERL_SAVE_ROLLOUT_VIDEOS="${VERL_SAVE_ROLLOUT_VIDEOS:-0}"

# Ray can spawn many background processes and reserve a large object store by default.
# These knobs avoid OS OOM-killer incidents (which can kill VSCode/Electron even if training continues).
export VERL_RAY_DISABLE_DASHBOARD="${VERL_RAY_DISABLE_DASHBOARD:-1}"

# Slurm often allocates fewer CPUs than the physical node (and nodes may report hyperthreads).
# Let Ray see the allocated CPUs to avoid spawning hundreds of idle workers which can overload raylet/gcs.
export VERL_RAY_NUM_CPUS="${VERL_RAY_NUM_CPUS:-${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-128}}}"

# Reduce Ray control-plane load by disabling task-event reporting to GCS (helps avoid heartbeat timeouts).
export RAY_task_events_report_interval_ms="${RAY_task_events_report_interval_ms:-0}"

# Raise the file-descriptor limit if possible (Ray can open many fds for sockets/processes).
ulimit -n 1048576 2>/dev/null || ulimit -n 262144 2>/dev/null || ulimit -n 65536 2>/dev/null || true
echo "VERL_RAY_NUM_CPUS=${VERL_RAY_NUM_CPUS} RAY_task_events_report_interval_ms=${RAY_task_events_report_interval_ms} ulimit_nofile=$(ulimit -n)" >&2

# Persist Ray session + logs on the shared filesystem so Slurm jobs can be debugged from the head node.
# Ray uses AF_UNIX sockets for plasma store; the socket path length must be <= 107 bytes.
# The repo path on shared FS is long, so we use a short `/tmp/...` symlink that points to the shared target dir.
#
# IMPORTANT: we intentionally override any pre-set `RAY_TMPDIR` because reusing the same Ray temp dir across jobs
# can lead to stale Ray state and placement group name collisions (e.g. `global_poolverl_group_4:0 already exists`).
if [ -n "${SLURM_JOB_ID:-}" ]; then
    _ray_tmp_target="${REPO_ROOT}/logs/ray/${SLURM_JOB_ID}/${HOSTNAME}"
    _ray_tmp_link="/tmp/ray_${SLURM_JOB_ID}"
else
    _ray_tmp_target="${REPO_ROOT}/logs/ray/local/${HOSTNAME}"
    _ray_tmp_link="/tmp/ray_local_${USER:-user}"
fi
mkdir -p "${_ray_tmp_target}"
if ln -sfn "${_ray_tmp_target}" "${_ray_tmp_link}" 2>/dev/null; then
    export RAY_TMPDIR="${_ray_tmp_link}"
else
    # Fallback: still allow Ray to start (node-local), even if shared logging isn't possible.
    export RAY_TMPDIR="${_ray_tmp_link}"
    mkdir -p "${RAY_TMPDIR}"
fi
echo "RAY_TMPDIR=${RAY_TMPDIR} (target=${_ray_tmp_target})" >&2

# Optional (unset by default): override Ray object store memory. If set too large, Ray can fail to start.
# Example: `VERL_RAY_OBJECT_STORE_MEMORY_GB=2 ./examples/libero/vla_adapter_token/rl/run_vla_adapter_token_rl_libero_lora.sh`
if [ -n "${VERL_RAY_OBJECT_STORE_MEMORY_GB:-}" ]; then
    export VERL_RAY_OBJECT_STORE_MEMORY_GB
fi

# Optional (unset by default): tune Ray's worker-kill threshold.
if [ -n "${RAY_memory_usage_threshold:-}" ]; then
    export RAY_memory_usage_threshold
fi

# ---- User-editable defaults ----
PROJECT_NAME="${PROJECT_NAME:-SimpleVLA-RL}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-libero_object_vla_adapter_token_rl}"

# Your VLA-Adapter repo (needed for `prismatic/` imports inside the checkpoint's custom code)
VLA_ADAPTER_REPO_PATH="${VLA_ADAPTER_REPO_PATH:-${REPO_ROOT}/../VLA-Adapter}"
export VLA_ADAPTER_REPO_PATH

# VLA-Adapter token checkpoint directory (raw path is fine; code will create a safe symlink automatically)
SFT_MODEL_PATH="${SFT_MODEL_PATH:-models/configs+libero_object_no_noops+b64+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--token--libero_object_no_noops--2025-12-21_12-39-01--2500_chkpt}"

CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/runs}"
DATASET_NAME="${DATASET_NAME:-libero_object}"
VLA_NAME="${VLA_NAME:-vla-adapter-token}"

# Allow relative paths (resolved against repo root) to make scripts portable across machines.
_abspath() {
    local p="$1"
    if [[ "$p" != /* ]]; then
        p="${REPO_ROOT}/${p}"
    fi
    "$PYTHON" - <<PY "$p"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
}
SFT_MODEL_PATH="$(_abspath "$SFT_MODEL_PATH")"
CKPT_PATH="$(_abspath "$CKPT_PATH")"
VLA_ADAPTER_REPO_PATH="$(_abspath "$VLA_ADAPTER_REPO_PATH")"
export VLA_ADAPTER_REPO_PATH

# ---- Checkpoint HF dynamic-module sync (VLA-Adapter token checkpoints) ----
# Some locally copied checkpoints may miss or carry stale versions of:
#   - configuration_prismatic.py
#   - processing_prismatic.py
#   - modeling_prismatic.py
# VLA-Adapter eval syncs these; do the same here so behavior matches VLA-Adapter.
if [ -d "$SFT_MODEL_PATH" ] && [ -d "${VLA_ADAPTER_REPO_PATH}/prismatic/extern/hf" ]; then
    _CFG_SRC="${VLA_ADAPTER_REPO_PATH}/prismatic/extern/hf/configuration_prismatic.py"
    _PROC_SRC="${VLA_ADAPTER_REPO_PATH}/prismatic/extern/hf/processing_prismatic.py"
    _MODEL_SRC="${VLA_ADAPTER_REPO_PATH}/prismatic/extern/hf/modeling_prismatic.py"
    _ts() { date +%Y%m%d_%H%M%S; }

    if [ -f "$_CFG_SRC" ]; then
        if [ ! -f "${SFT_MODEL_PATH}/configuration_prismatic.py" ] || ! cmp -s "$_CFG_SRC" "${SFT_MODEL_PATH}/configuration_prismatic.py"; then
            if [ -f "${SFT_MODEL_PATH}/configuration_prismatic.py" ]; then
                cp -f "${SFT_MODEL_PATH}/configuration_prismatic.py" "${SFT_MODEL_PATH}/configuration_prismatic.py.back.$(_ts)" || true
            fi
            cp -f "$_CFG_SRC" "${SFT_MODEL_PATH}/configuration_prismatic.py"
            echo "Synced configuration_prismatic.py from VLA-Adapter into: ${SFT_MODEL_PATH}" >&2
        fi
    fi

    if [ -f "$_PROC_SRC" ]; then
        if [ ! -f "${SFT_MODEL_PATH}/processing_prismatic.py" ] || ! cmp -s "$_PROC_SRC" "${SFT_MODEL_PATH}/processing_prismatic.py"; then
            if [ -f "${SFT_MODEL_PATH}/processing_prismatic.py" ]; then
                cp -f "${SFT_MODEL_PATH}/processing_prismatic.py" "${SFT_MODEL_PATH}/processing_prismatic.py.back.$(_ts)" || true
            fi
            cp -f "$_PROC_SRC" "${SFT_MODEL_PATH}/processing_prismatic.py"
            echo "Synced processing_prismatic.py from VLA-Adapter into: ${SFT_MODEL_PATH}" >&2
        fi
    fi

    if [ -f "$_MODEL_SRC" ]; then
        if [ ! -f "${SFT_MODEL_PATH}/modeling_prismatic.py" ] || ! cmp -s "$_MODEL_SRC" "${SFT_MODEL_PATH}/modeling_prismatic.py"; then
            if [ -f "${SFT_MODEL_PATH}/modeling_prismatic.py" ]; then
                cp -f "${SFT_MODEL_PATH}/modeling_prismatic.py" "${SFT_MODEL_PATH}/modeling_prismatic.py.back.$(_ts)" || true
            fi
            cp -f "$_MODEL_SRC" "${SFT_MODEL_PATH}/modeling_prismatic.py"
            echo "Synced modeling_prismatic.py from VLA-Adapter into: ${SFT_MODEL_PATH}" >&2
        fi
    fi
fi

# RL fine-tuning defaults to LoRA.
# Note: avoid LoRA on `lm_head` by default (huge vocab -> very high memory during log-prob computation).
TARGET_MODULES="${TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]}"
# Whether to load a PEFT adapter from the checkpoint (e.g. `lora_adapter/`) or start a fresh adapter.
# Some checkpoints already have LoRA merged into `model.safetensors`, in which case loading the adapter again will
# double-apply LoRA and tank performance. Most VLA-Adapter exported checkpoints are already merged, so default to 0.
# Set `LORA_LOAD_FROM_CHECKPOINT=1` only if you know the adapter is NOT merged into `model.safetensors`.
LORA_LOAD_FROM_CHECKPOINT="${LORA_LOAD_FROM_CHECKPOINT:-0}"

if [ -z "${NUM_GPUS:-}" ]; then
    if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
        NUM_GPUS="${SLURM_GPUS_ON_NODE}"
    elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -ra _CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
        NUM_GPUS="${#_CUDA_DEVICES[@]}"
    else
        NUM_GPUS=1
    fi
fi
NUM_NODES="${NUM_NODES:-1}"

# Ray's default object store can reserve a large chunk of host RAM. Combined with LIBERO env multiprocessing,
# this can trip Ray's OOM killer on single-GPU workstations. Keep it conservative unless explicitly overridden.
if [ -z "${VERL_RAY_OBJECT_STORE_MEMORY_GB:-}" ] && [ "${NUM_GPUS}" -le 1 ]; then
    export VERL_RAY_OBJECT_STORE_MEMORY_GB=4
fi

ALIGN_PATH="${ALIGN_PATH:-${REPO_ROOT}/align.json}"
TRAINER_RUNTIME_ENV="${TRAINER_RUNTIME_ENV:-none}"

# If runtime_env file uses `excludes: ["*"]`, Ray workers won't inherit environment variables from this script.
# Default to `none` in that case (can override with `FORCE_RUNTIME_ENV=1`).
if [ -z "${FORCE_RUNTIME_ENV:-}" ] && [ -f "$TRAINER_RUNTIME_ENV" ]; then
    _exclude_all="$("$PYTHON" - <<'PY'
import json, os
path = os.environ.get("TRAINER_RUNTIME_ENV", "")
try:
    with open(path, "r") as f:
        cfg = json.load(f)
    excludes = cfg.get("excludes", [])
    print("1" if any(x == "*" for x in excludes) else "0")
except Exception:
    print("0")
PY
)"
    if [ "$_exclude_all" = "1" ]; then
        TRAINER_RUNTIME_ENV="none"
        echo "trainer.runtime_env set to 'none' (runtime_env excludes all env vars). Set FORCE_RUNTIME_ENV=1 to keep using $ALIGN_PATH."
    fi
fi

_disable_wandb=0
if [ -n "${DISABLE_WANDB:-}" ]; then
    _disable_wandb=1
fi

# Auto-disable WandB if runtime_env injects a placeholder/invalid key (prevents AuthenticationError).
if [ "$_disable_wandb" -eq 0 ] && [ -z "${FORCE_WANDB:-}" ] && [ -f "$ALIGN_PATH" ]; then
    _env_key="$("$PYTHON" - <<'PY'
import json, os, sys
path = os.environ.get("ALIGN_PATH", "")
try:
    with open(path, "r") as f:
        cfg = json.load(f)
    print(cfg.get("env_vars", {}).get("WANDB_API_KEY", ""))
except Exception:
    print("")
PY
)"
    if [ -z "$_env_key" ] || [ "$_env_key" = "YOUR WANDB_API_KEY" ]; then
        _disable_wandb=1
        echo "WandB disabled (ALIGN_PATH provides placeholder/missing WANDB_API_KEY). Set FORCE_WANDB=1 and a real key to enable."
    fi
fi

if [ "$_disable_wandb" -eq 1 ]; then
    LOGGER="['console']"
    WANDB_MODE="disabled"
else
    LOGGER="['console','wandb']"
    WANDB_MODE="${WANDB_MODE:-online}"
fi

mkdir -p "${CKPT_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME}"

LOG_FILE="${LOG_FILE:-${CKPT_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME}/console.log}"
mkdir -p "$(dirname "$LOG_FILE")"
set -o pipefail

ALIGN_PATH="$ALIGN_PATH" TRAINER_RUNTIME_ENV="$TRAINER_RUNTIME_ENV" HYDRA_FULL_ERROR=1 "$PYTHON" -u -m verl.trainer.main_ppo \
    trainer.total_steps=230 \
    trainer.max_val_batches=1 \
    data.task_suite_name=$DATASET_NAME \
    data.num_trials_per_task=50 \
    data.n_samples=8 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0 \
    data.accuracy_upper_bound=1 \
    data.oversample_factor=1 \
    data.train_batch_size=4 \
    data.val_batch_size=32 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.vla_adapter_repo_path=$VLA_ADAPTER_REPO_PATH \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.lora_load_from_checkpoint=$LORA_LOAD_FROM_CHECKPOINT \
    actor_rollout_ref.model.target_modules=$TARGET_MODULES \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=8 \
    actor_rollout_ref.actor.optim.lr=5e-5 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=$NUM_GPUS \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False  \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False  \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=2 \
    actor_rollout_ref.actor.traj_mini_batch_size=8 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.num_images_in_input=2 \
    actor_rollout_ref.rollout.use_minivlm=True \
    actor_rollout_ref.rollout.use_proprio=False \
    actor_rollout_ref.rollout.val_micro_batch_size=8 \
    actor_rollout_ref.rollout.temperature=1.6 \
    actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
    actor_rollout_ref.rollout.micro_batch_size=1 \
    actor_rollout_ref.rollout.unnorm_key=$DATASET_NAME \
    actor_rollout_ref.rollout.task_suite_name=$DATASET_NAME \
    actor_rollout_ref.rollout.pretrained_checkpoint=$SFT_MODEL_PATH \
    actor_rollout_ref.rollout.center_crop=True \
    actor_rollout_ref.rollout.max_prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.98 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.logger=$LOGGER \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=25 \
    trainer.test_freq=10 \
    trainer.total_epochs=100 \
    trainer.val_only=False \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    trainer.runtime_env=$TRAINER_RUNTIME_ENV \
    trainer.wandb_mode=$WANDB_MODE \
    trainer.val_before_train=True \
    "$@" |& tee -a "$LOG_FILE"
