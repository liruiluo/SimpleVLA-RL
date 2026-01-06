#!/bin/bash
set -euo pipefail
set -x

# LoRA-enabled LIBERO RL training launcher.
# Usage:
#   bash examples/libero/openvla_oft/run_openvla_oft_rl_libero_lora.sh
# Optional overrides (env vars):
#   SFT_MODEL_PATH=/path/to/sft_ckpt CKPT_PATH=/path/to/save \
#   DATASET_NAME=libero_10 NUM_GPUS=8 NUM_NODES=1 \
#   LORA_RANK=32 LORA_ALPHA=64 LORA_TARGET_MODULES=all-linear \
#   EXPERIMENT_NAME=lib10_openvla_oft_rl_lora

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
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

if [ -f "${REPO_ROOT}/activate_env.sh" ]; then
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/activate_env.sh"
else
    echo "WARNING: ${REPO_ROOT}/activate_env.sh not found. Skipping environment activation."
fi

export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export ROBOT_PLATFORM=LIBERO

# TensorFlow/XLA is used in image preprocessing (see `verl/utils/libero_utils.py`).
if [[ "${XLA_FLAGS:-}" != *"--xla_gpu_cuda_data_dir="* ]]; then
    _candidate_cuda_dirs=()
    for _d in "${XLA_GPU_CUDA_DATA_DIR:-}" "${CUDA_HOME:-}" "${CUDA_DIR:-}" "${CUDA_PATH:-}"; do
        if [ -n "${_d}" ]; then
            _candidate_cuda_dirs+=("${_d}")
        fi
    done
    if command -v nvcc >/dev/null 2>&1; then
        _nvcc_path="$(command -v nvcc)"
        _nvcc_real="$(readlink -f "${_nvcc_path}" 2>/dev/null || echo "${_nvcc_path}")"
        _candidate_cuda_dirs+=("$(cd "$(dirname "${_nvcc_real}")/.." && pwd)")
    fi
    _candidate_cuda_dirs+=("/cm/shared/apps/cuda12.2" "/usr/local/cuda" "/usr/local/cuda-12.2" "/usr/lib/cuda")

    for _cuda_dir in "${_candidate_cuda_dirs[@]}"; do
        if [ -f "${_cuda_dir}/nvvm/libdevice/libdevice.10.bc" ]; then
            export XLA_FLAGS="--xla_gpu_cuda_data_dir=${_cuda_dir} ${XLA_FLAGS:-}"
            echo "Set XLA_FLAGS for TF/XLA: ${XLA_FLAGS}"
            break
        fi
    done
fi

# ---- Experiment identifiers ----
PROJECT_NAME="${PROJECT_NAME:-SimpleVLA-RL}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-lib10_openvla_oft_rl_lora}"

# ---- Paths ----
SFT_MODEL_PATH="${SFT_MODEL_PATH:-${REPO_ROOT}/models/Openvla-oft-SFT-libero10-traj1}"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/runs}"

# ---- Task / model ----
DATASET_NAME="${DATASET_NAME:-libero_10}"
VLA_NAME="${VLA_NAME:-openvla-oft}"

# ---- LoRA knobs ----
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"

# ---- GPU / node settings ----
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

# Ray runtime env config (used to pass env vars like WANDB_API_KEY)
ALIGN_PATH="${ALIGN_PATH:-${REPO_ROOT}/align.json}"

# Persist Ray session + logs on the shared filesystem so Slurm jobs can be debugged from the head node.
# Ray uses AF_UNIX sockets for plasma store; the socket path length must be <= 107 bytes.
# The repo path on shared FS is long, so we use a short `/tmp/...` symlink that points to the shared target dir.
#
# IMPORTANT: override any pre-set `RAY_TMPDIR` to avoid reusing the same Ray temp dir across jobs
# (which can lead to stale Ray state and placement group name collisions).
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
    export RAY_TMPDIR="${_ray_tmp_link}"
    mkdir -p "${RAY_TMPDIR}"
fi
echo "RAY_TMPDIR=${RAY_TMPDIR} (target=${_ray_tmp_target})" >&2

# WandB logging:
if [ -n "${DISABLE_WANDB:-}" ]; then
    LOGGER="['console']"
    WANDB_MODE="disabled"
    echo "DISABLE_WANDB is set; WandB logging disabled."
else
    LOGGER="['console','wandb']"
    WANDB_MODE="${WANDB_MODE:-online}"
    echo "Using WandB logging. Make sure you have run 'wandb login'."
fi

mkdir -p "${CKPT_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME}"

# ---- Preflight checks (fail fast) ----
if [ ! -d "${SFT_MODEL_PATH}" ]; then
    echo "ERROR: SFT_MODEL_PATH does not exist: ${SFT_MODEL_PATH}"
    echo "       Set SFT_MODEL_PATH to your downloaded OpenVLA-OFT checkpoint directory."
    exit 1
fi

if ! "${REPO_ROOT}/env/bin/python" -c "import numpy as np; import sys; sys.exit(0 if int(np.__version__.split('.')[0]) < 2 else 1)"; then
    "${REPO_ROOT}/env/bin/python" -c "import numpy as np; print('Detected numpy:', np.__version__)"
    echo "ERROR: numpy>=2 detected, but TensorFlow/OpenVLA-OFT in this env typically requires numpy<2."
    echo "       Fix (example): pip install -U --force-reinstall 'numpy<2'  (then retry)"
    exit 1
fi

if ! "${REPO_ROOT}/env/bin/python" -c "import libero" >/dev/null 2>&1; then
    echo "ERROR: 'libero' is not importable in this environment."
    echo "       Install LIBERO (example): git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && pip install -e LIBERO"
    echo "       Or set LIBERO_ROOT and add it to PYTHONPATH before running."
    exit 1
fi

# Ensure the VLA checkpoint has the latest OpenVLA-OFT code
bash "${REPO_ROOT}/examples/utils/overwrite_vla_ckpt_utils.sh" "$SFT_MODEL_PATH"

HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
    data.task_suite_name=$DATASET_NAME \
    data.num_trials_per_task=50 \
    data.n_samples=8 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0 \
    data.accuracy_upper_bound=1 \
    data.oversample_factor=1 \
    data.train_batch_size=32 \
    data.val_batch_size=256 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=8 \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.target_modules=$LORA_TARGET_MODULES \
    actor_rollout_ref.actor.optim.lr=5e-5 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=$NUM_GPUS \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=8 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.use_proprio=False \
    actor_rollout_ref.rollout.val_micro_batch_size=8 \
    actor_rollout_ref.rollout.temperature=1.6 \
    actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
    actor_rollout_ref.rollout.micro_batch_size=1 \
    actor_rollout_ref.rollout.unnorm_key=$DATASET_NAME \
    actor_rollout_ref.rollout.model_family=openvla \
    actor_rollout_ref.rollout.task_suite_name=$DATASET_NAME \
    actor_rollout_ref.rollout.num_steps_wait=10 \
    actor_rollout_ref.rollout.pretrained_checkpoint=$SFT_MODEL_PATH \
    actor_rollout_ref.rollout.center_crop=True \
    actor_rollout_ref.rollout.max_prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
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
    trainer.runtime_env=$ALIGN_PATH \
    trainer.wandb_mode=$WANDB_MODE \
    trainer.val_before_train=True
