#!/bin/bash
set -x

# Determine repo root.
# 优先使用外部传入的 REPO_ROOT；否则根据当前工作目录推断：
# - 若当前目录本身是仓库根（包含 verl/ 和 examples/），用 $PWD
# - 若上一级目录是仓库根，用 $PWD/..
if [ -z "${REPO_ROOT:-}" ]; then
    if [ -d "$PWD/verl" ] && [ -d "$PWD/examples" ]; then
        REPO_ROOT="$PWD"
    elif [ -d "$PWD/../verl" ] && [ -d "$PWD/../examples" ]; then
        REPO_ROOT="$(cd "$PWD/.." && pwd)"
    else
        # Fallback: use script location (可能在 Slurm spool 下，仅作为兜底)
        REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    fi
fi

# ---- Environment setup (Python / Conda) ----
# Ensure user shell init is loaded so that `conda` is available
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

# Activate the repo-local Conda env via helper script (if present)
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
export ROBOT_PLATFORM=LIBERO # Use LIBERO: ROBOT_PLATFORM=LIBERO  Use Robotwin ROBOT_PLATFORM=ALOHA

# Basic experiment identifiers (can be overridden by env vars)
PROJECT_NAME="${PROJECT_NAME:-SimpleVLA-RL}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-lib10_openvla_oft_rl}"

# Default paths:
# - SFT_MODEL_PATH points to the included Libero-10 SFT checkpoint
# - CKPT_PATH is where RL checkpoints will be saved
SFT_MODEL_PATH="${SFT_MODEL_PATH:-${REPO_ROOT}/models/Openvla-oft-SFT-libero10-traj1}"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/runs}"

# DATASET_NAME can be libero_10 (libero_Long), libero_90, libero_spatial, libero_object, libero_goal
DATASET_NAME="${DATASET_NAME:-libero_10}"
VLA_NAME="${VLA_NAME:-openvla-oft}"

# GPU / node settings
# 优先使用外部传入的 NUM_GPUS；否则从 Slurm / CUDA_VISIBLE_DEVICES 自动推断
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

# If you want to use 2*8 GPU to RL. Set NUM_NODES=2
NUM_NODES="${NUM_NODES:-1}"

# Ray runtime env config (used to pass env vars like WANDB_API_KEY)
ALIGN_PATH="${ALIGN_PATH:-${REPO_ROOT}/align.json}"

# WandB logging:
# - 默认启用 WandB（假设你已经在登录节点上运行过 `wandb login`）
# - 如需禁用 WandB，可在提交前 `export DISABLE_WANDB=1`
if [ -n "${DISABLE_WANDB:-}" ]; then
    LOGGER="['console']"
    WANDB_MODE="disabled"
    echo "DISABLE_WANDB is set; WandB logging disabled."
else
    LOGGER="['console','wandb']"
    WANDB_MODE="${WANDB_MODE:-online}"
    echo "Using WandB logging. Make sure you have run 'wandb login'."
fi

# Make sure ckpt base dir exists
mkdir -p "${CKPT_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME}"

# Ensure the VLA checkpoint has the latest OpenVLA-OFT code
bash "${REPO_ROOT}/examples/overwrite_vla_ckpt_utils.sh" "$SFT_MODEL_PATH"

HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
    data.task_suite_name=$DATASET_NAME \
    data.num_trials_per_task=50 \
    data.n_samples=8 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.1 \
    data.accuracy_upper_bound=0.9 \
    data.oversample_factor=1 \
    data.train_batch_size=64 \
    data.val_batch_size=256 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=8 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
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
    actor_rollout_ref.actor.traj_mini_batch_size=16 \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
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
    trainer.test_freq=4 \
    trainer.total_epochs=100 \
    trainer.val_only=False \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    trainer.runtime_env=$ALIGN_PATH \
    trainer.wandb_mode=$WANDB_MODE \
    trainer.val_before_train=True \
