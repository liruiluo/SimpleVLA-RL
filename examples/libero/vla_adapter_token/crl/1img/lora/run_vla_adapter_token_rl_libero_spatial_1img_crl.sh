#!/bin/bash
set -euo pipefail
set -x

# Make CUDA errors synchronous for clearer stack traces (debugging).
export CUDA_LAUNCH_BLOCKING=1

# CRL (sequential tasks) launcher for LIBERO-Spatial, 1-image, LoRA.

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

_args=(data.use_crl=True data.val_batch_size=32 trainer.crl_eval_on_switch=True trainer.crl_save_on_switch=True actor_rollout_ref.model.save_merged_model=False)
if [ -n "${CRL_STEPS_PER_TASK:-}" ]; then
  _args+=(trainer.crl_steps_per_task="${CRL_STEPS_PER_TASK}")
fi
if [ -n "${CRL_TASK_IDS:-}" ]; then
  _args+=(data.crl_task_ids="${CRL_TASK_IDS}")
fi

bash "${REPO_ROOT}/examples/libero/vla_adapter_token/rl/1img/lora/run_vla_adapter_token_rl_libero_spatial_1img.sh" \
  "${_args[@]}" \
  "$@"
