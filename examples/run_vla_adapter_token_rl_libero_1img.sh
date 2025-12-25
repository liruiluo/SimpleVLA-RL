#!/bin/bash
set -euo pipefail
set -x

# 1-image launcher for VLA-Adapter token RL on LIBERO.
# Wraps `examples/run_vla_adapter_token_rl_libero_lora.sh` and forces `num_images_in_input=1`.

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

# Default checkpoint (override via env `SFT_MODEL_PATH=...`).
# This defaults to the 1-image LIBERO-Spatial token-VLA checkpoint.
SFT_MODEL_PATH="${SFT_MODEL_PATH:-${REPO_ROOT}/models/token-1img/configs+libero_spatial_no_noops+b64+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--token--1img--libero_spatial_no_noops--2025-12-25_00-30-24--10000_chkpt}"
export SFT_MODEL_PATH

# Keep HF caches repo-local to avoid stale dynamic-module caches under ~/.cache.
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"

# Default VLA-Adapter repo path (required for token-VLA checkpoints).
VLA_ADAPTER_REPO_PATH="${VLA_ADAPTER_REPO_PATH:-${REPO_ROOT}/../VLA-Adapter}"
export VLA_ADAPTER_REPO_PATH

# Some local token-VLA checkpoints may miss HF dynamic-module files referenced by `auto_map`.
# For Qwen-based token-VLA, these must come from VLA-Adapter's `prismatic/extern/hf/` (NOT SimpleVLA-RL's OpenVLA).
if [ -d "$SFT_MODEL_PATH" ]; then
    _CFG_SRC="${VLA_ADAPTER_REPO_PATH}/prismatic/extern/hf/configuration_prismatic.py"
    _PROC_SRC="${VLA_ADAPTER_REPO_PATH}/prismatic/extern/hf/processing_prismatic.py"
    _MODEL_SRC="${VLA_ADAPTER_REPO_PATH}/prismatic/extern/hf/modeling_prismatic.py"

    if [ -f "$_CFG_SRC" ]; then
        _NEED_CFG=0
        if [ ! -f "${SFT_MODEL_PATH}/configuration_prismatic.py" ]; then
            _NEED_CFG=1
        else
            # If checkpoint config says it's Qwen-based but the local configuration file doesn't mention qwen,
            # it's likely the wrong file (e.g., copied from OpenVLA).
            _PY_DETECT="python"
            if ! command -v "$_PY_DETECT" >/dev/null 2>&1; then
                _PY_DETECT="python3"
            fi
            _LLM_ID="$("$_PY_DETECT" - <<'PY' "$SFT_MODEL_PATH"
import json, os, sys
cfg = os.path.join(sys.argv[1], "config.json")
try:
    obj = json.load(open(cfg, "r"))
except Exception:
    print("")
    raise SystemExit(0)
print(obj.get("llm_backbone_id", "") or "")
PY
)"
            if [[ "${_LLM_ID}" == qwen* ]] && ! grep -qi "qwen" "${SFT_MODEL_PATH}/configuration_prismatic.py" 2>/dev/null; then
                _NEED_CFG=1
            fi
        fi

        if [ "$_NEED_CFG" -eq 1 ]; then
            if [ -f "${SFT_MODEL_PATH}/configuration_prismatic.py" ]; then
                cp -f "${SFT_MODEL_PATH}/configuration_prismatic.py" "${SFT_MODEL_PATH}/configuration_prismatic.py.back.$(date +%Y%m%d_%H%M%S)" || true
            fi
            cp -f "$_CFG_SRC" "${SFT_MODEL_PATH}/configuration_prismatic.py"
            echo "Patched configuration_prismatic.py from VLA-Adapter into: ${SFT_MODEL_PATH}" >&2
        fi
    fi

    if [ -f "$_PROC_SRC" ] && [ ! -f "${SFT_MODEL_PATH}/processing_prismatic.py" ]; then
        cp -f "$_PROC_SRC" "${SFT_MODEL_PATH}/processing_prismatic.py"
        echo "Patched missing processing_prismatic.py from VLA-Adapter into: ${SFT_MODEL_PATH}" >&2
    fi

    if [ -f "$_MODEL_SRC" ] && [ ! -f "${SFT_MODEL_PATH}/modeling_prismatic.py" ]; then
        cp -f "$_MODEL_SRC" "${SFT_MODEL_PATH}/modeling_prismatic.py"
        echo "Patched missing modeling_prismatic.py from VLA-Adapter into: ${SFT_MODEL_PATH}" >&2
    fi

    # If this checkpoint was ever loaded with a wrong configuration_prismatic.py, Transformers may have cached it
    # under ~/.cache/huggingface/modules/transformers_modules/hf_safe_<md5>. Remove that stale cache if it doesn't
    # contain Qwen mappings but the checkpoint expects Qwen.
    _PY_DETECT="python"
    if ! command -v "$_PY_DETECT" >/dev/null 2>&1; then
        _PY_DETECT="python3"
    fi
    if command -v "$_PY_DETECT" >/dev/null 2>&1; then
        _LLM_ID="$("$_PY_DETECT" - <<'PY' "$SFT_MODEL_PATH"
import json, os, sys
cfg = os.path.join(sys.argv[1], "config.json")
try:
    obj = json.load(open(cfg, "r"))
except Exception:
    print("")
    raise SystemExit(0)
print(obj.get("llm_backbone_id", "") or "")
PY
)"
        if [[ "${_LLM_ID}" == qwen* ]]; then
            _DIGEST="$("$_PY_DETECT" - <<'PY' "$SFT_MODEL_PATH"
import hashlib, os, sys
p = os.path.abspath(sys.argv[1])
print(hashlib.md5(p.encode()).hexdigest()[:12])
PY
)"
            _STALE_DIR="$HOME/.cache/huggingface/modules/transformers_modules/hf_safe_${_DIGEST}"
            if [ -f "${_STALE_DIR}/configuration_prismatic.py" ] && ! grep -qi "qwen" "${_STALE_DIR}/configuration_prismatic.py" 2>/dev/null; then
                if [ "${VERL_CLEAN_HF_MODULES_CACHE:-0}" = "1" ]; then
                    rm -rf "${_STALE_DIR}"
                    echo "Removed stale HF dynamic-module cache: ${_STALE_DIR}" >&2
                else
                    echo "WARNING: Stale HF dynamic-module cache detected at: ${_STALE_DIR}" >&2
                    echo "         Set VERL_CLEAN_HF_MODULES_CACHE=1 to auto-remove it, or delete it manually." >&2
                fi
            fi
        fi
    fi
fi

# Auto-detect `DATASET_NAME` from `${SFT_MODEL_PATH}/dataset_statistics.json` if not provided.
if [ -z "${DATASET_NAME:-}" ]; then
    _PY_DETECT="python"
    if ! command -v "$_PY_DETECT" >/dev/null 2>&1; then
        _PY_DETECT="python3"
    fi

    if command -v "$_PY_DETECT" >/dev/null 2>&1 && [ -f "${SFT_MODEL_PATH}/dataset_statistics.json" ]; then
        _DETECTED_DATASET_NAME="$("$_PY_DETECT" - <<'PY' "$SFT_MODEL_PATH"
import json, sys, os
ckpt = sys.argv[1]
path = os.path.join(ckpt, "dataset_statistics.json")
try:
    obj = json.load(open(path, "r"))
except Exception:
    print("")
    raise SystemExit(0)
keys = list(obj.keys())
if len(keys) != 1:
    print("")
    raise SystemExit(0)
k = keys[0]
if k.endswith("_no_noops"):
    k = k[: -len("_no_noops")]
print(k)
PY
)"
        if [ -n "${_DETECTED_DATASET_NAME}" ]; then
            DATASET_NAME="${_DETECTED_DATASET_NAME}"
            export DATASET_NAME
            echo "Auto-detected DATASET_NAME=${DATASET_NAME} from ${SFT_MODEL_PATH}/dataset_statistics.json" >&2
        fi
    fi
fi

# Avoid confusing defaults from the base script when switching checkpoints (e.g. libero_object vs libero_spatial).
if [ -z "${EXPERIMENT_NAME:-}" ] && [ -n "${DATASET_NAME:-}" ]; then
    EXPERIMENT_NAME="${DATASET_NAME}_vla_adapter_token_rl_1img"
    export EXPERIMENT_NAME
fi

NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-1}"

bash "${REPO_ROOT}/examples/run_vla_adapter_token_rl_libero_lora.sh" \
  actor_rollout_ref.actor.num_images_in_input="${NUM_IMAGES_IN_INPUT}" \
  actor_rollout_ref.rollout.num_images_in_input="${NUM_IMAGES_IN_INPUT}" \
  "$@"
