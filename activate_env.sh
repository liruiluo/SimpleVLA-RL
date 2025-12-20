#!/bin/bash

# Helper script to activate the repo-local Conda environment.
# This script is meant to be sourced, not executed as a separate process.
#
# Usage (from other scripts):
#   REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "${REPO_ROOT}/activate_env.sh"

# Compute repo root as the directory containing this script
REPO_ROOT_ACTIVATE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PATH="${REPO_ROOT_ACTIVATE}/env"

if [ ! -d "$ENV_PATH" ]; then
    echo "WARNING: Conda env directory not found at: $ENV_PATH"
    echo "         Please create it or adjust ENV_PATH in activate_env.sh."
    return 0 2>/dev/null || exit 0
fi

# Try to locate and initialize Conda properly
if command -v conda >/dev/null 2>&1; then
    # Use `conda info --base` to find the base prefix, then source conda.sh
    CONDA_BASE="$(conda info --base 2>/dev/null)"
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1090
        . "$CONDA_BASE/etc/profile.d/conda.sh"
    fi
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "WARNING: 'conda' shell function not available. Unable to activate $ENV_PATH."
    return 0 2>/dev/null || exit 0
fi

echo "Activating Conda environment at: $ENV_PATH"
conda activate "$ENV_PATH"

# ---- HuggingFace cache locations ----
# Avoid writing to small/quotad scratch by default.
# You can override any of these vars before sourcing this script.
HF_HOME_DEFAULT="${REPO_ROOT_ACTIVATE}/.cache/huggingface"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"
echo "HF_HOME=$HF_HOME"

# Optionally add LIBERO source tree to PYTHONPATH so that `import libero` works.
# You can override LIBERO_ROOT before sourcing this script if your path differs.
LIBERO_ROOT_DEFAULT="/share/ml/luolirui/LIBERO"
if [ -z "${LIBERO_ROOT+x}" ]; then
    LIBERO_ROOT="$LIBERO_ROOT_DEFAULT"
    _LIBERO_ROOT_FROM_ENV=0
else
    _LIBERO_ROOT_FROM_ENV=1
fi

# If the default path is missing, try common local layouts:
#   your_workspace/
#   ├── SimpleVLA-RL/   (this repo)
#   └── LIBERO/
if [ "$_LIBERO_ROOT_FROM_ENV" -eq 0 ] && [ ! -d "$LIBERO_ROOT" ]; then
    _LIBERO_SIBLING="${REPO_ROOT_ACTIVATE}/../LIBERO"
    if [ -d "$_LIBERO_SIBLING" ]; then
        LIBERO_ROOT="$_LIBERO_SIBLING"
    fi
fi

if [ -d "$LIBERO_ROOT" ]; then
    case ":${PYTHONPATH:-}:" in
        *":$LIBERO_ROOT:"*) ;;
        *) export PYTHONPATH="$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}";;
    esac
    echo "Added LIBERO to PYTHONPATH: $LIBERO_ROOT"
else
    echo "NOTE: LIBERO root directory not found at: $LIBERO_ROOT"
    echo "      If you need LIBERO, set LIBERO_ROOT before sourcing activate_env.sh."
fi
