from __future__ import annotations

import os
import re
import sys
import hashlib
from pathlib import Path
from typing import Optional

import torch


def ensure_vla_adapter_prismatic(vla_adapter_repo_path: Optional[str] = None, hint_path: Optional[str] = None) -> str:
    """
    Ensure imports inside VLA-Adapter token checkpoints (e.g. `modeling_prismatic.py`) resolve the correct
    `prismatic` package (with Qwen-specific constants like `NUM_TOKENS`).

    Returns the resolved repo path inserted into `sys.path`.
    """
    repo = (vla_adapter_repo_path or os.environ.get("VLA_ADAPTER_REPO_PATH") or "").strip()
    if not repo:
        raise RuntimeError(
            "VLA-Adapter repo path is not set.\n"
            "Set env `VLA_ADAPTER_REPO_PATH=/path/to/VLA-Adapter` or config `model.vla_adapter_repo_path`."
        )

    repo_path = Path(repo).resolve()
    if not (repo_path / "prismatic").is_dir():
        raise RuntimeError(f"Invalid VLA-Adapter repo path (missing `prismatic/`): {repo_path}")

    chosen_str = str(repo_path)
    if chosen_str not in sys.path:
        sys.path.insert(0, chosen_str)

    # If `prismatic` was already imported from another location (e.g. openvla-oft), reload it from VLA-Adapter.
    existing = sys.modules.get("prismatic")
    if existing is not None:
        existing_file = getattr(existing, "__file__", "") or ""
        if not str(existing_file).startswith(chosen_str):
            for name in [n for n in sys.modules.keys() if n == "prismatic" or n.startswith("prismatic.")]:
                sys.modules.pop(name, None)

    return chosen_str


_PY_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def ensure_hf_trust_remote_code_safe_path(path: str, link_root: Optional[str] = None) -> str:
    """
    HF `trust_remote_code=True` caches modules under a Python package name derived from the model path/name.
    If the directory basename contains characters like '+' or '-', relative imports inside custom code can fail.

    This creates a stable symlink with a Python-identifier-safe name and returns that symlink path.
    """
    original = Path(path).resolve()
    if original.is_file():
        original = original.parent

    name = original.name
    if _PY_IDENTIFIER_RE.match(name):
        return str(original)

    digest = hashlib.md5(str(original).encode()).hexdigest()[:12]
    safe_name = f"hf_safe_{digest}"

    root = Path(link_root).resolve() if link_root else (original.parent / "_hf_safe")
    root.mkdir(parents=True, exist_ok=True)
    safe_path = root / safe_name

    if safe_path.exists():
        return str(safe_path)

    try:
        safe_path.symlink_to(original, target_is_directory=True)
    except FileExistsError:
        pass

    return str(safe_path)


def lm_head_logits_subset(lm_head: torch.nn.Module, hidden_states: torch.Tensor, start_token: int, end_token: int) -> torch.Tensor:
    """
    Compute logits for a contiguous vocab slice [start_token:end_token) from hidden states, including PEFT-LoRA
    contributions if `lm_head` is a LoRA-wrapped layer.

    Args:
        lm_head: A (possibly PEFT-wrapped) lm_head module.
        hidden_states: (..., hidden) tensor.
        start_token: Start vocab index (inclusive).
        end_token: End vocab index (exclusive).

    Returns:
        logits: (..., end_token-start_token) in fp32.
    """
    if hasattr(lm_head, "base_layer"):
        base_layer = lm_head.base_layer
    else:
        base_layer = lm_head

    weight = base_layer.weight[start_token:end_token]
    logits = torch.matmul(hidden_states, weight.t()).to(torch.float32)
    bias = getattr(base_layer, "bias", None)
    if bias is not None:
        logits = logits + bias[start_token:end_token].to(torch.float32)

    # PEFT LoRA: delta = (x @ A^T) @ B^T * scaling
    if hasattr(lm_head, "lora_A") and hasattr(lm_head, "lora_B") and len(getattr(lm_head, "lora_A", {})) > 0:
        name = None
        active = getattr(lm_head, "active_adapters", None)
        if isinstance(active, (list, tuple, set)) and len(active) > 0:
            name = next(iter(active))
        elif isinstance(active, str) and active:
            name = active
        else:
            active = getattr(lm_head, "active_adapter", None)
            if isinstance(active, str) and active:
                name = active
            else:
                name = next(iter(lm_head.lora_A.keys()))

        A = lm_head.lora_A[name].weight  # (r, hidden)
        B = lm_head.lora_B[name].weight  # (vocab, r)
        scaling = float(getattr(lm_head, "scaling", {}).get(name, 1.0))

        x_a = torch.matmul(hidden_states.to(A.dtype), A.t())  # (..., r)
        B_slice = B[start_token:end_token]  # (slice, r)
        delta = torch.matmul(x_a.to(B_slice.dtype), B_slice.t()) * scaling
        logits = logits + delta.to(torch.float32)

    return logits
