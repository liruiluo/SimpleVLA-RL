from __future__ import annotations

import os
import re
import sys
import hashlib
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


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
    # Match PEFT's LoRA Linear.forward numerics as closely as possible by using F.linear
    # (so dtype / rounding match the underlying kernels), then cast to fp32 at the end.
    base_layer = lm_head.base_layer if hasattr(lm_head, "base_layer") else lm_head

    width = int(end_token) - int(start_token)
    if width <= 0:
        raise ValueError(f"Invalid vocab slice [{start_token}:{end_token}); width must be > 0.")

    # NOTE: Avoid `param[start:end]` views under FSDP. Those can reference ephemeral unsharded storage that gets
    # freed before backward, leading to "storage size 0" / setStorage errors.
    #
    # Additionally, some checkpoints/configs may have `action_vocab_size` slightly larger than the actual
    # `lm_head` rows (e.g., after tokenizer resize). Python slicing would silently clamp; `index_select` would
    # hard-error (device-side assert). We emulate "take the last `width` rows up to `end_token`", clamped to
    # the real weight size.
    weight_rows = int(base_layer.weight.size(0))
    effective_end = min(int(end_token), weight_rows)
    effective_start = effective_end - width
    if effective_start < 0:
        effective_start = max(0, weight_rows - width)
        effective_end = effective_start + width

    if not (0 <= effective_start < effective_end <= weight_rows) or (effective_end - effective_start) != width:
        raise RuntimeError(
            f"Cannot slice lm_head rows safely: requested=[{start_token}:{end_token}) width={width}, "
            f"weight_rows={weight_rows}, effective=[{effective_start}:{effective_end})."
        )

    token_idx = torch.arange(effective_start, effective_end, device=base_layer.weight.device, dtype=torch.long)
    weight_slice = torch.index_select(base_layer.weight, dim=0, index=token_idx)
    bias = getattr(base_layer, "bias", None)
    bias_slice = torch.index_select(bias, dim=0, index=token_idx) if bias is not None else None

    # Most RL runs in this repo fine-tune LoRA adapters only; base model weights are frozen.
    # If the base lm_head weights are frozen, detach the gathered slice to avoid running autograd
    # (index_select backward) through FSDP-managed sharded parameters.
    if not bool(getattr(base_layer.weight, "requires_grad", False)):
        weight_slice = weight_slice.detach()
        if bias_slice is not None:
            bias_slice = bias_slice.detach()

    # Base projection (match base_layer forward dtype).
    hs = hidden_states
    if hs.dtype != weight_slice.dtype:
        hs = hs.to(weight_slice.dtype)
    result = F.linear(hs, weight_slice, bias_slice)
    torch_result_dtype = result.dtype

    # PEFT LoRA (peft==0.11.1): result += lora_B(lora_A(dropout(x))) * scaling for each active adapter.
    # If adapters are disabled or merged, base_layer already contains the effective weights.
    if hasattr(lm_head, "disable_adapters") and (getattr(lm_head, "disable_adapters") or getattr(lm_head, "merged", False)):
        return result.to(torch.float32)

    def _get(obj, key, default=None):
        if obj is None:
            return default
        if hasattr(obj, "get"):
            return obj.get(key, default)
        if hasattr(obj, "__contains__") and key in obj:
            return obj[key]
        return default

    lora_A = getattr(lm_head, "lora_A", None)
    lora_B = getattr(lm_head, "lora_B", None)
    # PEFT stores adapters in nn.ModuleDict (dict-like but not an actual `dict` / Mapping).
    if lora_A is not None and lora_B is not None and hasattr(lora_A, "keys") and hasattr(lora_B, "keys") and len(lora_A) > 0:
        active_adapters = getattr(lm_head, "active_adapters", None)
        if isinstance(active_adapters, str):
            active_adapters = [active_adapters]
        if not isinstance(active_adapters, (list, tuple, set)):
            active_adapters = list(lora_A.keys())

        scaling_map = getattr(lm_head, "scaling", None)
        dropout_map = getattr(lm_head, "lora_dropout", None)
        use_dora_map = getattr(lm_head, "use_dora", None)

        for name in active_adapters:
            if name not in lora_A or name not in lora_B:
                continue
            if bool(_get(use_dora_map, name, False)):
                raise NotImplementedError("DoRA is not supported in lm_head_logits_subset")

            A_mod = lora_A[name]  # Linear(in_features=hidden, out_features=r)
            B_mod = lora_B[name]  # Linear(in_features=r, out_features=vocab)
            dropout = _get(dropout_map, name, None)
            scaling = float(_get(scaling_map, name, 1.0))

            x = hidden_states.to(A_mod.weight.dtype)
            if dropout is not None:
                x = dropout(x)
            # lora_A(dropout(x)) using module forward (matches PEFT)
            x_a = A_mod(x)

            # lora_B(...) but only for the vocab slice
            idx_b = token_idx if token_idx.device == B_mod.weight.device else token_idx.to(B_mod.weight.device)
            B_weight = torch.index_select(B_mod.weight, dim=0, index=idx_b)
            delta = F.linear(x_a, B_weight, None)

            # Multiply in-module dtype (PEFT uses python float scaling, which follows tensor dtype).
            delta = delta * delta.new_tensor(scaling)
            result = result + delta

        result = result.to(torch_result_dtype)

    return result.to(torch.float32)
