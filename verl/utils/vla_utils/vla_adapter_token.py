from __future__ import annotations

import os
import re
import sys
import hashlib
from pathlib import Path
from typing import Any, Optional, Tuple

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


class VLAAdapterTokenActionLogitsWrapper(torch.nn.Module):
    """
    FSDP-friendly wrapper for VLA-Adapter token models.

    - Default: delegates to the wrapped model's forward() (HF-compatible).
    - Special path: when `action_input_ids` is provided, computes only action-token logits in the 256-bin
      action vocabulary slice without materializing full-vocab logits.
    """

    def __init__(self, wrapped: torch.nn.Module, action_token_len: int, action_chunks_len: int):
        super().__init__()
        self.wrapped = wrapped
        self.action_token_len = int(action_token_len)
        self.action_chunks_len = int(action_chunks_len)

    def __getattr__(self, name: str) -> Any:  # pragma: no cover
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.wrapped, name)

    def forward(
        self,
        *args: Any,
        action_input_ids: Optional[torch.Tensor] = None,
        action_attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        prompt_lens: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> Any:
        if action_input_ids is None:
            return self.wrapped(*args, **kwargs)

        if action_attention_mask is None or pixel_values is None or labels is None or prompt_lens is None:
            raise ValueError(
                "VLAAdapterTokenActionLogitsWrapper.forward() requires "
                "`action_attention_mask`, `pixel_values`, `labels`, and `prompt_lens` when `action_input_ids` is set."
            )

        device = action_input_ids.device
        batch_size = action_input_ids.size(0)
        action_token_count = int(self.action_token_len * self.action_chunks_len)

        module = self.wrapped
        if hasattr(module, "get_base_model"):
            module = module.get_base_model()

        input_embeddings = module.get_input_embeddings()(action_input_ids)  # (B, seq, D)
        all_actions_mask = module._process_action_masks(labels)
        language_embeddings = input_embeddings[~all_actions_mask].reshape(batch_size, -1, input_embeddings.shape[2])

        projected_patch_embeddings = module._process_vision_features(pixel_values, language_embeddings, use_film=False)

        action_queries = module.action_queries.weight
        action_queries = action_queries.view(1, action_queries.shape[0], action_queries.shape[1]).repeat(batch_size, 1, 1)
        input_embeddings = module._replace_input_embeddings(input_embeddings, all_actions_mask, action_queries)

        multimodal_embeddings, multimodal_attention_mask = module._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, action_attention_mask
        )

        lm = module.language_model
        lm_outputs = lm.model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = lm_outputs[0]  # (B, seq_total, hidden)

        num_patches = int(module.vision_backbone.get_num_patches() * module.vision_backbone.get_num_images_in_input())
        start_positions = num_patches + prompt_lens.to(torch.long) - 1  # (B,)
        positions = start_positions[:, None] + torch.arange(action_token_count, device=device)[None, :]
        action_hidden = hidden_states[torch.arange(batch_size, device=device)[:, None], positions]  # (B, 56, hidden)

        action_vocab_size = int(module.action_vocab_size)
        start_token = int(action_vocab_size - 256)
        action_logits = lm_head_logits_subset(lm.lm_head, action_hidden, start_token, action_vocab_size)  # (B, 56, 256)

        temperature = float(temperature)
        if temperature <= 0:
            temperature = 1.0
        action_logits = action_logits / temperature

        return action_logits, start_token
