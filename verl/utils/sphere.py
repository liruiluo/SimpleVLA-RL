from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


def _topk_scatter_softmax_probs(logits: torch.Tensor, top_k: int, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    if logits.dim() < 2:
        raise ValueError(f"Expected logits with shape (..., n_experts), got {tuple(logits.shape)}")

    n_experts = int(logits.size(-1))
    if top_k > n_experts:
        raise ValueError(f"top_k must be <= n_experts, got top_k={top_k} n_experts={n_experts}")

    # logits: [..., E]
    scores = logits / float(temperature)  # [..., E]
    topk_scores, topk_idx = torch.topk(scores, k=int(top_k), dim=-1)  # [..., K], [..., K]
    topk_probs = torch.softmax(topk_scores, dim=-1)  # [..., K]
    probs = torch.zeros_like(scores)  # [..., E]
    probs.scatter_(dim=-1, index=topk_idx, src=topk_probs)  # [..., E]
    return probs


def sphere_parseval_feature_loss(phi: torch.Tensor) -> torch.Tensor:
    """SPHERE Parseval penalty on the feature Gram matrix.

    Matches the paper-style form used in lm_reward_jax:
      A = (Phi^T Phi) / T
      L = ||A||_F^2 - tr(A)^2 / D

    Args:
        phi: (T, D) feature matrix.
    """
    if phi.dim() != 2:
        raise ValueError(f"phi must be 2D (T, D), got shape={tuple(phi.shape)}")

    # phi: [T, D]
    T, D = phi.shape
    if T <= 0 or D <= 0:
        return phi.new_tensor(0.0)

    phi = phi.to(dtype=torch.float32)
    T_f = float(T)

    if D <= T:
        # A = (Phi^T Phi) / T: [D, D]
        A = (phi.transpose(0, 1) @ phi) / T_f  # [D, D]
        fro2 = torch.sum(A * A)
        tr = torch.trace(A)
    else:
        # K = (Phi Phi^T) / T: [T, T] (shares eigenvalues with A)
        K = (phi @ phi.transpose(0, 1)) / T_f  # [T, T]
        fro2 = torch.sum(K * K)
        tr = torch.trace(K)

    D_f = float(D)
    return fro2 - (tr * tr) / D_f


@dataclass
class SphereMoETrace:
    module: nn.Module
    hidden: torch.Tensor
    gate_logits: Optional[torch.Tensor] = None
    lora_A_outputs: Optional[dict[str, torch.Tensor]] = None


class SphereMoELoRATracer:
    """Collect the last MoE-LoRA router site during a forward pass.

    This is intentionally heuristic: it targets MoE-LoRA layers from VLA-Adapter (`prismatic.util.moe_lora`)
    without importing them directly.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._installed = False
        self._candidates: list[nn.Module] = []
        self.last: Optional[SphereMoETrace] = None
        self._active_module: nn.Module | None = None
        self._active_gate_logits: torch.Tensor | None = None
        self._active_lora_A_outputs: dict[str, torch.Tensor] = {}

    @staticmethod
    def _is_vla_adapter_moe_lora_linear(module: nn.Module) -> bool:
        # VLA-Adapter MoE-LoRA implementation:
        # - module.moe: MoELoRAConfig(num_experts, top_k)
        # - module.gate: nn.Linear(in_features, num_experts, bias=False)
        # - module.lora_A: nn.ModuleDict with per-expert adapters
        moe = getattr(module, "moe", None)
        if moe is None:
            return False
        gate = getattr(module, "gate", None)
        if not isinstance(gate, nn.Linear):
            return False
        lora_A = getattr(module, "lora_A", None)
        if lora_A is None or not hasattr(lora_A, "keys"):
            return False
        return True

    def ensure_installed(self) -> None:
        if self._installed:
            return

        self._candidates = [m for m in self.model.modules() if self._is_vla_adapter_moe_lora_linear(m)]
        for m in self._candidates:
            self._handles.append(m.register_forward_pre_hook(self._pre_hook))
            self._handles.append(m.register_forward_hook(self._post_hook))

            gate = getattr(m, "gate", None)
            if isinstance(gate, nn.Module):
                self._handles.append(gate.register_forward_hook(self._make_gate_hook(parent=m)))

            lora_A = getattr(m, "lora_A", None)
            if lora_A is not None and hasattr(lora_A, "items"):
                for name, A in lora_A.items():
                    if isinstance(A, nn.Module):
                        self._handles.append(A.register_forward_hook(self._make_lora_A_hook(parent=m, expert_name=str(name))))
        self._installed = True

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._installed = False
        self._candidates = []
        self.last = None
        self._active_module = None
        self._active_gate_logits = None
        self._active_lora_A_outputs = {}

    def clear(self) -> None:
        self.last = None
        self._active_module = None
        self._active_gate_logits = None
        self._active_lora_A_outputs = {}

    def _pre_hook(self, module: nn.Module, _inputs: tuple) -> None:
        self._active_module = module
        self._active_gate_logits = None
        self._active_lora_A_outputs = {}

    def _post_hook(self, module: nn.Module, inputs: tuple, _output) -> None:
        if not inputs:
            return
        hidden = inputs[0]
        if not isinstance(hidden, torch.Tensor):
            return
        if self._active_module is module:
            self.last = SphereMoETrace(
                module=module,
                hidden=hidden,
                gate_logits=self._active_gate_logits,
                lora_A_outputs=dict(self._active_lora_A_outputs),
            )
        else:
            self.last = SphereMoETrace(module=module, hidden=hidden)

        self._active_module = None
        self._active_gate_logits = None
        self._active_lora_A_outputs = {}

    def _make_gate_hook(self, *, parent: nn.Module):
        def hook(_gate: nn.Module, _inputs: tuple, output):
            if self._active_module is not parent:
                return
            if isinstance(output, torch.Tensor):
                self._active_gate_logits = output

        return hook

    def _make_lora_A_hook(self, *, parent: nn.Module, expert_name: str):
        def hook(_A: nn.Module, _inputs: tuple, output):
            if self._active_module is not parent:
                return
            if isinstance(output, torch.Tensor):
                self._active_lora_A_outputs[expert_name] = output

        return hook


def _extract_moe_lora_topk_gate_probs_from_logits(
    module: nn.Module,
    gate_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    moe = getattr(module, "moe", None)
    if moe is None:
        raise RuntimeError(f"MoE-LoRA module missing `moe` config: type={type(module).__name__}")

    num_experts = int(getattr(moe, "num_experts", 0) or 0)
    top_k = int(getattr(moe, "top_k", 0) or 0)
    if num_experts <= 0 or top_k <= 0:
        raise RuntimeError(f"Invalid MoE-LoRA config: num_experts={num_experts} top_k={top_k}")

    # gate_logits: [B, S, E] or [T, E]
    if gate_logits.dim() == 3:
        gate_logits = gate_logits.reshape(-1, gate_logits.size(-1))  # [T, E]
    elif gate_logits.dim() == 2:
        # [T, E]
        pass
    else:
        raise ValueError(f"Unexpected gate_logits rank for SPHERE: shape={tuple(gate_logits.shape)}")

    if int(gate_logits.size(-1)) != num_experts:
        raise ValueError(
            f"Gate logits expert dim mismatch: logits_E={int(gate_logits.size(-1))} num_experts={num_experts}"
        )

    return _topk_scatter_softmax_probs(gate_logits.to(torch.float32), top_k=top_k, temperature=temperature)


def _extract_moe_lora_expert_features_A_from_hook(
    module: nn.Module,
    lora_A_outputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return per-expert features captured during the MoE-LoRA forward."""
    active = getattr(module, "active_adapters", None)
    if not isinstance(active, (list, tuple)):
        active = list(lora_A_outputs.keys())

    # lora_A_outputs[name]: [B, S, r] or [T, r]
    feats = []
    for name in active:
        if name not in lora_A_outputs:
            continue
        out = lora_A_outputs[name]
        if out.dim() == 3:
            out = out.reshape(-1, out.size(-1))  # [T, r]
        elif out.dim() == 2:
            # [T, r]
            pass
        else:
            raise ValueError(f"Unexpected lora_A output rank for SPHERE: shape={tuple(out.shape)} expert={name}")
        feats.append(out.to(torch.float32))

    if not feats:
        raise RuntimeError("No MoE-LoRA lora_A outputs captured for SPHERE feature extraction.")

    return torch.stack(feats, dim=1)  # [T, E, r]


def compute_sphere_loss_from_last_moe_lora(
    tracer: SphereMoELoRATracer,
    *,
    token_mask_flat: Optional[torch.Tensor],
    temperature: float,
) -> torch.Tensor:
    if tracer.last is None:
        raise RuntimeError("SPHERE enabled but no MoE-LoRA layer was observed in the forward pass.")

    module = tracer.last.module
    hidden = tracer.last.hidden
    gate_logits = tracer.last.gate_logits
    lora_A_outputs = tracer.last.lora_A_outputs

    if gate_logits is None:
        raise RuntimeError("SPHERE expected MoE-LoRA gate logits to be captured during forward, but got None.")
    if lora_A_outputs is None:
        raise RuntimeError("SPHERE expected MoE-LoRA lora_A outputs to be captured during forward, but got None.")

    # Flatten token axis T (batch*seq) without touching parameters again.
    if hidden.dim() == 3:
        T = int(hidden.shape[0] * hidden.shape[1])
    elif hidden.dim() == 2:
        T = int(hidden.shape[0])
    else:
        raise ValueError(f"Unexpected hidden rank for SPHERE: shape={tuple(hidden.shape)}")

    probs = _extract_moe_lora_topk_gate_probs_from_logits(module, gate_logits, temperature=temperature)  # [T, E]
    expert_features = _extract_moe_lora_expert_features_A_from_hook(module, lora_A_outputs)  # [T, E, r]
    if probs.size(0) != expert_features.size(0):
        raise ValueError(
            f"SPHERE token count mismatch: probs_T={int(probs.size(0))} expert_T={int(expert_features.size(0))}"
        )

    if token_mask_flat is not None:
        token_mask_flat = token_mask_flat.to(device=probs.device)
        if token_mask_flat.numel() != T:
            raise ValueError(f"token_mask_flat size mismatch: mask={int(token_mask_flat.numel())} tokens={T}")
        token_mask_flat = token_mask_flat.to(dtype=torch.bool)
        if not bool(token_mask_flat.any()):
            return probs.new_tensor(0.0)
        probs = probs[token_mask_flat]  # [T', E]
        expert_features = expert_features[token_mask_flat]  # [T', E, r]

    # Φ(token) = concat_e (p_e(token) * a_e(token))
    # probs: [T', E] -> [T', E, 1]
    # expert_features: [T', E, r]
    # phi: [T', E*r]
    phi = (probs.unsqueeze(-1) * expert_features).reshape(expert_features.size(0), -1)  # [T', E*r]
    return sphere_parseval_feature_loss(phi)
