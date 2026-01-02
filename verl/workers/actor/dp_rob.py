# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, log_probs_from_logits_all_rmpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F
from codetiming import Timer
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['RobDataParallelPPOActor']



class RobDataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        print(f'PRM use dynamic bsz={self.config.get("use_dynamic_bsz", False)}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = False #self.ulysses_sequence_parallel_size > 1
        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)
       
    def process_tensor(self, tensor, pad_id):
        mask = tensor != pad_id
        if not torch.all(mask == mask[0:1], dim=1).all():
            raise ValueError("Padding error!")
        base_mask = mask[0]
        valid_len = base_mask.sum().item()
        return tensor[:, base_mask], valid_len
    
    def generate_traj_mask(self, end_step, traj_len):
        """
        Args:
            end_step: (batch_size,), 
            traj_len: 
        Returns:
            mask: (batch_size, traj_len),
        """
        steps = torch.arange(traj_len, device=end_step.device)  # (traj_len,)
        steps_expanded = steps.unsqueeze(0).expand(end_step.size(0), -1)
        mask = steps_expanded < end_step.unsqueeze(1)  # (batch_size, traj_len)
        return mask
    
    def apply_mask_with_grad_control(self, log_probs, entropy, mask):
        """
        Args:
            log_probs: (batch_size, traj_len, ...)
            entropy:   (batch_size, traj_len, ...)
            mask:      (batch_size, traj_len)
        Returns:
            log_probs_masked: 
            entropy_masked:   
        """
        mask_expanded = mask.unsqueeze(-1)  

        log_probs_masked = torch.where(
            mask_expanded,
            log_probs,
            torch.zeros_like(log_probs, requires_grad=False)  
        )

        entropy_masked = torch.where(
            mask_expanded,
            entropy,
            torch.zeros_like(entropy, requires_grad=False)   
        )

        return log_probs_masked, entropy_masked

    def _build_vla_adapter_token_action_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Build action-prediction inputs for VLA-Adapter token models.

        We keep prompt tokens as a prefix, then insert `NUM_TOKENS` placeholder tokens and a STOP token. We also
        construct `labels` so that `_process_action_masks(labels)` marks the placeholder positions as action tokens.

        Returns:
            action_input_ids: (B, L')
            action_attention_mask: (B, L')
            labels: (B, L')
            prompt_lens: (B,)
        """
        from prismatic.vla.constants import ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX, NUM_TOKENS, STOP_INDEX

        if self.pad_token_id is None:
            raise ValueError("pad_token_id is not set; expected it in data.meta_info['pad_token_id'].")

        device = input_ids.device
        batch_size = input_ids.size(0)

        # Right-padded => prompt is a prefix.
        prompt_lens = attention_mask.to(torch.long).sum(dim=-1)
        max_prompt_len = int(prompt_lens.max().item())

        total_len = max_prompt_len + int(NUM_TOKENS) + 1
        action_input_ids = torch.full((batch_size, total_len), self.pad_token_id, dtype=input_ids.dtype, device=device)
        action_attention_mask = torch.zeros((batch_size, total_len), dtype=attention_mask.dtype, device=device)
        labels = torch.full((batch_size, total_len), IGNORE_INDEX, dtype=torch.long, device=device)

        placeholder_token_id = torch.tensor(1, dtype=input_ids.dtype, device=device)
        arbitrary_action_token_id = torch.tensor(ACTION_TOKEN_BEGIN_IDX + 1, dtype=torch.long, device=device)

        for i in range(batch_size):
            plen = int(prompt_lens[i].item())
            if plen > 0:
                action_input_ids[i, :plen] = input_ids[i, :plen]
            action_input_ids[i, plen:plen + int(NUM_TOKENS)] = placeholder_token_id
            action_input_ids[i, plen + int(NUM_TOKENS)] = STOP_INDEX

            action_attention_mask[i, : plen + int(NUM_TOKENS) + 1] = 1

            labels[i, plen:plen + int(NUM_TOKENS)] = arbitrary_action_token_id
            labels[i, plen + int(NUM_TOKENS)] = STOP_INDEX

        return action_input_ids, action_attention_mask, labels, prompt_lens

    def _extract_vla_adapter_token_logits(self, full_logits: torch.Tensor, prompt_lens: torch.Tensor, temperature: float):
        """
        Extract per-token logits for the (action_chunks_len * action_token_len) action tokens, and restrict to the
        256-bin action-token vocabulary range.

        Returns:
            action_logits: (B, action_token_count, 256)
            start_token: int  # token id base for the 256-bin slice
        """
        batch_size = full_logits.size(0)
        device = full_logits.device
        action_token_count = int(self.config.action_token_len * self.config.action_chunks_len)

        num_patches = int(self.actor_module.vision_backbone.get_num_patches() * self.actor_module.vision_backbone.get_num_images_in_input())
        start_positions = num_patches + prompt_lens.to(torch.long) - 1  # (B,)
        positions = start_positions[:, None] + torch.arange(action_token_count, device=device)[None, :]
        token_logits = full_logits[torch.arange(batch_size, device=device)[:, None], positions]  # (B, 56, vocab)

        action_vocab_size = int(self.actor_module.action_vocab_size)
        start_token = action_vocab_size - 256
        action_logits = token_logits[..., start_token:action_vocab_size]

        if temperature <= 0:
            temperature = 1.0
        action_logits = action_logits / float(temperature)

        return action_logits, start_token

    def _forward_vla_adapter_token_action_logits(
        self,
        action_input_ids: torch.Tensor,
        action_attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: torch.Tensor,
        prompt_lens: torch.Tensor,
        temperature: float,
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute action-token logits for VLA-Adapter token models without materializing full-vocab logits.

        Returns:
            action_logits: (B, action_token_count, 256)
            start_token: int  # token id base for the 256-bin slice
        """
        module = self.actor_module
        action_logits, start_token = module(
            action_input_ids=action_input_ids,
            action_attention_mask=action_attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            prompt_lens=prompt_lens,
            temperature=temperature,
        )
        return action_logits, int(start_token)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        micro_batch:
        
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        
        batch_size = micro_batch['responses'].size(0)
        traj_len = micro_batch['responses'].size(1)
        tot_pad_len = micro_batch['input_ids'].size(2)
        
        assert all(micro_batch[key].size(0) == batch_size for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(1) == traj_len for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(2) == tot_pad_len for key in [ 'input_ids', 'attention_mask'])
        if self.config.use_proprio:
            assert micro_batch["proprio"].size(0) == batch_size and micro_batch["proprio"].size(1) == traj_len and micro_batch["proprio"].size(2) == self.config.action_token_len
            
        response_length = micro_batch['responses'].size(-1) # 7*8
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids_3d = micro_batch['input_ids']
            attention_mask_3d = micro_batch['attention_mask']
            pixel_values_3d = micro_batch["pixel_values"]
            responses_3d = micro_batch["responses"]

            if self.config.use_proprio:
                proprio_3d = micro_batch["proprio"]
            else:
                proprio_3d = None

            if self.config.vla == "openvla-oft":
                input_ids = input_ids_3d.reshape((batch_size * traj_len,) + input_ids_3d.shape[2:])
                attention_mask = attention_mask_3d.reshape((batch_size * traj_len,) + attention_mask_3d.shape[2:])
                pixel_values = pixel_values_3d.reshape((batch_size * traj_len,) + pixel_values_3d.shape[2:])
                responses = responses_3d.reshape((batch_size * traj_len,) + responses_3d.shape[2:])
                proprio = proprio_3d.reshape((batch_size * traj_len,) + proprio_3d.shape[2:]) if proprio_3d is not None else None

                input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
                attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)
                logits = self.actor_module(input_ids=input_ids_unpad,
                                        attention_mask=attention_mask_unpad,
                                        pixel_values=pixel_values,
                                        proprio=proprio,
                                        )  # prevent model thinks we are generating
                
                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256 
                logits = logits[..., -256-64:-64]  # Shape: [batch_size, seq_len, 256]
                responses = responses - start_index
                #assert (0<=responses<=255).all()
            
                logits = logits.div(temperature) 
                
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
            
                assert len(log_probs.shape)==2 and len(entropy.shape)==2 
                log_probs = log_probs.reshape((batch_size, traj_len*self.config.action_chunks_len,self.config.action_token_len) ) #*
                entropy = entropy.reshape((batch_size, traj_len*self.config.action_chunks_len,self.config.action_token_len) )

                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len*self.config.action_chunks_len) #, self.config.action_token_len
                log_probs, entropy = self.apply_mask_with_grad_control(log_probs, entropy, mask)
                
                log_probs = log_probs.reshape((batch_size, traj_len*response_length))
                entropy = entropy.reshape((batch_size, traj_len*response_length)) 
                
            elif self.config.vla == "openvla":
                input_ids = input_ids_3d.reshape((batch_size * traj_len,) + input_ids_3d.shape[2:])
                attention_mask = attention_mask_3d.reshape((batch_size * traj_len,) + attention_mask_3d.shape[2:])
                pixel_values = pixel_values_3d.reshape((batch_size * traj_len,) + pixel_values_3d.shape[2:])
                responses = responses_3d.reshape((batch_size * traj_len,) + responses_3d.shape[2:])

                input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
                attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)
                output = self.actor_module(input_ids=input_ids_unpad,
                                    attention_mask=attention_mask_unpad,
                                    pixel_values=pixel_values,
                                    use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                logits = logits.div(temperature) 
                
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                #ADD
                
                log_probs = log_probs.reshape((batch_size, traj_len,) + log_probs.shape[1:])
                entropy = entropy.reshape((batch_size, traj_len,) + entropy.shape[1:])

                
                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len)
                log_probs, entropy = self.apply_mask_with_grad_control(log_probs, entropy, mask)
                
                log_probs = log_probs.reshape((batch_size, traj_len*response_length))
                entropy = entropy.reshape((batch_size, traj_len*response_length))
                
                
            elif self.config.vla == "vla-adapter-token":
                # VLA-Adapter token models have very large vocab (Qwen); avoid flattening all traj steps at once.
                # Chunk over trajectory dimension to prevent allocating (B*T, seq, vocab) logits tensors.
                traj_chunk = int(getattr(self.config, "traj_mini_batch_size", 1) or 1)
                traj_chunk = max(1, min(traj_len, traj_chunk))

                log_probs_parts = []
                entropy_parts = []
                for t0 in range(0, traj_len, traj_chunk):
                    t1 = min(traj_len, t0 + traj_chunk)
                    chunk_len = t1 - t0

                    input_ids = input_ids_3d[:, t0:t1, :].reshape((batch_size * chunk_len,) + input_ids_3d.shape[2:])
                    attention_mask = attention_mask_3d[:, t0:t1, :].reshape((batch_size * chunk_len,) + attention_mask_3d.shape[2:])
                    pixel_values = pixel_values_3d[:, t0:t1, ...].reshape((batch_size * chunk_len,) + pixel_values_3d.shape[2:])
                    responses = responses_3d[:, t0:t1, :].reshape((batch_size * chunk_len,) + responses_3d.shape[2:])

                    action_input_ids, action_attention_mask, labels, prompt_lens = self._build_vla_adapter_token_action_inputs(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    action_logits, start_token = self._forward_vla_adapter_token_action_logits(
                        action_input_ids=action_input_ids,
                        action_attention_mask=action_attention_mask,
                        pixel_values=pixel_values,
                        labels=labels,
                        prompt_lens=prompt_lens,
                        temperature=temperature,
                    )
                    token_offsets = (responses.to(torch.long) - start_token).clamp(min=0, max=255)
                    chunk_log_probs = logprobs_from_logits(action_logits, token_offsets)  # (B*chunk, 56)
                    chunk_entropy = verl_F.entropy_from_logits(action_logits)  # (B*chunk, 56)

                    chunk_log_probs = chunk_log_probs.reshape(batch_size, chunk_len, -1)
                    chunk_entropy = chunk_entropy.reshape(batch_size, chunk_len, -1)
                    log_probs_parts.append(chunk_log_probs)
                    entropy_parts.append(chunk_entropy)

                log_probs = torch.cat(log_probs_parts, dim=1)  # (B, traj_len, 56)
                entropy = torch.cat(entropy_parts, dim=1)  # (B, traj_len, 56)

                log_probs = log_probs.reshape((batch_size, traj_len * self.config.action_chunks_len, self.config.action_token_len))
                entropy = entropy.reshape((batch_size, traj_len * self.config.action_chunks_len, self.config.action_token_len))
                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len*self.config.action_chunks_len)
                log_probs, entropy = self.apply_mask_with_grad_control(log_probs, entropy, mask)

                log_probs = log_probs.reshape((batch_size, traj_len*response_length))
                entropy = entropy.reshape((batch_size, traj_len*response_length))

            return entropy, log_probs
    
    def _forward_micro_batch_update(self, input_ids, attention_mask, pixel_values, responses, temperature, proprio) -> Tuple[torch.Tensor, torch.Tensor]:
       
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if self.config.vla == "openvla-oft":
                
                input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
                attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)

                
                logits = self.actor_module(input_ids=input_ids_unpad,
                                                attention_mask=attention_mask_unpad,
                                                pixel_values=pixel_values,
                                                proprio=proprio,
                                                )  
                
                assert logits.requires_grad 
                
                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256 
                logits = logits[..., -256-64:-64]  # Shape: [batch_size, seq_len, 256]
                responses = responses - start_index
                
                logits = logits.div(temperature) 
                
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                
                log_probs = log_probs.reshape((1, -1))
                entropy = entropy.reshape((1, -1))
                
                return entropy, log_probs
            
            elif self.config.vla == "openvla":
                response_length = responses.size(-1)
                input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
                attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)
                output = self.actor_module(input_ids=input_ids_unpad,
                                        attention_mask=attention_mask_unpad,
                                        pixel_values=pixel_values,
                                        use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                #
                
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                logits = logits.div(temperature) 
                
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                
                
                log_probs = log_probs.reshape((1, -1))
                entropy = entropy.reshape((1, -1))

                return entropy, log_probs
            
            elif self.config.vla == "vla-adapter-token":
                # Chunk to avoid OOM when (N, seq, vocab) becomes too large.
                n = input_ids.size(0)
                traj_chunk = int(getattr(self.config, "traj_mini_batch_size", 1) or 1)
                traj_chunk = max(1, min(n, traj_chunk))

                log_probs_parts = []
                entropy_parts = []
                for s0 in range(0, n, traj_chunk):
                    s1 = min(n, s0 + traj_chunk)
                    ids = input_ids[s0:s1]
                    mask = attention_mask[s0:s1]
                    pix = pixel_values[s0:s1]
                    resp = responses[s0:s1]

                    action_input_ids, action_attention_mask, labels, prompt_lens = self._build_vla_adapter_token_action_inputs(
                        input_ids=ids, attention_mask=mask
                    )
                    action_logits, start_token = self._forward_vla_adapter_token_action_logits(
                        action_input_ids=action_input_ids,
                        action_attention_mask=action_attention_mask,
                        pixel_values=pix,
                        labels=labels,
                        prompt_lens=prompt_lens,
                        temperature=temperature,
                    )
                    token_offsets = (resp.to(torch.long) - start_token).clamp(min=0, max=255)
                    chunk_log_probs = logprobs_from_logits(action_logits, token_offsets)  # (chunk, 56)
                    chunk_entropy = verl_F.entropy_from_logits(action_logits)  # (chunk, 56)
                    log_probs_parts.append(chunk_log_probs)
                    entropy_parts.append(chunk_entropy)

                log_probs = torch.cat(log_probs_parts, dim=0).reshape((1, -1))
                entropy = torch.cat(entropy_parts, dim=0).reshape((1, -1))
                return entropy, log_probs
                

    def _forward_micro_batch_entropy(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = micro_batch['responses'].size(0)
        traj_len = micro_batch['responses'].size(1)
        tot_pad_len = micro_batch['input_ids'].size(2)
 
        assert all(micro_batch[key].size(0) == batch_size for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(1) == traj_len for key in ['responses', 'input_ids', 'attention_mask', 'pixel_values'])
        assert all(micro_batch[key].size(2) == tot_pad_len for key in [ 'input_ids', 'attention_mask'])
            
        if self.config.use_proprio:
            assert micro_batch["proprio"].size(0) == batch_size and micro_batch["proprio"].size(1) == traj_len and micro_batch["proprio"].size(2) == self.config.action_token_len
            
        response_length = micro_batch['responses'].size(-1)
        #assert response_length == 7*8
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            #batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            pixel_values = micro_batch["pixel_values"]
            
            input_ids = input_ids.reshape((batch_size * traj_len,) + input_ids.shape[2:])
            attention_mask = attention_mask.reshape((batch_size * traj_len,) + attention_mask.shape[2:])
            pixel_values = pixel_values.reshape((batch_size * traj_len,) + pixel_values.shape[2:])
            
            if self.config.use_proprio:
                proprio = micro_batch["proprio"]
                proprio = proprio.reshape((batch_size * traj_len,) + proprio.shape[2:])
            else:
                proprio = None
            
            if  self.config.vla == "openvla-oft":
                input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
                attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)
            
                logits = self.actor_module(input_ids=input_ids_unpad,
                                                attention_mask=attention_mask_unpad,
                                                pixel_values=pixel_values,
                                                proprio=proprio,
                                                ) 
            
                assert self.actor_module.vocab_size == 32000
                start_index = self.actor_module.vocab_size - 256 
                logits = logits[..., -256-64:-64]  # Shape: [batch_size, seq_len, 256]
            
                logits = logits.div(temperature) 
            
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

                assert len(entropy.shape)==2 
                entropy = entropy.reshape((batch_size, traj_len*self.config.action_chunks_len, self.config.action_token_len) ) 
                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len*self.config.action_chunks_len) 
                _, entropy = self.apply_mask_with_grad_control(entropy, entropy, mask)
                entropy = entropy.reshape((batch_size, traj_len*response_length))
                return entropy
            
            elif self.config.vla == "openvla":
                input_ids_unpad, _ = self.process_tensor(input_ids, self.pad_token_id)
                attention_mask_unpad, _ = self.process_tensor(attention_mask, 0)
                output = self.actor_module(input_ids=input_ids_unpad,
                                        attention_mask=attention_mask_unpad,
                                        pixel_values=pixel_values,
                                        use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                #
                
                
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                logits = logits.div(temperature) 
                
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                #ADD

                entropy = entropy.reshape((batch_size, traj_len,) + entropy.shape[1:])
                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len)
                _, entropy = self.apply_mask_with_grad_control(entropy, entropy, mask)
                entropy = entropy.reshape((batch_size, traj_len*response_length))
                return entropy
            
            elif self.config.vla == "vla-adapter-token":
                traj_chunk = int(getattr(self.config, "traj_mini_batch_size", 1) or 1)
                traj_chunk = max(1, min(traj_len, traj_chunk))

                entropy_parts = []
                for t0 in range(0, traj_len, traj_chunk):
                    t1 = min(traj_len, t0 + traj_chunk)
                    chunk_len = t1 - t0

                    input_ids = micro_batch['input_ids'][:, t0:t1, :].reshape((batch_size * chunk_len,) + micro_batch['input_ids'].shape[2:])
                    attention_mask = micro_batch['attention_mask'][:, t0:t1, :].reshape((batch_size * chunk_len,) + micro_batch['attention_mask'].shape[2:])
                    pixel_values = micro_batch["pixel_values"][:, t0:t1, ...].reshape((batch_size * chunk_len,) + micro_batch["pixel_values"].shape[2:])

                    action_input_ids, action_attention_mask, labels, prompt_lens = self._build_vla_adapter_token_action_inputs(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    action_logits, _ = self._forward_vla_adapter_token_action_logits(
                        action_input_ids=action_input_ids,
                        action_attention_mask=action_attention_mask,
                        pixel_values=pixel_values,
                        labels=labels,
                        prompt_lens=prompt_lens,
                        temperature=temperature,
                    )
                    chunk_entropy = verl_F.entropy_from_logits(action_logits)  # (B*chunk, 56)
                    chunk_entropy = chunk_entropy.reshape(batch_size, chunk_len, -1)
                    entropy_parts.append(chunk_entropy)

                entropy = torch.cat(entropy_parts, dim=1)  # (B, traj_len, 56)
                entropy = entropy.reshape((batch_size, traj_len * self.config.action_chunks_len, self.config.action_token_len))
                mask = self.generate_traj_mask(micro_batch['finish_step'], traj_len*self.config.action_chunks_len)
                _, entropy = self.apply_mask_with_grad_control(entropy, entropy, mask)
                entropy = entropy.reshape((batch_size, traj_len*response_length))
                return entropy


    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size'] #256
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error # 1
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz'] #trues
        self.pad_token_id = data.meta_info['pad_token_id']
        
        select_keys = ['responses', 'input_ids', 'attention_mask', 'pixel_values',"finish_step"]
        if self.config.use_proprio:
            select_keys.append("proprio")
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        self.pad_token_id = data.meta_info.get('pad_token_id', getattr(self, 'pad_token_id', None))

        select_keys = ['responses', 'input_ids', 'attention_mask', 'pixel_values', 'old_log_probs', 'advantages',"finish_step"]
        if self.config.use_proprio:
            select_keys.append("proprio")
        batch = data.select(batch_keys=select_keys).batch
        assert self.config.ppo_micro_batch_size == 1

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)
        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad(set_to_none=True)

            for test_idx, data in enumerate(micro_batches):
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                
                response_length = responses.size(1) *  responses.size(2)
                finish_step = data['finish_step'] * self.config.action_token_len
                steps = torch.arange(response_length, device=data['responses'].device)  # (traj_len,)
                steps_expanded = steps.unsqueeze(0).expand(data['responses'].size(0), -1)
                response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)
                
                response_mask_sum = response_mask.sum(axis=None)

                old_log_prob = data['old_log_probs']
                advantages = data['advantages']
                
                #clip_ratio = self.config.clip_ratio
                clip_ratio_high = self.config.clip_ratio_high
                clip_ratio_low = self.config.clip_ratio_low
                entropy_coeff = self.config.entropy_coeff

                batch_size = data['responses'].size(0)
                traj_len = data['responses'].size(1)
                tot_pad_len = data['input_ids'].size(2)
                
                
                input_ids = data['input_ids']
                attention_mask = data['attention_mask']
                pixel_values = data["pixel_values"]
                responses = data["responses"]
                
                
                input_ids = input_ids.reshape((batch_size * traj_len,) + input_ids.shape[2:])
                attention_mask = attention_mask.reshape((batch_size * traj_len,) + attention_mask.shape[2:])
                pixel_values = pixel_values.reshape((batch_size * traj_len,) + pixel_values.shape[2:])
                responses = responses.reshape((batch_size * traj_len,) + responses.shape[2:])
                
                if self.config.use_proprio:
                    proprio = data["proprio"]
                    proprio = proprio.reshape((batch_size * traj_len,) + proprio.shape[2:])
                else:
                    proprio = None
                
                
                loss_info = {
                    #'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss':0,
                    'actor/pg_clipfrac': 0,
                    'actor/ppo_kl': 0,
                }
                
                assert traj_len % self.config.traj_mini_batch_size ==0
                traj_split_num = int(traj_len/self.config.traj_mini_batch_size)
                

                for i in range(0, traj_len, int(traj_len/traj_split_num)):
                   
                    entropy, log_prob = self._forward_micro_batch_update(input_ids=input_ids[i:i+int(traj_len/traj_split_num)], 
                                                                         attention_mask=attention_mask[i:i+int(traj_len/traj_split_num)], 
                                                                         pixel_values=pixel_values[i:i+int(traj_len/traj_split_num)], 
                                                                         responses=responses[i:i+int(traj_len/traj_split_num)], 
                                                                         temperature=temperature,
                                                                         proprio=proprio[i:i+int(traj_len/traj_split_num)] if proprio is not None  else None)
                    
                    slice_id = i*self.config.action_token_len*self.config.action_chunks_len
                    next_slice_id = (i+int(traj_len/traj_split_num))*self.config.action_token_len*self.config.action_chunks_len
                    old_log_prob_tmp = old_log_prob[:, slice_id: next_slice_id]
                    advantages_tmp = advantages[:, slice_id: next_slice_id]
                    response_mask_tmp = response_mask[:, slice_id: next_slice_id]
                        
                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob_tmp,
                                                                            log_prob=log_prob,
                                                                            advantages=advantages_tmp,
                                                                            eos_mask=response_mask_tmp,
                                                                            clip_ratio_high=clip_ratio_high,
                                                                            clip_ratio_low=clip_ratio_low)
                    
                    response_mask_tmp_sum = response_mask_tmp.sum(axis=None)
                    pg_loss = pg_loss* response_mask_tmp_sum
                    pg_clipfrac = pg_clipfrac* response_mask_tmp_sum / response_mask_sum
                    ppo_kl = ppo_kl* response_mask_tmp_sum / response_mask_sum
                    
                    policy_loss = pg_loss / response_mask_sum
                    
                    loss = policy_loss / self.gradient_accumulation
                    
                    loss.backward()
                    
                    loss_info['actor/pg_loss'] =  loss_info['actor/pg_loss'] + policy_loss.detach().item()
                    loss_info['actor/pg_clipfrac'] = loss_info['actor/pg_clipfrac'] + pg_clipfrac.detach().item()
                    loss_info['actor/ppo_kl'] = loss_info['actor/ppo_kl'] +  ppo_kl.detach().item()

                append_to_dict(metrics, loss_info)
               
            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
            torch.cuda.empty_cache()
        self.actor_optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return metrics

    
    def compute_entropy(self, bacth_data: DataProto):
        
        if bacth_data.meta_info['train_mode'] ==True:
            self.actor_module.train()
            print("train mode")
        else:
            self.actor_module.eval()
            print("eval mode")

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = bacth_data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'pixel_values', "finish_step"]
        if self.config.use_proprio:
            select_keys.append("proprio")
        batch = bacth_data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)
        print("dataloader_length:", len(dataloader))
        
        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1) *  responses.size(2)
                finish_step = data['finish_step'] * self.config.action_token_len
                steps = torch.arange(response_length, device=data['responses'].device)  # (traj_len,)
                steps_expanded = steps.unsqueeze(0).expand(data['responses'].size(0), -1)
                response_mask = steps_expanded < finish_step.unsqueeze(1)  # (batch_size, traj_len)
                

                with torch.no_grad():
                    entropy = self._forward_micro_batch_entropy(micro_batch=data, temperature=temperature)
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                if bacth_data.meta_info['is_filtered'] and bacth_data.meta_info['train_mode']:
                    data = {
                        'actor_after/entropy_loss_train': entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                elif bacth_data.meta_info['is_filtered'] and not bacth_data.meta_info['train_mode']:
                    data = {
                        'actor_after/entropy_loss_eval': entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                elif not bacth_data.meta_info['is_filtered'] and bacth_data.meta_info['train_mode']:
                    data = {
                        'actor_before/entropy_loss_train': entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                elif not bacth_data.meta_info['is_filtered'] and not bacth_data.meta_info['train_mode']:
                    data = {
                        'actor_before/entropy_loss_eval': entropy_loss.detach().item(),
                    }
                    append_to_dict(metrics, data)
                        
                
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return metrics
