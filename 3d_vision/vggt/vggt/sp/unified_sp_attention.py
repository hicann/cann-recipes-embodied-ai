# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch_npu
import torch.distributed as dist


@dataclass
class SPConfig:
    ulysses_degree: int = 1
    ring_degree: int = 1
    use_ring_overlap: bool = True
    
    @property
    def sp_degree(self):
        return self.ulysses_degree * self.ring_degree


@dataclass
class AttentionParams:
    """Encapsulates attention computation parameters."""
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    causal: bool = False
    softmax_scale: Optional[float] = None
    dropout_p: float = 0.0
    attn_mask: Optional[torch.Tensor] = None


@dataclass
class NPUAttentionConfig:
    """NPU attention API parameters configuration."""
    num_heads: int
    scale: float
    input_layout: str = "BNSD"
    num_key_value_heads: int = 0
    pre_tokens: int = 65535
    next_tokens: int = 65535
    sparse_mode: int = 0
    inner_precise: int = 0
    softmax_lse_flag: bool = False


class UnifiedSPAttention:
    """
    Unified sequence parallel attention using NPU fused attention.
    
    Features:
    1. Uses torch_npu.npu_fused_infer_attention_score
    2. Supports LSE (Log-Sum-Exp) for Ring Overlap
    3. NPU hardware acceleration
    """
    
    def __init__(
        self,
        ulysses_group: dist.ProcessGroup,
        ring_group: dist.ProcessGroup,
        use_ring_overlap: bool = True,
    ):
        self.ulysses_pg = ulysses_group
        self.ring_pg = ring_group
        self.use_ring_overlap = use_ring_overlap
        
        self.ulysses_world_size = dist.get_world_size(ulysses_group) if ulysses_group else 1
        self.ring_world_size = dist.get_world_size(ring_group) if ring_group else 1
        
        self.ulysses_rank = dist.get_rank(ulysses_group) if self.ulysses_world_size > 1 else 0
        self.ring_rank = dist.get_rank(ring_group) if self.ring_world_size > 1 else 0
        
        self._other_indices_cache = None
        self._cached_rank = None
        self._cached_ring_size = None

    def forward(
        self,
        attention_params: AttentionParams,
        **kwargs
    ) -> torch.Tensor:
        """
        Unified forward interface.
        
        Args:
            attention_params: Encapsulated attention parameters (AttentionParams)
            **kwargs: Additional keyword arguments for internal implementation
        
        Returns:
            output: Tensor in [B, S, num_heads, head_dim] format
        """
        return self._forward_impl(attention_params, **kwargs)
    
    def _forward_impl(
        self,
        params: AttentionParams,
        **kwargs
    ) -> torch.Tensor:
        """Internal forward implementation."""
        return self._forward_core(params, **kwargs)
    
    def _forward_core(
        self,
        params: AttentionParams,
        **kwargs
    ) -> torch.Tensor:
        """
        Core forward logic.
        
        Input: Each rank has [batch_size, shard_s, num_heads, head_dim]
        Output: Each rank returns [batch_size, shard_s, num_heads, head_dim]
        """
        q, k, v = params.q, params.k, params.v

        if self.ulysses_world_size > 1:
            num_heads = q.shape[2]
            if num_heads % self.ulysses_world_size != 0:
                raise ValueError(
                    f"Number of heads ({num_heads}) must be divisible by "
                    f"ulysses_world_size ({self.ulysses_world_size})"
                )
            
            q = self._all_to_all_head_to_seq(q)
            k = self._all_to_all_head_to_seq(k)
            v = self._all_to_all_head_to_seq(v)

        updated_params = AttentionParams(
            q=q, k=k, v=v,
            causal=params.causal,
            softmax_scale=params.softmax_scale,
            dropout_p=params.dropout_p,
            attn_mask=params.attn_mask
        )
        
        if self.ring_world_size > 1:
            if self.use_ring_overlap:
                out = self._ring_attention_overlap(updated_params, **kwargs)
            else:
                out = self._ring_attention_allgather(updated_params, **kwargs)
        else:
            out = self._compute_attention(updated_params)

        if self.ulysses_world_size > 1:
            out = self._all_to_all_seq_to_head(out)

        return out
    
    def _get_other_indices(self, rank, ring_size, device):
        rank_changed = self._cached_rank != rank
        ring_size_changed = self._cached_ring_size != ring_size
        cache_missing = self._other_indices_cache is None
        device_changed = (self._other_indices_cache is not None and 
                        self._other_indices_cache.device != device)
        
        needs_update = (rank_changed or ring_size_changed or 
                        cache_missing or device_changed)
        
        if needs_update:
            indices_list = [i for i in range(ring_size) if i != rank]
            self._other_indices_cache = torch.tensor(
                indices_list, dtype=torch.long, device=device
            )
            self._cached_rank = rank
            self._cached_ring_size = ring_size
        
        return self._other_indices_cache

    def _all_to_all_head_to_seq(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Ulysses all-to-all: head dimension split -> sequence dimension split.
        
        Input: [batch_size, shard_s, num_heads, head_dim]
        Output: [batch_size, seq_len, shard_hc, head_dim]
        """
        if self.ulysses_world_size == 1:
            return input_
        
        batch_size, shard_s, hc, head_dim = input_.shape
        world_size = self.ulysses_world_size
        
        seq_len = shard_s * world_size
        shard_hc = hc // world_size
        
        input_t = input_.reshape(batch_size, shard_s, world_size, shard_hc, head_dim)
        input_t = input_t.permute(2, 1, 0, 3, 4).contiguous()
        
        torch.npu.synchronize()
        
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=self.ulysses_pg)
        
        output = output.reshape(seq_len, batch_size, shard_hc, head_dim)
        output = output.permute(1, 0, 2, 3).contiguous()
        
        return output
    
    def _all_to_all_seq_to_head(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Ulysses all-to-all reverse: sequence dimension split -> head dimension split.
        
        Input: [batch_size, seq_len, shard_hc, head_dim]
        Output: [batch_size, shard_s, num_heads, head_dim]
        """
        if self.ulysses_world_size == 1:
            return input_
        
        batch_size, seq_len, shard_hc, head_dim = input_.shape
        world_size = self.ulysses_world_size
        
        hc = shard_hc * world_size
        shard_s = seq_len // world_size
        
        input_t = input_.reshape(batch_size, world_size, shard_s, shard_hc, head_dim)
        input_t = input_t.permute(1, 3, 2, 0, 4).contiguous()
        
        torch.npu.synchronize()
        
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=self.ulysses_pg)
        
        output = output.reshape(hc, shard_s, batch_size, head_dim)
        output = output.permute(2, 1, 0, 3).contiguous()
        
        return output
    
    def _ring_attention_overlap(
        self,
        params: AttentionParams,
        **kwargs
    ) -> torch.Tensor:
        """Ring Attention with overlap using NPU fused attention."""
        q, k, v = params.q, params.k, params.v
        batch_size, seq_len_local, num_heads, head_dim = k.shape
        device = k.device
        
        k_gathered = torch.empty(
            [self.ring_world_size, batch_size, seq_len_local, num_heads, head_dim],
            dtype=k.dtype, device=device
        )
        v_gathered = torch.empty(
            [self.ring_world_size, batch_size, seq_len_local, num_heads, head_dim],
            dtype=v.dtype, device=device
        )
        
        k_handle = dist.all_gather_into_tensor(
            k_gathered, k.contiguous(), 
            group=self.ring_pg, async_op=True
        )
        v_handle = dist.all_gather_into_tensor(
            v_gathered, v.contiguous(), 
            group=self.ring_pg, async_op=True
        )
        
        local_params = AttentionParams(
            q=q, k=k, v=v,
            causal=params.causal,
            softmax_scale=params.softmax_scale,
            dropout_p=params.dropout_p,
            attn_mask=params.attn_mask
        )
        out_local, lse_local = self._compute_attention_with_lse(local_params)
        
        k_handle.wait()
        v_handle.wait()
        
        other_indices = self._get_other_indices(
            self.ring_rank, self.ring_world_size, device
        )
        
        k_others = torch.index_select(k_gathered, dim=0, index=other_indices)
        v_others = torch.index_select(v_gathered, dim=0, index=other_indices)
        
        k_others = k_others.permute(1, 0, 2, 3, 4).contiguous().reshape(batch_size, -1, num_heads, head_dim)
        v_others = v_others.permute(1, 0, 2, 3, 4).contiguous().reshape(batch_size, -1, num_heads, head_dim)
        
        cross_params = AttentionParams(
            q=q, k=k_others, v=v_others,
            causal=False,
            softmax_scale=params.softmax_scale,
            dropout_p=params.dropout_p,
            attn_mask=None
        )
        out_others, lse_others = self._compute_attention_with_lse(cross_params)
        
        out_merged = self._merge_two_outputs(
            out_local, lse_local,
            out_others, lse_others
        )
        
        return out_merged
    
    def _ring_attention_allgather(
        self,
        params: AttentionParams,
        **kwargs
    ) -> torch.Tensor:
        """Ring Attention without overlap."""
        q, k, v = params.q, params.k, params.v
        batch_size, seq_len_local, num_heads, head_dim = k.shape
        
        q_gathered = torch.empty(
            batch_size, self.ring_world_size * seq_len_local, num_heads, head_dim,
            dtype=q.dtype, device=q.device
        )
        k_gathered = torch.empty(
            batch_size, self.ring_world_size * seq_len_local, num_heads, head_dim,
            dtype=k.dtype, device=k.device
        )
        v_gathered = torch.empty(
            batch_size, self.ring_world_size * seq_len_local, num_heads, head_dim,
            dtype=v.dtype, device=v.device
        )
        
        dist.all_gather_into_tensor(q_gathered, q.contiguous(), group=self.ring_pg)
        dist.all_gather_into_tensor(k_gathered, k.contiguous(), group=self.ring_pg)
        dist.all_gather_into_tensor(v_gathered, v.contiguous(), group=self.ring_pg)
        
        params_full = AttentionParams(
            q=q_gathered,
            k=k_gathered,
            v=v_gathered,
            causal=params.causal,
            softmax_scale=params.softmax_scale,
            dropout_p=params.dropout_p,
            attn_mask=params.attn_mask
        )
        out = self._compute_attention(params_full)
        
        out = out.reshape(batch_size, self.ring_world_size, seq_len_local, num_heads, head_dim)
        out = out[:, self.ring_rank, :, :, :].contiguous()
        
        return out
    
    @staticmethod
    def _compute_attention(
        params: AttentionParams
    ) -> torch.Tensor:
        """
        Core attention using torch_npu.npu_fused_infer_attention_score.
        
        Args:
            params: AttentionParams object
        
        Returns:
            output: [batch_size, seq_len, num_heads, head_dim]
        """
        
        q, k, v = params.q, params.k, params.v
        batch_size, seq_len, num_heads, head_dim = q.shape
        out_dtype = q.dtype
        
        scale = params.softmax_scale
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)
        
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        
        num_key_value_heads = k.shape[1]
        if num_heads == num_key_value_heads:
            num_key_value_heads = 0
        
        config = NPUAttentionConfig(
            num_heads=num_heads,
            scale=float(scale),
            num_key_value_heads=num_key_value_heads,
            next_tokens=65535 if not params.causal else 0,
            softmax_lse_flag=False,
        )
        
        out = torch_npu.npu_fused_infer_attention_score(
            q, k, v,
            num_heads=config.num_heads,
            scale=config.scale,
            input_layout=config.input_layout,
            num_key_value_heads=config.num_key_value_heads,
            pre_tokens=config.pre_tokens,
            next_tokens=config.next_tokens,
            sparse_mode=config.sparse_mode,
            inner_precise=config.inner_precise,
        )[0]
        
        out = out.transpose(1, 2).contiguous()
        
        return out.to(out_dtype)
    
    @staticmethod
    def _compute_attention_with_lse(
        params: AttentionParams
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Attention with LSE using torch_npu.npu_fused_infer_attention_score.
        
        Returns:
            output: [batch_size, seq_len, num_heads, head_dim]
            lse: [batch_size, num_heads, seq_len]
        """
        
        q, k, v = params.q, params.k, params.v
        batch_size, seq_len, num_heads, head_dim = q.shape
        out_dtype = q.dtype
        
        scale = params.softmax_scale
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)
        
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        
        num_key_value_heads = k.shape[1]
        
        if num_heads % num_key_value_heads != 0:
            raise ValueError(
                f"GQA requires num_heads divisible by num_key_value_heads, "
                f"got num_heads={num_heads}, num_key_value_heads={num_key_value_heads}"
            )
        
        if num_heads == num_key_value_heads:
            num_key_value_heads = 0
        
        config = NPUAttentionConfig(
            num_heads=num_heads,
            scale=float(scale),
            num_key_value_heads=num_key_value_heads,
            next_tokens=65535 if not params.causal else 0,
            softmax_lse_flag=True,
        )
        
        out, lse = torch_npu.npu_fused_infer_attention_score(
            q, k, v,
            num_heads=config.num_heads,
            scale=config.scale,
            input_layout=config.input_layout,
            num_key_value_heads=config.num_key_value_heads,
            pre_tokens=config.pre_tokens,
            next_tokens=config.next_tokens,
            sparse_mode=config.sparse_mode,
            inner_precise=config.inner_precise,
            softmax_lse_flag=config.softmax_lse_flag,
        )
        
        out = out.transpose(1, 2).contiguous()
        lse = lse.squeeze(-1)
        
        return out.to(out_dtype), lse
    
    @staticmethod
    def _merge_two_outputs(
        out1: torch.Tensor, 
        lse1: torch.Tensor,
        out2: torch.Tensor, 
        lse2: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge two attention outputs using LSE.
        
        Args:
            out1, out2: [batch_size, seq_len, num_heads, head_dim]
            lse1, lse2: [batch_size, num_heads, seq_len]
        
        Returns:
            merged output: [batch_size, seq_len, num_heads, head_dim]
        """
        lse1_expanded = lse1.transpose(1, 2).unsqueeze(-1)
        lse2_expanded = lse2.transpose(1, 2).unsqueeze(-1)
        
        max_lse = torch.maximum(lse1_expanded, lse2_expanded)
        lse_new = max_lse + torch.log(
            torch.exp(lse1_expanded - max_lse) + 
            torch.exp(lse2_expanded - max_lse)
        )
        
        weight1 = torch.exp(lse1_expanded - lse_new)
        weight2 = torch.exp(lse2_expanded - lse_new)
        
        out_merged = weight1 * out1 + weight2 * out2
        
        return out_merged