# coding=utf-8
# Adapted from
# https://github.com/facebookresearch/vggt/
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_npu
from torch import Tensor


@dataclass
class AttentionForwardParams:
    """Encapsulates parameters for Attention forward pass."""
    x: Tensor
    pos: Optional[Tensor] = None
    height: Optional[int] = None
    width: Optional[int] = None
    global_seq_len: Optional[int] = None
    attn_bias: Optional[Tensor] = None  # For MemEffAttention compatibility


@dataclass
class FIAComputeParams:
    """Encapsulates parameters for FIA attention computation."""
    q: Tensor
    k: Tensor
    v: Tensor
    causal: bool = False
    scale: Optional[float] = None
    dropout_p: float = 0.0


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
        # Sequence Parallel parameters
        is_global_attention: bool = False,
        sp_config: Optional['SPConfig'] = None,
        sp_ulysses_group: Optional[dist.ProcessGroup] = None,
        sp_ring_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        
        # Sequence Parallel configuration
        self.is_global_attention = is_global_attention
        self.sp_enabled = sp_config is not None and is_global_attention and sp_config.sp_degree > 1
        
        if self.sp_enabled:
            from vggt.sp import UnifiedSPAttention
            self.sp_attention = UnifiedSPAttention(
                ulysses_group=sp_ulysses_group,
                ring_group=sp_ring_group,
                use_ring_overlap=sp_config.use_ring_overlap,
            )
            logging.info(f"Sequence Parallel enabled for Global Attention: "
                        f"Ulysses degree={sp_config.ulysses_degree}, "
                        f"Ring degree={sp_config.ring_degree}")

    def forward(
        self, 
        params: AttentionForwardParams,
    ) -> Tensor:
        """
        Unified forward pass with operator selection based on attention type.
        
        Frame Attention: Uses SDPA
        Global Attention: Uses FIA (with or without SP)
        
        Args:
            params: Encapsulated attention forward parameters (AttentionForwardParams)
        
        Returns:
            output: [batch_size, num_tokens, channel]
        """
        x = params.x
        pos = params.pos
        height = params.height
        width = params.width
        global_seq_len = params.global_seq_len
        
        batch_size, num_tokens, channel = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # QK normalization
        q, k = self.q_norm(q), self.k_norm(k)
        
        # Apply RoPE if enabled
        if self.rope is not None:
            # Check if we have the extended RoPE parameters
            if height is not None and width is not None:
                # Use extended RoPE call with caching support (for SP mode)
                if global_seq_len is None:
                    global_seq_len = (height // 14) * (width // 14)
                
                q = self._apply_rope(q, pos, height, width, global_seq_len)
                k = self._apply_rope(k, pos, height, width, global_seq_len)
            else:
                q = self._apply_rope_simple(q, pos)
                k = self._apply_rope_simple(k, pos)
        
        # Attention computation
        if self.sp_enabled:
            # Global Attention with SP
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            
            sp_attn_params = AttentionForwardParams(
                x=None,
                pos=None
            )
            x = self.sp_attention.forward(
                attention_params=AttentionParams(
                    q=q, k=k, v=v,
                    causal=False,
                    softmax_scale=self.scale,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
            )
            x = x.reshape(batch_size, num_tokens, channel)
            
        elif self.is_global_attention:
            # Global Attention: always use NPU FIA (fp32 handled via dtype conversion inside)
            fia_params = FIAComputeParams(
                q=q, k=k, v=v,
                causal=False,
                scale=self.scale,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            x = self._compute_attention_with_fia(fia_params)
            x = x.transpose(1, 2).contiguous().reshape(batch_size, num_tokens, channel)
            
        else:
            # Frame Attention (uses SDPA, short sequences don't need FIA)
            out_dtype = q.dtype
            # SDPA dispatches to npu_flash_attention_score which also lacks fp32 support;
            # convert to bf16 for computation, restore original dtype afterwards.
            if out_dtype == torch.float32:
                q = q.to(torch.bfloat16)
                k = k.to(torch.bfloat16)
                v = v.to(torch.bfloat16)
            x = F.scaled_dot_product_attention(
                q, k, v, 
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,
                scale=self.scale,
            )
            if out_dtype == torch.float32:
                x = x.to(out_dtype)
            x = x.transpose(1, 2).contiguous().reshape(batch_size, num_tokens, channel)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

    
    def _apply_rope(self, tensor: Tensor, pos, height, width, global_seq_len):
        return self.rope.forward(
            tokens=tensor,
            positions=pos,
            height=height,
            width=width,
            global_seq_len=global_seq_len
        )

    def _apply_rope_simple(self, tensor: Tensor, pos):
        return self.rope.forward(
            tokens=tensor,
            positions=pos,
            height=None,
            width=None,
            global_seq_len=None
        )

    def _compute_attention_with_fia(
        self,
        params: FIAComputeParams,
    ) -> Tensor:
        """
        Compute attention using NPU FIA operator.
        
        Args:
            params: Encapsulated FIA computation parameters (FIAComputeParams)
        
        Returns:
            output: [batch_size, num_heads, num_tokens, head_dim]
        """
        q, k, v = params.q, params.k, params.v
        batch_size, num_heads, num_tokens, head_dim = q.shape
        out_dtype = q.dtype

        # FIA does not support fp32 natively on NPU; convert to bf16 for computation,
        # restore original dtype at the end (same pattern as custom_interpolate).
        if out_dtype == torch.float32:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

        scale = params.scale if params.scale is not None else 1.0 / math.sqrt(head_dim)
        
        # Check for GQA
        num_key_value_heads = k.shape[1]
        if num_heads == num_key_value_heads:
            num_key_value_heads = 0

        # 950: SDPA and npu_fused_infer_attention_score require contiguous q/k/v
        if "950" in torch.npu.get_device_name():
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        # FIA does not support dropout, fallback to SDPA if needed
        if params.dropout_p > 0.0 and self.training:
            logging.warning(
                f"FIA does not support dropout (dropout_p={params.dropout_p}), "
                f"falling back to SDPA"
            )
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=params.dropout_p,
                is_causal=params.causal,
                scale=scale,
            )
            return out.to(out_dtype)
        
        # Call NPU fused attention
        out = torch_npu.npu_fused_infer_attention_score(
            q, k, v,
            num_heads=num_heads,
            scale=float(scale),
            input_layout="BNSD",
            num_key_value_heads=num_key_value_heads,
            pre_tokens=65535,
            next_tokens=65535 if not params.causal else 0,
            sparse_mode=0,
            inner_precise=0,
        )[0]
        
        return out.to(out_dtype)


class MemEffAttention(Attention):
    """Memory-efficient attention variant (fallback to standard Attention)."""
    
    def forward(
        self, 
        params: AttentionForwardParams,
    ) -> Tensor:
        """
        Forward pass for MemEffAttention (compatible with parent class).
        
        Args:
            params: Encapsulated attention forward parameters (AttentionForwardParams)
        
        Returns:
            output: [batch_size, num_tokens, channel]
        """
        assert params.pos is None, "MemEffAttention does not support RoPE yet"
        
        if params.attn_bias is not None:
            raise AssertionError("xFormers is required for using nested tensors")
        
        return super().forward(params)


@dataclass
class AttentionParams:
    """Encapsulates attention computation parameters (for SP Attention compatibility)."""
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    causal: bool = False
    softmax_scale: Optional[float] = None
    dropout_p: float = 0.0
    attn_mask: Optional[torch.Tensor] = None