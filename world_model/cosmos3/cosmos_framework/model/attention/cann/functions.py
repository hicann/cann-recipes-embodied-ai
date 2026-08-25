# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
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
#
"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

CANN FIA backend implementation.
"""

from dataclasses import dataclass

import torch
import torch_npu
from torch import Tensor


def _causal_mask(q_len: int, kv_len: int, device: torch.device) -> Tensor:
    return torch.ones((1, 1, q_len, kv_len), dtype=torch.bool, device=device).triu(1).contiguous()


def _tnd_causal_mask(device: torch.device) -> Tensor:
    # FIA requires a 2048-wide optimized mask for TND causal sparse mode.
    return _causal_mask(2048, 2048, device)


def _actual_seq_lengths(cumulative_seqlen: Tensor) -> list[int]:
    # Cosmos cumulative seqlens include a leading 0; FIA TND expects cumulative
    # end offsets only.
    return [int(x.item()) for x in cumulative_seqlen[1:]]


@dataclass(frozen=True)
class _VarlenAttentionParams:
    cumulative_seqlen_q: Tensor
    cumulative_seqlen_kv: Tensor
    is_causal: bool
    scale: float


def _cann_varlen_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    params: _VarlenAttentionParams,
) -> Tensor:
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0] or query.shape[0] != 1:
        raise ValueError("CANN FIA TND attention expects Cosmos packed BSND tensors with batch size 1.")

    # Cosmos frontend passes packed BSND tensors in varlen mode. Remove the
    # singleton batch dimension and use FIA's native TND layout.
    q, k, v = (x.squeeze(0).contiguous() for x in (query, key, value))
    actual_seq_lengths_q = _actual_seq_lengths(params.cumulative_seqlen_q)
    actual_seq_lengths_kv = _actual_seq_lengths(params.cumulative_seqlen_kv)
    q_total = q.shape[0]
    q_valid = actual_seq_lengths_q[-1]
    kv_valid = actual_seq_lengths_kv[-1]
    atten_mask = _tnd_causal_mask(q.device) if params.is_causal else None
    output, _ = torch_npu.npu_fused_infer_attention_score(
        q[:q_valid],
        k[:kv_valid],
        v[:kv_valid],
        atten_mask=atten_mask,
        actual_seq_lengths=actual_seq_lengths_q,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        num_heads=q.shape[1],
        num_key_value_heads=k.shape[1],
        input_layout="TND",
        scale=params.scale,
        pre_tokens=65535,
        next_tokens=65535,
        sparse_mode=3 if params.is_causal else 0,
    )
    output = torch.cat([output, output.new_zeros((q_total - q_valid, *output.shape[1:]))], dim=0)
    return output.unsqueeze(0)


def _cann_attention(query: Tensor, key: Tensor, value: Tensor, is_causal: bool, scale: float) -> Tensor:
    # The frontend uses BSND while FIA's standard path expects BNSD.
    q, k, v = (x.permute(0, 2, 1, 3).contiguous() for x in (query, key, value))
    atten_mask = _causal_mask(q.shape[2], k.shape[2], q.device) if is_causal else None
    output, _ = torch_npu.npu_fused_infer_attention_score(
        q,
        k,
        v,
        atten_mask=atten_mask,
        num_heads=q.shape[1],
        num_key_value_heads=k.shape[1],
        input_layout="BNSD",
        scale=scale,
        pre_tokens=65535,
        next_tokens=65535,
    )
    return output.permute(0, 2, 1, 3)


def cann_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    scale: float | None = None,
    return_lse: bool = False,
    **kwargs,
) -> Tensor | tuple[Tensor, None]:
    cumulative_seqlen_q = kwargs.get("cumulative_seqlen_Q")
    cumulative_seqlen_kv = kwargs.get("cumulative_seqlen_KV")
    scale = scale if scale is not None else query.shape[-1] ** -0.5

    if cumulative_seqlen_q is not None:
        params = _VarlenAttentionParams(
            cumulative_seqlen_q=cumulative_seqlen_q,
            cumulative_seqlen_kv=cumulative_seqlen_kv,
            is_causal=is_causal,
            scale=scale,
        )
        output = _cann_varlen_attention(
            query,
            key,
            value,
            params,
        )
    else:
        output = _cann_attention(query, key, value, is_causal, scale)

    if return_lse:
        return output, None
    return output
