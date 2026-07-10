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
    is_varlen = cumulative_seqlen_q is not None
    scale = scale if scale is not None else query.shape[-1] ** -0.5


    if is_varlen:
        if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0] or query.shape[0] != 1:
            raise ValueError("CANN FIA TND attention expects Cosmos packed BSND tensors with batch size 1.")

        # Cosmos frontend still passes packed BSND tensors in varlen mode:
        # [1,total_tokens,H,D]. Match FlashAttention varlen by squeezing the
        # singleton batch dimension and using FIA's native TND layout.
        q, k, v = (x.squeeze(0).contiguous() for x in (query, key, value))

        atten_mask = _tnd_causal_mask(q.device) if is_causal else None
        output, _ = torch_npu.npu_fused_infer_attention_score(
            q,
            k,
            v,
            atten_mask=atten_mask,
            actual_seq_lengths=_actual_seq_lengths(cumulative_seqlen_q),
            actual_seq_lengths_kv=_actual_seq_lengths(cumulative_seqlen_kv),
            num_heads=q.shape[1],
            num_key_value_heads=k.shape[1],
            input_layout="TND",
            scale=scale,
            pre_tokens=65535,
            next_tokens=65535,
            sparse_mode=3 if is_causal else 0,
        )
        output = output.unsqueeze(0)
    else:
        # Non-varlen inputs use the frontend's standard BSND layout. FIA's
        # dense path expects BNSD, so move heads before sequence and restore the
        # frontend layout after the call.
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
        output = output.permute(0, 2, 1, 3)

    if return_lse:
        return output, None
    return output
