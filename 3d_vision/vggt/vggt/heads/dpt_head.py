# coding=utf-8
# Adapted from
# https://github.com/facebookresearch/vggt/
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Inspired by https://github.com/DepthAnything/Depth-Anything-V2 

import os
from typing import List, Dict, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .head_act import activate_head
from .utils import create_uv_grid, position_grid_to_embed


class DPTHead(nn.Module):
    """
    DPT Head for dense prediction tasks.
    This implementation follows the architecture described in "Vision Transformers for Dense Prediction"	 
    (https://arxiv.org/abs/2103.13413). The DPT head processes features from a vision transformer	 
    backbone and produces dense predictions by fusing multi-scale features.
    
    Improvements: Supports sequence parallel
    - Single-device mode: Input [B, S, P, C], output [B, S, H, W, out_dim]
    - SP mode: 
        - Input [B, shard_S, P, C]
        - Output [B, shard_S, H, W, out_dim] (default, sharded state)
        - Or output [B, S, H, W, out_dim] (allgather_output=True)
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 4,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        pos_embed_cache: Dict[Tuple, torch.Tensor] = None,
        use_dpt_pos_embed_cache: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
        sp_config: Optional['SPConfig'] = None,
        sp_global_group: Optional[dist.ProcessGroup] = None,
        allgather_output: bool = False,
    ) -> None:
        super(DPTHead, self).__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx
        self.use_dpt_pos_embed_cache = use_dpt_pos_embed_cache
        self.pos_embed_cache = pos_embed_cache if pos_embed_cache is not None else {}

        self.norm = nn.LayerNorm(dim_in)

        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        self.scratch = _make_scratch(out_channels, features, expand=False)

        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features
        head_features_2 = 32

        if feature_only:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        else:
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
            )
            conv2_in_channels = head_features_1 // 2

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )
        
        self.sp_config = sp_config
        self.sp_global_group = sp_global_group
        self.sp_enabled = sp_config is not None and sp_config.sp_degree > 1
        self.allgather_output = allgather_output
        
        if self.sp_enabled:
            self.sp_world_size = sp_config.sp_degree

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the DPT head.
        
        Args:
            aggregated_tokens_list: List of token tensors
                Single-device: Each [B, S, P, C]
                SP: Each [B, shard_S, P, C]
            images: Input images
                Single-device: [B, S, 3, H, W]
                SP: [B, shard_S, 3, H, W] (already sharded)
            patch_start_idx: Starting index for patch tokens
            frames_chunk_size: Chunk size for processing frames

        Returns:
            Single-device: (preds, conf) with shape [B, S, *, H, W]
            SP (default): (preds, conf) with shape [B, shard_S, *, H, W]
            SP (allgather_output=True): (preds, conf) with shape [B, S, *, H, W]
        """
        # 修复：B -> batch_size, shard_S -> shard_seq_len, H -> height, W -> width
        batch_size, shard_seq_len, _, height, width = images.shape
        
        if frames_chunk_size is None or frames_chunk_size >= shard_seq_len:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

        all_preds = []
        all_conf = []

        for start_idx in range(0, shard_seq_len, frames_chunk_size):
            end_idx = min(start_idx + frames_chunk_size, shard_seq_len)

            if self.feature_only:
                chunk_output = self._forward_impl(
                    aggregated_tokens_list, images, patch_start_idx, start_idx, end_idx
                )
                all_preds.append(chunk_output)
            else:
                chunk_preds, chunk_conf = self._forward_impl(
                    aggregated_tokens_list, images, patch_start_idx, start_idx, end_idx
                )
                all_preds.append(chunk_preds)
                all_conf.append(chunk_conf)

        if self.feature_only:
            result = torch.cat(all_preds, dim=1)
        else:
            result = (torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1))
        
        if self.sp_enabled and self.allgather_output:
            if self.feature_only:
                result = self._allgather_output(result)
            else:
                result = (self._allgather_output(result[0]), self._allgather_output(result[1]))
        
        return result

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Implementation of the forward pass."""
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        batch_size, shard_seq_len, _, height, width = images.shape
        patch_h, patch_w = height // self.patch_size, width // self.patch_size

        out = []
        dpt_idx = 0

        actual_batch_size = None

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            batch_from_tokens, shard_seq_len_from_tokens, patch_num_from_tokens, channel_from_tokens = x.shape
            
            if actual_batch_size is None:
                actual_batch_size = batch_from_tokens * shard_seq_len_from_tokens
            
            x = x.reshape(batch_from_tokens * shard_seq_len_from_tokens, patch_num_from_tokens, channel_from_tokens)
            
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, width, height)
            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        out = self.scratch_forward(out)
        
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, width, height)

        if self.feature_only:
            return out.view(batch_from_tokens, shard_seq_len_from_tokens, *out.shape[1:])

        out = self.scratch.output_conv2(out)
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)

        preds = preds.view(batch_from_tokens, shard_seq_len_from_tokens, *preds.shape[1:])
        conf = conf.view(batch_from_tokens, shard_seq_len_from_tokens, *conf.shape[1:])
        
        return preds, conf
    
    def _allgather_output(self, output: torch.Tensor) -> torch.Tensor:
        """
        Allgather output from all ranks (SP mode).
        
        Args:
            output: [B, shard_S, *, H, W]
        
        Returns:
            output_full: [B, S, *, H, W]
        """
        shape = output.shape
        batch_size, shard_seq_len = shape[0], shape[1]
        remaining_dims = shape[2:]
        
        output_flat = output.reshape(batch_size, shard_seq_len, -1)
        
        output_gathered = torch.empty(
            batch_size, self.sp_world_size * shard_seq_len, output_flat.shape[-1],
            dtype=output.dtype,
            device=output.device
        )
        
        dist.all_gather_into_tensor(
            output_gathered,
            output_flat.contiguous(),
            group=self.sp_global_group
        )
        
        output_full = output_gathered.reshape(batch_size, -1, *remaining_dims)
        
        return output_full

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """Apply positional embedding to tensor x."""
        # If cache is disabled, fall back to original (recompute every time)
        if not self.use_dpt_pos_embed_cache:
            return self._apply_pos_embed_original(x, W, H, ratio)

        cache_key = (W, H, x.shape)
        if cache_key not in self.pos_embed_cache:
            self.pos_embed_cache[cache_key] = self._compute_pos_embed(x, W, H, ratio)
        pos_embed = self.pos_embed_cache[cache_key]
        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through the fusion blocks."""
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out

    def _compute_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """Compute positional embedding from input tensor."""
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return pos_embed

    def _apply_pos_embed_original(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """Original implementation: recompute positional embedding every time (baseline)."""
        pos_embed = self._compute_pos_embed(x, W, H, ratio)
        return x + pos_embed


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    return FeatureFusionBlock(
        features, nn.ReLU(inplace=True), deconv=False, bn=False, expand=False,
        align_corners=True, size=size, has_residual=has_residual, groups=groups,
    )


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""
    def __init__(self, features, activation, bn, groups=1):
        super().__init__()
        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.norm1 = None
        self.norm2 = None
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""
    def __init__(self, features, activation, deconv=False, bn=False, expand=False,
                 align_corners=True, size=None, has_residual=True, groups=1):
        super(FeatureFusionBlock, self).__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups)

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]
        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


def custom_interpolate(x: torch.Tensor, size: Tuple[int, int] = None, scale_factor: float = None,
                      mode: str = "bilinear", align_corners: bool = True) -> torch.Tensor:
    """Custom interpolate to avoid INT_MAX issues."""
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    input_dtype = x.dtype
    use_dtype_cast = input_dtype != torch.float32
    if use_dtype_cast:
        # NPU nn.functional.interpolate (bilinear) only supports FP16/FP32/FP64.
        # Cast bf16 -> fp16 so the op can run on NPU AI Core; restore after.
        x = x.half()

    INT_MAX = 1610612736
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]
    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.to(input_dtype).contiguous() if use_dtype_cast else x.contiguous()
    else:
        x = nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)
        return x.to(input_dtype) if use_dtype_cast else x