# coding=utf-8
# Adapted from
# https://github.com/facebookresearch/vggt/
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import torch
import torch.nn as nn
import torch.distributed as dist
from huggingface_hub import PyTorchModelHubMixin

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    """
    VGGT model with optional Full Sequence Parallel support.
    
    Improvements:
    1. Single-device mode (sp_config=None): Fully compatible with original implementation
    2. SP mode: Aggregator outputs sharded state, Head layers handle gathering if needed
    3. NPU fused operators are enabled by default (RoPE npu_rotary_mul, Add+LayerNorm, FIA)
    """
    
    def __init__(
        self, 
        img_size=518, 
        patch_size=14, 
        embed_dim=1024,
        enable_camera=True, 
        enable_point=True, 
        enable_depth=True, 
        enable_track=True,
        # Sequence Parallel parameters
        sp_config: Optional['SPConfig'] = None,
        sp_ulysses_group: Optional[dist.ProcessGroup] = None,
        sp_ring_group: Optional[dist.ProcessGroup] = None,
        sp_global_group: Optional[dist.ProcessGroup] = None,
        # Computation redundancy elimination switches
        use_rope_cache: bool = True,
        use_dpt_pos_embed_cache: bool = True,
    ):
        super().__init__()

        self.aggregator = Aggregator(
            img_size=img_size, 
            patch_size=patch_size, 
            embed_dim=embed_dim,
            sp_config=sp_config,
            sp_ulysses_group=sp_ulysses_group,
            sp_ring_group=sp_ring_group,
            sp_global_group=sp_global_group,
            use_rope_cache=use_rope_cache,
        )
        
        # Store SP configuration
        self.sp_config = sp_config
        self.sp_global_group = sp_global_group
        self.sp_enabled = sp_config is not None and sp_config.sp_degree > 1

        # Head layers with SP information
        self.camera_head = CameraHead(
            dim_in=2 * embed_dim,
            sp_config=sp_config,
            sp_global_group=sp_global_group
        ) if enable_camera else None
        
        self.dpt_pos_embed_cache = {}
        
        self.point_head = DPTHead(
            dim_in=2 * embed_dim, 
            output_dim=4, 
            activation="inv_log", 
            conf_activation="expp1", 
            pos_embed_cache=self.dpt_pos_embed_cache,
            use_dpt_pos_embed_cache=use_dpt_pos_embed_cache,
            sp_config=sp_config,
            sp_global_group=sp_global_group
        ) if enable_point else None
        
        self.depth_head = DPTHead(
            dim_in=2 * embed_dim, 
            output_dim=2, 
            activation="exp", 
            conf_activation="expp1", 
            pos_embed_cache=self.dpt_pos_embed_cache,
            use_dpt_pos_embed_cache=use_dpt_pos_embed_cache,
            sp_config=sp_config,
            sp_global_group=sp_global_group
        ) if enable_depth else None
        
        self.track_head = TrackHead(
            dim_in=2 * embed_dim, 
            patch_size=patch_size,
            sp_config=sp_config,
            sp_global_group=sp_global_group
        ) if enable_track else None

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: Dictionary of predictions
                Single-device mode: All outputs are full sequences
                SP mode: 
                    - camera_head: Requires full sequence (internal allgather)
                    - depth/point: Returns sharded state (optional external allgather)
                    - track: Returns sharded state (optional external allgather)
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        # Aggregator forward
        # Single-device: returns [B, S, P, C]
        # SP mode: returns [B, shard_S, P, C]
        # aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        aggregated_tokens_list, patch_start_idx, original_seq_len = self.aggregator(images)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            # Camera Head: requires full sequence (internal allgather)
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list, original_seq_len=original_seq_len)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list
                
            # Depth Head: can return sharded state
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            # Point Head: can return sharded state
            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        # Track Head: can return sharded state
        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images

        return predictions
