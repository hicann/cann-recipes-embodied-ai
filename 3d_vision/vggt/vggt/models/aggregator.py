# coding=utf-8
# Adapted from
# https://github.com/facebookresearch/vggt/
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)


@dataclass
class AttentionProcessConfig:
    """Configuration for attention processing."""
    batch_size: int
    shard_seq_len: int
    total_seq_len: int
    num_patches: int
    channels: int
    pos: Optional[torch.Tensor] = None
    height: Optional[int] = None
    width: Optional[int] = None


class Aggregator(nn.Module):
    """
    Aggregator with Full Sequence Parallel support.
    
    Key Features:
    1. Single-device mode (sp_config=None): Fully compatible with original implementation
    2. SP mode: Splits sequence AFTER token concatenation (delayed sharding)
    3. Frame Attention: Operates on sharded sequence (no communication)
    4. Global Attention: Handles SP communication internally via Ring/Ulysses attention
    
    Improvements in this version:
    - Delayed sharding: All ranks process identical inputs until token concatenation
    - Better precision: No early sequence boundary effects
    - Simpler logic: Unified token expansion without rank-specific branches
    - Cleaner code: Removed unnecessary early sharding complexity
    """
    
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        # Sequence Parallel parameters
        sp_config: Optional['SPConfig'] = None,
        sp_ulysses_group: Optional[dist.ProcessGroup] = None,
        sp_ring_group: Optional[dist.ProcessGroup] = None,
        sp_global_group: Optional[dist.ProcessGroup] = None,
        # Computation redundancy elimination switches
        use_rope_cache: bool = True,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # RoPE with cache switch
        self.rope = RotaryPositionEmbedding2D(
            frequency=rope_freq, 
            use_rope_cache=use_rope_cache,
        ) if rope_freq > 0 else None
        self.position_getter = PositionGetter(use_cache=use_rope_cache) if self.rope is not None else None

        # Create Frame Blocks (NO sequence parallel)
        self.frame_blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                init_values=init_values,
                qk_norm=qk_norm,
                rope=self.rope,
                is_global_attention=False,
                sp_config=None,
                sp_ulysses_group=None,
                sp_ring_group=None,
            )
            for _ in range(depth)
        ])

        # Create Global Blocks (WITH sequence parallel)
        self.global_blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                init_values=init_values,
                qk_norm=qk_norm,
                rope=self.rope,
                is_global_attention=True,
                sp_config=sp_config,
                sp_ulysses_group=sp_ulysses_group,
                sp_ring_group=sp_ring_group,
            )
            for _ in range(depth)
        ])

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Camera and register tokens: [1, 2, num_tokens, C]
        # Dimension 1 has size 2: [query token for first frame (index 0), shared token for remaining frames (index 1)]
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        # 1 camera token + num_register_tokens
        self.patch_start_idx = 1 + num_register_tokens

        # std=1e-6: Initialize with very small values to minimize impact during early training
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # ImageNet normalization statistics: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # These are standard values used by pretrained models like DINOv2
        for name, value in (("_resnet_mean", [0.485, 0.456, 0.406]), ("_resnet_std", [0.229, 0.224, 0.225])):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False
        
        # Save sequence parallel configuration
        self.sp_config = sp_config
        self.sp_ulysses_group = sp_ulysses_group
        self.sp_ring_group = sp_ring_group
        self.sp_global_group = sp_global_group
        # SP requires at least 2 GPUs (sp_degree > 1)
        self.sp_enabled = sp_config is not None and sp_config.sp_degree > 1
        
        # Track original sequence length for padding handling
        self.original_seq_len = None
        self.padded_seq_len = None
        
        if self.sp_enabled:
            self.sp_rank = dist.get_rank(sp_global_group)
            self.sp_world_size = sp_config.sp_degree
        else:
            # Single device defaults
            self.sp_rank = 0
            self.sp_world_size = 1

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """Build the patch embed layer."""
        if "conv" in patch_embed:
            # 3: RGB channels
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """
        Forward with optional Full Sequence Parallel (delayed sharding version).
        Key improvement: Sharding happens AFTER token concatenation, not at input stage.
        This ensures all ranks process identical inputs through patch_embed and token expansion.
        Single-device mode (sp_enabled=False):
            - Input: [B, S, C, H, W]
            - Output: List of [B, S, P, 2*C] tensors
        SP mode (sp_enabled=True):
            - Input: [B, S, C, H, W] (all ranks receive and process full sequence)
            - After concatenation: Sharded to [B, shard_S, P_total, C]
            - Output: List of [B, shard_S, P, 2*C] tensors (sharded state)
        Returns:
            output_list: List of concatenated frame+global features from each block group
            patch_start_idx: Index where patch tokens begin (after special tokens)
        """
        batch_size, seq_len, channels_in, height, width = images.shape
        # Expect 3 RGB channels
        if channels_in != 3:
            raise ValueError(f"Expected 3 input channels, got {channels_in}")
        
        # Store original sequence length
        self.original_seq_len = seq_len
        
        # Normalization and reshape - FULL sequence
        images = (images - self._resnet_mean) / self._resnet_std
        # Note: Keep input dtype consistent with model dtype (fp32 or bf16)
        # Original code hardcoded bfloat16, which caused type mismatch when model is fp32
        images = images.view(batch_size * seq_len, channels_in, height, width)
        
        # Patch Embedding - on FULL sequence (all ranks identical)
        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        
        _, num_patches, channels = patch_tokens.shape
        
        # Token expansion - using FULL sequence length (all ranks identical)
        camera_token = self._expand_tokens_full(
            self.camera_token, batch_size, seq_len
        )
        
        register_token = self._expand_tokens_full(
            self.register_token, batch_size, seq_len
        )
        
        # Concatenate all tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        # Shard AFTER concatenation
        if self.sp_enabled:
            tokens, shard_seq_len = self._shard_tokens_after_concat(tokens, batch_size, seq_len)
            total_seq_len = self.padded_seq_len
        else:
            shard_seq_len = seq_len
            total_seq_len = seq_len
        
        # Position encoding
        pos = None
        if self.rope is not None:
            pos = self._get_position_for_seq(batch_size, shard_seq_len, height, width, images.device)

        _, num_patches, channels = tokens.shape
        
        # Execute Attention Blocks
        frame_idx = 0
        global_idx = 0
        output_list = []
        
        # Create attention config
        attn_config = AttentionProcessConfig(
            batch_size=batch_size,
            shard_seq_len=shard_seq_len,
            total_seq_len=total_seq_len,
            num_patches=num_patches,
            channels=channels,
            pos=pos,
            height=height,
            width=width
        )
        
        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, frame_idx, attn_config
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, global_idx, attn_config
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            # Merge intermediate results
            # dim=-1: Concatenate along channel dimension to get [B, shard_S, P, 2*C]
            for i in range(len(frame_intermediates)):
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)
        
        del concat_inter, frame_intermediates, global_intermediates
        return output_list, self.patch_start_idx, self.original_seq_len
    
    def _expand_tokens_full(
        self, 
        token_tensor: torch.Tensor, 
        batch_size: int, 
        seq_len: int
    ) -> torch.Tensor:
        """
        Simplified token expansion (all ranks use identical logic).
        
        Both camera_token and register_token follow the same logic:
        - First frame uses token_tensor[:, 0] (query)
        - Remaining frames use token_tensor[:, 1] (others)
        
        Args:
            token_tensor: [1, 2, num_tokens, C] - (query for frame 0, others for frame 1+)
            batch_size: batch size
            seq_len: FULL sequence length (not sharded)
        
        Returns:
            expanded: [batch_size*seq_len, num_tokens, C]
        """
        # 0:1 slice: Extract first token type (query) while keeping dimensions
        query = token_tensor[:, 0:1, ...].expand(batch_size, 1, *token_tensor.shape[2:])
        # 1:2 slice: Extract second token type (others for remaining frames)
        # seq_len - 1: Number of frames after the first one
        others = token_tensor[:, 1:2, ...].expand(batch_size, seq_len - 1, *token_tensor.shape[2:])
        # dim=1: Concatenate along sequence dimension
        combined = torch.cat([query, others], dim=1)
        
        # shape[2:]: Keep all dimensions after index 2 unchanged
        combined = combined.reshape(batch_size * seq_len, *combined.shape[2:])
        return combined

    def _shard_tokens_after_concat(self, tokens, batch_size, seq_len):
        """Shard tokens after concatenation for sequence parallel."""
        _, num_patches_total, channels = tokens.shape
        tokens = tokens.reshape(batch_size, seq_len, num_patches_total, channels)
        
        # Padding logic
        # Check if seq_len is divisible by world_size (remainder == 0)
        if seq_len % self.sp_world_size != 0:
            # Pad to next multiple of world_size
            padded_seq_len = math.ceil(seq_len / self.sp_world_size) * self.sp_world_size
            pad_len = padded_seq_len - seq_len
            
            # -1: slice: Use last frame for padding to avoid introducing random data
            last_frame = tokens[:, -1:, :, :].expand(batch_size, pad_len, num_patches_total, channels).clone()
            tokens = torch.cat([tokens, last_frame], dim=1)
            
            self.padded_seq_len = padded_seq_len
        else:
            padded_seq_len = seq_len
            self.padded_seq_len = seq_len
        
        # Sharding
        shard_seq_len = padded_seq_len // self.sp_world_size
        start_idx = self.sp_rank * shard_seq_len
        end_idx = start_idx + shard_seq_len
        
        tokens_sharded = tokens[:, start_idx:end_idx, :, :].contiguous()
        tokens_sharded = tokens_sharded.reshape(batch_size * shard_seq_len, num_patches_total, channels)
        
        return tokens_sharded, shard_seq_len

    def _get_position_for_seq(
        self, 
        batch_size: int, 
        shard_seq_len: int,
        height: int, 
        width: int, 
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate position encodings (adapted for single-device and SP modes).
        
        In SP mode, each rank generates positions with proper global frame offset.
        Position encoding includes both spatial (H, W) and temporal (frame index) information.
        
        """
        # Calculate global frame offset for this rank
        frame_offset = self.sp_rank * shard_seq_len if self.sp_enabled else 0
        
        # 1: Generate positions for single sample (batch_size=1), will be expanded later
        spatial_pos = self.position_getter(1, height // self.patch_size, width // self.patch_size, device)
        
        # Prepare position list for all local frames
        pos_list = []
        for local_idx in range(shard_seq_len):
            # Calculate global frame index
            global_frame_idx = frame_offset + local_idx
            
            # Clone spatial position for this frame
            frame_pos = spatial_pos.clone()
            
            # Add offset for special tokens
            if self.patch_start_idx > 0:
                # +1: Offset to avoid position collision with special tokens
                frame_pos = frame_pos + 1
                # 2: 2D position encoding (x, y coordinates)
                pos_special = torch.zeros(1, self.patch_start_idx, 2).to(device).to(frame_pos.dtype)
                frame_pos = torch.cat([pos_special, frame_pos], dim=1)
            
            pos_list.append(frame_pos)
        
        # dim=0: Concatenate along batch/sequence dimension
        pos = torch.cat(pos_list, dim=0)
        # 0: Add batch dimension at index 0
        pos = pos.unsqueeze(0).expand(batch_size, -1, -1, -1)
        pos = pos.reshape(batch_size * shard_seq_len, -1, 2)
        
        return pos

    def _process_frame_attention(
        self, 
        tokens: torch.Tensor, 
        frame_idx: int,
        config: AttentionProcessConfig
    ) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        """
        Process Frame Attention blocks.
        
        Frame attention operates on individual frames independently,
        so it works directly on the sharded sequence without communication.
        
        Args:
            tokens: [batch_size*shard_seq_len, num_patches, channels]
            frame_idx: current frame block index
            config: attention processing configuration
            
        Returns:
            tokens: [batch_size*shard_seq_len, num_patches, channels]
            frame_idx: updated block index
            intermediates: List of [batch_size, shard_seq_len, num_patches, channels] for each sub-block
        """
        batch_size = config.batch_size
        shard_seq_len = config.shard_seq_len
        num_patches = config.num_patches
        channels = config.channels
        pos = config.pos
        
        expected_shape = (batch_size * shard_seq_len, num_patches, channels)
        if tokens.shape != expected_shape:
            tokens = tokens.view(batch_size, shard_seq_len, num_patches, channels).view(*expected_shape)
        # 2: Last dimension is 2 for 2D position encoding (x, y)
        if pos is not None and pos.shape != (batch_size * shard_seq_len, num_patches, 2):
            pos = pos.view(batch_size, shard_seq_len, num_patches, 2).view(batch_size * shard_seq_len, num_patches, 2)
        
        intermediates = []
        
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(
                    self.frame_blocks[frame_idx], tokens, pos,
                    use_reentrant=self.use_reentrant
                )
            else:
                tokens = self.frame_blocks[frame_idx](
                    tokens, 
                    pos=pos, 
                    height=None, 
                    width=None, 
                    global_seq_len=None
                )
            
            frame_idx += 1
            intermediates.append(tokens.view(batch_size, shard_seq_len, num_patches, channels))
        
        return tokens, frame_idx, intermediates

    def _process_global_attention(
        self,
        tokens: torch.Tensor,
        global_idx: int,
        config: AttentionProcessConfig
    ) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        """
        Process Global Attention blocks.
        
        Global attention attends across all frames. In SP mode, the Global Block
        handles Ring/Ulysses communication internally.
        
        Input format expected by Global Block:
            Single-device: [B, S, P, C] -> reshape to [B, S*P, C]
            SP mode: [B, shard_S, P, C] -> reshape to [B, shard_S*P, C]
        
        Args:
            tokens: [batch_size*shard_seq_len, num_patches, channels]
            global_idx: current global block index
            config: attention processing configuration
            
        Returns:
            tokens: [batch_size, shard_seq_len*num_patches, channels]
            global_idx: updated block index
            intermediates: List of [batch_size, shard_seq_len, num_patches, channels] for each sub-block
        """
        batch_size = config.batch_size
        shard_seq_len = config.shard_seq_len
        num_patches = config.num_patches
        channels = config.channels
        pos = config.pos
        
        expected_shape = (batch_size, shard_seq_len * num_patches, channels)
        if tokens.shape != expected_shape:
            tokens = tokens.view(batch_size, shard_seq_len, num_patches, channels).view(*expected_shape)

        if pos is not None and pos.shape != (batch_size, shard_seq_len * num_patches, 2):
            pos = pos.view(batch_size, shard_seq_len, num_patches, 2).view(batch_size, shard_seq_len * num_patches, 2)
        
        intermediates = []
        
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(
                    self.global_blocks[global_idx], tokens, pos,
                    use_reentrant=self.use_reentrant
                )
            else:
                tokens = self.global_blocks[global_idx](
                    tokens, 
                    pos=pos, 
                    height=None, 
                    width=None, 
                    global_seq_len=None
                )
            
            global_idx += 1
            intermediates.append(tokens.view(batch_size, shard_seq_len, num_patches, channels))
        
        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, batch_size, seq_len):
    """
    Helper function for token expansion (kept for backward compatibility).
    
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    - Uses the first position (index=0) for the first frame only
    - Uses the second position (index=1) for all remaining frames
    
    Args:
        token_tensor: [1, 2, X, C]
        batch_size: batch size
        seq_len: sequence length
        
    Returns:
        torch.Tensor: [batch_size*seq_len, X, C]
    """
    query = token_tensor[:, 0:1, ...].expand(batch_size, 1, *token_tensor.shape[2:])
    others = token_tensor[:, 1:2, ...].expand(batch_size, seq_len - 1, *token_tensor.shape[2:])
    combined = torch.cat([query, others], dim=1)
    combined = combined.view(batch_size * seq_len, *combined.shape[2:])
    return combined
    