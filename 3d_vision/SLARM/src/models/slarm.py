# coding=utf-8

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import re
import time
import math
import logging
import functools
from collections import namedtuple
from typing import List

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter_max, scatter_add
from einops import rearrange, repeat
from huggingface_hub import PyTorchModelHubMixin

from tools import is_ascend_npu
from tools.export_ply import save_ply, PlyExportConfig
from src.dataset.constants import SEMANTIC_LABEL_LIST, SEMANTIC_ID_TO_COLOR

from .slarm_ply_export import SLARMPlyExportMixin
from .slarm_novelview import SLARMNovelViewMixin, _FinalizeForwardParams
from .decoder import ConvDecoder, DummyDecoder, ModulatedLinearLayer
from .layers import LayerNorm2d, Mlp
from .components.aggregator.aggregator import Aggregator
from .components.heads.camera_head import CameraHead
from .components.heads.dpt_head import DPTHead
from .components.utils.pose_enc import (
    pose_encoding_to_extri_intri, extri_intri_to_pose_encoding
)
from .components.utils.geometry import (
    unproject_depth_map_to_point_map, angular_velocity_to_quaternion,
    quaternion_multiply, angle_axis_to_quaternion,
    compute_normals_scales_torch, rot_from_normals_torch, scale_from_dxdy_torch
)

if is_ascend_npu():
    from src.utils.rasterizer import Rasterizer, new_ascend_rasterization
    from src.utils.render_types import GaussianData, CameraParams, RenderConfig
else:
    from gsplat.rendering import rasterization, rasterization_2dgs

if os.getenv("FEAT_DIST"):
    from tools.feats_tools import get_text_label_feats, feat2class

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

_SetBasicAttrsParams = namedtuple('_SetBasicAttrsParams', [
    'img_size', 'embed_dim', 'patch_size', 'num_cams', 'gs_dim', 'depth', 'num_register_tokens', 'patch_embed',
    'near', 'far', 'use_last_token', 'enable_depth_head', 'enable_camera_head', 'enable_point_head',
    'use_pred_camera_pose', 'use_pred_depth', 'render_context_view', 'render_context_frame_contribution',
    'concat_plucker_embed', 'add_patch_plucker_embed', 'add_camera_embed', 'in_chans', 'shortcut_rgb',
    'gs_dense_reg_head_type', 'feat_dense_reg_head_type', 'motion_dense_key_head_type', 'pred_gs_conf', 'voxelize',
    'voxel_size', 'similarity_probs_threshold', 'disable_pos_embed', 'projected_motion_dim', 'pred_feat_dim',
    'with_feat', 'decoder_type', 'grad_checkpointing', 'vggt_pretrained_weight_filepath', 'use_2dgs',
    'pesudo_3dgs', 'save_gaussian', 'gaussian_save_path', 'save_rendered_pc', 'rendered_pc_save_path',
    'use_render_novel_view', 'use_time_token', 'use_sky_token', 'use_affine_token', 'mode', 'use_ms3_motion',
    'add_angular_velocity', 'enable_lifespan', 'ms3_deg', 'omega_deg', 'ms3_deg_downmax_mult', 'sigmoid_ms3_bias',
    'sigmoid_ms3_min', 'sigmoid_ms3_max', 'ms3_clamp', 'num_motion_tokens', 'tau'
])
_SetCoreAttrsParams = namedtuple('_SetCoreAttrsParams', [
    'img_size', 'embed_dim', 'patch_size', 'num_cams', 'gs_dim', 'depth', 'num_register_tokens', 'patch_embed',
    'near', 'far', 'use_last_token', 'enable_depth_head', 'enable_camera_head', 'enable_point_head'
])
_SetRemainingAttrsParams = namedtuple('_SetRemainingAttrsParams', [
    'mode', 'use_time_token', 'use_sky_token', 'use_affine_token', 'disable_pos_embed', 'projected_motion_dim',
    'pred_feat_dim', 'with_feat', 'decoder_type', 'grad_checkpointing', 'vggt_pretrained_weight_filepath',
    'use_2dgs', 'pesudo_3dgs', 'save_gaussian', 'gaussian_save_path', 'save_rendered_pc', 'rendered_pc_save_path',
    'use_render_novel_view'
])
_SetupActivationsParams = namedtuple('_SetupActivationsParams', [
    'gs_marbles', 'max_scale', 'opacity_offset', 'near', 'far', 'sigmoid_rgb'
])
_SetMotionAttrsParams = namedtuple('_SetMotionAttrsParams', [
    'use_ms3_motion', 'add_angular_velocity', 'ms3_deg', 'omega_deg', 'ms3_deg_downmax_mult', 'sigmoid_ms3_bias',
    'sigmoid_ms3_min', 'sigmoid_ms3_max', 'ms3_clamp'
])
_ComputeMotionImgKeysParams = namedtuple('_ComputeMotionImgKeysParams', ['x', 'b', 't', 'v', 'h', 'w', 'dense_feat'])
_ComputeGsParamsRawParams = namedtuple('_ComputeGsParamsRawParams', ['x', 'b', 't', 'v', 'h', 'w', 'rgb', 'dense_feat'])
_ComputePseudo3dgsScalesQuatsParams = namedtuple('_ComputePseudo3dgsScalesQuatsParams', [
    'means', 'depths', 'b', 't', 'v', 'h', 'w'
])
_RenderProbsInChunksParams = namedtuple('_RenderProbsInChunksParams', [
    'means', 'quats', 'scales', 'opacities', 'probs_batched', 'viewmats_batched', 'ks_batched', 'tgt_h', 'tgt_w',
    'radius_clip', 'chunk_size'
])
_ExecuteRenderingParams = namedtuple('_ExecuteRenderingParams', [
    'data_dict', 'gs_attrs', 'gs_params', 'means_batched', 'scales_batched', 'quats_batched', 'opacities_batched',
    'color_batched', 'forward_v_batched', 'feats_batched', 'colors_batched', 'forward_flow', 'feats',
    'render_motion_seg', 'concat_feat_render', 'b', 't', 'v', 'h', 'w', 'tgt_t', 'tgt_v', 'tgt_h', 'tgt_w',
    'radius_clip'
])
_BuildGsAttrsParams = namedtuple('_BuildGsAttrsParams', [
    'means_batched', 'scales_batched', 'quats_batched', 'opacities_batched', 'color_batched', 'forward_v_batched',
    'feats', 'feats_batched'
])
_AppendMotionWeightsParams = namedtuple('_AppendMotionWeightsParams', [
    'gs_params', 'colors_batched', 'tgt_t', 'idx', 'v', 'h', 'w'
])
_PrepareGsBatchedParams = namedtuple('_PrepareGsBatchedParams', [
    'gs_params', 'data_dict', 'feats', 'b', 't', 'v', 'h', 'w', 'tgt_t'
])
_RunRasterizationParams = namedtuple('_RunRasterizationParams', [
    'means_batched', 'quats_batched', 'scales_batched', 'opacities_batched', 'colors_to_render',
    'viewmats_batched', 'ks_batched', 'tgt_h', 'tgt_w', 'radius_clip'
])
_ApplyMotionParams = namedtuple('_ApplyMotionParams', [
    'gs_params', 'ctx_time', 'tdiff_forward_batched', 'means_batched', 'quats_batched', 'opacities_batched',
    'tgt_t', 'time_step', 'static_render', 'data_dict'
])
_ApplyMs3MotionParams = namedtuple('_ApplyMs3MotionParams', [
    'gs_params', 'ctx_time', 'tdiff_forward_batched', 'means_batched', 'quats_batched', 'opacities_batched',
    'tgt_t', 'time_step', 'static_render', 'data_dict'
])
_SliceGsAttrsByIdxParams = namedtuple('_SliceGsAttrsByIdxParams', [
    'means_batched', 'scales_batched', 'quats_batched', 'opacities_batched', 'color_batched', 'forward_v_batched',
    'idx', 'v', 'h', 'w'
])
_ApplyVoxelizationParams = namedtuple('_ApplyVoxelizationParams', [
    'gs_attrs', 'gs_params', 'idx', 'v', 'h', 'w', 'b', 'tgt_t'
])
_PrepareRenderColorsParams = namedtuple('_PrepareRenderColorsParams', [
    'color_batched', 'forward_v_batched', 'feats_batched', 'concat_feat_render', 'idx', 'v', 'h', 'w'
])
_RenderMotionSegmentationParams = namedtuple('_RenderMotionSegmentationParams', [
    'colors_batched', 'means_batched', 'quats_batched', 'scales_batched', 'opacities_batched', 'viewmats_batched',
    'ks_batched', 'tgt_h', 'tgt_w', 'radius_clip', 'b', 'tgt_t', 'tgt_v'
])
_BuildRenderOutputParams = namedtuple('_BuildRenderOutputParams', [
    'color', 'depth', 'forward_flow', 'rendered_alpha', 'feat', 'motion_seg', 'gs_attrs', 'b', 'tgt_t', 'tgt_v',
    'tgt_h', 'tgt_w', 'feats', 'concat_feat_render', 'means_batched', 'quats_batched', 'scales_batched',
    'opacities_batched', 'feats_batched', 'viewmats_batched', 'ks_batched', 'radius_clip'
])
_RunHeadsParams = namedtuple('_RunHeadsParams', ['output_list', 'images', 'b', 't', 'v', 'camera_head_kv_cache_list'])
_RunPredictorsParams = namedtuple('_RunPredictorsParams', [
    'last_tokens', 'images', 'motion_tokens', 'ray_origins', 'ray_dirs', 'activated_depth', 'h', 'w', 't', 'v',
    'b', 'output_list', 'data_dict'
])
_ForwardGsPredictorParams = namedtuple('_ForwardGsPredictorParams', [
    'x', 'origins', 'directions', 'activated_depth', 'rgb', 'dense_feat'
])
_ForwardGsPredictorParams.__new__.__defaults__ = (None, None, False)
_PrepareGsBatchedResult = namedtuple('_PrepareGsBatchedResult', [
    'means_batched', 'scales_batched', 'quats_batched', 'opacities_batched', 'color_batched',
    'feats_batched', 'tdiff_forward_batched'
])
_SliceGsAttrsByIdxResult = namedtuple('_SliceGsAttrsByIdxResult', [
    'means_batched', 'scales_batched', 'quats_batched', 'opacities_batched', 'color_batched',
    'forward_v_batched'
])
_ApplyVoxelizationResult = namedtuple('_ApplyVoxelizationResult', [
    'gs_attrs', 'means_batched', 'scales_batched', 'quats_batched', 'opacities_batched',
    'color_batched', 'forward_v_batched', 'feats_batched'
])
_RunHeadsResult = namedtuple('_RunHeadsResult', [
    'pose_enc_list', 'camera_head_kv_cache_list', 'pred_ray_dict', 'pred_context_depth',
    'pred_context_depth_conf', 'pred_context_pts3d', 'pred_context_pts3d_conf'
])


class SLARM(nn.Module, PyTorchModelHubMixin, SLARMPlyExportMixin, SLARMNovelViewMixin):
    '''Streaming and Language-Aligned Reconstruction Model for Dynamic Scenes'''

    output_format = 'bcthw'

    def __init__(
            self,
            img_size=None,
            num_cams=3,  # to ablate
            gs_dim=3,
            in_chans=3,
            embed_dim=1024,
            patch_size=14,
            decoder_type="dummy",
            # depth
            near=0.2,
            far=400,
            # model config
            disable_pos_embed=False,
            use_sky_token=True,
            use_affine_token=False,  # only needed for multiview
            use_pred_camera_pose=False,
            use_pred_depth=False,
            use_time_token=True,
            add_patch_plucker_embed=True,
            add_camera_embed=True,
            concat_plucker_embed=True,
            shortcut_rgb=True,
            pred_gs_conf=False,
            enable_lifespan=False,
            use_last_token=False,
            enable_depth_head=False,
            enable_camera_head=False,
            enable_point_head=False,
            use_ms3_motion=False,
            add_angular_velocity=False,
            render_context_view=False,
            render_context_frame_contribution=False,
            voxelize=False,
            voxel_size=0.2,
            similarity_probs_threshold=0.2,
            # patch emb
            patch_embed="dinov2_vitl14_reg",
            num_register_tokens=4,
            # attention block
            depth=24,
            # head
            gs_dense_reg_head_type='mlp',  # 'conv'
            feat_dense_reg_head_type='mlp',  # 'conv'
            motion_dense_key_head_type='mlp',  # 'conv'
            num_motion_tokens=0,
            tau=0.5,
            projected_motion_dim=32,
            pred_feat_dim=64,
            with_feat=True,
            # gs activation
            gs_marbles=False,
            max_scale=0.5,
            opacity_offset=-2.0,
            sigmoid_rgb=True,  # legacy oversight: sigmoid was omitted
            # other
            grad_checkpointing=True,
            vggt_pretrained_weight_filepath='',
            use_2dgs=False,
            pesudo_3dgs=False,
            save_gaussian=False,
            gaussian_save_path='output_gs',
            save_rendered_pc=False,
            rendered_pc_save_path='output_rendered_pc',
            use_render_novel_view=False,
            # ms3 motion
            ms3_deg=3,
            omega_deg=3,
            ms3_deg_downmax_mult=1.0,
            sigmoid_ms3_bias=-6.9068,
            sigmoid_ms3_min=0.0,
            sigmoid_ms3_max=100,  # 2.0
            ms3_clamp=0.0001,
            # stream
            mode="full",
            **kwargs,
    ):
        super().__init__()

        num_velocity_channels = self._set_basic_attrs(_SetBasicAttrsParams(
            img_size, embed_dim, patch_size, num_cams, gs_dim, depth,
            num_register_tokens, patch_embed, near, far, use_last_token,
            enable_depth_head, enable_camera_head, enable_point_head,
            use_pred_camera_pose, use_pred_depth, render_context_view,
            render_context_frame_contribution, concat_plucker_embed,
            add_patch_plucker_embed, add_camera_embed, in_chans, shortcut_rgb,
            gs_dense_reg_head_type, feat_dense_reg_head_type, motion_dense_key_head_type,
            pred_gs_conf, voxelize, voxel_size, similarity_probs_threshold,
            disable_pos_embed, projected_motion_dim, pred_feat_dim, with_feat,
            decoder_type, grad_checkpointing, vggt_pretrained_weight_filepath,
            use_2dgs, pesudo_3dgs, save_gaussian, gaussian_save_path,
            save_rendered_pc, rendered_pc_save_path, use_render_novel_view,
            use_time_token, use_sky_token, use_affine_token, mode,
            use_ms3_motion, add_angular_velocity, enable_lifespan,
            ms3_deg, omega_deg, ms3_deg_downmax_mult, sigmoid_ms3_bias,
            sigmoid_ms3_min, sigmoid_ms3_max, ms3_clamp, num_motion_tokens, tau,
        ))

        self._configure_rendering()

        self._set_intermediate_layer_idx(depth)

        self._build_aggregator()

        self._build_heads(embed_dim)

        # ------- embedders -------
        self.plucker_embedder = self.aggregator.plucker_embedder

        self._build_motion_and_tokens(embed_dim, projected_motion_dim, num_velocity_channels)

        if self.use_last_token:
            self._build_last_token_heads(embed_dim)
        else:
            self._build_dense_reg_heads(embed_dim, projected_motion_dim)

        self._build_feat_decoder()

        self._setup_activations(_SetupActivationsParams(gs_marbles, max_scale, opacity_offset, near, far, sigmoid_rgb))

        self.init_weights()

        self._load_pretrained_weights()

    def init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

    def load_pretrained_vggt(self, vggt_ckpts_filepath=''):
        vggt_pretrained_weight = torch.load(vggt_ckpts_filepath)

        self._load_vggt_module(vggt_pretrained_weight, 'patch_embed', 'aggregator.patch_embed.',
                               self.aggregator.patch_embed)
        self._load_vggt_module(vggt_pretrained_weight, 'global_blocks', 'aggregator.global_blocks.',
                               self.aggregator.global_blocks)
        self._load_vggt_module(vggt_pretrained_weight, 'frame_blocks', 'aggregator.frame_blocks.',
                               self.aggregator.frame_blocks)

        if self.depth_head is not None:
            self._load_vggt_module(vggt_pretrained_weight, 'depth_head', 'depth_head.',
                                   self.depth_head)
        if self.camera_head is not None:
            self._load_vggt_module(vggt_pretrained_weight, 'camera_head', 'camera_head.',
                                   self.camera_head)
        if self.point_head is not None:
            self._load_vggt_module(vggt_pretrained_weight, 'point_head', 'point_head.',
                                   self.point_head)

        if self.aggregator.camera_token is not None:
            camera_token_state_dict = {}
            for old_key, value in vggt_pretrained_weight.items():
                if 'camera_token' in old_key:
                    camera_token_state_dict[old_key.removeprefix('aggregator.')] = value
            camera_token = camera_token_state_dict.get('camera_token')
            if camera_token is not None:
                self.aggregator.camera_token.data = camera_token

        if self.aggregator.register_token is not None:
            register_token_state_dict = {}
            for old_key, value in vggt_pretrained_weight.items():
                if 'register_token' in old_key:
                    register_token_state_dict[old_key.removeprefix('aggregator.')] = value
            register_token = register_token_state_dict.get('register_token')
            if register_token is not None:
                self.aggregator.register_token.data = register_token

    def unpatchify(self, x, hw=None, channel_first=True, patch_size=None) -> torch.Tensor:
        hw = hw or self.img_size
        imgs = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            p1=self.patch_size if patch_size is None else patch_size,
            p2=self.patch_size if patch_size is None else patch_size,
            h=hw[0] // (self.patch_size if patch_size is None else patch_size),
            w=hw[1] // (self.patch_size if patch_size is None else patch_size),
        )
        if not channel_first:
            imgs = rearrange(imgs, "b c h w -> b h w c")
        return imgs

    def forward_decoder(self, render_results):
        render_results["rgb_key"] = "rendered_image"
        render_results["depth_key"] = "rendered_depth"
        render_results["alpha_key"] = "rendered_alpha"
        render_results["flow_key"] = "rendered_flow"
        render_results["decoder_depth_key"] = None
        render_results["decoder_alpha_key"] = None
        render_results["decoder_flow_key"] = None
        render_results = self.decoder(render_results)
        decoded_depth_key = render_results["decoder_depth_key"]
        if decoded_depth_key is not None:
            decoded_depth = self.depth_act_fn(render_results[decoded_depth_key])
            render_results[decoded_depth_key] = decoded_depth
        return render_results

    def decode_flow(self, ms3):
        # Extract degree of marginal scale (number of scale components)
        ms3_deg = ms3.shape[-1] // 4
        # Extract speed components (every 4th element starting from index 3)
        speed = ms3[..., 3::4, None]  # [B, T, H, W, ms3_deg, 1]
        # Reshape spatial components (first 3 of every 4 elements)
        ms3 = torch.cat(
            [ms3[..., None, i * 4:i * 4 + 3] for i in range(ms3_deg)], dim=-2
        )  # [B, T, H, W, ms3_deg, 3]

        # Rescale speed with sigmoid and apply clamping threshold
        speed = (speed + self.sigmoid_ms3_bias).sigmoid() * (
                self.sigmoid_ms3_max - self.sigmoid_ms3_min
        ) + self.sigmoid_ms3_min
        speed = (speed - self.ms3_clamp).clamp(0)  # Zero out speeds below threshold

        # Apply decay factor to speed based on scale level
        # Higher scale levels get progressively smaller speeds
        speed = torch.cat(
            [speed[..., i:i + 1, :] / self.ms3_deg_downmax_mult ** i
             for i in range(ms3_deg)], dim=-2
        )  # [B, T, H, W, ms3_deg, 1]

        # Apply speed-modulated normalized marginal scales
        ms3 = speed * F.normalize(ms3[..., :3], dim=-1)  # Normalize and modulate by speed
        ms3 = ms3.reshape(ms3.shape[:-2] + (-1,))  # Flatten to [B, T, H, W, ms3_deg*3]
        return ms3

    def forward_motion_predictor(self, x, motion_tokens=None, gs_params=None, dense_feat=False):
        b, t, v, h, w, _ = gs_params["means"].shape
        img_keys = self._compute_motion_img_keys(_ComputeMotionImgKeysParams(x, b, t, v, h, w, dense_feat))

        if self.num_motion_tokens > 0:
            hyper_in_list = []
            for i in range(self.num_motion_tokens):
                hyper_in = self.motion_query_heads[i](motion_tokens[:, i])
                hyper_in_list.append(hyper_in)
            motion_token_queries = torch.stack(hyper_in_list, dim=1)
            dot_product_similarity = torch.einsum(
                "b k c, b t v h w c -> b t v h w k",
                motion_token_queries,
                img_keys,
            )
            motion_weights = torch.softmax(dot_product_similarity / self.tau, dim=-1)
            motion_bases = self.motion_basis_decoder(motion_tokens)
            motion_final = torch.einsum(
                "b t v h w k, b k c -> b t v h w c", motion_weights, motion_bases
            )
            gs_params["motion_weights"] = motion_weights
            gs_params["motion_bases"] = motion_bases
        else:
            # if there's no motion token, directly predict the velocity from the upsampled image features
            motion_final = self.motion_basis_decoder(img_keys)

        if self.use_ms3_motion:
            if self.add_angular_velocity:
                ms3, omega = motion_final.split([4 * self.ms3_deg, 4 * self.omega_deg], dim=-1)
                forward_ms3 = torch.concat([self.decode_flow(ms3), self.decode_flow(omega)], dim=-1)
            else:
                ms3 = motion_final
                forward_ms3 = self.decode_flow(ms3)
            gs_params["forward_ms3"] = forward_ms3
        else:
            gs_params["forward_flow"] = motion_final
        return {k: v for k, v in gs_params.items() if v is not None}

    def forward_gs_predictor(self, params: _ForwardGsPredictorParams):
        (x, origins, directions, activated_depth, rgb, dense_feat) = params
        b, t, v, h, w, _ = origins.shape
        gs_params = self._compute_gs_params_raw(_ComputeGsParamsRawParams(x, b, t, v, h, w, rgb, dense_feat))
        gs_params_dict = dict(zip(self.gs_params_name, gs_params.split(self.gs_params_size, dim=-1)))
        if activated_depth is not None:
            depths = activated_depth
        else:
            depths = self.depth_act_fn(gs_params_dict["depth"])
        means = origins + directions * depths

        # pesudo_3dgs
        if self.pesudo_3dgs:
            scales, quats = self._compute_pseudo_3dgs_scales_quats(
                _ComputePseudo3dgsScalesQuatsParams(means, depths, b, t, v, h, w)
            )
        else:
            scales = self.scale_act_fn(gs_params_dict["scales"])
            quats = self.quat_act_fn(gs_params_dict["quats"])
        colors = self.rgb_act_fn(gs_params_dict["colors"])
        opacitys = self.opacity_act_fn(gs_params_dict["opacitys"])
        output = {
            "means": means,
            "scales": scales,
            "quats": quats,
            "opacities": opacitys.squeeze(-1),
            "colors": colors,
            "depths": depths.squeeze(-1),
        }
        if self.pred_gs_conf:
            confs = self.gs_conf_act_fn(gs_params_dict["confs"])
            output["confs"] = confs
        if self.enable_lifespan:
            lifespans = self.lifespan_act_fn(gs_params_dict["lifespans"])
            output["lifespans"] = lifespans.squeeze(-1)
        return output

    def forward_feat_predictor(self, x, shape, dense_feat=False):
        h, w, t, v = shape
        if dense_feat:
            if self.feat_dense_reg_head_type == 'mlp':
                pred_feat = self.feat_dense_reg_head(rearrange(x, "b (t v) c h w -> b t v h w c", t=t))
            elif self.feat_dense_reg_head_type == 'conv':
                pred_feat = self.feat_dense_reg_head(rearrange(x, "b tv c h w -> (b tv) c h w"))
                pred_feat = rearrange(pred_feat, "(b t v) c h w -> b t v h w c", t=t, v=v)
        else:
            x = rearrange(x, "b tv hw c -> (b tv) hw c")
            pred_feat = self.feat_pred(x)  # [48, 600, 768] -> [48, 600, 768(12*8*8)]
            pred_feat = self.unpatchify(pred_feat, hw=(h, w), patch_size=self.unpatch_size)
            pred_feat = rearrange(pred_feat, "(b t v) c h w -> b t v h w c", t=t, v=v)
        pred_feat = torch.sigmoid(pred_feat)
        return pred_feat

    def forward_renderer_context_view(self, gs_params, data_dict, radius_clip=0.0):
        b, t, v, h, w, _ = gs_params["means"].shape
        tgt_h, tgt_w = data_dict["height"], data_dict["width"]
        tgt_t, tgt_v = data_dict["context_camtoworlds"].shape[1:3]
        means = rearrange(gs_params["means"], "b t v h w c -> (b t v) (h w) c")
        scales = rearrange(gs_params["scales"], "b t v h w c -> (b t v) (h w) c")
        quats = rearrange(gs_params["quats"], "b t v h w c -> (b t v) (h w) c")
        opacities = rearrange(gs_params["opacities"], "b t v h w -> (b t v) (h w)")
        colors = rearrange(gs_params["colors"], "b t v h w c -> (b t v) (h w) c")

        camtoworlds_batched = data_dict["context_camtoworlds"].view(b * tgt_t * v, -1, 4, 4)
        viewmats_batched = torch.linalg.inv(camtoworlds_batched.float())
        ks_batched = data_dict["context_intrinsics"].view(b * tgt_t * v, -1, 3, 3)

        with torch.autocast("cuda", enabled=False):
            rendered_color, rendered_alpha, _ = self.rasterization_func(
                gaussians=GaussianData(means=means.float(), quats=quats.float(), scales=scales.float(),
                                       opacities=opacities.float(), colors=colors.float()),
                camera=CameraParams(viewmats=viewmats_batched, ks=ks_batched, camera_model="pinhole"),
                config=RenderConfig(width=tgt_w, height=tgt_h, render_mode="RGB+ED",
                                    near_plane=self.near, far_plane=self.far,
                                    packed=False, radius_clip=radius_clip),
            )
        color, depth = rendered_color.split([self.gs_dim, 1], dim=-1)

        output_dict = {
            "rendered_image": color.view(b, tgt_t, tgt_v, tgt_h, tgt_w, -1),
            "rendered_depth": depth.view(b, tgt_t, tgt_v, tgt_h, tgt_w),
            "rendered_alpha": rendered_alpha.view(b, tgt_t, tgt_v, tgt_h, tgt_w),
        }
        return output_dict

    def forward_renderer_target_view_feat(self, render_results, data_dict, pred_feats, radius_clip=0.0, chunk_size=3):
        ''' render semantic probability '''
        b, t, v, _, h, w = data_dict['context_image'].shape
        tgt_h, tgt_w = data_dict["height"], data_dict["width"]
        tgt_t, tgt_v = data_dict["target_camtoworlds"].shape[1:3]

        means = rearrange(render_results["gs_means"], "b tgt_t (tvhw) c -> (b tgt_t) (tvhw) c")
        scales = rearrange(render_results["gs_scales"], "b tgt_t (tvhw) c -> (b tgt_t) (tvhw) c")
        quats = rearrange(render_results["gs_quats"], "b tgt_t (tvhw) c -> (b tgt_t) (tvhw) c")
        opacities = rearrange(render_results["gs_opacities"], "b tgt_t (tvhw) c -> (b tgt_t) (tvhw c)")

        pred_feats = rearrange(pred_feats, "... c -> (...) c")
        probs = feat2class(
            pred_feats,
            get_text_label_feats(SEMANTIC_LABEL_LIST),
            similarity_probs_threshold=self.similarity_probs_threshold,
            return_probs=True
        )
        probs = rearrange(probs, '(b t v h w) c -> b (t v h w) c', b=b, t=t, v=v, h=h, w=w)
        probs_batched = repeat(probs, "b ... -> (b t) ...", t=tgt_t)

        camtoworlds_batched = data_dict["target_camtoworlds"].view(b * tgt_t, -1, 4, 4)
        viewmats_batched = torch.linalg.inv(camtoworlds_batched.float())
        ks_batched = data_dict["target_intrinsics"].view(b * tgt_t, -1, 3, 3)

        # NOTE: current npu rendering only supports 3D color, pad with zeros if not divisible by 3
        class_num = probs.shape[-1]
        pad_len = (chunk_size - class_num % chunk_size) % chunk_size

        if pad_len > 0:
            probs_batched = F.pad(probs_batched, (0, pad_len), mode='constant', value=0)

        rendered_probs = self._render_probs_in_chunks(_RenderProbsInChunksParams(
            means, quats, scales, opacities, probs_batched,
            viewmats_batched, ks_batched, tgt_h, tgt_w, radius_clip, chunk_size
        ))
        rendered_probs = rendered_probs[..., :class_num]

        # argmax
        rendered_semantic = torch.argmax(rendered_probs, dim=-1)
        rendered_semantic = rendered_semantic.long()
        return rendered_semantic.view(b, tgt_t, tgt_v, tgt_h, tgt_w)

    def forward_renderer(self, gs_params, data_dict, feats=None, render_motion_seg=not is_ascend_npu(),
                         radius_clip=0.0, time_step=5, concat_feat_render=True, idx=None, static_render=False):
        if os.getenv("CONTEXT_FEAT"):
            feats = None
        b, t, v, h, w, _ = gs_params["means"].shape
        tgt_h, tgt_w = data_dict["height"], data_dict["width"]
        tgt_t, tgt_v = data_dict["target_camtoworlds"].shape[1:3]

        means_batched, scales_batched, quats_batched, opacities_batched, \
            color_batched, feats_batched, tdiff_forward_batched = self._prepare_gs_batched(_PrepareGsBatchedParams(
            gs_params, data_dict, feats, b, t, v, h, w, tgt_t
        ))
        ctx_time = data_dict["context_time"] * data_dict["timespan"]

        forward_v_batched, means_batched, quats_batched, opacities_batched = self._apply_motion(_ApplyMotionParams(
            gs_params, ctx_time, tdiff_forward_batched, means_batched, quats_batched,
            opacities_batched, tgt_t, time_step, static_render, data_dict
        ))

        if not self.training:
            forward_v_batched[forward_v_batched.norm(dim=-1) < 1.0] = 0.0

        if self.enable_lifespan:
            opacities_batched = self._apply_lifespan(
                gs_params, tdiff_forward_batched, opacities_batched, tgt_t
            )

        if idx is not None:
            means_batched, scales_batched, quats_batched, opacities_batched, \
                color_batched, forward_v_batched = self._slice_gs_attrs_by_idx(_SliceGsAttrsByIdxParams(
                means_batched, scales_batched, quats_batched, opacities_batched,
                color_batched, forward_v_batched, idx, v, h, w
            ))

        gs_attrs = self._build_gs_attrs(_BuildGsAttrsParams(
            means_batched, scales_batched, quats_batched, opacities_batched,
            color_batched, forward_v_batched, feats, feats_batched
        ))

        if self.voxelize and self.pred_gs_conf:
            gs_attrs, means_batched, scales_batched, quats_batched, opacities_batched, \
                color_batched, forward_v_batched, feats_batched = self._apply_voxelization(_ApplyVoxelizationParams(
                gs_attrs, gs_params, idx, v, h, w, b, tgt_t
            ))

        colors_batched, forward_flow = self._prepare_render_colors(_PrepareRenderColorsParams(
            color_batched, forward_v_batched, feats_batched, concat_feat_render, idx, v, h, w
        ))

        if not self.training and self.num_motion_tokens > 0 and render_motion_seg:
            colors_batched = self._append_motion_weights(_AppendMotionWeightsParams(
                gs_params, colors_batched, tgt_t, idx, v, h, w
            ))

        return self._execute_rendering(_ExecuteRenderingParams(
            data_dict, gs_attrs, gs_params, means_batched, scales_batched, quats_batched,
            opacities_batched, color_batched, forward_v_batched, feats_batched,
            colors_batched, forward_flow, feats, render_motion_seg, concat_feat_render,
            b, t, v, h, w, tgt_t, tgt_v, tgt_h, tgt_w, radius_clip
        ))

    def voxelizaton_using_confidence(self, gs_xyz, gs_conf, voxel_size):
        voxel_indices = (gs_xyz / voxel_size).round().int()  # [N, 3]
        unique_voxels, inverse_indices, counts = torch.unique(
            voxel_indices, dim=0, return_inverse=True, return_counts=True
        )

        # Compute softmax weights per voxel
        conf_voxel_max, _ = scatter_max(gs_conf, inverse_indices, dim=0)
        conf_exp = torch.exp(gs_conf - conf_voxel_max[inverse_indices])
        voxel_weights = scatter_add(conf_exp, inverse_indices, dim=0)  # [num_unique_voxels]
        weights = (conf_exp / (voxel_weights[inverse_indices] + 1e-6)).unsqueeze(-1)  # [N, 1]

        return weights, inverse_indices

    def get_ray_dict(self, data_dict):
        ray_dict = self.plucker_embedder(
            data_dict["context_intrinsics"],
            data_dict["context_camtoworlds"],
            image_size=data_dict["context_image"].shape[-2:],
        )
        if self.decoder_type != "dummy":
            feat_ray_dict = self.plucker_embedder(
                data_dict["context_intrinsics"],
                data_dict["context_camtoworlds"],
                image_size=data_dict["context_image"].shape[-2:],
                patch_size=self.patch_size,
            )
            ray_dict["origins"] = feat_ray_dict["origins"]
            ray_dict["dirs"] = feat_ray_dict["dirs"]

            tgt_intrinsics = data_dict["target_intrinsics"]
            tgt_intrinsics[..., 0, 0] = tgt_intrinsics[..., 0, 0] / self.patch_size
            tgt_intrinsics[..., 1, 1] = tgt_intrinsics[..., 1, 1] / self.patch_size
            tgt_intrinsics[..., 0, 2] = tgt_intrinsics[..., 0, 2] / self.patch_size
            tgt_intrinsics[..., 1, 2] = tgt_intrinsics[..., 1, 2] / self.patch_size
            data_dict["target_intrinsics"] = tgt_intrinsics
            data_dict["width"] //= self.patch_size
            data_dict["height"] //= self.patch_size
        return data_dict, ray_dict

    def forward(self,
                data_dict,
                stream_save=True,
                aggregator_kv_cache_list: List[List[torch.Tensor]] = None,
                camera_head_kv_cache_list: List[List[List[torch.Tensor]]] = None):
        if os.environ.get('TIME_COUNT_TYPE2'):
            start = time.time()
        images = data_dict["context_image"]
        b, t, v, c, h, w = images.size()

        # GT camera pose
        _, ray_dict = self.get_ray_dict(data_dict)

        # Normalize images and reshape for patch embed
        # NOTE: apply this normalization when rgb is in 0~1 range
        images = (images - self.aggregator.resnet_mean) / self.aggregator.resnet_std  # (b, t, v, c, h, w)

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(b * t * v, c, h, w)

        output_list, self.patch_start_idx, aggregator_kv_cache_list = self._run_aggregator(
            data_dict, aggregator_kv_cache_list
        )

        with torch.cuda.amp.autocast(enabled=False):
            last_tokens = output_list[-1]
            others_last_tokens = last_tokens[:, :, :self.patch_start_idx]  # Exclude patch token

            sky_token, affine_tokens, motion_tokens, time_tokens = self._extract_tokens(others_last_tokens)

            pose_enc_list, camera_head_kv_cache_list, pred_ray_dict, \
                pred_context_depth, pred_context_depth_conf, \
                pred_context_pts3d, pred_context_pts3d_conf = self._run_heads(_RunHeadsParams(
                output_list, images, b, t, v, camera_head_kv_cache_list
            ))

            # switch between camera pose + depth and pointmap
            if self.use_pred_camera_pose:
                ray_origins = pred_ray_dict["origins"]
                ray_dirs = pred_ray_dict["dirs"]
            else:
                ray_origins = ray_dict["origins"]
                ray_dirs = ray_dict["dirs"]
            if self.use_pred_depth:
                activated_depth = pred_context_depth
            else:
                activated_depth = None

            gs_params, pred_feat = self._run_predictors(_RunPredictorsParams(
                last_tokens, images, motion_tokens, ray_origins, ray_dirs,
                activated_depth, h, w, t, v, b, output_list, data_dict
            ))

        if os.environ.get('TIME_COUNT_TYPE2'):
            torch.cuda.synchronize()
            logger.info('Computation time - forward: %s', time.time() - start)

        return self._finalize_forward(_FinalizeForwardParams(
            data_dict, gs_params, ray_dict, pred_feat,
            sky_token, affine_tokens, pose_enc_list,
            pred_context_depth, pred_context_depth_conf,
            pred_context_pts3d, pred_context_pts3d_conf,
            stream_save, aggregator_kv_cache_list, camera_head_kv_cache_list
        ))

    def _set_intermediate_layer_idx(self, depth):
        if depth == 12:
            self.intermediate_layer_idx = [2, 5, 8, 11]
        elif depth == 24:
            self.intermediate_layer_idx = [4, 11, 17, 23]
        else:
            raise ValueError('only support depth layer 12 or 24!')

    def _build_aggregator(self):
        self.aggregator = Aggregator(
            in_chans=self.in_chans,
            num_cams=self.num_cams,
            img_size=self.img_size,
            patch_size=self.patch_size,
            decoder_type=self.decoder_type,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_register_tokens=self.num_register_tokens,
            patch_embed=self.patch_embed,
            num_motion_tokens=self.num_motion_tokens,
            use_time_token=self.use_time_token,
            use_sky_token=self.use_sky_token,
            use_affine_token=self.use_affine_token,
            concat_plucker_embed=self.concat_plucker_embed,
            add_patch_plucker_embed=self.add_patch_plucker_embed,
            add_camera_embed=self.add_camera_embed,
            grad_checkpointing=self.grad_checkpointing
        )

    def _build_last_token_heads(self, embed_dim):
        self.aggregated_last_tokens_norm = nn.LayerNorm(2 * embed_dim)
        self._build_gshead_and_motionhead_without_dpthead(
            embed_dim=embed_dim,
            decoder_upsample_ratio=self.decoder_upsample_ratio,
            projected_motion_dim=self.projected_motion_dim,
            decoder_type=self.decoder_type,
            grad_checkpointing=self.grad_checkpointing
        )
        if self.with_feat:
            self._build_feat_head_without_dpthead(
                embed_dim=embed_dim,
                decoder_upsample_ratio=self.decoder_upsample_ratio,
                decoder_type=self.decoder_type
            )

    def _build_feat_decoder(self):
        if self.with_feat:
            self.feat_decoder = nn.Sequential(  # default 64 -> 512
                nn.Linear(self.pred_feat_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512)
            )
            # Adapt to existing model checkpoints: feat_decoders.0.xx
            self.feat_decoders = nn.ModuleList([self.feat_decoder])
            '''
            Will still report: Missing key(s) in state_dict: "feat_decoder.xxx"
            But won't report: Unexpected key(s) in state_dict: "feat_decoders.0.xxx"
            '''

    def _load_pretrained_weights(self):
        if os.path.exists(self.vggt_pretrained_weight_filepath):
            if self.aggregator.depth != 24 or self.aggregator.embed_dim != 1024 or self.patch_size != 14:
                raise ValueError(
                    "vggt_pretrained_weight_filepath requires aggregator.depth==24, "
                    "embed_dim==1024, patch_size==14"
                )
            self.load_pretrained_vggt(self.vggt_pretrained_weight_filepath)  # additional learning rate

            def zero_module(module):
                """
                Zero out the parameters of a module and return it.
                """
                for p in module.parameters():
                    p.detach().zero_()
                return module

            # zero initialization
            if self.add_patch_plucker_embed:
                self.aggregator.patch_plucker_embed_mlp = zero_module(self.aggregator.patch_plucker_embed_mlp)
            if self.add_camera_embed:
                self.aggregator.pose_encoding_mlp = zero_module(self.aggregator.pose_encoding_mlp)

    def _set_prediction_render_attrs(self, params: _SetBasicAttrsParams):
        self.use_pred_camera_pose = params.use_pred_camera_pose
        self.use_pred_depth = params.use_pred_depth
        self.render_context_view = params.render_context_view
        self.render_context_frame_contribution = params.render_context_frame_contribution
        self.concat_plucker_embed = params.concat_plucker_embed
        self.add_patch_plucker_embed = params.add_patch_plucker_embed
        self.add_camera_embed = params.add_camera_embed
        self.in_chans = params.in_chans
        self.shortcut_rgb = params.shortcut_rgb
        self.gs_dense_reg_head_type = params.gs_dense_reg_head_type
        self.feat_dense_reg_head_type = params.feat_dense_reg_head_type
        self.motion_dense_key_head_type = params.motion_dense_key_head_type
        self.pred_gs_conf = params.pred_gs_conf
        self.voxelize = params.voxelize
        self.voxel_size = params.voxel_size

    def _set_basic_attrs(self, params: _SetBasicAttrsParams):
        (
            img_size, embed_dim, patch_size, num_cams, gs_dim, depth, num_register_tokens, patch_embed, near, far,
            use_last_token, enable_depth_head, enable_camera_head, enable_point_head, use_pred_camera_pose,
            use_pred_depth, render_context_view, render_context_frame_contribution, concat_plucker_embed,
            add_patch_plucker_embed, add_camera_embed, in_chans, shortcut_rgb, gs_dense_reg_head_type,
            feat_dense_reg_head_type, motion_dense_key_head_type, pred_gs_conf, voxelize, voxel_size,
            similarity_probs_threshold, disable_pos_embed, projected_motion_dim, pred_feat_dim, with_feat,
            decoder_type, grad_checkpointing, vggt_pretrained_weight_filepath, use_2dgs, pesudo_3dgs,
            save_gaussian, gaussian_save_path, save_rendered_pc, rendered_pc_save_path, use_render_novel_view,
            use_time_token, use_sky_token, use_affine_token, mode, use_ms3_motion, add_angular_velocity,
            enable_lifespan, ms3_deg, omega_deg, ms3_deg_downmax_mult, sigmoid_ms3_bias, sigmoid_ms3_min,
            sigmoid_ms3_max, ms3_clamp, num_motion_tokens, tau
        ) = params
        if img_size is None:
            img_size = [168, 252]
        self._set_core_attrs(_SetCoreAttrsParams(
            img_size, embed_dim, patch_size, num_cams, gs_dim, depth,
            num_register_tokens, patch_embed, near, far, use_last_token,
            enable_depth_head, enable_camera_head, enable_point_head
        ))
        self._set_prediction_render_attrs(params)
        self._validate_config()
        self.similarity_probs_threshold = similarity_probs_threshold
        self._setup_gs_params(enable_lifespan)
        self.num_motion_tokens = num_motion_tokens
        self.tau = tau
        self.use_ms3_motion = use_ms3_motion
        self.add_angular_velocity = add_angular_velocity
        num_velocity_channels = self._set_motion_attrs(_SetMotionAttrsParams(
            use_ms3_motion, add_angular_velocity, ms3_deg, omega_deg,
            ms3_deg_downmax_mult, sigmoid_ms3_bias, sigmoid_ms3_min,
            sigmoid_ms3_max, ms3_clamp
        ))
        if mode != "full" and mode != "causal" and not bool(re.match(r'^window_(\d+)$', mode)):
            raise ValueError(f"mode must be 'full', 'causal', or match pattern 'window_(\\d+)', got {mode}")
        self._set_remaining_attrs(_SetRemainingAttrsParams(
            mode, use_time_token, use_sky_token, use_affine_token, disable_pos_embed,
            projected_motion_dim, pred_feat_dim, with_feat, decoder_type,
            grad_checkpointing, vggt_pretrained_weight_filepath, use_2dgs, pesudo_3dgs,
            save_gaussian, gaussian_save_path, save_rendered_pc,
            rendered_pc_save_path, use_render_novel_view
        ))
        return num_velocity_channels

    def _set_core_attrs(self, params: _SetCoreAttrsParams):
        (
            img_size, embed_dim, patch_size, num_cams, gs_dim, depth, num_register_tokens, patch_embed, near, far,
            use_last_token, enable_depth_head, enable_camera_head, enable_point_head
        ) = params
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_cams = num_cams
        self.gs_dim = gs_dim
        self.depth = depth
        self.num_register_tokens = num_register_tokens
        self.patch_embed = patch_embed
        self.near = near
        self.far = far
        self.use_last_token = use_last_token
        self.enable_depth_head = enable_depth_head
        self.enable_camera_head = enable_camera_head
        self.enable_point_head = enable_point_head

    def _set_remaining_attrs(self, params: _SetRemainingAttrsParams):
        (
            mode, use_time_token, use_sky_token, use_affine_token, disable_pos_embed, projected_motion_dim,
            pred_feat_dim, with_feat, decoder_type, grad_checkpointing, vggt_pretrained_weight_filepath, use_2dgs,
            pesudo_3dgs, save_gaussian, gaussian_save_path, save_rendered_pc, rendered_pc_save_path,
            use_render_novel_view
        ) = params
        self.mode = mode
        self.use_time_token = use_time_token
        self.use_sky_token = use_sky_token
        self.use_affine_token = use_affine_token
        self.disable_pos_embed = disable_pos_embed
        self.projected_motion_dim = projected_motion_dim
        self.pred_feat_dim = pred_feat_dim
        self.with_feat = with_feat
        self.decoder_type = decoder_type
        if self.decoder_type == "dummy":
            self.decoder = DummyDecoder()
        self.decoder_upsample_ratio = self.patch_size
        self.grad_checkpointing = grad_checkpointing
        self.vggt_pretrained_weight_filepath = vggt_pretrained_weight_filepath
        self.use_2dgs = use_2dgs
        self.pesudo_3dgs = pesudo_3dgs
        self.save_gaussian = save_gaussian
        self.gaussian_save_path = gaussian_save_path
        self.save_rendered_pc = save_rendered_pc
        self.rendered_pc_save_path = rendered_pc_save_path
        self.use_render_novel_view = use_render_novel_view
        self.use_reentrant = False

    def _configure_rendering(self):
        if is_ascend_npu():
            self.use_2dgs = False
            self.gs_renderer_npu = Rasterizer(tile_size=32, camera_model='pinhole')
            self.rasterization_func = functools.partial(
                new_ascend_rasterization,
                ascend_render=self.gs_renderer_npu
            )
        else:
            if not self.use_2dgs:
                self.rasterization_func = rasterization
            else:
                self.rasterization_func = rasterization_2dgs

    def _build_heads(self, embed_dim):
        # 3D annotation head
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if self.use_pred_camera_pose else None
        self.depth_head = (
            DPTHead(dim_in=2 * embed_dim, output_dim=2,
                    intermediate_layer_idx=self.intermediate_layer_idx,
                    patch_size=self.patch_size,
                    activation="exp", conf_activation="expp1")
            if self.use_pred_depth else None
        )
        self.point_head = (
            DPTHead(dim_in=2 * embed_dim, output_dim=4,
                    intermediate_layer_idx=self.intermediate_layer_idx,
                    patch_size=self.patch_size,
                    activation="inv_log", conf_activation="expp1")
            if self.enable_point_head else None
        )

    def _build_motion_and_tokens(self, embed_dim, projected_motion_dim, num_velocity_channels):
        if self.num_motion_tokens > 0:
            self.motion_token_norm = nn.LayerNorm(2 * embed_dim)
            self.motion_query_heads = nn.ModuleList(
                [
                    Mlp(2 * embed_dim, 2 * embed_dim, projected_motion_dim)
                    for _ in range(self.num_motion_tokens)
                ]
            )
            self.motion_basis_decoder = Mlp(2 * embed_dim, 256, num_velocity_channels)
        else:
            self.motion_basis_decoder = Mlp(projected_motion_dim, 256, num_velocity_channels)

        if self.use_affine_token:
            self.affine_token_norm = nn.LayerNorm(2 * embed_dim)
            self.affine_linear = nn.Linear(2 * embed_dim, self.gs_dim * (self.gs_dim + 1))

        if self.use_sky_token:
            self.sky_token_norm = nn.LayerNorm(2 * embed_dim)
            self.sky_head = ModulatedLinearLayer(
                3,
                hidden_channels=512,
                condition_channels=2 * embed_dim,
                out_channels=self.gs_dim,
            )

    def _build_dense_reg_heads(self, embed_dim, projected_motion_dim):
        # gs head
        self.dense_feats_dim = 256  # 256 is dpt default feature dim
        self.gs_feature_head = DPTHead(
            dim_in=2 * embed_dim,
            feature_only=True,
            intermediate_layer_idx=self.intermediate_layer_idx,
            patch_size=self.patch_size
        )

        if self.gs_dense_reg_head_type == 'mlp':
            if self.shortcut_rgb:
                self.gs_dense_reg_head = Mlp(self.dense_feats_dim + 3, 2 * self.dense_feats_dim, self.out_channels)
            else:
                self.gs_dense_reg_head = Mlp(self.dense_feats_dim, 2 * self.dense_feats_dim, self.out_channels)
        elif self.gs_dense_reg_head_type == 'conv':
            self.gs_dense_reg_head = self._build_conv_head(
                self.dense_feats_dim, self.out_channels, embed_dim
            )
        else:
            raise ValueError(f"Unsupported gs_dense_reg_head_type: {self.gs_dense_reg_head_type}")

        # motion head
        self.motion_feature_head = DPTHead(
            dim_in=2 * embed_dim,
            feature_only=True,
            intermediate_layer_idx=self.intermediate_layer_idx,
            patch_size=self.patch_size
        )
        if self.motion_dense_key_head_type == 'mlp':
            self.motion_dense_key_head = Mlp(self.dense_feats_dim, 2 * self.dense_feats_dim, projected_motion_dim)
        elif self.motion_dense_key_head_type == 'conv':
            self.motion_dense_key_head = self._build_conv_head(
                self.dense_feats_dim, projected_motion_dim, embed_dim
            )
        else:
            raise ValueError(f"Unsupported motion_dense_key_head_type: {self.motion_dense_key_head_type}")

        # feat head
        if self.with_feat:
            self.feat_feature_head = DPTHead(
                dim_in=2 * embed_dim,
                feature_only=True,
                intermediate_layer_idx=self.intermediate_layer_idx,
                patch_size=self.patch_size
            )

            if self.feat_dense_reg_head_type == 'mlp':
                self.feat_dense_reg_head = Mlp(self.dense_feats_dim, 2 * self.dense_feats_dim, self.pred_feat_dim)
            elif self.feat_dense_reg_head_type == 'conv':
                self.feat_dense_reg_head = self._build_conv_head(
                    self.dense_feats_dim, self.pred_feat_dim, embed_dim
                )
            else:
                raise ValueError(f"Unsupported feat_dense_reg_head_type: {self.feat_dense_reg_head_type}")

    def _build_conv_head(self, in_channels, out_channels, embed_dim):
        if self.shortcut_rgb:
            in_channels = in_channels + 3
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                (2 * embed_dim) // 16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.GELU(),
            nn.Conv2d(
                (2 * embed_dim) // 16,
                (2 * embed_dim) // 32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.GELU(),
            nn.Conv2d(
                (2 * embed_dim) // 32,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

    def _setup_activations(self, params: _SetupActivationsParams):
        (gs_marbles, max_scale, opacity_offset, near, far, sigmoid_rgb) = params
        self.gs_marbles = gs_marbles
        self.max_scale = nn.Parameter(torch.tensor([float(max_scale)]), requires_grad=False)
        self.scale_offset = float(torch.log(torch.tensor([self.max_scale])))  # NOTE: learn from large to small

        if self.gs_marbles:
            self.scale_act_fn = self._scale_act_marbles
            self.quat_act_fn = self._quat_act_marbles
        else:
            self.scale_act_fn = self._scale_act_anisotropic
            self.quat_act_fn = self._quat_act_anisotropic

        self.opacity_act_fn = functools.partial(self._opacity_act, opacity_offset=opacity_offset)
        self.depth_act_fn = functools.partial(self._depth_act, near=near, far=far)
        self.rgb_act_fn = functools.partial(self._rgb_act, sigmoid_rgb=sigmoid_rgb)

        if self.pred_gs_conf:
            self.gs_conf_act_fn = self._gs_conf_act

        if self.enable_lifespan:
            self.lifespan_act_fn = self._lifespan_act  # default to 1.2s lifespan

    def _set_motion_attrs(self, params: _SetMotionAttrsParams):
        (
            use_ms3_motion, add_angular_velocity, ms3_deg, omega_deg, ms3_deg_downmax_mult, sigmoid_ms3_bias,
            sigmoid_ms3_min, sigmoid_ms3_max, ms3_clamp
        ) = params
        if use_ms3_motion:
            self.ms3_deg = ms3_deg
            self.omega_deg = omega_deg
            self.ms3_factorials = torch.tensor([math.factorial(i + 1) for i in range(self.ms3_deg)])
            self.omega_factorials = torch.tensor([math.factorial(i + 1) for i in range(self.omega_deg)])
            self.ms3_deg_downmax_mult = ms3_deg_downmax_mult
            self.sigmoid_ms3_bias = sigmoid_ms3_bias
            self.sigmoid_ms3_min = sigmoid_ms3_min
            self.sigmoid_ms3_max = sigmoid_ms3_max
            self.ms3_clamp = ms3_clamp
            num_velocity_channels = 4 * self.ms3_deg
            if add_angular_velocity:
                num_velocity_channels += 4 * self.omega_deg
        else:
            num_velocity_channels = 3
            if add_angular_velocity:
                num_velocity_channels += 3
        return num_velocity_channels

    def _validate_config(self):
        if self.use_pred_camera_pose:
            if not self.enable_camera_head:
                raise ValueError("enable_camera_head must be True when use_pred_camera_pose is True")
        if self.use_pred_depth:
            if not self.enable_depth_head:
                raise ValueError("enable_depth_head must be True when use_pred_depth is True")
        if self.voxelize:
            if not self.pred_gs_conf:
                raise ValueError('Voxelization requires gs confidence calculation weights.')

    def _setup_gs_params(self, enable_lifespan):
        self.gs_params_name = ["depth", "scales", "quats", "opacitys", "colors"]
        self.gs_params_size = [1, 3, 4, 1, self.gs_dim]
        self.out_channels = sum(self.gs_params_size)
        if self.pred_gs_conf:
            self.out_channels += 1
            self.gs_params_name.append("confs")
            self.gs_params_size.append(1)
        self.enable_lifespan = enable_lifespan
        if self.enable_lifespan:
            self.out_channels += 1
            self.gs_params_name.append("lifespans")
            self.gs_params_size.append(1)

    def _load_vggt_module(self, vggt_pretrained_weight, key_substr, prefix_to_remove, target_module):
        state_dict = {}
        for old_key, value in vggt_pretrained_weight.items():
            if key_substr in old_key:
                state_dict[old_key.removeprefix(prefix_to_remove)] = value
        target_module.load_state_dict(state_dict, strict=True)

    def _compute_motion_img_keys(self, params: _ComputeMotionImgKeysParams):
        (x, b, t, v, h, w, dense_feat) = params
        if dense_feat:
            if self.motion_dense_key_head_type == 'mlp':
                img_keys = self.motion_dense_key_head(rearrange(x, 'b (t v) c h w -> b t v h w c', t=t, v=v))
            elif self.motion_dense_key_head_type == 'conv':
                img_keys = self.motion_dense_key_head(rearrange(x, "b (t v) c h w -> (b t v) c h w", t=t, v=v))
                img_keys = rearrange(img_keys, "(b t v) c h w -> b t v h w c", t=t, v=v)
        else:
            img_embeds = self.unpatchify(
                rearrange(x, "b (t v) hw c -> (b t v) hw c", t=t, v=v),
                hw=(h // self.unpatch_size, w // self.unpatch_size),
                patch_size=1,
            )
            if self.grad_checkpointing:
                img_embeds = checkpoint(self.output_upscaling, img_embeds, use_reentrant=self.use_reentrant)
            else:
                img_embeds = self.output_upscaling(img_embeds)
            img_embeds = rearrange(img_embeds, "(b t v) c h w -> b t v h w c", t=t, v=v)
            img_keys = self.motion_key_head(img_embeds)
        return img_keys

    def _compute_gs_params_raw(self, params: _ComputeGsParamsRawParams):
        (x, b, t, v, h, w, rgb, dense_feat) = params
        if dense_feat:
            # shortcut rgb
            if self.shortcut_rgb and rgb is not None:
                x = torch.concat([rearrange(rgb, '(b t v) c h w -> b (t v) c h w', b=b, t=t, v=v), x], dim=2)
            if self.gs_dense_reg_head_type == 'mlp':
                gs_params = self.gs_dense_reg_head(rearrange(x, "b (t v) c h w -> b t v h w c", t=t, v=v))
            elif self.gs_dense_reg_head_type == 'conv':
                gs_params = self.gs_dense_reg_head(rearrange(x, "b (t v) c h w -> (b t v) c h w", t=t, v=v))
                gs_params = rearrange(gs_params, "(b t v) c h w -> b t v h w c", t=t, v=v)
        else:
            x = rearrange(x, "b (t v) hw c -> (b t v) hw c", t=t, v=v)
            gs_params = self.gs_pred(x)
            gs_params = self.unpatchify(gs_params, hw=(h, w), patch_size=self.unpatch_size)
            # shortcut rgb
            if self.shortcut_rgb and rgb is not None:
                gs_params = torch.concat([gs_params, rgb], dim=1)
                gs_params = self.gs_pred_with_rgb(rearrange(gs_params, '(b t v) c h w -> (b t v) h w c', t=t, v=v))
                gs_params = rearrange(gs_params, "(b t v) h w c -> b t v h w c", t=t, v=v)
            else:
                gs_params = rearrange(gs_params, "(b t v) c h w -> b t v h w c", t=t, v=v)
        return gs_params

    def _compute_pseudo_3dgs_scales_quats(self, params: _ComputePseudo3dgsScalesQuatsParams):
        (means, depths, b, t, v, h, w) = params
        scale_limit = 4
        normals, delta_x, delta_y, dx, dy = compute_normals_scales_torch(
            rearrange(means, 'b t v h w c -> (b t v) h w c')
        )  # B, H, W, C
        quats = rot_from_normals_torch(normals.reshape(-1, 3), up=dy)
        quats = rearrange(quats, '(b t v h w) c -> b t v h w c', b=b, t=t, v=v, h=h, w=w)
        scale_limit = (scale_limit * depths * repeat(self.azimuth_tan, '... -> ... 1 1 3'))  # azimuth angle limit
        scales = scale_from_dxdy_torch(delta_x, delta_y)
        scales = rearrange(scales, '(b t v) h w c -> b t v h w c', b=b, t=t, v=v)
        scales = torch.where(scales > scale_limit, scale_limit, scales)
        return scales, quats

    def _render_probs_in_chunks(self, params: _RenderProbsInChunksParams):
        (
            means, quats, scales, opacities, probs_batched, viewmats_batched, ks_batched, tgt_h, tgt_w,
            radius_clip, chunk_size
        ) = params
        probs_list = []
        for slice_i in range(0, probs_batched.shape[-1], chunk_size):
            with torch.autocast("cuda", enabled=False):
                rendered_res, _, _ = self.rasterization_func(
                    gaussians=GaussianData(means=means.float(), quats=quats.float(), scales=scales.float(),
                                           opacities=opacities.float(),
                                           colors=probs_batched[..., slice_i:slice_i + chunk_size].float()),
                    camera=CameraParams(viewmats=viewmats_batched, ks=ks_batched, camera_model="pinhole"),
                    config=RenderConfig(width=tgt_w, height=tgt_h, render_mode="RGB+ED",
                                        near_plane=self.near, far_plane=self.far,
                                        packed=False, radius_clip=radius_clip),
                )
            rendered_probs, _ = rendered_res.split([self.gs_dim, 1], dim=-1)
            probs_list.append(rendered_probs)
        return torch.concat(probs_list, dim=-1)

    def _execute_rendering(self, params: _ExecuteRenderingParams):
        (
            data_dict, gs_attrs, gs_params, means_batched, scales_batched, quats_batched, opacities_batched,
            color_batched, forward_v_batched, feats_batched, colors_batched, forward_flow, feats,
            render_motion_seg, concat_feat_render, b, t, v, h, w, tgt_t, tgt_v, tgt_h, tgt_w, radius_clip
        ) = params
        camtoworlds_batched = data_dict["target_camtoworlds"].view(b * tgt_t, -1, 4, 4)
        viewmats_batched = torch.linalg.inv(camtoworlds_batched.float())
        ks_batched = data_dict["target_intrinsics"].view(b * tgt_t, -1, 3, 3)

        if self.use_2dgs:
            colors_batched = colors_batched[:, None]

        self._validate_npu_constraints(feats, render_motion_seg, colors_batched)

        if os.environ.get('TIME_COUNT_TYPE2'):
            start = time.time()

        motion_seg = None
        feat = None
        colors_to_render = self._select_colors_to_render(colors_batched, render_motion_seg)

        rendered_color, rendered_alpha = self._run_rasterization(_RunRasterizationParams(
            means_batched, quats_batched, scales_batched, opacities_batched,
            colors_to_render, viewmats_batched, ks_batched, tgt_h, tgt_w, radius_clip
        ))

        color, forward_flow, feat, depth = self._split_rendered_color(
            rendered_color, feats, concat_feat_render
        )

        if not self.training and self.num_motion_tokens > 0 and render_motion_seg:
            motion_seg = self._render_motion_segmentation(_RenderMotionSegmentationParams(
                colors_batched, means_batched, quats_batched, scales_batched,
                opacities_batched, viewmats_batched, ks_batched,
                tgt_h, tgt_w, radius_clip, b, tgt_t, tgt_v
            ))

        if os.environ.get('TIME_COUNT_TYPE2'):
            torch.cuda.synchronize()
            logger.info('Rendering time - forward: %s', time.time() - start)

        return self._build_render_output(_BuildRenderOutputParams(
            color, depth, forward_flow, rendered_alpha, feat, motion_seg,
            gs_attrs, b, tgt_t, tgt_v, tgt_h, tgt_w,
            feats, concat_feat_render, means_batched, quats_batched,
            scales_batched, opacities_batched, feats_batched,
            viewmats_batched, ks_batched, radius_clip
        ))

    def _build_gs_attrs(self, params: _BuildGsAttrsParams):
        (
            means_batched, scales_batched, quats_batched, opacities_batched, color_batched, forward_v_batched,
            feats, feats_batched
        ) = params
        gs_attrs = {
            'means': means_batched,
            'scales': scales_batched,
            'quats': quats_batched,
            'opacities': opacities_batched.unsqueeze(-1),
            'color': color_batched,
            'forward_v': forward_v_batched,
        }
        if feats is not None:
            gs_attrs['feats'] = feats_batched
        return gs_attrs

    def _append_motion_weights(self, params: _AppendMotionWeightsParams):
        (gs_params, colors_batched, tgt_t, idx, v, h, w) = params
        motion_weights = rearrange(gs_params["motion_weights"], "b t v h w k -> b (t v h w) k")
        weights_batched = repeat(motion_weights, "b ... -> (b t) ...", t=tgt_t)
        if idx is not None:
            weights_batched = weights_batched[:, (idx) * v * h * w: (idx + 1) * v * h * w]
        return torch.cat([colors_batched, weights_batched], dim=-1)

    def _prepare_gs_batched(self, params: _PrepareGsBatchedParams):
        (gs_params, data_dict, feats, b, t, v, h, w, tgt_t) = params
        means = rearrange(gs_params["means"], "b t v h w c -> b (t v h w) c")
        scales = rearrange(gs_params["scales"], "b t v h w c -> b (t v h w) c")
        quats = rearrange(gs_params["quats"], "b t v h w c -> b (t v h w) c")
        opacities = rearrange(gs_params["opacities"], "b t v h w -> b (t v h w)")
        colors = rearrange(gs_params["colors"], "b t v h w c -> b (t v h w) c")
        feats = rearrange(feats, "b t v h w c -> b (t v h w) c") if feats is not None else None

        means_batched = repeat(means, "b ... -> (b t) ...", t=tgt_t)
        scales_batched = repeat(scales, "b ... -> (b t) ...", t=tgt_t)
        quats_batched = repeat(quats, "b ... -> (b t) ...", t=tgt_t)
        opacities_batched = repeat(opacities, "b ... -> (b t) ...", t=tgt_t)
        color_batched = repeat(colors, "b ... -> (b t) ...", t=tgt_t)
        feats_batched = repeat(feats, "b ... -> (b t) ...", t=tgt_t) if feats is not None else None

        ctx_time = data_dict["context_time"] * data_dict["timespan"]
        tgt_time = data_dict["target_time"] * data_dict["timespan"]
        if tgt_time.ndim == 3:
            tdiff_forward = tgt_time.unsqueeze(2) - ctx_time.unsqueeze(1)
            tdiff_forward = tdiff_forward.view(b * tgt_t, t * v, 1)
            tdiff_forward_batched = repeat(tdiff_forward, "bt tv 1 -> bt (tv hw) 1", hw=h * w)
        else:
            tdiff_forward = tgt_time.unsqueeze(-1) - ctx_time.unsqueeze(-2)
            tdiff_forward = tdiff_forward.view(b * tgt_t, t, 1)
            tdiff_forward_batched = repeat(tdiff_forward, "bt t 1 -> bt (t vhw) 1", hw=v * h * w)
        return _PrepareGsBatchedResult(
            means_batched, scales_batched, quats_batched, opacities_batched,
            color_batched, feats_batched, tdiff_forward_batched
        )

    def _validate_npu_constraints(self, feats, render_motion_seg, colors_batched):
        if is_ascend_npu():
            if feats is not None:
                raise ValueError("NPU does not support feature rendering, feats must be None")
            if render_motion_seg:
                raise ValueError("NPU does not support motion segmentation rendering, render_motion_seg must be False")
            if self.training:
                if colors_batched.shape[-1] != 3:
                    raise ValueError(
                        f"colors_batched.shape[-1] must be 3 during training on NPU, "
                        f"got {colors_batched.shape[-1]}"
                    )

    def _select_colors_to_render(self, colors_batched, render_motion_seg):
        if not self.training and self.num_motion_tokens > 0 and render_motion_seg:
            return colors_batched[..., :-self.num_motion_tokens].float()
        return colors_batched.float()

    def _run_rasterization(self, params: _RunRasterizationParams):
        (
            means_batched, quats_batched, scales_batched, opacities_batched, colors_to_render, viewmats_batched,
            ks_batched, tgt_h, tgt_w, radius_clip
        ) = params
        with torch.autocast("cuda", enabled=False):
            rendered_color, rendered_alpha, *_ = self.rasterization_func(
                gaussians=GaussianData(means=means_batched.float(), quats=quats_batched.float(),
                                       scales=scales_batched.float(),
                                       opacities=opacities_batched.float(), colors=colors_to_render),
                camera=CameraParams(viewmats=viewmats_batched, ks=ks_batched, camera_model="pinhole"),
                config=RenderConfig(width=tgt_w, height=tgt_h, render_mode="RGB+ED",
                                    near_plane=self.near, far_plane=self.far,
                                    packed=False, radius_clip=radius_clip),
            )
        return rendered_color, rendered_alpha

    def _apply_motion(self, params: _ApplyMotionParams):
        (
            gs_params, ctx_time, tdiff_forward_batched, means_batched, quats_batched, opacities_batched, tgt_t,
            time_step, static_render, data_dict
        ) = params
        if not self.use_ms3_motion:
            forward_v_batched, means_batched, quats_batched = self._apply_linear_motion(
                gs_params, tdiff_forward_batched, means_batched, quats_batched, tgt_t
            )
        else:
            forward_v_batched, means_batched, quats_batched, opacities_batched = self._apply_ms3_motion(
                _ApplyMs3MotionParams(
                    gs_params, ctx_time, tdiff_forward_batched, means_batched, quats_batched,
                    opacities_batched, tgt_t, time_step, static_render, data_dict
                )
            )
        return forward_v_batched, means_batched, quats_batched, opacities_batched

    def _apply_linear_motion(self, gs_params, tdiff_forward_batched, means_batched,
                             quats_batched, tgt_t):
        forward_v = rearrange(gs_params["forward_flow"], "b t v h w c -> b (t v h w) c")
        if self.add_angular_velocity:
            forward_v, forward_angular_v = forward_v.split([3, 3], dim=-1)
        forward_v_batched = repeat(forward_v, "b ... -> (b t) ...", t=tgt_t)
        if self.add_angular_velocity:
            forward_angular_v_batched = repeat(forward_angular_v, "b ... -> (b t) ...", t=tgt_t)

        forward_translation = forward_v_batched * tdiff_forward_batched
        means_batched = means_batched + forward_translation

        if self.add_angular_velocity:
            quats_offset_batched = angular_velocity_to_quaternion(forward_angular_v_batched, tdiff_forward_batched)
            quats_batched = quaternion_multiply(quats_batched, quats_offset_batched)
        return forward_v_batched, means_batched, quats_batched

    def _apply_angular_velocity(self, gs_params, tdiff_forward_batched, quats_batched, tgt_t):
        forward_omega = gs_params["forward_ms3"][..., -self.omega_deg * 3:]
        forward_omega = rearrange(forward_omega, "b t v h w c -> b (t v h w) c")
        forward_omega_batched = repeat(forward_omega, "b ... -> (b t) ...", t=tgt_t)
        angle_axis_offset_batched = torch.stack(
            [
                forward_omega_batched[..., i * 3:(i + 1) * 3] *
                tdiff_forward_batched ** (i + 1) /
                self.omega_factorials[i]
                for i in range(self.omega_deg)
            ]
        ).sum(0)
        quats_offset_batched = angle_axis_to_quaternion(angle_axis_offset_batched)
        quats_batched = quaternion_multiply(quats_batched, quats_offset_batched)
        return quats_batched

    def _compute_ms3_translations(self, forward_ms3_batched, tdiff_forward_batched, delta_time):
        forward_translation_cur = torch.stack(
            [
                forward_ms3_batched[..., i * 3:(i + 1) * 3] *
                tdiff_forward_batched ** (i + 1) /
                self.ms3_factorials[i]
                for i in range(self.ms3_deg)
            ]
        ).sum(0)
        forward_translation_prev = torch.stack(
            [
                forward_ms3_batched[..., i * 3:(i + 1) * 3]
                * (tdiff_forward_batched - delta_time) ** (i + 1)
                / self.ms3_factorials[i]
                for i in range(self.ms3_deg)
            ]
        ).sum(0)
        return forward_translation_cur, forward_translation_prev

    def _apply_ms3_motion(self, params: _ApplyMs3MotionParams):
        (
            gs_params, ctx_time, tdiff_forward_batched, means_batched, quats_batched, opacities_batched, tgt_t,
            time_step, static_render, data_dict
        ) = params
        forward_ms3 = gs_params["forward_ms3"][..., :self.ms3_deg * 3]
        forward_ms3 = rearrange(forward_ms3, "b t v h w c -> b (t v h w) c")
        forward_ms3_batched = repeat(forward_ms3, "b ... -> (b t) ...", t=tgt_t)
        if self.add_angular_velocity:
            quats_batched = self._apply_angular_velocity(
                gs_params, tdiff_forward_batched, quats_batched, tgt_t
            )

        delta_time = float(1 / data_dict['fps'])

        forward_translation_cur, forward_translation_prev = self._compute_ms3_translations(
            forward_ms3_batched, tdiff_forward_batched, delta_time
        )

        forward_v_batched = (forward_translation_cur - forward_translation_prev) / delta_time

        delta_ctx_time = self._compute_delta_ctx_time(ctx_time, time_step, delta_time)

        time_mask_backward, time_mask_forward, final_mask = self._compute_motion_masks(
            tdiff_forward_batched, forward_v_batched, delta_time, delta_ctx_time
        )

        if not static_render:
            if "window" in self.mode or "causal" in self.mode:
                forward_translation_cur = forward_translation_cur * time_mask_backward
            means_batched = means_batched + forward_translation_cur

        if "window" in self.mode or "causal" in self.mode:
            forward_v_batched = forward_v_batched * time_mask_backward
            opacities_batched = opacities_batched * final_mask.squeeze(-1)

        gs_params["forward_flow"] = gs_params["forward_ms3"]
        return forward_v_batched, means_batched, quats_batched, opacities_batched

    def _compute_delta_ctx_time(self, ctx_time, time_step, delta_time):
        if len(ctx_time[0]) > 1:
            return ctx_time[0][1][0] - ctx_time[0][0][0]
        return time_step * delta_time

    def _compute_motion_masks(self, tdiff_forward_batched, forward_v_batched,
                              delta_time, delta_ctx_time):
        time_mask_backward = (
                (tdiff_forward_batched < 0.5 * delta_time) &
                (tdiff_forward_batched > - 1.0 * delta_ctx_time + 0.5 * delta_time)
        )
        time_mask_forward = (
                (tdiff_forward_batched < delta_ctx_time + 0.5 * delta_time) &
                (tdiff_forward_batched > 0.5 * delta_time)
        )
        static_mask = forward_v_batched.norm(dim=-1) < 1.0
        final_mask = (static_mask.unsqueeze(-1) & time_mask_forward) | time_mask_backward
        return time_mask_backward, time_mask_forward, final_mask

    def _apply_lifespan(self, gs_params, tdiff_forward_batched, opacities_batched, tgt_t):
        lifespans = rearrange(gs_params["lifespans"], "b t v h w -> b (t v h w)")
        lifespans_batched = repeat(lifespans, "b ... -> (b t) ...", t=tgt_t)
        reduction_factor = 0.05
        sigma = (lifespans_batched ** 2) / (torch.log(torch.tensor(reduction_factor)) / -0.5)
        life_span_coef = torch.exp(-0.5 * (tdiff_forward_batched.squeeze(-1)) ** 2 / sigma)
        opacities_batched = opacities_batched * life_span_coef
        return opacities_batched

    def _slice_gs_attrs_by_idx(self, params: _SliceGsAttrsByIdxParams):
        (
            means_batched, scales_batched, quats_batched, opacities_batched, color_batched, forward_v_batched, idx,
            v, h, w
        ) = params
        slice_range = slice((idx) * v * h * w, (idx + 1) * v * h * w)
        return _SliceGsAttrsByIdxResult(
            means_batched[:, slice_range], scales_batched[:, slice_range],
            quats_batched[:, slice_range], opacities_batched[:, slice_range],
            color_batched[:, slice_range], forward_v_batched[:, slice_range]
        )

    def _apply_voxelization(self, params: _ApplyVoxelizationParams):
        (gs_attrs, gs_params, idx, v, h, w, b, tgt_t) = params
        gs_confs = rearrange(gs_params["confs"], "b t v h w c -> b (t v h w) c")
        gs_confs_batched = repeat(gs_confs, "b ... -> (b t) ...", t=tgt_t)
        if idx is not None:
            gs_confs_batched = gs_confs_batched[:, (idx) * v * h * w: (idx + 1) * v * h * w]

        gs_attrs_voxel_padded = self._voxelize_and_pad_gs_attrs(gs_attrs, gs_confs_batched, b, tgt_t)

        means_batched = gs_attrs_voxel_padded['means']
        scales_batched = gs_attrs_voxel_padded['scales']
        quats_batched = gs_attrs_voxel_padded['quats']
        opacities_batched = gs_attrs_voxel_padded['opacities'].squeeze(-1)
        color_batched = gs_attrs_voxel_padded['color']
        forward_v_batched = gs_attrs_voxel_padded['forward_v']
        gs_attrs = {
            'means': means_batched,
            'scales': scales_batched,
            'quats': quats_batched,
            'opacities': opacities_batched.unsqueeze(-1),
            'color': color_batched,
            'forward_v': forward_v_batched,
        }
        feats_batched = None
        if 'feats' in gs_attrs_voxel_padded:
            feats_batched = gs_attrs_voxel_padded['feats']
            gs_attrs['feats'] = feats_batched
        return _ApplyVoxelizationResult(
            gs_attrs, means_batched, scales_batched, quats_batched,
            opacities_batched, color_batched, forward_v_batched, feats_batched
        )

    def _prepare_render_colors(self, params: _PrepareRenderColorsParams):
        (color_batched, forward_v_batched, feats_batched, concat_feat_render, idx, v, h, w) = params
        forward_flow = None
        if self.training:
            colors_batched = color_batched
        else:
            colors_batched = torch.cat([color_batched, forward_v_batched], dim=-1)
        if feats_batched is not None and concat_feat_render:
            if idx is not None:
                feats_batched = feats_batched[:, (idx) * v * h * w: (idx + 1) * v * h * w]
            colors_batched = torch.cat([colors_batched, feats_batched], dim=-1)
        return colors_batched, forward_flow

    def _split_rendered_color(self, rendered_color, feats, concat_feat_render):
        forward_flow = None
        feat = None
        if feats is not None and concat_feat_render:
            if not self.training:
                color, forward_flow, feat, depth = rendered_color.split(
                    [self.gs_dim, 3, self.pred_feat_dim, 1], dim=-1
                )
            else:
                color, feat, depth = rendered_color.split(
                    [self.gs_dim, self.pred_feat_dim, 1], dim=-1
                )
        else:
            if not self.training:
                color, forward_flow, depth = rendered_color.split([self.gs_dim, 3, 1], dim=-1)
            else:
                color, depth = rendered_color.split([self.gs_dim, 1], dim=-1)
        return color, forward_flow, feat, depth

    def _render_motion_segmentation(self, params: _RenderMotionSegmentationParams):
        (
            colors_batched, means_batched, quats_batched, scales_batched, opacities_batched, viewmats_batched,
            ks_batched, tgt_h, tgt_w, radius_clip, b, tgt_t, tgt_v
        ) = params
        with torch.autocast("cuda", enabled=False):
            chunksize = 32
            assignment_map = []
            rendered_colors = colors_batched[..., -self.num_motion_tokens:]
            for i in range(0, self.num_motion_tokens, chunksize):
                weights, *_ = self.rasterization_func(
                    gaussians=GaussianData(means=means_batched.float(), quats=quats_batched.float(),
                                           scales=scales_batched.float(),
                                           opacities=opacities_batched.float(),
                                           colors=rendered_colors[..., i:i + chunksize]),
                    camera=CameraParams(viewmats=viewmats_batched, ks=ks_batched, camera_model="pinhole"),
                    config=RenderConfig(width=tgt_w, height=tgt_h, render_mode="RGB+ED",
                                        near_plane=self.near, far_plane=self.far,
                                        packed=False, radius_clip=radius_clip),
                )
                weights = weights.split([weights.size(-1) - 1, 1], dim=-1)[0]
                assignment_map.append(weights)
            motion_seg = torch.cat(assignment_map, dim=-1)
            motion_seg = motion_seg.reshape(b, tgt_t, tgt_v, tgt_h, tgt_w, -1).argmax(
                dim=-1
            )
        return motion_seg

    def _build_render_output(self, params: _BuildRenderOutputParams):
        (
            color, depth, forward_flow, rendered_alpha, feat, motion_seg, gs_attrs, b, tgt_t, tgt_v, tgt_h, tgt_w,
            feats, concat_feat_render, means_batched, quats_batched, scales_batched, opacities_batched,
            feats_batched, viewmats_batched, ks_batched, radius_clip
        ) = params
        output_dict = {
            "rendered_image": color.view(b, tgt_t, tgt_v, tgt_h, tgt_w, -1),
            "rendered_depth": depth.view(b, tgt_t, tgt_v, tgt_h, tgt_w),
        }
        if forward_flow is not None:
            output_dict["rendered_flow"] = forward_flow.view(b, tgt_t, tgt_v, tgt_h, tgt_w, -1)
        if rendered_alpha is not None:
            output_dict["rendered_alpha"] = rendered_alpha.view(b, tgt_t, tgt_v, tgt_h, tgt_w)
        else:
            output_dict["rendered_alpha"] = None

        if feats is not None and not concat_feat_render:
            with torch.autocast("cuda", enabled=False):
                rendered_feat, _, _ = self.rasterization_func(
                    gaussians=GaussianData(means=means_batched.detach().float(),
                                           quats=quats_batched.detach().float(),
                                           scales=scales_batched.detach().float(),
                                           opacities=opacities_batched.detach().float(),
                                           colors=feats_batched.float()),
                    camera=CameraParams(viewmats=viewmats_batched.detach(), ks=ks_batched.detach(),
                                        camera_model="pinhole"),
                    config=RenderConfig(width=tgt_w, height=tgt_h, render_mode="RGB+ED",
                                        near_plane=self.near, far_plane=self.far,
                                        packed=False, radius_clip=radius_clip),
                )
                feat, _ = rendered_feat.split([self.pred_feat_dim, 1], dim=-1)

        if feat is not None:
            output_dict["rendered_feat"] = feat.view(b, tgt_t, tgt_v, tgt_h, tgt_w, -1)
        if motion_seg is not None:
            output_dict["rendered_motion_seg"] = motion_seg.squeeze(-1)
        if self.save_gaussian or (self.with_feat and os.getenv("CONTEXT_FEAT")):
            for k, val in gs_attrs.items():
                output_dict[f"gs_{k}"] = val.unsqueeze(0)
        return output_dict

    def _run_aggregator(self, data_dict, aggregator_kv_cache_list):
        if aggregator_kv_cache_list is not None:
            output_list, self.patch_start_idx, aggregator_kv_cache_list = self.aggregator(
                data_dict, mode=self.mode, kv_cache_list=aggregator_kv_cache_list
            )
        else:
            output_list, self.patch_start_idx = self.aggregator(data_dict, mode=self.mode)
        return output_list, self.patch_start_idx, aggregator_kv_cache_list

    def _extract_tokens(self, others_last_tokens):
        sky_token, affine_tokens, motion_tokens, time_tokens = None, None, None, None
        if self.use_sky_token:
            sky_token = others_last_tokens[:, :, -1:]
            sky_token = self.sky_token_norm(sky_token)  # NOTE: token need LayerNorm
            sky_token = sky_token.mean(1)  # remove extra copied parts above
            others_last_tokens = others_last_tokens[:, :, :-1]

        if self.use_affine_token:
            affine_tokens = others_last_tokens[:, :, -self.num_cams:]
            affine_tokens = self.affine_token_norm(affine_tokens)  # NOTE: token need LayerNorm
            affine_tokens = affine_tokens.mean(1)  # remove extra copied parts above
            others_last_tokens = others_last_tokens[:, :, :-self.num_cams]

        if self.num_motion_tokens > 0:
            motion_tokens = others_last_tokens[:, :, -self.num_motion_tokens:]
            motion_tokens = self.motion_token_norm(motion_tokens)  # NOTE: token need LayerNorm
            motion_tokens = motion_tokens.mean(1)  # remove extra copied parts above
            others_last_tokens = others_last_tokens[:, :, :-self.num_motion_tokens]

        if self.use_time_token:
            time_tokens = others_last_tokens[:, :, -1:]
            others_last_tokens = others_last_tokens[:, :, :-1]
        return sky_token, affine_tokens, motion_tokens, time_tokens

    def _run_heads(self, params: _RunHeadsParams):
        (output_list, images, b, t, v, camera_head_kv_cache_list) = params
        pose_enc_list = None
        pred_ray_dict = None
        if self.camera_head is not None:
            if camera_head_kv_cache_list is not None:
                pose_enc_list, camera_head_kv_cache_list = self.camera_head(
                    output_list, t, v, mode=self.mode, kv_cache_list=camera_head_kv_cache_list
                )
            else:
                pose_enc_list = self.camera_head(output_list, t, v, mode=self.mode)
            pose_enc = pose_enc_list[-1]
            pred_extrinsic, pred_intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            # world to camera -> camera to world
            pred_camtoworlds = torch.concat(
                [
                    pred_extrinsic,
                    repeat(torch.tensor([[0, 0, 0, 1]]).to(images.device), '... -> b tv ...', b=b, tv=t * v)
                ],
                dim=-2
            ).inverse()

            pred_ray_dict = self.plucker_embedder(
                rearrange(pred_intrinsic, 'b (t v) ... -> b t v ...', t=t, v=v),
                rearrange(pred_camtoworlds, 'b (t v) ... -> b t v ...', t=t, v=v),
                image_size=images.shape[-2:],
            )

        pred_context_depth, pred_context_depth_conf = None, None
        if self.depth_head is not None and self.use_pred_depth:
            pred_context_depth, pred_context_depth_conf = self.depth_head(
                output_list,
                images=rearrange(images, '(b t v) c h w -> b (t v) c h w', b=b, t=t, v=v),
                patch_start_idx=self.patch_start_idx
            )
            # apply sigmoid activation
            pred_context_depth = torch.clamp(pred_context_depth, min=self.near, max=(self.far - self.near))
            pred_context_depth = rearrange(pred_context_depth, 'b (t v) ... -> b t v ...', b=b, t=t, v=v)
            pred_context_depth_conf = rearrange(pred_context_depth_conf, 'b (t v) ... -> b t v ...', b=b, t=t, v=v)

        pred_context_pts3d, pred_context_pts3d_conf = None, None
        if self.point_head is not None:
            pred_context_pts3d, pred_context_pts3d_conf = self.point_head(
                output_list,
                images=rearrange(images, '(b t v) c h w -> b (t v) c h w', b=b, t=t, v=v),
                patch_start_idx=self.patch_start_idx
            )
            pred_context_pts3d = rearrange(pred_context_pts3d, 'b (t v) ... -> b t v ...', b=b, t=t, v=v)
            pred_context_pts3d_conf = rearrange(pred_context_pts3d_conf, 'b (t v) ... -> b t v ...', b=b, t=t, v=v)

        return _RunHeadsResult(
            pose_enc_list, camera_head_kv_cache_list, pred_ray_dict,
            pred_context_depth, pred_context_depth_conf,
            pred_context_pts3d, pred_context_pts3d_conf
        )

    def _run_predictors(self, params: _RunPredictorsParams):
        (
            last_tokens, images, motion_tokens, ray_origins, ray_dirs, activated_depth, h, w, t, v, b, output_list,
            data_dict
        ) = params
        pred_feat = None
        if self.use_last_token:
            # last layer's token
            aggregated_last_tokens = last_tokens[:, :, self.patch_start_idx:]  # aggregated patch token
            # NOTE: token need LayerNorm
            aggregated_last_tokens = self.aggregated_last_tokens_norm(aggregated_last_tokens)

            if self.pesudo_3dgs:
                self.azimuth_tan = 1 / data_dict['context_intrinsics'][:, :, :, 0, 0]  # compute_azimuth_tan

            # Gaussian head
            gs_params = self.forward_gs_predictor(_ForwardGsPredictorParams(
                x=aggregated_last_tokens, origins=ray_origins, directions=ray_dirs,
                activated_depth=activated_depth, rgb=images
            ))

            # Motion head
            gs_params = self.forward_motion_predictor(aggregated_last_tokens, motion_tokens, gs_params)

            # Feature head
            if self.with_feat:
                pred_feat = self.forward_feat_predictor(aggregated_last_tokens, shape=(h, w, t, v))
        else:
            # Gaussian head (Dpt head)
            gs_dense_feats = self.gs_feature_head(
                output_list,
                images=rearrange(images, '(b t v) c h w -> b (t v) c h w', b=b, t=t, v=v),
                patch_start_idx=self.patch_start_idx
            )
            gs_params = self.forward_gs_predictor(_ForwardGsPredictorParams(
                x=gs_dense_feats, origins=ray_origins, directions=ray_dirs,
                activated_depth=activated_depth, rgb=images, dense_feat=True
            ))

            # Motion head (Dpt head)
            motion_dense_feats = self.motion_feature_head(
                output_list,
                images=rearrange(images, '(b t v) c h w -> b (t v) c h w', b=b, t=t, v=v),
                patch_start_idx=self.patch_start_idx
            )
            gs_params = self.forward_motion_predictor(
                motion_dense_feats, motion_tokens, gs_params, dense_feat=True
            )

            # Feature head (Dpt head)
            if self.with_feat:
                feat_dense_feats = self.feat_feature_head(
                    output_list,
                    images=rearrange(images, '(b t v) c h w -> b (t v) c h w', b=b, t=t, v=v),
                    patch_start_idx=self.patch_start_idx
                )
                pred_feat = self.forward_feat_predictor(feat_dense_feats, shape=(h, w, t, v), dense_feat=True)
        return gs_params, pred_feat

    def _build_gshead_and_motionhead_without_dpthead(
            self,
            embed_dim,
            decoder_upsample_ratio,
            projected_motion_dim,
            decoder_type,
            grad_checkpointing,
    ):
        # ------- gs predictor and mask decoder -------
        gs_pred_out_chans = 32 if self.shortcut_rgb else self.out_channels
        if decoder_type == "dummy":
            self.gs_pred = nn.Linear(2 * embed_dim, decoder_upsample_ratio ** 2 * gs_pred_out_chans)
            self.unpatch_size = decoder_upsample_ratio
            self.output_upscaling = self._build_dummy_output_upscaling(embed_dim)
        elif decoder_type == "conv":
            self.gs_pred = nn.Linear(2 * embed_dim, self.out_channels)
            # latent-XXX decoder
            self.decoder = ConvDecoder(
                latent_dim=self.gs_dim,
                out_channels=4,  # 3 for RGB, 1 for depth
                num_res_blocks=3,
                channels=[512, 256, 256, 128],  # 8 times upsample
                grad_checkpointing=grad_checkpointing,
            )
            self.unpatch_size = 1
            # upscaling the low-resolution image features to the pixel-resolution
            # the "pixel" resolution here is essentially the feature map resolution
            # which is 1/patch_size of the image resolution
            self.output_upscaling = nn.Sequential(
                nn.Conv2d(2 * embed_dim, 512, kernel_size=1),
                LayerNorm2d(512),
                nn.GELU(),
                nn.Conv2d(512, 256, kernel_size=1),
                LayerNorm2d(256),
                nn.GELU(),
                nn.Conv2d(256, 128, kernel_size=1),
                LayerNorm2d(128),
                nn.GELU(),
            )

        if self.shortcut_rgb:
            self.gs_pred_with_rgb = nn.Linear(gs_pred_out_chans + 3, self.out_channels)

        self.motion_key_head = Mlp(128, 256, projected_motion_dim)

    def _build_dummy_output_upscaling(self, embed_dim):
        # used for upscaling the low-resolution image features to the pixel-resolution
        # very handcrafted and never tuned
        if self.decoder_upsample_ratio == 8:
            return nn.Sequential(
                nn.ConvTranspose2d(2 * embed_dim, 512, kernel_size=2, stride=2),
                LayerNorm2d(512),
                nn.GELU(),
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                LayerNorm2d(256),
                nn.GELU(),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                LayerNorm2d(128),
                nn.GELU(),
            )
        elif self.decoder_upsample_ratio == 16:
            return nn.Sequential(
                nn.ConvTranspose2d(2 * embed_dim, 512, kernel_size=2, stride=2),
                LayerNorm2d(512),
                nn.GELU(),
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                LayerNorm2d(256),
                nn.GELU(),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                LayerNorm2d(128),
                nn.GELU(),
                nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
                LayerNorm2d(128),
                nn.GELU(),
            )
        elif self.decoder_upsample_ratio == 14:
            return nn.Sequential(
                nn.ConvTranspose2d(2 * embed_dim, 512, kernel_size=2, stride=2),
                LayerNorm2d(512),
                nn.GELU(),
                nn.ConvTranspose2d(512, 256, kernel_size=1, stride=1),
                LayerNorm2d(256),
                nn.GELU(),
                nn.ConvTranspose2d(256, 128, kernel_size=7, stride=7),
                LayerNorm2d(128),
                nn.GELU(),
                nn.ConvTranspose2d(128, 128, kernel_size=1, stride=1),
                LayerNorm2d(128),
                nn.GELU(),
            )
        else:
            raise ValueError(f"Unsupported decoder_upsample_ratio: {self.decoder_upsample_ratio}")

    def _build_feat_head_without_dpthead(
            self,
            embed_dim,
            decoder_upsample_ratio,
            decoder_type,
    ):
        # ------- feat predictor -------
        if decoder_type == "dummy":
            self.feat_pred = nn.Linear(2 * embed_dim, decoder_upsample_ratio ** 2 * self.pred_feat_dim)
            self.unpatch_size = decoder_upsample_ratio
        else:
            raise NotImplementedError

    def _pos_embed(self, x: Tensor) -> Tensor:
        if not self.disable_pos_embed:
            return super()._pos_embed(x)
        return rearrange(x, "b h w c -> b (h w) c")

    def _time_embed(self, x: Tensor, time_tensor: Tensor, num_views=1) -> Tensor:
        if time_tensor.ndim == 3:
            b, t, v = time_tensor.shape
            time_embedding = (
                self.time_embedder(time_tensor.flatten())  # (bt, c)
                .view(b, t, v, -1)  # (b, t, v, c)
                .view(-1, 1, self.embed_dim)  # (btv, 1, c)
                .repeat(1, x.shape[1], 1)  # (btv, n, c)
            )
        else:
            time_embedding = (
                self.time_embedder(time_tensor.flatten())  # (bt, c)
                .view(time_tensor.shape[0], time_tensor.shape[1], 1, -1)  # (b, t, 1, c)
                .repeat(1, 1, num_views, 1)  # (b, t, v, c)
                .view(-1, 1, self.embed_dim)  # (btv, 1, c)
                .repeat(1, x.shape[1], 1)  # (btv, n, c)
            )
        return x + time_embedding

    def _scale_act_marbles(self, x):
        """Activation function for scale in marbles mode."""
        return torch.minimum(torch.exp(x.mean(-1, True).expand_as(x) + self.scale_offset), self.max_scale)

    def _quat_act_marbles(self, x):
        """Activation function for quaternion in marbles mode."""
        return x.new_tensor((1, 0, 0, 0)).expand(*x.shape[:-1], 4)

    def _scale_act_anisotropic(self, x):
        """Activation function for scale in anisotropic mode."""
        return torch.minimum(torch.exp(x + self.scale_offset), self.max_scale)

    def _quat_act_anisotropic(self, x):
        """Activation function for quaternion in anisotropic mode."""
        return x

    def _opacity_act(self, x, opacity_offset):
        """Activation function for opacity."""
        return torch.sigmoid(x + opacity_offset)

    def _depth_act(self, x, near, far):
        """Activation function for depth."""
        return near + torch.sigmoid(x) * (far - near)

    def _rgb_act(self, x, sigmoid_rgb):
        """Activation function for RGB colors."""
        return torch.sigmoid(x) if sigmoid_rgb else x

    def _gs_conf_act(self, x):
        """Activation function for Gaussian confidence."""
        return torch.sigmoid(x)

    def _lifespan_act(self, x):
        """Activation function for lifespan."""
        return F.sigmoid(x - 4.0) * (100 - 0.1) + 0.1
