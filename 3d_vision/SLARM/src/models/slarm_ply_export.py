# coding=utf-8

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
from collections import namedtuple

import numpy as np
import torch
from einops import rearrange
from torch_scatter import scatter_add

from tools.export_ply import save_ply, PlyExportConfig
from src.dataset.constants import SEMANTIC_LABEL_LIST, SEMANTIC_ID_TO_COLOR

logger = logging.getLogger(__name__)

if os.getenv("FEAT_DIST"):
    from tools.feats_tools import get_text_label_feats, feat2class

_SaveGsParamsToPlyParams = namedtuple('_SaveGsParamsToPlyParams', [
    'data_dict', 'render_results', 'target_sky', 'affine', 'opacity_threshold', 'save_path'
], defaults=[0.1, 'output_gs'])
_ComputeGsMaskParams = namedtuple('_ComputeGsMaskParams', [
    'opacities', 'opacity_threshold', 'target_sky', 'data_dict', 't', 'v', 'h', 'w'
])
_RenderAndMergeContextContributionsParams = namedtuple('_RenderAndMergeContextContributionsParams', [
    'gs_params', 'data_dict', 'render_results', 't', 'time_step', 'static_render', 'chunk_start'
])
_SaveCameraPoseParams = namedtuple('_SaveCameraPoseParams', [
    'c2ws', 'data_dict', 'save_path', 'target_frame_idxs', 't_idx', 'v', 'xyzs'
])


class SLARMPlyExportMixin:
    """Mixin for SLARM PLY export and point cloud saving functionality."""

    @staticmethod
    def pad_tensor_list(tensor_list, pad_shape, value=0.0):
        padded = []
        for t in tensor_list:
            pad_len = pad_shape[0] - t.shape[0]
            if pad_len > 0:
                padding = torch.full(
                    (pad_len, *t.shape[1:]), value, device=t.device, dtype=t.dtype
                )
                t = torch.cat([t, padding], dim=0)
            padded.append(t)
        return torch.stack(padded)

    def save_gs_params_to_ply(self, params: _SaveGsParamsToPlyParams):
        (data_dict, render_results, target_sky, affine, opacity_threshold, save_path) = params
        input_image = data_dict['context_image']
        target_frame_idxs = data_dict['target_frame_idx'][0].to(torch.int16).tolist()
        b, t, v, c, h, w = input_image.shape
        if b != 1:
            raise ValueError(f"batch size must be 1 for save_gs_params_to_ply, got {b}")
        _, tgt_t, _, _ = render_results['gs_means'].shape  # [(b tgt_t), (t v h w), c]

        for t_idx in range(tgt_t):
            xyz = render_results['gs_means'][:, t_idx]
            color = render_results['gs_color'][:, t_idx]
            opacities = render_results['gs_opacities'][:, t_idx]
            scales = render_results['gs_scales'][:, t_idx]
            quats = render_results['gs_quats'][:, t_idx]

            mask = self._compute_gs_mask(
                _ComputeGsMaskParams(opacities, opacity_threshold, target_sky, data_dict, t, v, h, w)
            )

            # affine transform
            if affine is not None:
                color = rearrange(color, 'b (t v h w) c -> b t v h w c', t=t, v=v, h=h, w=w)
                color = torch.einsum('btvhwi,bvij->btvhwj', color, affine['linear']) + affine['translation']
                color = rearrange(color, 'b t v h w c -> b (t v h w) c')

            # filter gs
            xyz = xyz[mask]
            color = color[mask]
            opacities = opacities[mask]
            scales = scales[mask]
            quats = quats[mask]

            # align to the same coordinate system (first frame in first segment)
            if 'segment_to_ref' in data_dict:
                segment_to_ref = data_dict['segment_to_ref'][0].to(torch.float32).cpu().numpy()
                xyz = xyz.to(torch.float32).cpu().numpy()
                # right multiply
                xyz = (np.concatenate([xyz, np.ones_like(xyz)[:, :1]], axis=-1) @ segment_to_ref.T)[:, :3]
                xyz = torch.from_numpy(xyz)

            gaussians_ply_format = self._build_gs_ply_base_format(xyz, color, opacities, scales, quats)
            save_ply(gaussians_ply_format, os.path.join(save_path, f'gs_{target_frame_idxs[t_idx]}.ply'))

            # gs: use input rgb
            color = rearrange(input_image, 'b t v c h w -> b (t v h w) c')
            color = color[mask]
            gaussians_ply_format[:, :, 3:6] = color
            save_ply(gaussians_ply_format, os.path.join(save_path, f'gs_rgb_{target_frame_idxs[t_idx]}.ply'))

            if self.with_feat:
                gaussians_ply_format = self._build_gs_ply_semantic_format(
                    render_results, gaussians_ply_format, mask, t_idx
                )
                ply_config = PlyExportConfig(
                    semantic_start_idx=14 if self.with_feat else None,
                    mask_indices_start_idx=15
                )
                save_ply(gaussians_ply_format, os.path.join(save_path, f'gs_semantic_{target_frame_idxs[t_idx]}.ply'),
                         ply_config)
            logger.info(f'Save frame_{target_frame_idxs[t_idx]} gs in {save_path}.')

    def save_rendered_pointcloud(self, data_dict, output, save_path, save_orig_results=False):
        rendered_image = output["render_results"]['rendered_image']
        rendered_depth = output["render_results"]['rendered_depth']
        rendered_flow = output["render_results"]['rendered_flow']
        b, tgt_t, v, h, w, c = rendered_image.shape
        if b != 1:
            raise ValueError(f"batch size must be 1 for save_rendered_pointcloud, got {b}")
        target_frame_idxs = data_dict['target_frame_idx'][0].to(torch.int16).tolist()
        c2ws = data_dict['target_camtoworlds'][0].view(-1, 4, 4)

        # save rendered rgb, depth, flow
        if save_orig_results:
            self._save_orig_rendered_results(output, target_frame_idxs, tgt_t, save_path)

        target_ray_dict = self.plucker_embedder(
            data_dict["target_intrinsics"],
            data_dict["target_camtoworlds"],
            image_size=(data_dict["height"], data_dict["width"]),
        )
        xyzs = target_ray_dict['origins'] + target_ray_dict['dirs'] * rendered_depth.unsqueeze(-1)

        for t_idx in range(tgt_t):
            xyz = xyzs[0, t_idx].reshape(-1, 3)
            color = rendered_image[0, t_idx].reshape(-1, 3)
            flow = rendered_flow[0, t_idx].reshape(-1, 3)
            flow = flow * (1 / data_dict['fps'])  # to next frame

            # align to the same coordinate system (first frame in first segment)
            xyz, flow = self._align_xyz_and_flow_to_ref(xyz, flow, data_dict)

            gaussians_ply_format, semantic = self._build_pc_ply_format_with_semantic(
                output, xyz, color, flow, t_idx
            )
            ply_config = PlyExportConfig(
                semantic_start_idx=17 if self.with_feat else None,
                flow_start_idx=14
            )
            save_ply(gaussians_ply_format, os.path.join(save_path, f'pc_rgb_{target_frame_idxs[t_idx]}.ply'),
                     ply_config)

            # pc: semantic color
            if self.with_feat:
                gaussians_ply_format = self._apply_semantic_color_to_ply(
                    gaussians_ply_format, rendered_image, semantic, t_idx
                )
                ply_config = PlyExportConfig(
                    semantic_start_idx=17 if self.with_feat else None,
                    flow_start_idx=14
                )
                save_ply(gaussians_ply_format, os.path.join(save_path, f'pc_semantic_{target_frame_idxs[t_idx]}.ply'),
                         ply_config)

            # save camera pose
            self._save_camera_pose(_SaveCameraPoseParams(c2ws, data_dict, save_path, target_frame_idxs, t_idx, v, xyzs))
            logger.info(f'Save frame_{target_frame_idxs[t_idx]} rendered pc in {save_path}.')

    @staticmethod
    def _compute_gs_mask(params: _ComputeGsMaskParams):
        (opacities, opacity_threshold, target_sky, data_dict, t, v, h, w) = params
        mask = (opacities > opacity_threshold).squeeze(-1)
        # sky mask
        if target_sky is not None:
            context_frame_idx = (data_dict['context_frame_idx'][0]).to(torch.int16).tolist()
            context_frame_idx = [x - min(context_frame_idx) for x in context_frame_idx]
            context_sky = target_sky[:, context_frame_idx].squeeze(-1)
            context_sky_mask = rearrange(context_sky, 'b t v h w -> b (t v h w)') < opacity_threshold
            mask = mask & context_sky_mask
        return mask

    @staticmethod
    def _build_gs_ply_base_format(xyz, color, opacities, scales, quats):
        # gs: use gs rgb
        gaussians_ply_format = torch.zeros(torch.Size([1, color.shape[0], 14]))
        gaussians_ply_format[:, :, 0:3] = xyz
        gaussians_ply_format[:, :, 3:6] = color
        gaussians_ply_format[:, :, 6:7] = opacities
        gaussians_ply_format[:, :, 7:10] = scales
        gaussians_ply_format[:, :, 10:14] = quats
        return gaussians_ply_format

    def _build_gs_ply_semantic_format(self, render_results, gaussians_ply_format, mask, t_idx):
        feats = render_results['gs_decoded_feats']
        feats = rearrange(feats, "b t v h w c -> (b t v h w) c")
        semantic = feat2class(
            feats,
            get_text_label_feats(SEMANTIC_LABEL_LIST),
            similarity_probs_threshold=self.similarity_probs_threshold
        )
        semantic = semantic.view(1, -1)

        color = torch.zeros_like(render_results['gs_color'][:, t_idx])
        for class_idx, _ in enumerate(SEMANTIC_LABEL_LIST):
            color[semantic == class_idx] = torch.tensor(SEMANTIC_ID_TO_COLOR[class_idx]).to(color.device)
        color /= 255.

        # gs: use semantic rgb
        color = color[mask]
        gaussians_ply_format[:, :, 3:6] = color
        # concat semantic
        semantic = semantic[mask]
        semantic = semantic.to(gaussians_ply_format.device)
        semantic = semantic.unsqueeze(0).unsqueeze(-1)
        gaussians_ply_format = torch.concat([gaussians_ply_format, semantic], dim=2)
        # concat mask
        mask_indices = torch.nonzero(mask.view(-1)).squeeze()
        mask_indices = mask_indices.to(gaussians_ply_format.device)
        mask_indices = mask_indices.unsqueeze(0).unsqueeze(-1)
        gaussians_ply_format = torch.concat([gaussians_ply_format, mask_indices], dim=2)
        return gaussians_ply_format

    @staticmethod
    def _save_orig_rendered_results(output, target_frame_idxs, tgt_t, save_path):
        dir_path = os.path.join(save_path, 'orig_results')
        os.makedirs(dir_path, exist_ok=True)
        import cv2
        from src.visualization.visualization_tools import depth_visualizer, scene_flow_to_rgb
        for t_idx in range(tgt_t):
            rgb = output["render_results"]['rendered_image'][0, t_idx]
            rgb = rearrange(rgb, 'v h w c -> h (v w) c')
            rgb = rgb.to(torch.float16).detach().cpu().numpy()[:, :, [2, 1, 0]] * 255
            cv2.imwrite(f'{dir_path}/frame_{target_frame_idxs[t_idx]}_rgb.png', rgb)

            depth = output["render_results"]['rendered_depth'][0, t_idx]
            alpha = output["render_results"]['rendered_alpha'][0, t_idx]
            depth = depth.to(torch.float16).detach().cpu().numpy()
            alpha = alpha.to(torch.float16).detach().cpu().numpy()
            depth_image = depth_visualizer(depth, alpha)
            depth_image = rearrange(depth_image, 'v h w c -> h (v w) c')
            cv2.imwrite(f'{dir_path}/frame_{target_frame_idxs[t_idx]}_depth.png',
                        depth_image[:, :, [2, 1, 0]] * 255)

            flow = output["render_results"]['rendered_flow'][0, t_idx]
            flow = scene_flow_to_rgb(flow, flow_max_radius=15)
            flow = rearrange(flow, 'v h w c -> h (v w) c')
            flow = flow.to(torch.float16).detach().cpu().numpy()[:, :, [2, 1, 0]] * 255
            cv2.imwrite(f'{dir_path}/frame_{target_frame_idxs[t_idx]}_flow.png', flow)

            logger.info(f'Save frame_{target_frame_idxs[t_idx]} rendered pc in {dir_path}.')

    @staticmethod
    def _align_xyz_and_flow_to_ref(xyz, flow, data_dict):
        # align to the same coordinate system (first frame in first segment)
        if 'segment_to_ref' in data_dict:
            segment_to_ref = data_dict['segment_to_ref'][0].to(torch.float32).cpu().numpy()
            xyz = xyz.to(torch.float32).cpu().numpy()
            # right multiply
            xyz = (np.concatenate([xyz, np.ones_like(xyz)[:, :1]], axis=-1) @ segment_to_ref.T)[:, :3]
            xyz = torch.from_numpy(xyz)
            flow = flow.to(torch.float32).cpu().numpy()
            # flow only apply rotations, not translations.
            flow = flow @ segment_to_ref.T[:3, :3]
            flow = torch.from_numpy(flow)
        return xyz, flow

    def _build_pc_ply_format_with_semantic(self, output, xyz, color, flow, t_idx):
        gaussians_ply_format = torch.zeros(torch.Size([1, xyz.shape[0], 14]))
        gaussians_ply_format[:, :, 0:3] = xyz.unsqueeze(0)
        gaussians_ply_format[:, :, 3:6] = color.unsqueeze(0)
        gaussians_ply_format[:, :, 6:7] = torch.ones([1, xyz.shape[0], 1])
        gaussians_ply_format[:, :, 7:10] = torch.ones([1, xyz.shape[0], 3]) * 0.01
        gaussians_ply_format[:, :, 10:14] = torch.ones([1, xyz.shape[0], 4]) * torch.tensor([1, 0, 0, 0])

        # flow
        flow = flow.to(gaussians_ply_format.device)
        flow = flow.unsqueeze(0)
        gaussians_ply_format = torch.concat([gaussians_ply_format, flow], dim=2)

        # pc: rgb color
        semantic = None
        if self.with_feat:
            if os.getenv("CONTEXT_FEAT"):
                semantic = output["render_results"]["rendered_semantic"]
                semantic = semantic[0, t_idx]
                semantic = rearrange(semantic, "v h w -> (v h w)")
            else:
                feats = output["render_results"]["gs_rendered_decoded_feat"]
                feats = feats[0, t_idx]
                feats = rearrange(feats, "v h w c -> (v h w) c")
                semantic = feat2class(
                feats,
                get_text_label_feats(SEMANTIC_LABEL_LIST),
                similarity_probs_threshold=self.similarity_probs_threshold
            )
            semantic = semantic.to(gaussians_ply_format.device)
            semantic = semantic.unsqueeze(0).unsqueeze(-1)
            gaussians_ply_format = torch.concat([gaussians_ply_format, semantic], dim=2)
        return gaussians_ply_format, semantic

    @staticmethod
    def _apply_semantic_color_to_ply(gaussians_ply_format, rendered_image, semantic, t_idx):
        # pc: semantic color
        color = torch.zeros_like(rendered_image[0, t_idx].reshape(-1, 3)).to(torch.float32)
        for class_idx, _ in enumerate(SEMANTIC_LABEL_LIST):
            semantic_color = torch.tensor(SEMANTIC_ID_TO_COLOR[class_idx]).to(color.device)
            color[semantic.squeeze(0).squeeze(-1) == class_idx] = semantic_color
        color /= 255.
        gaussians_ply_format[:, :, 3:6] = color
        return gaussians_ply_format

    def _voxelize_and_pad_gs_attrs(self, gs_attrs, gs_confs_batched, b, tgt_t):
        gs_attrs_b_t_lists = {attr: [] for attr in gs_attrs.keys()}
        for b_idx in range(b):
            for t_idx in range(tgt_t):
                b_t_idx = b_idx * tgt_t + t_idx

                # Voxelize using a specific voxelsize, and calculate the weight by confidence
                weights, inverse_indices = self.voxelizaton_using_confidence(gs_attrs['means'][b_t_idx],
                                                                             gs_confs_batched[b_t_idx].squeeze(1),
                                                                             self.voxel_size)

                # Loop through each gaussian attribute
                for name, attr in gs_attrs.items():
                    # Compute weighted average of gaussian attribute
                    # aggregate on feature dimension or after activation
                    weighted_attrs_b_t = attr[b_t_idx] * weights
                    # Aggregate per voxel
                    voxel_attrs_b_t = scatter_add(weighted_attrs_b_t, inverse_indices, dim=0)

                    gs_attrs_b_t_lists[name].append(voxel_attrs_b_t)

        # NOTE: dynamic shape
        max_voxels = max(f.shape[0] for attr_b_t in gs_attrs_b_t_lists.values() for f in attr_b_t)
        min_voxels = min(f.shape[0] for attr_b_t in gs_attrs_b_t_lists.values() for f in attr_b_t)

        # Padding
        gs_attrs_voxel_padded = {attr: None for attr in gs_attrs.keys()}
        # Loop through each gaussian attribute
        for name, gs_attrs_b_t_list in gs_attrs_b_t_lists.items():
            gs_attrs_voxel_padded[name] = self.pad_tensor_list(
                gs_attrs_b_t_list, (max_voxels,), value=0.0
            )
        return gs_attrs_voxel_padded

    def _render_and_merge_context_contributions(self, params: _RenderAndMergeContextContributionsParams):
        (gs_params, data_dict, render_results, t, time_step, static_render, chunk_start) = params
        for idx in range(t):
            chunk_context_render_results = self.forward_renderer(
                gs_params, data_dict, idx=idx,
                time_step=time_step, static_render=static_render
            )
            if chunk_start == 0:
                render_results[f'context_{idx}_rendered_image'] = \
                    chunk_context_render_results['rendered_image']
                render_results[f'context_{idx}_rendered_depth'] = \
                    chunk_context_render_results['rendered_depth']
                render_results[f'context_{idx}_rendered_alpha'] = \
                    chunk_context_render_results['rendered_alpha']
                render_results[f'context_{idx}_rendered_flow'] = \
                    chunk_context_render_results['rendered_flow']
            else:
                render_results[f'context_{idx}_rendered_image'] = torch.cat(
                    [render_results[f'context_{idx}_rendered_image'],
                     chunk_context_render_results['rendered_image']], dim=1
                )
                render_results[f'context_{idx}_rendered_depth'] = torch.cat(
                    [render_results[f'context_{idx}_rendered_depth'],
                     chunk_context_render_results['rendered_depth']], dim=1
                )
                render_results[f'context_{idx}_rendered_alpha'] = torch.cat(
                    [render_results[f'context_{idx}_rendered_alpha'],
                     chunk_context_render_results['rendered_alpha']], dim=1
                )
                render_results[f'context_{idx}_rendered_flow'] = torch.cat(
                    [render_results[f'context_{idx}_rendered_flow'],
                     chunk_context_render_results['rendered_flow']], dim=1
                )
            del chunk_context_render_results
        return render_results

    @staticmethod
    def _save_camera_pose(params: _SaveCameraPoseParams):
        (c2ws, data_dict, save_path, target_frame_idxs, t_idx, v, xyzs) = params
        filepath = os.path.join(save_path, f'camera_pose_{target_frame_idxs[t_idx]}.txt')
        with open(filepath, 'w') as f:
            for v_idx in range(v):
                c2w = c2ws[t_idx * v + v_idx]
                if 'segment_to_ref' in data_dict:
                    segment_to_ref = data_dict['segment_to_ref'][0].to(xyzs.device)
                    c2w = segment_to_ref @ c2w  # (row.T @ segment_to_ref.T).T
                c2w_str = ' '.join(str(x) for x in (c2w).view(-1, 16).tolist())
                f.write(c2w_str + '\n')
