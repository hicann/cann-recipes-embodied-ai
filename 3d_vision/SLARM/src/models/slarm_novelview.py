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

from .slarm_ply_export import _RenderAndMergeContextContributionsParams, _SaveGsParamsToPlyParams

logger = logging.getLogger(__name__)

_FinalizeForwardParams = namedtuple('_FinalizeForwardParams', [
    'data_dict', 'gs_params', 'ray_dict', 'pred_feat', 'sky_token', 'affine_tokens', 'pose_enc_list',
    'pred_context_depth', 'pred_context_depth_conf', 'pred_context_pts3d', 'pred_context_pts3d_conf',
    'stream_save', 'aggregator_kv_cache_list', 'camera_head_kv_cache_list'
])
_RenderAllTargetsParams = namedtuple('_RenderAllTargetsParams', [
    'data_dict', 'gs_params', 'pred_feat', 'step', 't', 'time_step', 'static_render'
])
_ApplySkyTokenParams = namedtuple('_ApplySkyTokenParams', [
    'data_dict', 'gs_params', 'images', 'opacities', 'render_results', 'sky_token', 't', 'step', 'context_images',
    'context_opacities'
])
_ApplyAffineTokenParams = namedtuple('_ApplyAffineTokenParams', [
    'gs_params', 'affine_tokens', 'images', 'render_results', 'context_images', 't', 'v', 'b'
])
_BuildPostOutputParams = namedtuple('_BuildPostOutputParams', [
    'ray_dict', 'gs_params', 'render_results', 'pred_feat', 'sky_token', 'affine_tokens', 'context_images',
    'context_depths', 'context_opacities', 'pose_enc_list', 'pred_context_depth', 'pred_context_depth_conf',
    'pred_context_pts3d', 'pred_context_pts3d_conf', 'b', 't', 'v'
])
_RenderChunkedTargetsParams = namedtuple('_RenderChunkedTargetsParams', [
    'data_dict', 'gs_params', 'pred_feat', 'step', 't', 'time_step', 'static_render'
])
_ApplyChunkedTargetSkyParams = namedtuple('_ApplyChunkedTargetSkyParams', [
    'target_ray_dict', 'sky_token', 'images', 'opacities', 'render_results', 't', 'step', 'data_dict'
])
_ApplyChunkedContextSkyParams = namedtuple('_ApplyChunkedContextSkyParams', [
    'context_ray_dict', 'sky_token', 'context_images', 'context_opacities', 'step', 'data_dict'
])
_ApplyNovelViewRtParams = namedtuple('_ApplyNovelViewRtParams', [
    'data_dict', 'degree_x', 'degree_y', 'degree_z', 'trans_x', 'trans_y', 'trans_z', 'fix_cam_pos'
])
_PostProcessingParams = namedtuple('_PostProcessingParams', [
    'data_dict', 'gs_params', 'time_step', 'ray_dict', 'pred_feat', 'sky_token', 'affine_tokens',
    'pose_enc_list', 'pred_context_depth', 'pred_context_depth_conf', 'pred_context_pts3d',
    'pred_context_pts3d_conf', 'static_render'
])


class SLARMNovelViewMixin:
    """Mixin for SLARM novel view rendering and post-processing functionality."""

    @staticmethod
    def apply_novelview_rt(params: _ApplyNovelViewRtParams):
        (data_dict, degree_x, degree_y, degree_z, trans_x, trans_y, trans_z, fix_cam_pos) = params
        def rotation_matrix_y(theta_degrees):
            theta = np.radians(theta_degrees)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            return np.array([
                [cos_theta, 0, sin_theta],
                [0, 1, 0],
                [-sin_theta, 0, cos_theta]
            ])

        def rotation_matrix_z(theta_degrees):
            theta = np.radians(theta_degrees)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            return np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ])

        def rotation_matrix_x(theta_degrees):
            theta = np.radians(theta_degrees)  # convert degrees to radians
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            return np.array([
                [1, 0, 0],
                [0, cos_theta, -sin_theta],
                [0, sin_theta, cos_theta]
            ])

        r_z = torch.from_numpy(rotation_matrix_z(degree_z))[None, None, None, ...].to(torch.float32).cuda()
        r_y = torch.from_numpy(rotation_matrix_y(degree_y))[None, None, None, ...].to(torch.float32).cuda()
        r_x = torch.from_numpy(rotation_matrix_x(degree_x))[None, None, None, ...].to(torch.float32).cuda()

        # translation
        data_dict["target_camtoworlds"][..., :3, 3] += torch.tensor([[[[trans_x, trans_y, trans_z]]]]).cuda()
        # fix_camera_t
        if fix_cam_pos:
            first_cam_pos = data_dict["target_camtoworlds"][:, 0:1, :, :3, 3]
            data_dict["target_camtoworlds"][..., :3, 3] = first_cam_pos

        # rotation
        data_dict["target_camtoworlds"][..., :3, :3] @= (r_z @ r_y @ r_x)

    def post_processing(self, params: _PostProcessingParams):
        (data_dict, gs_params, time_step, ray_dict, pred_feat, sky_token, affine_tokens,
         pose_enc_list, pred_context_depth, pred_context_depth_conf, pred_context_pts3d,
         pred_context_pts3d_conf, static_render) = params
        images = data_dict["context_image"]
        b, t, v, c, h, w = images.size()

        step = 20

        render_results = self._render_all_targets(_RenderAllTargetsParams(
            data_dict, gs_params, pred_feat, step, t, time_step, static_render
        ))

        images, opacities = render_results["rendered_image"], render_results["rendered_alpha"]
        # Rendering every context frame in its own view.
        context_images = None
        context_depths = None
        context_opacities = None
        if self.render_context_view:
            context_render_results = self.forward_renderer_context_view(gs_params, data_dict)
            context_images = context_render_results['rendered_image'].clone()
            context_depths = context_render_results['rendered_depth'].clone()
            context_opacities = context_render_results['rendered_alpha'].clone()

        if self.use_sky_token:
            images, opacities, context_images = self._apply_sky_token(_ApplySkyTokenParams(
                data_dict, gs_params, images, opacities, render_results,
                sky_token, t, step, context_images, context_opacities
            ))

        if self.use_affine_token:
            images, context_images = self._apply_affine_token(_ApplyAffineTokenParams(
                gs_params, affine_tokens, images, render_results,
                context_images, t, v, b
            ))

        render_results["rendered_image"] = images
        render_results = self.forward_decoder(render_results)

        if self.with_feat:
            render_results = self._process_feat(
                render_results, data_dict, pred_feat
            )

        return self._build_post_output(_BuildPostOutputParams(
            ray_dict, gs_params, render_results, pred_feat, sky_token, affine_tokens,
            context_images, context_depths, context_opacities,
            pose_enc_list, pred_context_depth, pred_context_depth_conf,
            pred_context_pts3d, pred_context_pts3d_conf, b, t, v
        ))

    @staticmethod
    def _apply_sky_to_context_frames(render_results, sky, t, chunk_start=None, step=None):
        for idx in range(t):
            image_key = f'context_{idx}_rendered_image'
            alpha_key = f'context_{idx}_rendered_alpha'
            if chunk_start is None:
                render_results[image_key] = (
                    render_results[image_key] +
                    (1 - render_results[alpha_key][..., None]) * sky
                )
            else:
                render_results[image_key][:, chunk_start:chunk_start + step] += (
                    (1 - render_results[alpha_key]
                     [:, chunk_start:chunk_start + step][..., None]) *
                    sky
                )

    def _apply_chunked_context_sky(self, params: _ApplyChunkedContextSkyParams):
        (context_ray_dict, sky_token, context_images, context_opacities, step, data_dict) = params
        for chunk_start in range(0, data_dict["context_camtoworlds"].shape[1], step):
            context_dirs = context_ray_dict["dirs"][:, chunk_start:chunk_start + step]
            chunk_context_sky = self.sky_head(context_dirs, sky_token)
            context_images[:, chunk_start:chunk_start + step] += (
                1 - context_opacities[:, chunk_start:chunk_start + step][..., None]
            ) * chunk_context_sky

    def _apply_chunked_target_sky(self, params: _ApplyChunkedTargetSkyParams):
        (target_ray_dict, sky_token, images, opacities, render_results, t, step, data_dict) = params
        for chunk_start in range(0, data_dict["target_camtoworlds"].shape[1], step):
            target_dirs = target_ray_dict["dirs"][:, chunk_start:chunk_start + step]
            chunk_target_sky = self.sky_head(target_dirs, sky_token)
            images[:, chunk_start:chunk_start + step] += (
                1 - opacities[:, chunk_start:chunk_start + step][..., None]
            ) * chunk_target_sky
            # per-context in target view
            if self.training or not self.render_context_frame_contribution:
                continue
            self._apply_sky_to_context_frames(render_results, chunk_target_sky, t, chunk_start, step)

    def _apply_sky_token(self, params: _ApplySkyTokenParams):
        (
            data_dict, gs_params, images, opacities, render_results, sky_token, t, step, context_images,
            context_opacities
        ) = params
        target_ray_dict = self.plucker_embedder(
            data_dict["target_intrinsics"],
            data_dict["target_camtoworlds"],
            image_size=(data_dict["height"], data_dict["width"]),
        )
        if data_dict["target_camtoworlds"].shape[1] <= step:
            # target
            target_sky = self.sky_head(target_ray_dict["dirs"], sky_token)
            images = images + (1 - opacities[..., None]) * target_sky
            # per-context in target view
            if not self.training and self.render_context_frame_contribution:
                self._apply_sky_to_context_frames(render_results, target_sky, t)
        else:
            self._apply_chunked_target_sky(_ApplyChunkedTargetSkyParams(
                target_ray_dict, sky_token, images, opacities, render_results, t, step, data_dict
            ))
        if self.render_context_view:
            context_ray_dict = self.plucker_embedder(
                data_dict["context_intrinsics"],
                data_dict["context_camtoworlds"],
                image_size=(data_dict["height"], data_dict["width"]),
            )
            if data_dict["context_camtoworlds"].shape[1] <= step:
                context_sky = self.sky_head(context_ray_dict["dirs"], sky_token)
                context_images = context_images + (1 - context_opacities[..., None]) * context_sky
            else:
                self._apply_chunked_context_sky(_ApplyChunkedContextSkyParams(
                    context_ray_dict, sky_token, context_images, context_opacities, step, data_dict
                ))
        if "target_sky" not in gs_params.keys() or gs_params["target_sky"] is None:
            gs_params["target_sky"] = 1 - opacities[..., None]
        if "sky_token" not in gs_params.keys() or gs_params["sky_token"] is None:
            gs_params["sky_token"] = sky_token
        return images, opacities, context_images

    def _apply_affine_token(self, params: _ApplyAffineTokenParams):
        (gs_params, affine_tokens, images, render_results, context_images, t, v, b) = params
        affine = self.affine_linear(affine_tokens)  # b v (gs_dim * (gs_dim + 1))
        affine_matrix = rearrange(affine, "b v (p q) -> b v p q", p=self.gs_dim)
        linear_part = affine_matrix[..., :3]  # b, v, 3, 3
        translation_part = affine_matrix[..., 3]  # b, v, 3
        translation_part = translation_part.view(b, 1, v, 1, 1, 3)
        # whether to add regularization to keep it consistent with original image
        gs_params["images_without_affine"] = images.clone()
        # apply linear and translation
        images = torch.einsum('btvhwi,bvij->btvhwj', images, linear_part) + translation_part
        if self.render_context_view:
            context_images = torch.einsum('btvhwi,bvij->btvhwj', context_images, linear_part) + translation_part
        if not self.training and self.render_context_frame_contribution:
            for idx in range(t):
                context_key = f'context_{idx}_rendered_image'
                render_results[context_key] = (
                    torch.einsum(
                        'btvhwi,bvij->btvhwj',
                        render_results[context_key],
                        linear_part
                    ) + translation_part
                )
        if "affine" not in gs_params.keys() or gs_params["affine"] is None:
            gs_params["affine"] = {
                'linear': linear_part,
                'translation': translation_part,
            }
        return images, context_images

    def _build_post_output(self, params: _BuildPostOutputParams):
        (
            ray_dict, gs_params, render_results, pred_feat, sky_token, affine_tokens, context_images,
            context_depths, context_opacities, pose_enc_list, pred_context_depth, pred_context_depth_conf,
            pred_context_pts3d, pred_context_pts3d_conf, b, t, v
        ) = params
        output = dict(
            ray_dict=ray_dict,
            gs_params=gs_params,
            render_results=render_results,
            pred_feat=pred_feat,
            sky_token=sky_token,
            affine_tokens=affine_tokens
        )
        # context
        if self.render_context_view:
            output["rendered_context_image"] = context_images
            output["rendered_context_depth"] = context_depths
            output["rendered_context_alpha"] = context_opacities

        if self.pred_gs_conf:
            output['pred_gs_conf'] = rearrange(gs_params['confs'], 'b t v h w 1 -> b 1 (t v) h w')  # gs confidence
        if self.camera_head is not None:
            if pose_enc_list is None:
                raise ValueError("pose_enc_list must not be None when camera_head is not None")
            output['pred_context_camera_enc_list'] = pose_enc_list
        if self.depth_head is not None:
            if pred_context_depth is None or pred_context_depth_conf is None:
                raise ValueError(
                    "pred_context_depth and pred_context_depth_conf must not be None "
                    "when depth_head is not None"
                )
            output['pred_context_depth'] = pred_context_depth.squeeze(-1)
            output['pred_context_depth_conf'] = pred_context_depth_conf
        if self.point_head is not None:
            if pred_context_pts3d is None or pred_context_pts3d_conf is None:
                raise ValueError(
                    "pred_context_pts3d and pred_context_pts3d_conf must not be None "
                    "when point_head is not None"
                )
            output['pred_context_pts3d'] = pred_context_pts3d
            output['pred_context_pts3d_conf'] = pred_context_pts3d_conf
        return output

    def _render_all_targets(self, params: _RenderAllTargetsParams):
        (data_dict, gs_params, pred_feat, step, t, time_step, static_render) = params
        if data_dict["target_camtoworlds"].shape[1] <= step:
            # Rendering the results of context frame aggregation.
            render_results = self.forward_renderer(
                gs_params, data_dict, feats=pred_feat,
                time_step=time_step, static_render=static_render
            )

            # Rendering every context frame (only itself, no aggregation) in target view.
            if not self.training and self.render_context_frame_contribution:
                for idx in range(t):
                    context_render_results = self.forward_renderer(
                        gs_params, data_dict, idx=idx,
                        time_step=time_step, static_render=static_render
                    )
                    render_results[f'context_{idx}_rendered_image'] = context_render_results['rendered_image']
                    render_results[f'context_{idx}_rendered_depth'] = context_render_results['rendered_depth']
                    render_results[f'context_{idx}_rendered_alpha'] = context_render_results['rendered_alpha']
                    render_results[f'context_{idx}_rendered_flow'] = context_render_results['rendered_flow']
                del context_render_results
        else:
            render_results = self._render_chunked_targets(_RenderChunkedTargetsParams(
                data_dict, gs_params, pred_feat, step, t, time_step, static_render
            ))
        return render_results

    def _render_chunked_targets(self, params: _RenderChunkedTargetsParams):
        (data_dict, gs_params, pred_feat, step, t, time_step, static_render) = params
        render_results = None
        chunk_data_dict = data_dict.copy()
        for chunk_start in range(0, data_dict["target_camtoworlds"].shape[1], step):
            chunk_end = min(chunk_start + step, data_dict["target_camtoworlds"].shape[1])
            chunk_data_dict["target_camtoworlds"] = data_dict["target_camtoworlds"][
                :, chunk_start:chunk_end
            ]
            chunk_data_dict["target_intrinsics"] = data_dict["target_intrinsics"][
                :, chunk_start:chunk_end
            ]
            chunk_data_dict["target_time"] = data_dict["target_time"][:, chunk_start:chunk_end]
            chunk_render_results = self.forward_renderer(
                gs_params, chunk_data_dict, feats=pred_feat,
                time_step=time_step, static_render=static_render
            )
            if chunk_start == 0:
                render_results = chunk_render_results
            else:
                for key, value in chunk_render_results.items():
                    render_results[key] = torch.cat([render_results[key], value], dim=1)

            # Rendering every context frame (only itself, no aggregation) in target view.
            if self.training or not self.render_context_frame_contribution:
                continue
            render_results = self._render_and_merge_context_contributions(_RenderAndMergeContextContributionsParams(
                gs_params, chunk_data_dict, render_results, t,
                time_step, static_render, chunk_start
            ))
        return render_results

    def _process_feat(self, render_results, data_dict, pred_feat):
        if os.getenv("CONTEXT_FEAT"):
            render_results["rendered_feat"] = pred_feat
        render_results["rendered_feat"] = self.feat_decoder(render_results["rendered_feat"])
        if os.getenv("CONTEXT_FEAT") and not self.training:
            render_results['rendered_semantic'] = self.forward_renderer_target_view_feat(
                render_results, data_dict, render_results["rendered_feat"]
            )

        if self.save_gaussian:
            render_results["gs_decoded_feats"] = self.feat_decoder(pred_feat)
        if self.save_rendered_pc:
            render_results["gs_rendered_decoded_feat"] = render_results["rendered_feat"]
        return render_results

    def _save_outputs(self, data_dict, output, gs_params, stream_save):
        # save gaussian
        if not self.training and self.save_gaussian and stream_save:
            self.save_gs_params_to_ply(_SaveGsParamsToPlyParams(data_dict, output["render_results"],
                                       target_sky=gs_params["target_sky"] if self.use_sky_token else None,
                                       affine=gs_params["affine"] if self.use_affine_token else None,
                                       save_path=self.gaussian_save_path))

        # save rendered pointcloud
        if not self.training and self.save_rendered_pc and stream_save:
            self.save_rendered_pointcloud(data_dict, output, save_path=self.rendered_pc_save_path)

    def _finalize_forward(self, params: _FinalizeForwardParams):
        (
            data_dict, gs_params, ray_dict, pred_feat, sky_token, affine_tokens, pose_enc_list, pred_context_depth,
            pred_context_depth_conf, pred_context_pts3d, pred_context_pts3d_conf, stream_save,
            aggregator_kv_cache_list, camera_head_kv_cache_list
        ) = params
        if not self.training and self.use_render_novel_view:
            self.apply_novelview_rt(_ApplyNovelViewRtParams(
                data_dict, degree_x=0, degree_y=-10, degree_z=0, trans_x=-7, trans_y=-5, trans_z=2, fix_cam_pos=False
            ))

        output = self.post_processing(_PostProcessingParams(
            data_dict, gs_params, 5, ray_dict, pred_feat, sky_token, affine_tokens,
            pose_enc_list, pred_context_depth, pred_context_depth_conf,
            pred_context_pts3d, pred_context_pts3d_conf, False
        ))

        self._save_outputs(data_dict, output, gs_params, stream_save)

        output["aggregator_kv_cache_list"] = None
        output["camera_head_kv_cache_list"] = None

        if aggregator_kv_cache_list is not None:
            output["aggregator_kv_cache_list"] = aggregator_kv_cache_list

        if camera_head_kv_cache_list is not None:
            output["camera_head_kv_cache_list"] = camera_head_kv_cache_list

        return output
