# coding=utf-8
# Adapted from
# https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/rendering.py

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Tuple

import torch
from torch import Tensor

import acl
from meta_gauss_render import (
    spherical_harmonics,
    flash_gaussian_build_mask,
    gaussian_sort, calc_render,
    get_render_schedule
)
from src.utils.projection_three_dims_gaussian_fused import (
    projection_three_dims_gaussian_fused,
)
from src.utils.render_types import GaussianData, CameraParams, RenderConfig, ProjectionResult, SortData


def _compute_sh_colors(gaussians: GaussianData, camera: CameraParams,
                       config: RenderConfig, num_gaussians: int,
                       num_cameras: int, batch_size: int) -> Tensor:
    colors = gaussians.colors[:, None]
    if config.sh_degree is None:
        if colors.dim() == 2:
            colors = colors.expand(num_cameras, -1, -1)
    else:
        camtoworlds = torch.inverse(camera.viewmats)
        if colors.dim() == 3:
            shs = colors.expand(num_cameras, -1, -1, -1)
        else:
            shs = colors

        def build_color(means3d, sh_coeffs, degree, camera_center, current_batch_size):
            rays_o = camera_center
            rays_d = means3d - rays_o
            rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
            coeff_count = (degree + 1) ** 2
            color = spherical_harmonics(
                degree,
                rays_d.reshape(current_batch_size, num_gaussians, 3),
                sh_coeffs[:, :coeff_count, :].reshape(
                    current_batch_size, num_gaussians, coeff_count, 3),
            )
            return (color + 0.5).clip(min=0.0)

        colors = build_color(
            means3d=gaussians.means,
            sh_coeffs=shs[0],
            degree=config.sh_degree,
            camera_center=camtoworlds[0, :3, 3],
            current_batch_size=batch_size,
        )

    if config.sh_degree is None:
        colors = colors.permute(1, 2, 0).contiguous()
    return colors


class Rasterizer:
    def __init__(self, tile_size=32, camera_model="pinhole") -> None:
        self.tile_size = tile_size
        self.camera_model = camera_model
        self.padded_width = None
        self.padded_height = None
        self.tile_grid = None
        self.pix_coord = None

    def tile2image(self, rendered_image, height, width):
        tile_rows = math.ceil(self.padded_height / self.tile_size)
        tile_cols = math.ceil(self.padded_width / self.tile_size)
        rendered_image = rendered_image.reshape(
            tile_rows, tile_cols, self.tile_size, self.tile_size, -1
        )
        rendered_image = rendered_image.permute(0, 2, 1, 3, 4)
        rendered_image = rendered_image.reshape(
            tile_rows * self.tile_size,
            tile_cols * self.tile_size,
            -1,
        )
        return rendered_image.permute(2, 0, 1)[:, :height, :width]

    def ascend_rasterize_splats(
            self,
            gaussians: GaussianData,
            camera: CameraParams,
            config: RenderConfig,
    ) -> Tuple[Tensor, Tensor, Dict]:
        tile_size = self.tile_size
        if self.tile_grid is None:
            self._init_tile_grid(config.width, config.height, tile_size, gaussians.means.device)

        colors = gaussians.colors
        flow = None
        if colors.shape[-1] == 6:
            flow = colors[:, 3:]
            colors = colors[:, :3]

        rgb_gaussians = GaussianData(gaussians.means, gaussians.quats, gaussians.scales,
                                     gaussians.opacities, colors)
        render_colors, render_depths, render_alphas, info = self._ascend_rasterization(
            rgb_gaussians, camera, config, tile_size)

        if flow is not None:
            flow_gaussians = GaussianData(gaussians.means, gaussians.quats, gaussians.scales,
                                          gaussians.opacities, flow)
            render_flows, _, _, _ = self._ascend_rasterization(
                flow_gaussians, camera, config, tile_size)
            render_colors = torch.cat([render_colors, render_flows, render_depths], dim=-1)
        else:
            render_colors = torch.cat([render_colors, render_depths], dim=-1)

        return render_colors, render_alphas, info

    def _init_tile_grid(self, width: int, height: int, tile_size: int, device):
        self.padded_width = math.ceil(width / tile_size) * tile_size
        self.padded_height = math.ceil(height / tile_size) * tile_size
        tile_y, tile_x = torch.meshgrid(
            torch.arange(0, self.padded_height, tile_size),
            torch.arange(0, self.padded_width, tile_size),
            indexing="ij",
        )
        self.tile_grid = torch.stack((tile_y, tile_x), dim=-1).view(-1, 2).to(device)
        pix_x, pix_y = torch.meshgrid(
            torch.arange(self.padded_width),
            torch.arange(self.padded_height),
            indexing="xy",
        )
        self.pix_coord = torch.stack((pix_x, pix_y), dim=-1).to(device)

    def _project_and_sort(self, proj: ProjectionResult, config: RenderConfig,
                          tile_size: int) -> SortData:
        with torch.no_grad():
            tile_sums, tile_offsets, tile_depths, tile_gauss_ids = flash_gaussian_build_mask(
                proj.means2d, proj.opacities[None, :], proj.conics, proj.covars2d,
                proj.depths, proj.cnt[None, :],
                self.tile_grid.float(), config.width, config.height, tile_size,
            )
            sorted_cnts = tile_offsets.squeeze(-1)[:, :, -1]
            sorted_offset = torch.cumsum(sorted_cnts.flatten(), dim=0)
            tile_sums_3d = tile_sums.squeeze(-1)
            tile_sums_cpu = tile_sums_3d.cpu().to(torch.int64)

            vector_num = acl.get_device_capability(0, 1)[0]
            tile_num = tile_sums_cpu.shape[-1]

            sort_bins = min(vector_num, tile_num)
            sort_sched_tensor_cpu = get_render_schedule(tile_sums_cpu, sort_bins)
            sort_sched_tensor = sort_sched_tensor_cpu.npu()
            render_sched_tensor_cpu = get_render_schedule(tile_sums_cpu, vector_num)
            render_sched_tensor = render_sched_tensor_cpu.npu()
            max_tile_gauss = torch.amax(tile_sums).item()

            sorted_gs_ids = gaussian_sort(
                sort_sched_tensor, tile_sums, tile_depths, tile_gauss_ids,
                sorted_offset, max_tile_gauss,
            )
        return SortData(sorted_gs_ids, sorted_offset, render_sched_tensor)

    def _render_single_view(self, cam_view: int, proj: ProjectionResult,
                            sort_data: SortData, config: RenderConfig, tile_size: int):
        cf_means2 = proj.means2d[0, cam_view]
        cf_colors3 = proj.colors[0, cam_view]
        cf_opacity = proj.opacities[0, cam_view]
        inv_x_0 = proj.conics[0, cam_view, 0, :]
        inv_x_1 = proj.conics[0, cam_view, 1, :]
        inv_x_2 = proj.conics[0, cam_view, 2, :]
        cf_depths = proj.depths[0, cam_view][None, :]

        padded_height = self.padded_height
        padded_width = self.padded_width
        tile_rows = padded_height // tile_size
        tile_cols = padded_width // tile_size
        pix_coords = (
            self.pix_coord.reshape(tile_rows, tile_size, tile_cols, tile_size, 2)
            .permute(0, 2, 1, 3, 4)
            .reshape(tile_rows * tile_cols, tile_size * tile_size, 2)
            .permute(0, 2, 1)
            .to(torch.float32)
            .contiguous()
        )
        start_index = sort_data.sorted_offset[cam_view - 1] if cam_view > 0 else 0
        end_index = sort_data.sorted_offset[cam_view]
        sorted_gs_id = sort_data.sorted_gs_ids[start_index:end_index]
        lb_sched = sort_data.render_sched_tensor[0, cam_view, :]

        cf_render_colors, cf_render_depths = calc_render(
            cf_means2, inv_x_0, inv_x_1, inv_x_2,
            cf_opacity, cf_colors3, cf_depths,
            pix_coords, lb_sched, sorted_gs_id,
        )
        cf_colors3_for_alphas = torch.ones_like(cf_colors3)
        cf_render_alphas, _ = calc_render(
            cf_means2, inv_x_0, inv_x_1, inv_x_2,
            cf_opacity, cf_colors3_for_alphas, cf_depths,
            pix_coords, lb_sched, sorted_gs_id,
        )
        cf_render_alphas = cf_render_alphas[0:1]

        cf_render_colors = self.tile2image(cf_render_colors.permute(1, 2, 0),
                                           config.height, config.width)
        cf_render_depths = self.tile2image(cf_render_depths.permute(1, 2, 0),
                                           config.height, config.width)
        cf_render_alphas = self.tile2image(cf_render_alphas.permute(1, 2, 0),
                                           config.height, config.width)

        return (cf_render_colors.permute(1, 2, 0),
                cf_render_depths.permute(1, 2, 0),
                cf_render_alphas.permute(1, 2, 0))

    def _ascend_rasterization(
            self,
            gaussians: GaussianData,
            camera: CameraParams,
            config: RenderConfig,
            tile_size: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict]:
        num_gaussians = gaussians.means.shape[0]
        num_cameras = camera.viewmats.shape[0]
        batch_size = 1

        colors = _compute_sh_colors(gaussians, camera, config,
                                    num_gaussians, num_cameras, batch_size)

        means2d, depths, conics, opacities, radius, covars2d, colors, cnt = \
            projection_three_dims_gaussian_fused(
                gaussians.means.reshape(batch_size, num_gaussians, 3),
                colors.contiguous(), None,
                gaussians.quats.reshape(batch_size, num_gaussians, 4),
                gaussians.scales.reshape(batch_size, num_gaussians, 3),
                gaussians.opacities.reshape(batch_size, num_gaussians),
                camera.viewmats.reshape(batch_size, num_cameras, 4, 4).contiguous(),
                camera.ks.reshape(batch_size, num_cameras, 3, 3),
                config.width, config.height, 0.3, config.near_plane,
            )

        proj_result = ProjectionResult(
            means2d=means2d, opacities=opacities, conics=conics,
            covars2d=covars2d, depths=depths, cnt=cnt, colors=colors,
        )
        sort_data = self._project_and_sort(proj_result, config, tile_size)

        render_colors, render_depths, render_alphas = [], [], []
        for cam_view in range(num_cameras):
            cr, cd, ca = self._render_single_view(
                cam_view, proj_result, sort_data, config, tile_size)
            render_colors.append(cr)
            render_depths.append(cd)
            render_alphas.append(ca)
        render_colors = torch.stack(render_colors)
        render_depths = torch.stack(render_depths)
        render_alphas = torch.stack(render_alphas)

        meta = {
            "camera_ids": None, "gaussian_ids": None,
            "means2d": proj_result.means2d, "depths": proj_result.depths,
            "conics": proj_result.conics, "opacities": proj_result.opacities,
            "tile_width": self.padded_width // tile_size,
            "tile_height": self.padded_height // tile_size,
            "width": config.width, "height": config.height,
            "tile_size": tile_size, "n_cameras": num_cameras,
        }
        return render_colors, render_depths, render_alphas, meta


def new_ascend_rasterization(
        ascend_render: Rasterizer,
        gaussians: GaussianData,
        camera: CameraParams,
        config: RenderConfig,
) -> Tuple[Tensor, Tensor, Dict]:
    batch_size = gaussians.means.shape[0]
    render_colors = []
    render_alphas = []
    for batch_idx in range(batch_size):
        batch_gaussians = GaussianData(
            means=gaussians.means[batch_idx],
            quats=gaussians.quats[batch_idx],
            scales=gaussians.scales[batch_idx],
            opacities=gaussians.opacities[batch_idx],
            colors=gaussians.colors[batch_idx],
        )
        num_views = camera.viewmats[batch_idx].shape[0]
        cols_single_sample, alphas_single_sample = [], []
        for view_idx in range(num_views):
            view_camera = CameraParams(
                viewmats=camera.viewmats[batch_idx][view_idx:view_idx + 1].contiguous(),
                ks=camera.ks[batch_idx][view_idx:view_idx + 1],
                camera_model="pinhole",
            )
            cols, alphas, _ = ascend_render.ascend_rasterize_splats(
                batch_gaussians, view_camera, config,
            )
            cols_single_sample.append(cols)
            alphas_single_sample.append(alphas)
        cols_single_sample = torch.cat(cols_single_sample)
        alphas_single_sample = torch.cat(alphas_single_sample)
        render_colors.append(cols_single_sample)
        render_alphas.append(alphas_single_sample)
    render_colors = torch.stack(render_colors)
    render_alphas = torch.stack(render_alphas)
    return render_colors, render_alphas, None