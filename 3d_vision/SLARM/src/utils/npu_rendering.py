# coding=utf-8
# Adapted from
# https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/_torch_impl.py

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal, assert_never

from gsplat.cuda._torch_impl import (
    _world_to_cam, _ortho_proj, _fisheye_proj, _persp_proj,
    _quat_scale_to_covar_preci,
)

from meta_gauss_render import AscendGaussRender
from meta_gauss_render.npu import CalcRender, get_render_schedule, get_num_vector_core

from src.utils.render_types import (
    CameraParams,
    GaussianData,
    ProjectionConfig,
    ProjectionInput,
    ProjectionOutput,
    RenderConfig,
)


# Consistent with _fully_fused_projection in gsplat/cuda/_torch_impl.py, with additional covars2d output.
def _fully_fused_projection(
        proj_input: ProjectionInput,
        camera: CameraParams,
        config: ProjectionConfig,
) -> ProjectionOutput:
    """PyTorch implementation of `gsplat.cuda._wrapper.fully_fused_projection()`

    .. note::

        This is a minimal implementation of fully fused version, which has more
        arguments. Not all arguments are supported.
    """

    if camera.camera_model == "ftheta":
        raise ValueError("ftheta camera is only supported via UT, please set with_ut=True in the rasterization()")

    means_c, covars_c = _world_to_cam(proj_input.means, proj_input.covars, camera.viewmats)
    if not os.getenv('DEBUG_NAN'):
        means_c[means_c == 0] = 1e-6  # 0.0 in means_c would raise nan in covars2d and det

    if camera.camera_model == "ortho":
        means2d, covars2d = _ortho_proj(means_c, covars_c, camera.ks, config.width, config.height)
    elif camera.camera_model == "fisheye":
        means2d, covars2d = _fisheye_proj(means_c, covars_c, camera.ks, config.width, config.height)
    elif camera.camera_model == "pinhole":
        means2d, covars2d = _persp_proj(means_c, covars_c, camera.ks, config.width, config.height)
    else:
        assert_never(camera.camera_model)

    det_orig = (covars2d[..., 0, 0] * covars2d[..., 1, 1] - covars2d[..., 0, 1] * covars2d[..., 1, 0])
    covars2d = covars2d + torch.eye(2, device=proj_input.means.device, dtype=proj_input.means.dtype) * config.eps2d
    det = (covars2d[..., 0, 0] * covars2d[..., 1, 1] - covars2d[..., 0, 1] * covars2d[..., 1, 0])
    det = det.clamp(min=1e-10)

    if config.calc_compensations:
        compensations = torch.sqrt(torch.clamp(det_orig / det, min=0.0))
    else:
        compensations = None

    conics = torch.stack(
        [covars2d[..., 1, 1] / det, -(covars2d[..., 0, 1] + covars2d[..., 1, 0]) / 2.0 / det,
         covars2d[..., 0, 0] / det], dim=-1, )

    depths = means_c[..., 2]  # [..., C, N]

    radius_x = torch.ceil(3.33 * torch.sqrt(covars2d[..., 0, 0]))
    radius_y = torch.ceil(3.33 * torch.sqrt(covars2d[..., 1, 1]))

    radius = torch.stack([radius_x, radius_y], dim=-1)  # [..., C, N, 2]

    valid = (det > 0) & (depths > config.near_plane) & (depths < config.far_plane)
    radius[~valid] = 0.0

    inside = ((means2d[..., 0] + radius[..., 0] > 0) & (means2d[..., 0] - radius[..., 0] < config.width) & (
            means2d[..., 1] + radius[..., 1] > 0) & (means2d[..., 1] - radius[..., 1] < config.height))
    radius[~inside] = 0.0

    radii = radius.int()
    return ProjectionOutput(radii=radii, means2d=means2d, depths=depths, conics=conics,
                            compensations=compensations, covars2d=covars2d)


def ascend_rasterization_single_view(
        ascend_render: AscendGaussRender,
        gaussians: GaussianData,
        camera: CameraParams,
        config: RenderConfig,
) -> Tuple[Tensor, Tensor, Dict]:
    """A version of rasterization() that utilizes PyTorch's autograd."""

    # 解包参数
    params = _unpack_rasterization_params(gaussians, camera, config)

    # 参数验证
    _validate_rasterization_params(params)

    # 投影高斯到2D
    projection_result = _project_gaussians_to_2d(params)

    # 构建tile掩码并排序高斯
    sorted_data = _build_and_sort_tiles(ascend_render, params, projection_result)

    # 渲染每个相机
    render_colors, render_alphas = _render_all_cameras(
        ascend_render, params, projection_result, sorted_data
    )

    # 构建元数据
    meta = _build_metadata(ascend_render, params, projection_result)

    return render_colors, render_alphas, meta


def _unpack_rasterization_params(gaussians: GaussianData, camera: CameraParams, config: RenderConfig):
    """解包所有参数到一个字典中便于传递"""
    return {
        'means': gaussians.means,
        'quats': gaussians.quats,
        'scales': gaussians.scales,
        'opacities': gaussians.opacities,
        'colors': gaussians.colors,
        'viewmats': camera.viewmats,
        'ks': camera.ks,
        'camera_model': camera.camera_model,
        'width': config.width,
        'height': config.height,
        'near_plane': config.near_plane,
        'far_plane': config.far_plane,
        'eps2d': config.eps2d,
        'sh_degree': config.sh_degree,
        'tile_size': config.tile_size,
        'backgrounds': config.backgrounds,
        'render_mode': config.render_mode,
        'rasterize_mode': config.rasterize_mode,
        'channel_chunk': config.channel_chunk,
        'batch_per_iter': config.batch_per_iter,
        'packed': config.packed,
        'radius_clip': config.radius_clip,
    }


def _validate_rasterization_params(params: dict):
    """验证所有输入参数的合法性"""
    num_gaussians = params['means'].shape[0]
    num_cameras = params['viewmats'].shape[0]

    _validate_gaussian_shapes(params, num_gaussians)
    _validate_camera_shapes(params, num_cameras)
    _validate_render_settings(params)
    _validate_colors_shape(params, num_gaussians, num_cameras)


def _validate_gaussian_shapes(params: dict, num_gaussians: int):
    """验证高斯数据形状"""
    shapes = {
        'means': (num_gaussians, 3),
        'quats': (num_gaussians, 4),
        'scales': (num_gaussians, 3),
        'opacities': (num_gaussians,),
    }
    for name, expected_shape in shapes.items():
        if params[name].shape != expected_shape:
            raise ValueError(f"Invalid {name} shape: {params[name].shape}")


def _validate_camera_shapes(params: dict, num_cameras: int):
    """验证相机参数形状"""
    if params['viewmats'].shape != (num_cameras, 4, 4):
        raise ValueError(f"Invalid viewmats shape: {params['viewmats'].shape}")
    if params['ks'].shape != (num_cameras, 3, 3):
        raise ValueError(f"Invalid Ks shape: {params['ks'].shape}")


def _validate_render_settings(params: dict):
    """验证渲染设置"""
    valid_modes = ["RGB", "D", "ED", "RGB+D", "RGB+ED"]
    if params['render_mode'] not in valid_modes:
        raise ValueError(f"Invalid render_mode: {params['render_mode']}")
    if params['packed'] is not False:
        raise ValueError("packed must be False")
    if abs(params['radius_clip']) > 1e-12:
        raise ValueError("radius_clip must be 0.0")
    if params['sh_degree'] is not None:
        raise ValueError("sh_degree must be None")


def _validate_colors_shape(params: dict, num_gaussians: int, num_cameras: int):
    """验证颜色数据形状"""
    colors = params['colors']
    shape_valid = (colors.dim() == 2 and colors.shape[0] == num_gaussians) or \
                  (colors.dim() == 3 and colors.shape[:2] == (num_cameras, num_gaussians))
    if not shape_valid:
        raise ValueError(f"Invalid colors shape: {colors.shape}")


def _project_gaussians_to_2d(params: dict):
    """将3D高斯投影到2D"""
    covars, _ = _quat_scale_to_covar_preci(
        params['quats'], params['scales'], True, False, triu=False
    )

    result = _fully_fused_projection(
        ProjectionInput(params['means'], covars),
        CameraParams(params['viewmats'], params['ks'], params['camera_model']),
        ProjectionConfig(
            params['width'], params['height'],
            eps2d=params['eps2d'],
            near_plane=params['near_plane'],
            far_plane=params['far_plane'],
            calc_compensations=(params['rasterize_mode'] == "antialiased")
        ),
    )

    # 处理补偿
    opacities = params['opacities'].repeat(params['viewmats'].shape[0], 1)
    if result.compensations is not None:
        opacities = opacities * result.compensations

    # 扩展颜色
    colors = params['colors']
    if colors.dim() == 2:
        colors = colors.expand(params['viewmats'].shape[0], -1, -1)

    return {
        'radii': result.radii,
        'means2d': result.means2d,
        'depths': result.depths,
        'conics': result.conics,
        'covars2d': result.covars2d,
        'opacities': opacities,
        'colors': colors,
        'num_cameras': params['viewmats'].shape[0],
        'tile_size': params['tile_size'],
    }


def _build_and_sort_tiles(ascend_render: AscendGaussRender, params: dict, proj_result: dict):
    """构建tile掩码并排序高斯"""
    num_cameras = proj_result['num_cameras']

    # 构建所有相机的tile掩码
    all_in_mask = _build_tile_masks(ascend_render, params, proj_result, num_cameras)

    # 排序所有相机的高斯
    sorted_gs_ids, tile_offsets = _sort_gaussians_for_all_cameras(
        ascend_render, all_in_mask, proj_result['depths'], num_cameras
    )

    return {
        'sorted_gs_ids': sorted_gs_ids,
        'tile_offsets': tile_offsets,
        'all_in_mask': all_in_mask,
    }


def _build_tile_masks(ascend_render: AscendGaussRender, params: dict, proj_result: dict, num_cameras: int):
    """为每个相机构建tile掩码"""
    return [
        ascend_render.build_tile_gs_mask(
            proj_result['means2d'][cam, :, 0],
            proj_result['means2d'][cam, :, 1],
            proj_result['radii'][cam].to(torch.int64).max(dim=1)[0],
            params['width'],
            params['height'],
            cov00=proj_result['covars2d'][cam, :, 0, 0],
            cov01=proj_result['covars2d'][cam, :, 0, 1],
            cov11=proj_result['covars2d'][cam, :, 1, 1],
            opacity=proj_result['opacities'][cam],
        )
        for cam in range(num_cameras)
    ]


def _sort_gaussians_for_all_cameras(ascend_render, all_in_mask, depths, num_cameras):
    """为所有相机排序高斯"""
    sorted_gs_ids = []
    tile_offsets = []

    for cam in range(num_cameras):
        sorted_ids, offsets = ascend_render.sort_gs(all_in_mask[cam], depths[cam])
        sorted_gs_ids.append(sorted_ids)
        tile_offsets.append(offsets)

    return sorted_gs_ids, tile_offsets


def _render_all_cameras(ascend_render: AscendGaussRender, params: dict,
                        proj_result: dict, sorted_data: dict):
    """渲染所有相机"""
    render_colors = []
    render_alphas = []

    for cam in range(proj_result['num_cameras']):
        render_color, render_alpha = _render_single_camera(
            ascend_render, params, proj_result, sorted_data, cam
        )
        render_colors.append(render_color)
        render_alphas.append(render_alpha)

    return torch.stack(render_colors), torch.stack(render_alphas)


def _render_single_camera(ascend_render: AscendGaussRender, params: dict,
                          proj_result: dict, sorted_data: dict, cam: int):
    """渲染单个相机"""
    sorted_ids = sorted_data['sorted_gs_ids'][cam]
    tile_offsets = sorted_data['tile_offsets'][cam]

    # 准备渲染数据
    render_data = _prepare_render_data_for_camera(params, proj_result, cam, sorted_ids)

    # 获取调度信息
    lb_sched = _get_render_schedule(tile_offsets, render_data['means2d'].device)

    # 执行渲染
    rendered = _execute_rendering(ascend_render, render_data, lb_sched)

    # 处理渲染结果
    return _post_process_render_result(ascend_render, rendered, params, cam)


def _prepare_render_data_for_camera(params: dict, proj_result: dict, cam: int, sorted_ids: Tensor):
    """为单个相机准备渲染数据"""
    means2d = proj_result['means2d'][cam]
    conics = proj_result['conics'][cam]
    opacities = proj_result['opacities'][cam]
    colors = proj_result['colors'][cam]
    depths = proj_result['depths'][cam]

    return {
        'means2d': _index_and_reshape(means2d, sorted_ids, 2),
        'conics0': _index_conic(conics, sorted_ids, 0),
        'conics1': _index_conic(conics, sorted_ids, 1),
        'conics2': _index_conic(conics, sorted_ids, 2),
        'opacity': torch.index_select(opacities, 0, sorted_ids).contiguous(),
        'colors3': _index_and_reshape(colors[:, 0:3], sorted_ids, 2),
        'flows3': _get_flows_if_present(colors, sorted_ids),
        'depths': torch.index_select(depths, 0, sorted_ids)[None, :].contiguous(),
    }


def _index_and_reshape(data: Tensor, indices: Tensor, dim: int) -> Tensor:
    """索引并重塑数据"""
    return torch.index_select(data, 0, indices).transpose(0, 1).contiguous()[:dim]


def _index_conic(conics: Tensor, indices: Tensor, idx: int) -> Tensor:
    """索引conic数据"""
    return torch.index_select(conics[:, idx], 0, indices).contiguous()


def _get_flows_if_present(colors: Tensor, sorted_ids: Tensor) -> Optional[Tensor]:
    """如果有flow数据则返回"""
    if colors.shape[1] == 6:
        return torch.index_select(colors[:, 3:6], 0, sorted_ids).transpose(0, 1).contiguous()
    elif colors.shape[1] != 3:
        raise ValueError(f"Invalid color channel count: {colors.shape[1]}")
    return None


def _get_render_schedule(tile_offsets: Tensor, device: torch.device) -> Tensor:
    """获取渲染调度计划"""
    nums = torch.cat([tile_offsets[:1], tile_offsets[1:] - tile_offsets[:-1]])
    schedule = get_render_schedule(nums.cpu(), get_num_vector_core())
    return torch.tensor(schedule, dtype=torch.int64, device=device)


def _execute_rendering(ascend_render: AscendGaussRender, render_data: dict, lb_sched: Tensor) -> dict:
    """执行实际的渲染计算"""
    pix_coords = _get_pixel_coordinates(ascend_render)

    # 渲染颜色
    colors, depths, _ = CalcRender.apply(
        render_data['means2d'],
        render_data['conics0'],
        render_data['conics1'],
        render_data['conics2'],
        render_data['opacity'],
        render_data['colors3'],
        render_data['depths'],
        pix_coords,
        lb_sched,
    )

    # 渲染alpha
    alpha_colors = torch.ones_like(render_data['colors3'])
    alphas, _, _ = CalcRender.apply(
        render_data['means2d'],
        render_data['conics0'],
        render_data['conics1'],
        render_data['conics2'],
        render_data['opacity'],
        alpha_colors,
        render_data['depths'],
        pix_coords,
        lb_sched,
    )
    alphas = alphas[0:1]

    # 渲染flows（如果存在）
    flows = None
    if render_data['flows3'] is not None:
        flows, _, _ = CalcRender.apply(
            render_data['means2d'],
            render_data['conics0'],
            render_data['conics1'],
            render_data['conics2'],
            render_data['opacity'],
            render_data['flows3'],
            render_data['depths'],
            pix_coords,
            lb_sched,
        )

    return {
        'colors': colors,
        'depths': depths,
        'alphas': alphas,
        'flows': flows,
    }


def _get_pixel_coordinates(ascend_render: AscendGaussRender) -> Tensor:
    """获取像素坐标"""
    tile_rows = ascend_render.padded_height // ascend_render.tile_size
    tile_cols = ascend_render.padded_width // ascend_render.tile_size

    return (ascend_render.pix_coord
            .reshape(tile_rows, ascend_render.tile_size, tile_cols, ascend_render.tile_size, 2)
            .permute(0, 2, 1, 3, 4)
            .reshape(tile_rows * tile_cols, ascend_render.tile_size * ascend_render.tile_size, 2)
            .permute(0, 2, 1)
            .to(torch.float32)
            .contiguous())


def _post_process_render_result(ascend_render: AscendGaussRender, rendered: dict,
                                params: dict, cam: int) -> Tuple[Tensor, Tensor]:
    """后处理渲染结果"""
    # 转换tile格式为图像
    colors = ascend_render.tile2image(rendered['colors'].permute(1, 2, 0), tile_size=params['tile_size'])
    alphas = ascend_render.tile2image(rendered['alphas'].permute(1, 2, 0), tile_size=params['tile_size'])
    depths = ascend_render.tile2image(rendered['depths'].permute(1, 2, 0), tile_size=params['tile_size'])

    # 组合输出
    if rendered['flows'] is not None:
        flows = ascend_render.tile2image(rendered['flows'].permute(1, 2, 0), tile_size=params['tile_size'])
        colors = torch.cat([colors, flows, depths], dim=0)
    else:
        dummy_flows = colors.detach()
        colors = torch.cat([colors, dummy_flows, depths], dim=0)

    return colors.permute(1, 2, 0), alphas.permute(1, 2, 0)


def _build_metadata(ascend_render: AscendGaussRender, params: dict, proj_result: dict) -> Dict:
    """构建元数据"""
    return {
        "camera_ids": None,
        "gaussian_ids": None,
        "radii": proj_result['radii'],
        "means2d": proj_result['means2d'],
        "depths": proj_result['depths'],
        "conics": proj_result['conics'],
        "opacities": proj_result['opacities'],
        "tile_width": ascend_render.padded_width // params['tile_size'],
        "tile_height": ascend_render.padded_height // params['tile_size'],
        "width": params['width'],
        "height": params['height'],
        "tile_size": params['tile_size'],
        "n_cameras": proj_result['num_cameras'],
    }


def _validate_batched_tensors(means, quats, scales, opacities, colors, viewmats, ks):
    if not (len(quats.shape) == 3 and len(scales.shape) == 3 and len(colors.shape) == 3):
        raise ValueError("quats, scales and colors must be 3D tensors for batched input")
    if len(opacities.shape) != 2:
        raise ValueError("opacities must be a 2D tensor for batched input")
    if not (len(viewmats.shape) == 4 and len(ks.shape) == 4):
        raise ValueError("viewmats and Ks must be 4D tensors for batched input")
    batch_size = means.shape[0]
    same_gaussian_batch = (
        batch_size == quats.shape[0]
        and batch_size == scales.shape[0]
        and batch_size == colors.shape[0]
        and batch_size == opacities.shape[0]
    )
    if not (same_gaussian_batch and batch_size == viewmats.shape[0] and batch_size == ks.shape[0]):
        raise ValueError("Batched input tensors must have the same batch size")


def _validate_unbatched_tensors(means, quats, scales, opacities, colors, viewmats, ks):
    if not (len(means.shape) == 2 and len(quats.shape) == 2 and len(scales.shape) == 2 and len(colors.shape) == 2):
        raise ValueError("means, quats, scales and colors must be 2D tensors")
    if len(opacities.shape) != 1:
        raise ValueError("opacities must be a 1D tensor")
    if not (len(viewmats.shape) == 3 and len(ks.shape) == 3):
        raise ValueError("viewmats and Ks must be 3D tensors")
    if not (means.shape[0] == quats.shape[0] == scales.shape[0] == colors.shape[0] == opacities.shape[0]):
        raise ValueError("Input tensors must have the same number of gaussians")
    if viewmats.shape[0] != ks.shape[0]:
        raise ValueError("viewmats and Ks must have the same number of cameras")


def _validate_batched_inputs(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    viewmats: Tensor,
    ks: Tensor,
) -> bool:
    is_batched = len(means.shape) == 3
    if is_batched:
        _validate_batched_tensors(means, quats, scales, opacities, colors, viewmats, ks)
    else:
        _validate_unbatched_tensors(means, quats, scales, opacities, colors, viewmats, ks)
    return is_batched


def ascend_rasterization(
        ascend_render: AscendGaussRender,
        gaussians: GaussianData,
        camera: CameraParams,
        config: RenderConfig,
) -> Tuple[Tensor, Tensor, Dict]:
    """A version of rasterization() that utilies on PyTorch's autograd.

    .. note::
        This function still relies on gsplat's CUDA backend for some computation, but the
        entire differentiable graph is on of PyTorch (and nerfacc) so could use Pytorch's
        autograd for backpropagation.

    .. note::
        This function relies on installing latest nerfacc, via:
        pip install git+https://github.com/nerfstudio-project/nerfacc

    .. note::
        Compared to rasterization(), this function does not support some arguments such as
        `packed`, `sparse_grad` and `absgrad`.
    """

    is_batched = _validate_batched_inputs(
        gaussians.means, gaussians.quats, gaussians.scales,
        gaussians.opacities, gaussians.colors,
        camera.viewmats, camera.ks,
    )

    if is_batched:
        num_views = gaussians.means.shape[0]
        render_colors = []
        render_alphas = []
        for view_idx in range(num_views):
            col, alp, _ = ascend_rasterization_single_view(
                ascend_render,
                GaussianData(
                    gaussians.means[view_idx],
                    gaussians.quats[view_idx],
                    gaussians.scales[view_idx],
                    gaussians.opacities[view_idx],
                    gaussians.colors[view_idx],
                ),
                CameraParams(camera.viewmats[view_idx], camera.ks[view_idx], camera.camera_model),
                config,
            )
            render_colors.append(col)
            render_alphas.append(alp)
        render_colors = torch.stack(render_colors)
        render_alphas = torch.stack(render_alphas)
        return render_colors, render_alphas, None
    else:
        return ascend_rasterization_single_view(ascend_render, gaussians, camera, config)
