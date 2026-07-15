# coding=utf-8
# Adapted from
# https://github.com/NVlabs/GaussianSTORM/blob/main/storm/visualization/video_maker.py

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) NVIDIA. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import time
from dataclasses import dataclass

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from matplotlib import cm

from src.dataset.constants import MEAN, STD
from src.dataset.data_utils import (
    prepare_inputs_and_targets,
    prepare_inputs_and_targets_novel_view,
    to_batch_tensor,
)

from .annotation import add_label
from .layout import add_border, hcat, prep_image, vcat
from .visualization_tools import depth_visualizer, scene_flow_to_rgb, feat_visualizer, get_global_pac_linear

logger = logging.getLogger("PerceptualModel")


@dataclass
class FrameInfo:
    """Frame index information."""
    t: int
    current_frame_idx: int


@dataclass
class TargetSize:
    """Target spatial dimensions."""
    h_tgt: int
    w_tgt: int


@dataclass
class AV2ResizeInfo:
    """AV2 resize parameters."""
    resize: object
    w: int
    h: int


@dataclass
class ClusterContext:
    """Cluster image and related context for visualization."""
    cluster_image: object
    alpha_image: object = None


@dataclass
class SkipFlags:
    """Boolean skip flags for visualization options."""
    skip_depth: bool = False
    skip_plot_gt_depth_and_flow: bool = False
    skip_gt_feat: bool = False
    skip_gt_rgb: bool = False


@dataclass
class VisOptions:
    """Visualization options combining skip flags and font size."""
    skip_flags: SkipFlags = None
    font_size: int = 24
    tgt_size: TargetSize = None
    av2_resize: object = None


@dataclass
class FrameRenderData:
    """Data needed for rendering a single video frame."""
    target_dict: dict = None
    input_dict: dict = None
    pred_images: object = None
    target_images: object = None
    render_results: dict = None
    target_v: int = 0
    n_ctx_per_row: int = 1
    vis_context_item: list = None
    font_size: int = 24
    context_frames: object = None


@dataclass
class AV2FrameRenderData:
    """Data needed for rendering a single AV2 video frame."""
    target_dict: dict = None
    pred_images: object = None
    target_images: object = None
    render_results: dict = None
    target_v: int = 0
    n_ctx_per_row: int = 1
    context_frames: object = None
    input_dict: dict = None


@dataclass
class PredContextLists:
    """Prediction context lists for video frames."""
    image_list: list = None
    depth_list: list = None
    alpha_list: list = None
    flow_list: list = None


@dataclass
class FeatureVizParams:
    """Parameters for feature visualization."""
    c_idx: int = 0
    c_idx_list: list = None
    pca_linear: object = None
    font_size: int = 24
    skip_flags: SkipFlags = None


@dataclass
class FrameVizConfig:
    """Visualization configuration for a single frame."""
    cluster_context: ClusterContext = None
    feat_params: FeatureVizParams = None
    skip_flags: SkipFlags = None
    tgt_size: TargetSize = None
    vis_options: VisOptions = None


@dataclass
class VideoInputs:
    """Data and model inputs for video generation."""
    dataset: object = None
    model: object = None
    device: object = None
    scene_id: object = None
    data_dict: dict = None
    input_dict: dict = None
    target_dict: dict = None
    pred_dict: dict = None
    feat_extractor: object = None


@dataclass
class GridLayout:
    """Grid layout parameters for labeled grid."""
    context_t: int = 0
    context_v: int = 0
    n_ctx_per_row: int = 1


@dataclass
class RgbFrameBuildOptions:
    """Parameters for building RGB frames in video visualization."""
    t: int = 0
    target_v: int = 0
    current_frame_idx: int = 0
    font_size: int = 24
    rotate_pred_images: object = None
    skip_gt_rgb: bool = False


@dataclass
class VideoRenderFramesConfig:
    """Configuration for _make_video_render_frames."""
    context_t: int = 0
    target_t: int = 0
    target_v: int = 0
    n_ctx_per_row: int = 1
    vis_context_item: list = None
    font_size: int = 24
    cluster_image: object = None
    tgt_size: TargetSize = None
    pca_linear: object = None
    skip_plot_gt_depth_and_flow: bool = False


@dataclass
class CleanVideoRenderFramesConfig:
    """Configuration for _make_clean_video_render_frames."""
    target_t: int = 0
    target_v: int = 0
    pred_images: object = None
    target_images: object = None
    rotate_pred_images: object = None
    render_results: dict = None
    n_ctx_per_row: int = 1
    font_size: int = 24
    cluster_image: object = None
    tgt_size: TargetSize = None
    skip_flags: SkipFlags = None
    c_idx: int = 0
    c_idx_list: list = None
    pca_linear: object = None


def get_pca_map(x):
    x_shape = x.shape
    x = x.view(-1, x.shape[-1])
    x = x @ torch.pca_lowrank(x, q=3, niter=20)[2]
    x = (x - x.min(dim=0)[0]) / (x.max(dim=0)[0] - x.min(dim=0)[0])
    return x.view(*x_shape[:-1], 3)


def _prepare_motion_segmentation_image(render_results, pred_dict, target_t, target_v, tgt_size: TargetSize):
    """Process motion segmentation and generate cluster image.
    
    Args:
        render_results: Render results dictionary
        pred_dict: Prediction dictionary
        target_t: Number of target frames
        target_v: Number of target views
        h_tgt: Target height
        w_tgt: Target width
    
    Returns:
        cluster_image: Processed cluster image or None
    """
    if "rendered_motion_seg" not in render_results:
        return None
    
    max_idx = render_results["rendered_motion_seg"][0]
    unique_clusters = torch.unique(max_idx)
    
    try:
        velocities = pred_dict["gs_params"]["motion_bases"][0][unique_clusters]
    except (KeyError, IndexError, TypeError):
        velocities = pred_dict["gs_params"]["motion_bases"][0].mean(dim=0)[unique_clusters]
    
    velocity_norm = torch.norm(velocities, dim=-1)
    sorted_indices = torch.argsort(velocity_norm)
    sorted_clusters = unique_clusters[sorted_indices]
    
    num_unique_clusters = len(unique_clusters)
    cmap = cm.get_cmap("rainbow", num_unique_clusters)
    
    cluster_to_color_map = torch.tensor([cmap(i) for i in range(num_unique_clusters)])[
        :, :3
    ].to(max_idx.device)
    
    cluster_mapping = torch.zeros_like(max_idx)
    for new_cluster_idx, original_cluster in enumerate(sorted_clusters):
        cluster_mapping[max_idx == original_cluster] = new_cluster_idx
    
    cluster_image = cluster_to_color_map[cluster_mapping]
    if cluster_image.shape[-3] != tgt_size.h_tgt or cluster_image.shape[-2] != tgt_size.w_tgt:
        cluster_image = F.interpolate(
            rearrange(cluster_image, "t v h w c -> (t v) c h w"),
            size=(tgt_size.h_tgt, tgt_size.w_tgt),
            mode="nearest",
        )
        cluster_image = rearrange(
            cluster_image, "(t v) c h w -> t v h w c", t=target_t, v=target_v
        )
    
    return cluster_image


def _process_depth_decoder_frame(render_results, t, current_frame_idx, font_size):
    depth_image = render_results[render_results["decoder_depth_key"]][0][t]
    depth_image = depth_image.detach().cpu().numpy()
    depth_image = depth_visualizer(depth_image, None)
    depth_image = torch.from_numpy(depth_image)
    depth_image = rearrange(depth_image, "v h w c -> v c h w")
    return add_label(
        hcat(*depth_image),
        f"Predicted Decoder Depth (t={current_frame_idx})",
        font_size=font_size,
        align="center",
    )


def _process_depth_gs_frame(render_results, t, current_frame_idx, tgt_size, font_size):
    depth_image = render_results[render_results["depth_key"]][0][t]
    alpha_image = render_results[render_results["alpha_key"]][0][t]
    if depth_image.shape[-2] != tgt_size.h_tgt or depth_image.shape[-1] != tgt_size.w_tgt:
        depth_image = F.interpolate(
            depth_image.unsqueeze(-3),
            size=(tgt_size.h_tgt, tgt_size.w_tgt),
            mode="bilinear",
            align_corners=False,
        ).squeeze(-3)
        alpha_image = F.interpolate(
            alpha_image.unsqueeze(-3),
            size=(tgt_size.h_tgt, tgt_size.w_tgt),
            mode="bilinear",
            align_corners=False,
        ).squeeze(-3)
    depth_image = depth_image.detach().cpu().numpy()
    alpha_image = alpha_image.detach().cpu().numpy()
    depth_image = depth_visualizer(depth_image, alpha_image)
    depth_image = torch.from_numpy(depth_image)
    depth_image = rearrange(depth_image, "v h w c -> v c h w")
    return add_label(
        hcat(*depth_image),
        f"Predicted Depth (t={current_frame_idx})",
        font_size=font_size,
        align="center",
    )


def _process_depth_visualization(
    render_results, frame_info: FrameInfo, tgt_size: TargetSize,
    skip_flags: SkipFlags, font_size
):
    """Process depth visualization for a single frame.

    Args:
        render_results: Render results dictionary
        frame_info: Frame index information (t, current_frame_idx)
        tgt_size: Target spatial dimensions (h_tgt, w_tgt)
        skip_flags: Boolean skip flags
        font_size: Font size for labels

    Returns:
        List of depth frame items
    """
    t = frame_info.t
    current_frame_idx = frame_info.current_frame_idx
    skip_depth = skip_flags.skip_depth
    frame_list = []
    if not skip_depth and render_results["decoder_depth_key"] is not None:
        frame_list.append(
            _process_depth_decoder_frame(render_results, t, current_frame_idx, font_size))
    if not skip_depth and render_results["depth_key"] is not None:
        frame_list.append(
            _process_depth_gs_frame(render_results, t, current_frame_idx, tgt_size, font_size))
    return frame_list


def _process_gt_depth_frame(
    target_dict, frame_info: FrameInfo, target_v, tgt_size: TargetSize,
    vis_options: VisOptions
):
    """Process GT depth frame visualization.
    
    Args:
        target_dict: Target dictionary
        frame_info: Frame index information (t, current_frame_idx)
        target_v: Number of target views
        tgt_size: Target spatial dimensions (h_tgt, w_tgt)
        skip_flags: Boolean skip flags
        font_size: Font size for labels
    
    Returns:
        GT depth frame item or None
    """
    t = frame_info.t
    current_frame_idx = frame_info.current_frame_idx
    skip_flags = vis_options.skip_flags
    font_size = vis_options.font_size
    skip_depth = skip_flags.skip_depth
    skip_plot_gt_depth_and_flow = skip_flags.skip_plot_gt_depth_and_flow
    if not skip_depth and "target_depth" in target_dict.keys():
        gt_depth = target_dict["target_depth"][0][t]
        gt_depth = gt_depth.detach().cpu().numpy()
        gt_depth = depth_visualizer(gt_depth, gt_depth > 0)
        gt_depth = torch.from_numpy(gt_depth)
        gt_depth = rearrange(gt_depth, "v h w c -> v c h w")
        gt_depth = add_label(
            hcat(*gt_depth),
            f"Target GT Depth (t={current_frame_idx})",
            font_size=font_size,
            align="center",
        )
        return gt_depth
    elif not skip_plot_gt_depth_and_flow and not skip_depth:
        gt_depth = torch.full((target_v, 3, tgt_size.h_tgt, tgt_size.w_tgt), 0.5)
        gt_depth = add_label(
            hcat(*gt_depth),
            f"Target GT Depth (t={current_frame_idx})",
            font_size=font_size,
            align="center",
        )
        return gt_depth
    
    return None


def _build_pred_feat_frame(render_results, t_idx, current_frame_idx, pca_linear, font_size):
    feat_image = render_results["rendered_feat"][0][t_idx]
    feat_image = rearrange(feat_image, "v h w c -> v c h w")
    feat_image = feat_visualizer(feat_image, pca_linear)
    return add_label(
        hcat(*feat_image),
        f"Predicted Feat (t={current_frame_idx})",
        font_size=font_size,
        align="center",
    )


def _build_gt_feat_frame(gt_feat, t_idx, current_frame_idx, pca_linear, font_size):
    gt_feat = gt_feat[0][t_idx]
    gt_feat = feat_visualizer(gt_feat, pca_linear)
    return add_label(
        hcat(*gt_feat),
        f"Target GT Feat (t={current_frame_idx})",
        font_size=font_size,
        align="center",
    )


def _resolve_gt_feat(input_dict, target_dict):
    if os.getenv("CONTEXT_FEAT"):
        if "context_feat" not in input_dict.keys():
            raise KeyError("context_feat must be in input_dict when CONTEXT_FEAT is set")
        return input_dict["context_feat"]
    if "target_feat" not in target_dict.keys():
        raise KeyError("target_feat must be in target_dict")
    return target_dict["target_feat"]


def _process_feature_visualization(
    render_results, input_dict, target_dict, frame_info: FrameInfo,
    feat_params: FeatureVizParams
):
    """Process feature visualization for a single frame.

    Args:
        render_results: Render results dictionary
        input_dict: Input dictionary
        target_dict: Target dictionary
        frame_info: Frame index information (t, current_frame_idx)
        feat_params: Feature visualization parameters (c_idx, c_idx_list, pca_linear)
        skip_flags: Boolean skip flags
        font_size: Font size for labels

    Returns:
        Tuple of (feature frame list, updated c_idx)
    """
    t = frame_info.t
    current_frame_idx = frame_info.current_frame_idx
    c_idx = feat_params.c_idx
    c_idx_list = feat_params.c_idx_list
    pca_linear = feat_params.pca_linear
    skip_flags = feat_params.skip_flags
    font_size = feat_params.font_size
    skip_gt_feat = skip_flags.skip_gt_feat if skip_flags else False
    frame_list = []
    if "rendered_feat" not in render_results:
        return frame_list, c_idx
    gt_feat = _resolve_gt_feat(input_dict, target_dict)
    if os.getenv("CONTEXT_FEAT") and c_idx < len(c_idx_list) - 1 and t == c_idx_list[c_idx + 1]:
        c_idx += 1
    t_idx = c_idx if os.getenv("CONTEXT_FEAT") else t
    frame_list.append(
        _build_pred_feat_frame(render_results, t_idx, current_frame_idx, pca_linear, font_size))
    if not skip_gt_feat:
        frame_list.append(
            _build_gt_feat_frame(gt_feat, t_idx, current_frame_idx, pca_linear, font_size))
    return frame_list, c_idx


def _make_video_depth_decoder_frame(render_results, t, current_frame_idx, font_size):
    depth_image = render_results[render_results["decoder_depth_key"]][0][t]
    depth_image = depth_image.detach().cpu().numpy()
    depth_image = depth_visualizer(depth_image, None)
    depth_image = torch.from_numpy(depth_image)
    depth_image = rearrange(depth_image, "v h w c -> v c h w")
    return add_label(
        hcat(*depth_image),
        f"Predicted Decoder Depth (t={current_frame_idx})",
        font_size=font_size,
        align="center",
    )


def _make_video_depth_gs_frame(render_results, t, current_frame_idx, tgt_size, font_size):
    depth_image = render_results[render_results["depth_key"]][0][t]
    alpha_image = render_results[render_results["alpha_key"]][0][t]
    if depth_image.shape[-2] != tgt_size.h_tgt or depth_image.shape[-1] != tgt_size.w_tgt:
        depth_image = F.interpolate(
            depth_image.unsqueeze(-3),
            size=(tgt_size.h_tgt, tgt_size.w_tgt),
            mode="bilinear",
            align_corners=False,
        ).squeeze(-3)
        alpha_image = F.interpolate(
            alpha_image.unsqueeze(-3),
            size=(tgt_size.h_tgt, tgt_size.w_tgt),
            mode="bilinear",
            align_corners=False,
        ).squeeze(-3)
    depth_image = depth_image.detach().cpu().numpy()
    alpha_image = alpha_image.detach().cpu().numpy()
    depth_image = depth_visualizer(depth_image, alpha_image)
    depth_image = torch.from_numpy(depth_image)
    depth_image = rearrange(depth_image, "v h w c -> v c h w")
    pred_depth = add_label(
        hcat(*depth_image),
        f"Predicted Depth (t={current_frame_idx})",
        font_size=font_size,
        align="center",
    )
    return pred_depth, alpha_image


def _make_video_process_depth_visualization(render_results, frame_info: FrameInfo, tgt_size: TargetSize, font_size):
    """Process depth visualization for make_video function.

    Args:
        render_results: Render results dictionary
        frame_info: Frame index information (t, current_frame_idx)
        tgt_size: Target spatial dimensions (h_tgt, w_tgt)
        font_size: Font size for labels

    Returns:
        Tuple of (depth frame list, alpha_image)
    """
    t = frame_info.t
    current_frame_idx = frame_info.current_frame_idx
    frame_list = []
    alpha_image = None
    if render_results["decoder_depth_key"] is not None:
        frame_list.append(
            _make_video_depth_decoder_frame(render_results, t, current_frame_idx, font_size))
    if render_results["depth_key"] is not None:
        pred_depth, alpha_image = _make_video_depth_gs_frame(
            render_results, t, current_frame_idx, tgt_size, font_size)
        frame_list.append(pred_depth)
    return frame_list, alpha_image


def _make_video_process_gt_depth(
    target_dict, frame_info: FrameInfo, target_v, tgt_size: TargetSize,
    vis_options: VisOptions
):
    """Process GT depth for make_video function.
    
    Args:
        target_dict: Target dictionary
        frame_info: Frame index information (t, current_frame_idx)
        target_v: Number of target views
        tgt_size: Target spatial dimensions (h_tgt, w_tgt)
        skip_flags: Boolean skip flags
        font_size: Font size for labels
    
    Returns:
        GT depth frame item or None
    """
    t = frame_info.t
    current_frame_idx = frame_info.current_frame_idx
    skip_flags = vis_options.skip_flags
    font_size = vis_options.font_size
    skip_plot_gt_depth_and_flow = skip_flags.skip_plot_gt_depth_and_flow
    if "target_depth" in target_dict.keys():
        gt_depth = target_dict["target_depth"][0][t]
        gt_depth = gt_depth.detach().cpu().numpy()
        gt_depth = depth_visualizer(gt_depth, gt_depth > 0)
        gt_depth = torch.from_numpy(gt_depth)
        gt_depth = rearrange(gt_depth, "v h w c -> v c h w")
        gt_depth = add_label(
            hcat(*gt_depth),
            f"Target GT Depth (t={current_frame_idx})",
            font_size=font_size,
            align="center",
        )
        return gt_depth
    elif not skip_plot_gt_depth_and_flow:
        gt_depth = torch.full((target_v, 3, tgt_size.h_tgt, tgt_size.w_tgt), 0.5)
        gt_depth = add_label(
            hcat(*gt_depth),
            f"Target GT Depth (t={current_frame_idx})",
            font_size=font_size,
            align="center",
        )
        return gt_depth
    
    return None


def _make_video_process_feature_visualization(
    render_results, input_dict, target_dict, frame_info: FrameInfo,
    feat_params: FeatureVizParams
):
    """Process feature visualization for make_video function.

    Args:
        render_results: Render results dictionary
        input_dict: Input dictionary
        target_dict: Target dictionary
        frame_info: Frame index information (t, current_frame_idx)
        feat_params: Feature visualization parameters (c_idx, c_idx_list, pca_linear)
        font_size: Font size for labels

    Returns:
        Tuple of (feature frame list, updated c_idx)
    """
    t = frame_info.t
    current_frame_idx = frame_info.current_frame_idx
    c_idx = feat_params.c_idx
    c_idx_list = feat_params.c_idx_list
    pca_linear = feat_params.pca_linear
    font_size = feat_params.font_size
    frame_list = []
    if "rendered_feat" not in render_results:
        return frame_list, c_idx
    gt_feat = _resolve_gt_feat(input_dict, target_dict)
    if os.getenv("CONTEXT_FEAT") and c_idx < len(c_idx_list) - 1 and t == c_idx_list[c_idx + 1]:
        c_idx += 1
    t_idx = c_idx if os.getenv("CONTEXT_FEAT") else t
    frame_list.append(
        _build_pred_feat_frame(render_results, t_idx, current_frame_idx, pca_linear, font_size))
    frame_list.append(
        _build_gt_feat_frame(gt_feat, t_idx, current_frame_idx, pca_linear, font_size))
    return frame_list, c_idx


def _make_video_prepare_pred_flow(render_results, frame_info: FrameInfo, tgt_size: TargetSize, font_size):
    t = frame_info.t
    current_frame_idx = frame_info.current_frame_idx
    h_tgt = tgt_size.h_tgt
    w_tgt = tgt_size.w_tgt
    flow_image = render_results[render_results["flow_key"]][0][t]
    flow_image = scene_flow_to_rgb(flow_image, flow_max_radius=15)
    flow_image = rearrange(flow_image, "v h w c -> v c h w")
    if flow_image.shape[-2] != h_tgt or flow_image.shape[-1] != w_tgt:
        flow_image = F.interpolate(
            flow_image,
            size=(h_tgt, w_tgt),
            mode="bilinear",
            align_corners=False,
        )
    flow_image = add_label(
        hcat(*flow_image),
        f"Predicted Flow (t={current_frame_idx})",
        font_size=font_size,
        align="center",
    )
    return flow_image


def _make_video_prepare_gt_flow(target_dict, t, current_frame_idx, font_size):
    gt_flow = target_dict["target_flow"][0][t]
    gt_flow = scene_flow_to_rgb(gt_flow, flow_max_radius=15)
    gt_flow = rearrange(gt_flow, "v h w c -> v c h w")
    gt_flow = add_label(
        hcat(*gt_flow),
        f"Target GT Flow (t={current_frame_idx})",
        font_size=font_size,
        align="center",
    )
    return gt_flow


def _make_video_prepare_opacity(alpha_image, current_frame_idx, font_size):
    alpha_image = torch.from_numpy(alpha_image).unsqueeze(1)
    alpha_image = alpha_image.repeat(1, 3, 1, 1)
    alpha_image = add_label(
        hcat(*alpha_image),
        f"Predicted Opacity (t={current_frame_idx})",
        font_size=font_size,
        align="center",
    )
    return alpha_image


def _make_video_prepare_sky_mask(target_dict, t, current_frame_idx, font_size):
    sky_mask = target_dict["target_sky_masks"][0][t].unsqueeze(1)
    sky_mask = sky_mask.repeat(1, 3, 1, 1)
    sky_mask = add_label(
        hcat(*sky_mask),
        f"GT Sky Mask (t={current_frame_idx})",
        font_size=font_size,
        align="center",
    )
    return sky_mask


def _make_video_prepare_cluster(cluster_image, t, current_frame_idx, font_size):
    cluster_image_t = cluster_image[t]
    cluster_image_t = rearrange(cluster_image_t, "v h w c -> v c h w")
    cluster_image_t = add_label(
        hcat(*cluster_image_t),
        f"Motion Segmentation (t={current_frame_idx})",
        font_size=font_size,
        align="center",
    )
    return cluster_image_t


def _make_video_process_flow_and_related(
    render_results, target_dict, frame_info: FrameInfo,
    cluster_context: ClusterContext, vis_options: VisOptions
):
    """Process flow and related visualizations for make_video function.
    
    Args:
        render_results: Render results dictionary
        target_dict: Target dictionary
        frame_info: Frame index information (t, current_frame_idx)
        alpha_image: Alpha image from depth processing
        cluster_image: Cluster image from motion segmentation
        tgt_size: Target spatial dimensions (h_tgt, w_tgt)
        font_size: Font size for labels
    
    Returns:
        List of flow and related frame items
    """
    t = frame_info.t
    current_frame_idx = frame_info.current_frame_idx
    alpha_image = cluster_context.alpha_image
    cluster_image = cluster_context.cluster_image
    font_size = vis_options.font_size
    tgt_size = vis_options.tgt_size
    frame_list = []
    
    if render_results["flow_key"] is None:
        return frame_list
    
    flow_image = _make_video_prepare_pred_flow(
        render_results, frame_info, tgt_size, font_size
    )
    frame_list.append(flow_image)
    
    if "target_flow" in target_dict.keys():
        gt_flow = _make_video_prepare_gt_flow(target_dict, t, current_frame_idx, font_size)
        frame_list.append(gt_flow)
    
    if render_results["depth_key"] is not None and alpha_image is not None:
        alpha_image = _make_video_prepare_opacity(alpha_image, current_frame_idx, font_size)
        frame_list.append(alpha_image)
    
    if "target_sky_masks" in target_dict.keys():
        sky_mask = _make_video_prepare_sky_mask(target_dict, t, current_frame_idx, font_size)
        frame_list.append(sky_mask)
    
    if cluster_image is not None:
        cluster_image_t = _make_video_prepare_cluster(cluster_image, t, current_frame_idx, font_size)
        frame_list.append(cluster_image_t)
    
    return frame_list


def _build_labeled_grid(images, grid_layout: GridLayout, input_dict, label_prefix, font_size):
    context_t = grid_layout.context_t
    context_v = grid_layout.context_v
    n_ctx_per_row = grid_layout.n_ctx_per_row
    frames = []
    for t in range(context_t):
        current_frame_idx = int(input_dict["context_frame_idx"][0][t].item())
        row = add_label(
            hcat(*[images[t][v_id] for v_id in range(context_v)]),
            f"{label_prefix} (t={current_frame_idx})",
            font_size=font_size,
            align="center",
        )
        frames.append(row)
    num_rows = max(1, len(frames) // n_ctx_per_row)
    frames = vcat(
        *[
            hcat(
                *frames[row * n_ctx_per_row: (row + 1) * n_ctx_per_row],
                gap=24,
            )
            for row in range(num_rows)
        ]
    )
    return frames


def _make_video_prepare_inputs(video_inputs: VideoInputs, save_video, output_filename):
    dataset = video_inputs.dataset
    model = video_inputs.model
    device = video_inputs.device
    scene_id = video_inputs.scene_id
    data_dict = video_inputs.data_dict
    input_dict = video_inputs.input_dict
    target_dict = video_inputs.target_dict
    pred_dict = video_inputs.pred_dict
    feat_extractor = video_inputs.feat_extractor
    if save_video:
        if output_filename is None:
            raise ValueError("output_filename must be provided when save_video is True")

    if data_dict is None:
        if dataset is None:
            raise ValueError("dataset must be provided when data_dict is None")
        if scene_id is None:
            scene_id = np.random.randint(0, len(dataset))
        data_dict = dataset.__getitem__(scene_id, np.random.randint(10, 100), return_all=True)
        data_dict = to_batch_tensor(data_dict)
    if not isinstance(data_dict['num_max_cams'], int):
        num_max_cams = int(data_dict['num_max_cams'][0])
    else:
        num_max_cams = data_dict['num_max_cams']
    if num_max_cams > 1:
        font_size = 24
    else:
        font_size = 8
    if input_dict is None or target_dict is None:
        if device is None:
            raise ValueError("device must be provided when input_dict or target_dict is None")
        input_dict, target_dict = prepare_inputs_and_targets(data_dict, device, v=num_max_cams,
                                                             feat_extractor=feat_extractor,
                                                             is_vis=True)

    if pred_dict is None:
        if model is None:
            raise ValueError("model must be provided when pred_dict is None")
        if device is None:
            raise ValueError("device must be provided when pred_dict is None")
        model = model.eval()  # model is needed only pred_dict is None
        dtype = torch.float16 if os.environ.get('DISABLE_BFLOAT') else torch.bfloat16
        with torch.autocast(device_type=device.type, dtype=dtype):
            start_time = time.perf_counter()
            pred_dict = model(input_dict)
            end_time = time.perf_counter()
            logger.info(f"Time taken to forward rendered results: {end_time - start_time} seconds")
    return data_dict, input_dict, target_dict, pred_dict, font_size


def _make_video_build_vis_context(input_dict, pred_dict, context_t, context_v, font_size):
    def denormalize(x, already_channel_last=False):
        if not already_channel_last:
            x = rearrange(x, "t v c h w -> t v h w c")
        return rearrange(x, "t v h w c -> t v c h w")

    context_images = input_dict["context_image"][0]
    context_images = denormalize(context_images)

    if context_v == 1:
        n_ctx_per_row = 4  # better visualize for 1 view
    elif context_v <= 3:
        n_ctx_per_row = 2
    else:
        n_ctx_per_row = 1
    grid_layout = GridLayout(context_t=context_t, context_v=context_v, n_ctx_per_row=n_ctx_per_row)
    context_frames = _build_labeled_grid(
        context_images, grid_layout, input_dict, "Context RGB", font_size)
    vis_context_item = [context_frames]

    if "rendered_context_image" in pred_dict.keys():
        rendered_context_images = pred_dict["rendered_context_image"][0].permute(0, 1, 4, 2, 3)
        rendered_context_images = denormalize(rendered_context_images)
        vis_context_item.append(_build_labeled_grid(
            rendered_context_images, grid_layout, input_dict,
            "Rendered Context RGB", font_size))

    if "rendered_context_depth" in pred_dict.keys():
        depth_image = rearrange(pred_dict["rendered_context_depth"][0], 't v h w -> (t v) h w').detach().cpu().numpy()
        alpha_image = None
        depth_image = depth_visualizer(depth_image, alpha_image)
        depth_image = torch.from_numpy(depth_image)
        rendered_context_depths = rearrange(depth_image, "(t v) ... c -> t v c ...", v=context_v)
        vis_context_item.append(_build_labeled_grid(
            rendered_context_depths, grid_layout, input_dict,
            "Rendered Context Depth", font_size))

    if "rendered_context_alpha" in pred_dict.keys():
        rendered_context_alphas = pred_dict["rendered_context_alpha"][0].unsqueeze(2).repeat(1, 1, 3, 1, 1)
        vis_context_item.append(_build_labeled_grid(
            rendered_context_alphas, grid_layout, input_dict,
            "Rendered Context Alpha", font_size))

    return vis_context_item, n_ctx_per_row


@dataclass
class PredContextResult:
    target_images: object
    pred_images: object
    render_results: object
    pred_context_image_list: list
    pred_context_depth_list: list
    pred_context_alpha_list: list
    pred_context_flow_list: list


def _make_video_prepare_images_and_pred_context(input_dict, target_dict, pred_dict, context_t, target_t):
    def denormalize(x, already_channel_last=False):
        if not already_channel_last:
            x = rearrange(x, "t v c h w -> t v h w c")
        return rearrange(x, "t v h w c -> t v c h w")

    target_images = target_dict["target_image"][0]
    target_images = denormalize(target_images)
    render_results = pred_dict["render_results"]
    pred_images = render_results[render_results["rgb_key"]][0]
    pred_images = denormalize(pred_images, already_channel_last=True)

    pred_context_image_list = []
    pred_context_depth_list = []
    pred_context_alpha_list = []
    pred_context_flow_list = []
    for t in range(context_t):
        if f"context_{t}_rendered_image" in render_results.keys():
            pred_context_image_list.append(denormalize(
                render_results[f"context_{t}_rendered_image"][0],
                already_channel_last=True,
            ))
        if f"context_{t}_rendered_depth" in render_results.keys():
            rendered_depth = render_results[f"context_{t}_rendered_depth"][0]
            rendered_depth = rearrange(rendered_depth, "t v ... -> (t v) ...")
            pred_context_depth_list.append(rearrange(
                torch.from_numpy(depth_visualizer(
                    rendered_depth.detach().cpu().numpy(), None,
                )),
                "(t v) h w c -> t v c h w",
                t=target_t,
            ))
        if f"context_{t}_rendered_alpha" in render_results.keys():
            pred_context_alpha_list.append(
                render_results[f"context_{t}_rendered_alpha"][0].unsqueeze(2).repeat(1, 1, 3, 1, 1)
            )
        if f"context_{t}_rendered_flow" in render_results.keys():
            rendered_flow = render_results[f"context_{t}_rendered_flow"][0]
            pred_context_flow_list.append(rearrange(
                scene_flow_to_rgb(rearrange(rendered_flow, "t v ... -> (t v) ..."), flow_max_radius=15),
                "(t v) h w c -> t v c h w",
                t=target_t,
            ))
    return PredContextResult(
        target_images=target_images,
        pred_images=pred_images,
        render_results=render_results,
        pred_context_image_list=pred_context_image_list,
        pred_context_depth_list=pred_context_depth_list,
        pred_context_alpha_list=pred_context_alpha_list,
        pred_context_flow_list=pred_context_flow_list,
    )


def _make_video_build_pred_context_frames(grid_layout: GridLayout, frame_info: FrameInfo,
                                          font_size, ctx_lists: PredContextLists):
    context_t = grid_layout.context_t
    target_v = grid_layout.context_v
    t = frame_info.t
    current_frame_idx = frame_info.current_frame_idx
    pred_context_image_list = ctx_lists.image_list if ctx_lists.image_list else []
    pred_context_depth_list = ctx_lists.depth_list if ctx_lists.depth_list else []
    pred_context_alpha_list = ctx_lists.alpha_list if ctx_lists.alpha_list else []
    pred_context_flow_list = ctx_lists.flow_list if ctx_lists.flow_list else []
    frame_list = []
    for c_t in range(context_t):
        if len(pred_context_image_list) > 0:
            frame_list.append(add_label(
                hcat(*[pred_context_image_list[c_t][t][v_id] for v_id in range(target_v)]),
                f"Predicted Context {c_t} RGB (t={current_frame_idx})",
                font_size=font_size,
                align="center",
            ))
        if len(pred_context_depth_list) > 0:
            frame_list.append(add_label(
                hcat(*[pred_context_depth_list[c_t][t][v_id] for v_id in range(target_v)]),
                f"Predicted Context {c_t} Depth (t={current_frame_idx})",
                font_size=font_size,
                align="center",
            ))
        if len(pred_context_alpha_list) > 0:
            frame_list.append(add_label(
                hcat(*[pred_context_alpha_list[c_t][t][v_id] for v_id in range(target_v)]),
                f"Predicted Context {c_t} Opacity (t={current_frame_idx})",
                font_size=font_size,
                align="center",
            ))
        if len(pred_context_flow_list) > 0:
            frame_list.append(add_label(
                hcat(*[pred_context_flow_list[c_t][t][v_id] for v_id in range(target_v)]),
                f"Predicted Context {c_t} Flow (t={current_frame_idx})",
                font_size=font_size,
                align="center",
            ))
    return frame_list


def _assemble_video_frame(frame_list, n_ctx_per_row, vis_context_item, input_dict):
    num_rows = len(frame_list) // n_ctx_per_row
    frame = vcat(
        *vis_context_item,
        vcat(
            *[
                hcat(
                    *frame_list[row * n_ctx_per_row: (row + 1) * n_ctx_per_row],
                    gap=24,
                )
                for row in range(num_rows)
            ]
        ),
    )
    if len(frame_list) % n_ctx_per_row != 0:
        frame = vcat(
            frame,
            hcat(
                *frame_list[num_rows * n_ctx_per_row:],
                gap=24,
            ),
        )
    frame = add_border(
        add_label(
            frame,
            f"Scene{input_dict['scene_id']:03d}-{input_dict['scene_name'][:15]}",
            font_size=24,
            align="center",
        )
    )
    return prep_image(frame)


def _make_video_build_rgb_frames(pred_images, target_images, options: RgbFrameBuildOptions):
    t = options.t
    target_v = options.target_v
    current_frame_idx = options.current_frame_idx
    font_size = options.font_size
    frame_list = []
    pred_rgb = add_label(
        hcat(*[pred_images[t][v_id] for v_id in range(target_v)]),
        f"Predicted RGB (t={current_frame_idx})",
        font_size=font_size,
        align="center",
    )
    frame_list.append(pred_rgb)
    gt_rgb = add_label(
        hcat(*[target_images[t][v_id] for v_id in range(target_v)]),
        f"Target GT RGB (t={current_frame_idx})",
        font_size=font_size,
        align="center",
    )
    frame_list.append(gt_rgb)
    return frame_list


def _make_video_process_frame(frame_info: FrameInfo, render_data: FrameRenderData,
                              grid_layout: GridLayout, ctx_lists: PredContextLists,
                              viz_config: FrameVizConfig):
    t = frame_info.t
    target_dict = render_data.target_dict
    input_dict = render_data.input_dict
    pred_images = render_data.pred_images
    target_images = render_data.target_images
    render_results = render_data.render_results
    target_v = render_data.target_v
    n_ctx_per_row = render_data.n_ctx_per_row
    vis_context_item = render_data.vis_context_item
    font_size = render_data.font_size
    c_idx = viz_config.feat_params.c_idx
    c_idx_list = viz_config.feat_params.c_idx_list
    pca_linear = viz_config.feat_params.pca_linear
    cluster_image = viz_config.cluster_context.cluster_image
    skip_flags = viz_config.skip_flags
    tgt_size = viz_config.tgt_size
    vis_options = viz_config.vis_options
    current_frame_idx = int(target_dict["target_frame_idx"][0][t].item())
    frame_list = []
    frame_list.extend(_make_video_build_pred_context_frames(
        grid_layout, frame_info, font_size, ctx_lists))
    frame_list.extend(_make_video_build_rgb_frames(
        pred_images, target_images,
        RgbFrameBuildOptions(t=t, target_v=target_v, current_frame_idx=current_frame_idx, font_size=font_size)))
    depth_frames, alpha_image = _make_video_process_depth_visualization(
        render_results, frame_info, tgt_size, font_size)
    frame_list.extend(depth_frames)
    gt_depth_frame = _make_video_process_gt_depth(
        target_dict, frame_info, target_v, tgt_size, vis_options)
    if gt_depth_frame is not None:
        frame_list.append(gt_depth_frame)
    feat_frames, c_idx = _make_video_process_feature_visualization(
        render_results, input_dict, target_dict, frame_info, viz_config.feat_params)
    frame_list.extend(feat_frames)
    flow_frames = _make_video_process_flow_and_related(
        render_results, target_dict, frame_info, viz_config.cluster_context, vis_options)
    frame_list.extend(flow_frames)
    return _assemble_video_frame(
        frame_list, n_ctx_per_row, vis_context_item, input_dict), c_idx


def _make_video_finalize(video_frames, data_dict, reverse_video, save_video, output_filename):
    if data_dict is not None and 'relative_clip_start_id' in data_dict:
        video_frames = video_frames[data_dict['relative_clip_start_id']:]
        logger.info("Video relative_clip_start_id: %s", data_dict['relative_clip_start_id'])
    if reverse_video:
        video_frame_reversed = video_frames[::-1][1:-1]
        video_frames.extend(video_frame_reversed)
    if save_video:
        imageio.mimsave(output_filename, video_frames, fps=data_dict["fps"])
    return video_frames


def _make_video_render_frames(
        pred_context, target_dict, input_dict, config: VideoRenderFramesConfig):
    context_t = config.context_t
    target_t = config.target_t
    target_v = config.target_v
    n_ctx_per_row = config.n_ctx_per_row
    vis_context_item = config.vis_context_item
    font_size = config.font_size
    cluster_image = config.cluster_image
    tgt_size = config.tgt_size
    pca_linear = config.pca_linear
    skip_plot_gt_depth_and_flow = config.skip_plot_gt_depth_and_flow
    target_images = pred_context.target_images
    pred_images = pred_context.pred_images
    render_results = pred_context.render_results
    pred_context_image_list = pred_context.pred_context_image_list
    pred_context_depth_list = pred_context.pred_context_depth_list
    pred_context_alpha_list = pred_context.pred_context_alpha_list
    pred_context_flow_list = pred_context.pred_context_flow_list
    video_frames = []
    if os.getenv("CONTEXT_FEAT"):
        c_idx_list = (input_dict['context_frame_idx'][0] - target_dict['target_frame_idx'][0][0]).int().tolist()
        c_idx = 0  # init
    for t in range(target_t):
        current_frame_idx = int(target_dict["target_frame_idx"][0][t].item())
        frame_info = FrameInfo(t=t, current_frame_idx=current_frame_idx)
        grid_layout = GridLayout(
            context_t=context_t, context_v=target_v, n_ctx_per_row=n_ctx_per_row)
        ctx_lists = PredContextLists(
            image_list=pred_context_image_list, depth_list=pred_context_depth_list,
            alpha_list=pred_context_alpha_list, flow_list=pred_context_flow_list)
        render_data = FrameRenderData(
            target_dict=target_dict, input_dict=input_dict, pred_images=pred_images,
            target_images=target_images, render_results=render_results,
            target_v=target_v, n_ctx_per_row=n_ctx_per_row,
            vis_context_item=vis_context_item, font_size=font_size)
        vis_options = VisOptions(
            skip_flags=SkipFlags(skip_plot_gt_depth_and_flow=skip_plot_gt_depth_and_flow),
            font_size=font_size, tgt_size=tgt_size)
        viz_config = FrameVizConfig(
            cluster_context=ClusterContext(cluster_image=cluster_image, alpha_image=None),
            feat_params=FeatureVizParams(
                c_idx=c_idx, c_idx_list=c_idx_list, pca_linear=pca_linear, font_size=font_size),
            skip_flags=SkipFlags(skip_plot_gt_depth_and_flow=skip_plot_gt_depth_and_flow),
            tgt_size=tgt_size, vis_options=vis_options)
        frame, c_idx = _make_video_process_frame(
            frame_info, render_data, grid_layout, ctx_lists, viz_config)
        video_frames.append(frame)
    return video_frames


@torch.no_grad()
def make_video(
        dataset=None,
        model=None,
        device=None,
        output_filename=None,
        scene_id=None,
        skip_plot_gt_depth_and_flow: bool = False,
        data_dict=None,
        input_dict=None,
        target_dict=None,
        pred_dict=None,
        feat_extractor=None,
        reverse_video=True,
        save_video=True,
):
    video_inputs = VideoInputs(
        dataset=dataset, model=model, device=device, scene_id=scene_id,
        data_dict=data_dict, input_dict=input_dict, target_dict=target_dict,
        pred_dict=pred_dict, feat_extractor=feat_extractor)
    data_dict, input_dict, target_dict, pred_dict, font_size = _make_video_prepare_inputs(
        video_inputs, save_video, output_filename)
    b, context_t, context_v, _, h, w = input_dict["context_image"].shape
    _, target_t, target_v, _, h_tgt, w_tgt = target_dict["target_image"].shape
    device = input_dict["context_image"].device
    mean, std = torch.tensor([[MEAN]], device=device), torch.tensor([[STD]], device=device)
    vis_context_item, n_ctx_per_row = _make_video_build_vis_context(
        input_dict, pred_dict, context_t, context_v, font_size)
    pred_context = _make_video_prepare_images_and_pred_context(
        input_dict, target_dict, pred_dict, context_t, target_t)
    tgt_size = TargetSize(h_tgt=h_tgt, w_tgt=w_tgt)
    cluster_image = _prepare_motion_segmentation_image(
        pred_context.render_results, pred_dict, target_t, target_v, tgt_size)
    if not os.getenv("CONTEXT_FEAT") and "target_feat" in target_dict.keys():
        pca_linear = get_global_pac_linear(target_dict["target_feat"])
    if os.getenv("CONTEXT_FEAT") and "context_feat" in input_dict.keys():
        pca_linear = get_global_pac_linear(input_dict["context_feat"])
    video_frames = _make_video_render_frames(
        pred_context, target_dict, input_dict,
        VideoRenderFramesConfig(context_t=context_t, target_t=target_t, target_v=target_v,
                                n_ctx_per_row=n_ctx_per_row, vis_context_item=vis_context_item,
                                font_size=font_size, cluster_image=cluster_image,
                                tgt_size=tgt_size, pca_linear=pca_linear,
                                skip_plot_gt_depth_and_flow=skip_plot_gt_depth_and_flow))
    return _make_video_finalize(
        video_frames, data_dict, reverse_video, save_video, output_filename)


def _make_clean_video_prepare_inputs(video_inputs: VideoInputs, font_size, output_filename):
    dataset = video_inputs.dataset
    model = video_inputs.model
    device = video_inputs.device
    scene_id = video_inputs.scene_id
    data_dict = video_inputs.data_dict
    input_dict = video_inputs.input_dict
    target_dict = video_inputs.target_dict
    pred_dict = video_inputs.pred_dict
    feat_extractor = video_inputs.feat_extractor
    if output_filename is None:
        raise ValueError("output_filename must be provided")

    if data_dict is None:
        if dataset is None:
            raise ValueError("dataset must be provided when data_dict is None")
        if scene_id is None:
            scene_id = np.random.randint(0, len(dataset))
        data_dict = dataset.__getitem__(scene_id, np.random.randint(10, 100), return_all=True)
        data_dict = to_batch_tensor(data_dict)
    if not isinstance(data_dict['num_max_cams'], int):
        num_max_cams = int(data_dict['num_max_cams'][0])
    else:
        num_max_cams = data_dict['num_max_cams']
    if font_size is None:
        if num_max_cams > 1:
            font_size = 24
        else:
            font_size = 8
    if input_dict is None or target_dict is None:
        if device is None:
            raise ValueError("device must be provided when input_dict or target_dict is None")
        input_dict, target_dict = prepare_inputs_and_targets(data_dict, device, v=num_max_cams,
                                                             feat_extractor=feat_extractor)

    if pred_dict is None:
        if model is None:
            raise ValueError("model must be provided when pred_dict is None")
        if device is None:
            raise ValueError("device must be provided when pred_dict is None")
        model = model.eval()  # model is needed only pred_dict is None
        dtype = torch.float16 if os.environ.get('DISABLE_BFLOAT') else torch.bfloat16
        with torch.autocast(device_type=device.type, dtype=dtype):
            start_time = time.perf_counter()
            pred_dict = model(input_dict)
            end_time = time.perf_counter()
            logger.info(f"Time taken to forward rendered results: {end_time - start_time} seconds")
    return data_dict, input_dict, target_dict, pred_dict, font_size


def _make_clean_video_prepare_images(video_inputs: VideoInputs, context_v,
                                     rotate_render_results, skip_flags: SkipFlags):
    input_dict = video_inputs.input_dict
    target_dict = video_inputs.target_dict
    pred_dict = video_inputs.pred_dict
    skip_depth = skip_flags.skip_depth
    skip_gt_rgb = skip_flags.skip_gt_rgb

    def denormalize(x, already_channel_last=False):
        if not already_channel_last:
            x = rearrange(x, "t v c h w -> t v h w c")
        return rearrange(x, "t v h w c -> t v c h w")

    context_images = input_dict["context_image"][0]
    context_images = denormalize(context_images)

    if context_v == 1:
        n_ctx_per_row = 4  # better visualize for 1 view
    elif context_v <= 3:
        n_ctx_per_row = 2
    else:
        n_ctx_per_row = 1
    if skip_depth and skip_gt_rgb:
        n_ctx_per_row = 1

    target_images = target_dict["target_image"][0]
    target_images = denormalize(target_images)
    render_results = pred_dict["render_results"]

    pred_images = render_results[render_results["rgb_key"]][0]
    pred_images = denormalize(pred_images, already_channel_last=True)
    if rotate_render_results is not None:
        rotate_pred_images = rotate_render_results[render_results["rgb_key"]][0]
        rotate_pred_images = denormalize(rotate_pred_images, already_channel_last=True)
    else:
        rotate_pred_images = None
    return n_ctx_per_row, target_images, pred_images, rotate_pred_images, render_results


def _make_clean_video_assemble_frame(frame_list, n_ctx_per_row):
    num_rows = len(frame_list) // n_ctx_per_row
    frame_rows = [
        hcat(
            *frame_list[row * n_ctx_per_row: (row + 1) * n_ctx_per_row],
            gap=24,
        )
        for row in range(num_rows)
    ]
    frame = vcat(*frame_rows)

    if len(frame_list) % n_ctx_per_row != 0:
        frame = vcat(
            frame,
            hcat(
                *frame_list[num_rows * n_ctx_per_row:],
                gap=24,
            ),
        )
    return prep_image(frame)


def _make_clean_video_build_rgb_frames(
        pred_images, target_images, options: RgbFrameBuildOptions):
    t = options.t
    target_v = options.target_v
    current_frame_idx = options.current_frame_idx
    font_size = options.font_size
    rotate_pred_images = options.rotate_pred_images
    skip_gt_rgb = options.skip_gt_rgb
    frame_list = []
    pred_rgb = add_label(
        hcat(*[pred_images[t][v_id] for v_id in range(target_v)]),
        f"Predicted RGB (t={current_frame_idx})",
        font_size=font_size,
        align="center",
    )
    frame_list.append(pred_rgb)
    if rotate_pred_images is not None:
        rotate_pred_rgb = add_label(
            hcat(*[rotate_pred_images[t][v_id] for v_id in range(target_v)]),
            f"Predicted RGB (t={current_frame_idx})",
            font_size=font_size,
            align="center",
        )
        frame_list.append(rotate_pred_rgb)
    if not skip_gt_rgb:
        gt_rgb = add_label(
            hcat(*[target_images[t][v_id] for v_id in range(target_v)]),
            f"Target GT RGB (t={current_frame_idx})",
            font_size=font_size,
            align="center",
        )
        frame_list.append(gt_rgb)
    return frame_list


def _make_clean_video_process_frame(frame_info: FrameInfo, render_data: FrameRenderData,
                                    rotate_pred_images, viz_config: FrameVizConfig):
    t = frame_info.t
    target_dict = render_data.target_dict
    input_dict = render_data.input_dict
    pred_images = render_data.pred_images
    target_images = render_data.target_images
    render_results = render_data.render_results
    target_v = render_data.target_v
    n_ctx_per_row = render_data.n_ctx_per_row
    font_size = render_data.font_size
    current_frame_idx = frame_info.current_frame_idx
    skip_flags = viz_config.skip_flags
    cluster_image = viz_config.cluster_context.cluster_image if viz_config.cluster_context else None
    feat_params = viz_config.feat_params
    tgt_size = viz_config.tgt_size
    skip_gt_rgb = skip_flags.skip_gt_rgb
    skip_depth = skip_flags.skip_depth
    skip_gt_feat = skip_flags.skip_gt_feat
    c_idx = feat_params.c_idx
    c_idx_list = feat_params.c_idx_list
    pca_linear = feat_params.pca_linear
    frame_list = []
    current_frame_idx = int(target_dict["target_frame_idx"][0][t].item())
    rgb_frames = _make_clean_video_build_rgb_frames(
        pred_images, target_images,
        RgbFrameBuildOptions(t=t, target_v=target_v, current_frame_idx=current_frame_idx,
                             font_size=font_size, rotate_pred_images=rotate_pred_images,
                             skip_gt_rgb=skip_gt_rgb))
    frame_list.extend(rgb_frames)
    vis_options = VisOptions(skip_flags=skip_flags, font_size=font_size, tgt_size=tgt_size)
    depth_frames = _process_depth_visualization(
        render_results, frame_info, tgt_size, vis_options)
    frame_list.extend(depth_frames)
    gt_depth_frame = _process_gt_depth_frame(
        target_dict, frame_info, target_v, tgt_size, vis_options)
    if gt_depth_frame is not None:
        frame_list.append(gt_depth_frame)
    feat_frames, c_idx = _process_feature_visualization(
        render_results, input_dict, target_dict, frame_info, feat_params)
    frame_list.extend(feat_frames)
    return _make_clean_video_assemble_frame(frame_list, n_ctx_per_row), c_idx


def _make_clean_video_prepare_feat(target_dict, input_dict):
    if not os.getenv("CONTEXT_FEAT") and "target_feat" in target_dict.keys():
        pca_linear = get_global_pac_linear(target_dict["target_feat"])
    if os.getenv("CONTEXT_FEAT") and "context_feat" in input_dict.keys():
        pca_linear = get_global_pac_linear(input_dict["context_feat"])
    c_idx_list = None
    c_idx = None
    if os.getenv("CONTEXT_FEAT"):
        c_idx_list = (input_dict['context_frame_idx'][0] - target_dict['target_frame_idx'][0][0]).int().tolist()
        c_idx = 0  # init
    return pca_linear, c_idx, c_idx_list


def _make_clean_video_render_frames(
        target_dict, input_dict, config: CleanVideoRenderFramesConfig):
    target_t = config.target_t
    target_v = config.target_v
    pred_images = config.pred_images
    target_images = config.target_images
    rotate_pred_images = config.rotate_pred_images
    render_results = config.render_results
    n_ctx_per_row = config.n_ctx_per_row
    font_size = config.font_size
    cluster_image = config.cluster_image
    tgt_size = config.tgt_size
    skip_flags = config.skip_flags
    c_idx = config.c_idx
    c_idx_list = config.c_idx_list
    pca_linear = config.pca_linear
    video_frames = []
    for t in range(0, int(target_t)):
        current_frame_idx = int(target_dict["target_frame_idx"][0][t].item())
        frame_info = FrameInfo(t=t, current_frame_idx=current_frame_idx)
        render_data = FrameRenderData(
            target_dict=target_dict, input_dict=input_dict, pred_images=pred_images,
            target_images=target_images, render_results=render_results,
            target_v=target_v, n_ctx_per_row=n_ctx_per_row, font_size=font_size)
        viz_config = FrameVizConfig(
            cluster_context=ClusterContext(cluster_image=cluster_image),
            feat_params=FeatureVizParams(
                c_idx=c_idx, c_idx_list=c_idx_list, pca_linear=pca_linear,
                font_size=font_size, skip_flags=skip_flags),
            skip_flags=skip_flags, tgt_size=tgt_size,
            vis_options=VisOptions(skip_flags=skip_flags, font_size=font_size, tgt_size=tgt_size))
        frame, c_idx = _make_clean_video_process_frame(
            frame_info, render_data, rotate_pred_images, viz_config)
        video_frames.append(frame)
    return video_frames


@torch.no_grad()
def make_clean_video(
        dataset=None,
        model=None,
        device=None,
        output_filename=None,
        scene_id=None,
        skip_plot_gt_depth_and_flow: bool = False,
        data_dict=None,
        input_dict=None,
        target_dict=None,
        rotate_render_results=None,
        pred_dict=None,
        feat_extractor=None,
        skip_depth=False,
        skip_gt_rgb=False,
        skip_gt_feat=False,
        font_size=None,
):
    video_inputs = VideoInputs(
        dataset=dataset, model=model, device=device, scene_id=scene_id,
        data_dict=data_dict, input_dict=input_dict, target_dict=target_dict,
        pred_dict=pred_dict, feat_extractor=feat_extractor)
    data_dict, input_dict, target_dict, pred_dict, font_size = _make_clean_video_prepare_inputs(
        video_inputs, font_size, output_filename)
    b, context_t, context_v, _, h, w = input_dict["context_image"].shape
    _, target_t, target_v, _, h_tgt, w_tgt = target_dict["target_image"].shape
    device = input_dict["context_image"].device
    mean = torch.tensor([[MEAN]], device=device)
    std = torch.tensor([[STD]], device=device)
    skip_flags = SkipFlags(
        skip_depth=skip_depth, skip_plot_gt_depth_and_flow=skip_plot_gt_depth_and_flow,
        skip_gt_feat=skip_gt_feat, skip_gt_rgb=skip_gt_rgb)
    video_inputs_clean = VideoInputs(
        input_dict=input_dict, target_dict=target_dict, pred_dict=pred_dict)
    n_ctx_per_row, target_images, pred_images, rotate_pred_images, render_results = \
        _make_clean_video_prepare_images(
            video_inputs_clean, context_v, rotate_render_results, skip_flags)
    tgt_size = TargetSize(h_tgt=h_tgt, w_tgt=w_tgt)
    cluster_image = _prepare_motion_segmentation_image(
        render_results, pred_dict, target_t, target_v, tgt_size)
    pca_linear, c_idx, c_idx_list = _make_clean_video_prepare_feat(target_dict, input_dict)
    video_frames = _make_clean_video_render_frames(
        target_dict, input_dict,
        CleanVideoRenderFramesConfig(target_t=target_t, target_v=target_v,
                                     pred_images=pred_images, target_images=target_images,
                                     rotate_pred_images=rotate_pred_images, render_results=render_results,
                                     n_ctx_per_row=n_ctx_per_row, font_size=font_size,
                                     cluster_image=cluster_image, tgt_size=tgt_size,
                                     skip_flags=skip_flags, c_idx=c_idx, c_idx_list=c_idx_list,
                                     pca_linear=pca_linear))
    imageio.mimsave(output_filename, video_frames, fps=data_dict["fps"])
    return video_frames


def _vis_build_context_frames(input_dict, context_t, context_v):
    """Build context frames for video visualization."""
    def denormalize(x, already_channel_last=False):
        if not already_channel_last:
            x = rearrange(x, "t v c h w -> t v h w c")
        return rearrange(x, "t v h w c -> t v c h w")

    # t, v, c, h, w
    context_images = input_dict["context_image"][0]
    context_images = denormalize(context_images)

    if context_v <= 3:
        n_ctx_per_row = 2
    else:
        n_ctx_per_row = 1
        # concate context images horizontally
    context_frames = []
    for t in range(context_t):
        current_frame_idx = int(input_dict["context_frame_idx"][0][t].item())
        row = add_label(
            hcat(*[context_images[t][v_id] for v_id in range(context_v)]),
            f"Context RGB (t={current_frame_idx})",
            font_size=24,
            align="center",
        )
        context_frames.append(row)
    num_rows = max(1, len(context_frames) // n_ctx_per_row)
    context_frames = vcat(
        *[
            hcat(
                *context_frames[row * n_ctx_per_row: (row + 1) * n_ctx_per_row],
                gap=24,
            )
            for row in range(num_rows)
        ]
    )
    return context_frames, n_ctx_per_row


def _vis_prepare_images(input_dict, target_dict, pred_dict):
    """Prepare target and predicted images for visualization."""
    def denormalize(x, already_channel_last=False):
        if not already_channel_last:
            x = rearrange(x, "t v c h w -> t v h w c")
        return rearrange(x, "t v h w c -> t v c h w")

    target_images = target_dict["target_image"][0]
    target_images = denormalize(target_images)
    render_results = pred_dict["render_results"]

    pred_images = render_results[render_results["rgb_key"]][0]
    pred_images = denormalize(pred_images, already_channel_last=True)
    return target_images, pred_images, render_results


def _vis_compute_cluster_image(render_results, pred_dict, target_t, target_v, tgt_size: TargetSize):
    """Compute motion segmentation cluster image."""
    h_tgt = tgt_size.h_tgt
    w_tgt = tgt_size.w_tgt
    if "rendered_motion_seg" not in render_results:
        return None

    # Get the max index (clusters) from the rendered results
    max_idx = render_results["rendered_motion_seg"][0]

    # Identify unique clusters
    unique_clusters = torch.unique(max_idx)
    velocities = pred_dict["gs_params"]["motion_bases"][0][unique_clusters]
    velocity_norm = torch.norm(velocities, dim=-1)
    # Sort the unique clusters according to velocity norm (lowest first)
    sorted_indices = torch.argsort(velocity_norm)
    sorted_clusters = unique_clusters[sorted_indices]

    # Number of unique clusters
    num_unique_clusters = len(sorted_clusters)

    # Create a new colormap based on the sorted unique clusters
    cmap = cm.get_cmap("rainbow", num_unique_clusters)

    # Map sorted unique clusters to new colors
    cluster_to_color_map = torch.tensor([cmap(i) for i in range(num_unique_clusters)])[
        :, :3
    ].to(max_idx.device)

    # Create a mapping from original clusters to the reassigned clusters
    cluster_mapping = torch.zeros_like(max_idx)

    # Map each pixel in max_idx to the new cluster index based on sorted clusters
    for new_cluster_idx, original_cluster in enumerate(sorted_clusters):
        cluster_mapping[max_idx == original_cluster] = new_cluster_idx

    # Assign the new colors to the cluster image
    cluster_image = cluster_to_color_map[cluster_mapping]
    if cluster_image.shape[-3] != h_tgt or cluster_image.shape[-2] != w_tgt:
        cluster_image = F.interpolate(
            rearrange(cluster_image, "t v h w c -> (t v) c h w"),
            size=(h_tgt, w_tgt),
            mode="nearest",
        )
        cluster_image = rearrange(
            cluster_image, "(t v) c h w -> t v h w c", t=target_t, v=target_v
        )
    return cluster_image


def _vis_process_depth_decoder(render_results, t, current_frame_idx):
    if render_results["decoder_depth_key"] is None:
        return []
    depth_image = render_results[render_results["decoder_depth_key"]][0][t]
    alpha_image = None
    depth_image = depth_image.detach().cpu().numpy()
    depth_image = depth_visualizer(depth_image, alpha_image)
    depth_image = torch.from_numpy(depth_image)
    depth_image = rearrange(depth_image, "v h w c -> v c h w")
    return [add_label(
        hcat(*depth_image),
        f"Predicted Decoder Depth (t={current_frame_idx})",
        font_size=24,
        align="center",
    )]


def _vis_process_depth_gs(render_results, t, current_frame_idx, h_tgt, w_tgt):
    if render_results["depth_key"] is None:
        return [], None
    depth_image = render_results[render_results["depth_key"]][0][t]
    alpha_image = render_results[render_results["alpha_key"]][0][t]
    if depth_image.shape[-2] != h_tgt or depth_image.shape[-1] != w_tgt:
        depth_image = F.interpolate(
            depth_image.unsqueeze(-3), size=(h_tgt, w_tgt), mode="bilinear", align_corners=False,
        ).squeeze(-3)
        alpha_image = F.interpolate(
            alpha_image.unsqueeze(-3), size=(h_tgt, w_tgt), mode="bilinear", align_corners=False,
        ).squeeze(-3)
    depth_image = depth_image.detach().cpu().numpy()
    alpha_image = alpha_image.detach().cpu().numpy()
    depth_image = depth_visualizer(depth_image, alpha_image)
    depth_image = torch.from_numpy(depth_image)
    depth_image = rearrange(depth_image, "v h w c -> v c h w")
    frame_list = [add_label(
        hcat(*depth_image),
        f"Predicted Depth (t={current_frame_idx})",
        font_size=24, align="center",
    )]
    alpha_label = torch.from_numpy(alpha_image).unsqueeze(1).repeat(1, 3, 1, 1)
    frame_list.append(add_label(
        hcat(*alpha_label),
        f"Predicted Opacity (t={current_frame_idx})",
        font_size=24, align="center",
    ))
    return frame_list, alpha_image


def _vis_process_flow_and_related(render_results, target_dict, frame_info: FrameInfo,
                                  tgt_size: TargetSize, cluster_image):
    t = frame_info.t
    current_frame_idx = frame_info.current_frame_idx
    h_tgt = tgt_size.h_tgt
    w_tgt = tgt_size.w_tgt
    if render_results["flow_key"] is None:
        return []
    flow_image = render_results[render_results["flow_key"]][0][t]
    flow_image = scene_flow_to_rgb(flow_image, flow_max_radius=15)
    flow_image = rearrange(flow_image, "v h w c -> v c h w")
    if flow_image.shape[-2] != h_tgt or flow_image.shape[-1] != w_tgt:
        flow_image = F.interpolate(
            flow_image, size=(h_tgt, w_tgt), mode="bilinear", align_corners=False)
    frame_list = [add_label(
        hcat(*flow_image), f"Predicted Flow (t={current_frame_idx})",
        font_size=24, align="center",
    )]
    if "target_flow" in target_dict.keys():
        gt_flow = target_dict["target_flow"][0][t]
        gt_flow = scene_flow_to_rgb(gt_flow, flow_max_radius=15)
        gt_flow = rearrange(gt_flow, "v h w c -> v c h w")
        frame_list.append(add_label(
            hcat(*gt_flow),
            f"GT Flow (t={current_frame_idx}) (Not used as supervision)",
            font_size=24, align="center",
        ))
    if cluster_image is not None:
        cluster_image_t = cluster_image[t]
        cluster_image_t = rearrange(cluster_image_t, "v h w c -> v c h w")
        frame_list.append(add_label(
            hcat(*cluster_image_t),
            f"Motion Segmentation (t={current_frame_idx})",
            font_size=24, align="center",
        ))
    return frame_list


def _vis_assemble_frame(frame_list, n_ctx_per_row, context_frames, input_dict):
    num_rows = len(frame_list) // n_ctx_per_row
    frame = vcat(
        context_frames,
        vcat(
            *[
                hcat(
                    *frame_list[row * n_ctx_per_row: (row + 1) * n_ctx_per_row],
                    gap=24,
                )
                for row in range(num_rows)
            ]
        ),
    )
    if len(frame_list) % n_ctx_per_row != 0:
        frame = vcat(
            frame,
            hcat(
                *frame_list[num_rows * n_ctx_per_row:],
                gap=24,
            ),
        )
    frame = add_border(
        add_label(
            frame,
            f"Scene{input_dict['scene_id']:03d}-{input_dict['scene_name'][:15]}",
            font_size=24,
            align="center",
        )
    )
    return prep_image(frame)


def _vis_process_frame(
        frame_info: FrameInfo, render_data: FrameRenderData, tgt_size: TargetSize,
        cluster_image):
    """Process a single frame for video visualization."""
    t = frame_info.t
    target_dict = render_data.target_dict
    pred_images = render_data.pred_images
    target_images = render_data.target_images
    render_results = render_data.render_results
    target_v = render_data.target_v
    n_ctx_per_row = render_data.n_ctx_per_row
    context_frames = render_data.context_frames
    input_dict = render_data.input_dict
    current_frame_idx = frame_info.current_frame_idx
    h_tgt = tgt_size.h_tgt
    w_tgt = tgt_size.w_tgt
    frame_list = [add_label(
        hcat(*[pred_images[t][v_id] for v_id in range(target_v)]),
        f"Predicted RGB (t={current_frame_idx})", font_size=24, align="center",
    )]
    frame_list.append(add_label(
        hcat(*[target_images[t][v_id] for v_id in range(target_v)]),
        f"Target GT RGB (t={current_frame_idx})", font_size=24, align="center",
    ))
    frame_list.extend(_vis_process_depth_decoder(render_results, t, current_frame_idx))
    depth_gs_frames, _ = _vis_process_depth_gs(
        render_results, t, current_frame_idx, h_tgt, w_tgt)
    frame_list.extend(depth_gs_frames)
    frame_list.extend(_vis_process_flow_and_related(
        render_results, target_dict, frame_info, tgt_size, cluster_image))
    return _vis_assemble_frame(frame_list, n_ctx_per_row, context_frames, input_dict)


@torch.no_grad()
def make_video_vis(
        video_inputs: VideoInputs,
        output_filename,
        args=None,
        time_step=10,
):
    dataset = video_inputs.dataset
    model = video_inputs.model
    device = video_inputs.device
    scene_id = video_inputs.scene_id
    data_dict = dataset.__getitem__(scene_id, time_step, return_all=True)
    data_dict = to_batch_tensor(data_dict)
    input_dict, target_dict = prepare_inputs_and_targets(
        data_dict,
        device,
    )
    model = model.eval()
    pred_dict = model(input_dict)
    b, context_t, context_v, _, h, w = input_dict["context_image"].shape
    _, target_t, target_v, _, h_tgt, w_tgt = target_dict["target_image"].shape

    device = input_dict["context_image"].device
    mean = torch.tensor([[MEAN]], device=device)
    std = torch.tensor([[STD]], device=device)

    context_frames, n_ctx_per_row = _vis_build_context_frames(input_dict, context_t, context_v)
    target_images, pred_images, render_results = _vis_prepare_images(input_dict, target_dict, pred_dict)
    tgt_size = TargetSize(h_tgt=h_tgt, w_tgt=w_tgt)
    cluster_image = _vis_compute_cluster_image(render_results, pred_dict, target_t, target_v, tgt_size)

    video_frames = []
    for t in range(target_t):
        current_frame_idx = int(target_dict["target_frame_idx"][0][t].item())
        frame_info = FrameInfo(t=t, current_frame_idx=current_frame_idx)
        render_data = FrameRenderData(
            target_dict=target_dict, pred_images=pred_images, target_images=target_images,
            render_results=render_results, target_v=target_v, n_ctx_per_row=n_ctx_per_row,
            context_frames=context_frames, input_dict=input_dict)
        video_frames.append(_vis_process_frame(
            frame_info, render_data, tgt_size, cluster_image))
    video_frame_reversed = video_frames[::-1][1:-1]
    video_frames.extend(video_frame_reversed)
    imageio.mimsave(output_filename, video_frames, fps=data_dict["fps"])


def _av2_process_depth_decoder(render_results, frame_info: FrameInfo, target_v, av2_resize: AV2ResizeInfo):
    """Process decoder depth visualization for av2 video.
    
    Args:
        render_results: Render results dictionary
        frame_info: Frame index information (t, current_frame_idx)
        target_v: Number of target views
        av2_resize: AV2 resize parameters (resize, w, h)
    
    Returns:
        List containing depth frame item or empty list
    """
    t = frame_info.t
    current_frame_idx = frame_info.current_frame_idx
    resize = av2_resize.resize
    w = av2_resize.w
    h = av2_resize.h
    frame_list = []
    if render_results["decoder_depth_key"] is not None:
        depth_image = render_results[render_results["decoder_depth_key"]][0][t]
        alpha_image = None
        depth_image = depth_image.detach().cpu().numpy()
        depth_image = depth_visualizer(depth_image, alpha_image)
        depth_image = torch.from_numpy(depth_image)
        depth_image = rearrange(depth_image, "v h w c -> v c h w")
        pred_depth = add_label(
            hcat(
                *[
                    (
                        depth_image[v_id]
                        if v_id != target_v // 2
                        else resize(depth_image[v_id], (w, h))
                    )
                    for v_id in range(target_v)
                ],
                align="bottom",
            ),
            f"Predicted Decoder Depth (t={current_frame_idx})",
            font_size=24,
            align="center",
        )
        frame_list.append(pred_depth)
    return frame_list


def _av2_process_depth_gs_prepare(render_results, t, h_tgt, w_tgt):
    depth_image = render_results[render_results["depth_key"]][0][t]
    alpha_image = render_results[render_results["alpha_key"]][0][t]
    if depth_image.shape[-2] != h_tgt or depth_image.shape[-1] != w_tgt:
        depth_image = F.interpolate(
            depth_image.unsqueeze(-3),
            size=(h_tgt, w_tgt),
            mode="bilinear",
            align_corners=False,
        ).squeeze(-3)
        alpha_image = F.interpolate(
            alpha_image.unsqueeze(-3),
            size=(h_tgt, w_tgt),
            mode="bilinear",
            align_corners=False,
        ).squeeze(-3)
    depth_image = depth_image.detach().cpu().numpy()
    alpha_image = alpha_image.detach().cpu().numpy()
    depth_image = depth_visualizer(depth_image, alpha_image)
    depth_image = torch.from_numpy(depth_image)
    depth_image = rearrange(depth_image, "v h w c -> v c h w")
    return depth_image, alpha_image


def _av2_label_pred_depth(depth_image, target_v, av2_resize: AV2ResizeInfo, current_frame_idx):
    resize = av2_resize.resize
    w = av2_resize.w
    h = av2_resize.h
    return add_label(
        hcat(
            *[
                (
                    depth_image[v_id]
                    if v_id != target_v // 2
                    else resize(depth_image[v_id], (w, h))
                )
                for v_id in range(target_v)
            ],
            align="bottom",
        ),
        f"Predicted Depth (t={current_frame_idx})",
        font_size=24,
        align="center",
    )


def _av2_process_depth_gs(render_results, frame_info: FrameInfo, target_v, tgt_size: TargetSize,
                          av2_resize: AV2ResizeInfo):
    """Process Gaussian splat depth visualization for av2 video.

    Args:
        render_results: Render results dictionary
        frame_info: Frame index information (t, current_frame_idx)
        target_v: Number of target views
        tgt_size: Target spatial dimensions (h_tgt, w_tgt)
        av2_resize: AV2 resize parameters (resize, w, h)

    Returns:
        Tuple of (frame_list, alpha_image) where alpha_image may be used by opacity processing
    """
    t = frame_info.t
    current_frame_idx = frame_info.current_frame_idx
    h_tgt = tgt_size.h_tgt
    w_tgt = tgt_size.w_tgt
    resize = av2_resize.resize
    w = av2_resize.w
    h = av2_resize.h
    frame_list = []
    alpha_image = None
    if render_results["depth_key"] is not None:
        depth_image, alpha_image = _av2_process_depth_gs_prepare(render_results, t, h_tgt, w_tgt)
        frame_list.append(
            _av2_label_pred_depth(depth_image, target_v, av2_resize, current_frame_idx))
    return frame_list, alpha_image


def _av2_label_gt_depth(gt_depth, target_v, av2_resize: AV2ResizeInfo, current_frame_idx):
    resize = av2_resize.resize
    w = av2_resize.w
    h = av2_resize.h
    gt_depth = add_label(
        hcat(
            *[
                (
                    gt_depth[v_id]
                    if v_id != target_v // 2
                    else resize(gt_depth[v_id], (w, h))
                )
                for v_id in range(target_v)
            ],
            align="bottom",
        ),
        f"Target GT Depth (t={current_frame_idx})",
        font_size=24,
        align="center",
    )
    return gt_depth


def _av2_process_gt_depth(
    target_dict, frame_info: FrameInfo, target_v, vis_options: VisOptions,
    av2_resize: AV2ResizeInfo
):
    """Process ground truth depth visualization for av2 video.
    
    Args:
        target_dict: Target dictionary
        frame_info: Frame index information (t, current_frame_idx)
        target_v: Number of target views
        tgt_size: Target spatial dimensions (h_tgt, w_tgt)
        skip_flags: Boolean skip flags
        av2_resize: AV2 resize parameters (resize, w, h)
    
    Returns:
        List containing GT depth frame item or empty list
    """
    t = frame_info.t
    current_frame_idx = frame_info.current_frame_idx
    h_tgt = vis_options.tgt_size.h_tgt
    w_tgt = vis_options.tgt_size.w_tgt
    skip_plot_gt_depth_and_flow = vis_options.skip_flags.skip_plot_gt_depth_and_flow
    resize = av2_resize.resize
    w = av2_resize.w
    h = av2_resize.h
    frame_list = []
    if "target_depth" in target_dict.keys():
        gt_depth = target_dict["target_depth"][0][t]
        gt_depth = gt_depth.detach().cpu().numpy()
        gt_depth = depth_visualizer(gt_depth, gt_depth > 0)
        gt_depth = torch.from_numpy(gt_depth)
        gt_depth = rearrange(gt_depth, "v h w c -> v c h w")
        gt_depth = _av2_label_gt_depth(gt_depth, target_v, av2_resize, current_frame_idx)
        frame_list.append(gt_depth)
    else:
        if not skip_plot_gt_depth_and_flow:
            gt_depth = torch.full((target_v, 3, h_tgt, w_tgt), 0.5)
            gt_depth = _av2_label_gt_depth(gt_depth, target_v, av2_resize, current_frame_idx)
            frame_list.append(gt_depth)
    return frame_list


def _av2_process_pred_flow(render_results, frame_info: FrameInfo, tgt_size: TargetSize,
                           target_v, av2_resize: AV2ResizeInfo):
    t = frame_info.t
    current_frame_idx = frame_info.current_frame_idx
    h_tgt = tgt_size.h_tgt
    w_tgt = tgt_size.w_tgt
    resize = av2_resize.resize
    w = av2_resize.w
    h = av2_resize.h
    flow_image = render_results[render_results["flow_key"]][0][t]
    flow_image = scene_flow_to_rgb(flow_image, flow_max_radius=15)
    flow_image = rearrange(flow_image, "v h w c -> v c h w")
    if flow_image.shape[-2] != h_tgt or flow_image.shape[-1] != w_tgt:
        flow_image = F.interpolate(
            flow_image,
            size=(h_tgt, w_tgt),
            mode="bilinear",
            align_corners=False,
        )
    flow_image = add_label(
        hcat(
            *[
                (
                    flow_image[v_id]
                    if v_id != target_v // 2
                    else resize(flow_image[v_id], (w, h))
                )
                for v_id in range(target_v)
            ],
            align="bottom",
        ),
        f"Predicted Flow (t={current_frame_idx})",
        font_size=24,
        align="center",
    )
    return flow_image


def _av2_process_gt_flow(target_dict, frame_info: FrameInfo, target_v, av2_resize: AV2ResizeInfo):
    t = frame_info.t
    current_frame_idx = frame_info.current_frame_idx
    resize = av2_resize.resize
    w = av2_resize.w
    h = av2_resize.h
    gt_flow = target_dict["target_flow"][0][t]
    gt_flow = scene_flow_to_rgb(gt_flow, flow_max_radius=15)
    gt_flow = rearrange(gt_flow, "v h w c -> v c h w")
    gt_flow = add_label(
        hcat(
            *[
                (
                    gt_flow[v_id]
                    if v_id != target_v // 2
                    else resize(gt_flow[v_id], (w, h))
                )
                for v_id in range(target_v)
            ],
            align="bottom",
        ),
        f"Target GT Flow (t={current_frame_idx})",
        font_size=24,
        align="center",
    )
    return gt_flow


def _av2_process_flow(render_results, target_dict, frame_info: FrameInfo, target_v,
                      vis_options: VisOptions):
    """Process flow visualization for av2 video.
    
    Args:
        render_results: Render results dictionary
        target_dict: Target dictionary
        frame_info: Frame index information (t, current_frame_idx)
        target_v: Number of target views
        vis_options: Visualization options (contains tgt_size)
        av2_resize: AV2 resize parameters (resize, w, h)
    
    Returns:
        List containing flow frame items
    """
    t = frame_info.t
    current_frame_idx = frame_info.current_frame_idx
    h_tgt = vis_options.tgt_size.h_tgt
    w_tgt = vis_options.tgt_size.w_tgt
    av2_resize = vis_options.av2_resize
    resize = av2_resize.resize
    w = av2_resize.w
    h = av2_resize.h
    frame_list = []
    if render_results["flow_key"] is not None:
        flow_image = _av2_process_pred_flow(
            render_results, frame_info, vis_options.tgt_size, target_v, vis_options.av2_resize
        )
        frame_list.append(flow_image)

        if "target_flow" in target_dict.keys():
            gt_flow = _av2_process_gt_flow(
                target_dict, frame_info, target_v, vis_options.av2_resize
            )
            frame_list.append(gt_flow)
    return frame_list


def _av2_process_opacity(alpha_image, frame_info: FrameInfo, target_v, av2_resize: AV2ResizeInfo):
    """Process opacity visualization for av2 video.
    
    Args:
        alpha_image: Alpha image array from depth processing
        frame_info: Frame index information (t, current_frame_idx)
        target_v: Number of target views
        av2_resize: AV2 resize parameters (resize, w, h)
    
    Returns:
        List containing opacity frame item or empty list
    """
    current_frame_idx = frame_info.current_frame_idx
    resize = av2_resize.resize
    w = av2_resize.w
    h = av2_resize.h
    frame_list = []
    if alpha_image is not None:
        alpha_image = torch.from_numpy(alpha_image).unsqueeze(1)
        alpha_image = alpha_image.repeat(1, 3, 1, 1)
        alpha_image = add_label(
            hcat(
                *[
                    (
                        alpha_image[v_id]
                        if v_id != target_v // 2
                        else resize(alpha_image[v_id], (w, h))
                    )
                    for v_id in range(target_v)
                ],
                align="bottom",
            ),
            f"Predicted Opacity (t={current_frame_idx})",
            font_size=24,
            align="center",
        )
        frame_list.append(alpha_image)
    return frame_list


def _av2_process_sky_mask(target_dict, frame_info: FrameInfo, target_v, av2_resize: AV2ResizeInfo):
    """Process sky mask visualization for av2 video.
    
    Args:
        target_dict: Target dictionary
        frame_info: Frame index information (t, current_frame_idx)
        target_v: Number of target views
        av2_resize: AV2 resize parameters (resize, w, h)
    
    Returns:
        List containing sky mask frame item or empty list
    """
    t = frame_info.t
    current_frame_idx = frame_info.current_frame_idx
    resize = av2_resize.resize
    w = av2_resize.w
    h = av2_resize.h
    frame_list = []
    if "target_sky_masks" in target_dict.keys():
        sky_mask = target_dict["target_sky_masks"][0][t].unsqueeze(1)
        sky_mask = sky_mask.repeat(1, 3, 1, 1)
        sky_mask = add_label(
            hcat(
                *[
                    (
                        sky_mask[v_id]
                        if v_id != target_v // 2
                        else resize(sky_mask[v_id], (w, h))
                    )
                    for v_id in range(target_v)
                ],
                align="bottom",
            ),
            f"GT Sky&/Road Mask (t={current_frame_idx})",
            font_size=24,
            align="center",
        )
        frame_list.append(sky_mask)
    return frame_list


def _av2_resize(input_resize, size, mode="bilinear"):
    if len(input_resize.shape) == 3:
        input_resize = input_resize.unsqueeze(0)

    elif len(input_resize.shape) == 2:
        input_resize = input_resize.unsqueeze(0).unsqueeze(0)
    output = F.interpolate(input_resize, size=size, mode=mode, align_corners=False)
    return output.squeeze()


def _av2_build_context_frames(input_dict, context_t, context_v, w, h):
    """Build context frames for AV2 video visualization."""
    def denormalize(x, already_channel_last=False):
        if not already_channel_last:
            x = rearrange(x, "t v c h w -> t v h w c")
        return rearrange(x, "t v h w c -> t v c h w")

    # t, v, c, h, w
    context_images = input_dict["context_image"][0]
    context_images = denormalize(context_images)

    if context_v <= 3:
        n_ctx_per_row = 2
    else:
        n_ctx_per_row = 1
        # concate context images horizontally
    context_frames = []
    for t in range(context_t):
        current_frame_idx = int(input_dict["context_frame_idx"][0][t].item())
        row = add_label(
            hcat(
                *[
                    (
                        context_images[t][v_id]
                        if v_id != context_v // 2
                        else _av2_resize(context_images[t][v_id], (w, h))
                    )
                    for v_id in range(context_v)
                ],
                align="bottom",
            ),
            f"Context RGB (t={current_frame_idx})",
            font_size=24,
            align="center",
        )
        context_frames.append(row)
    num_rows = max(1, len(context_frames) // n_ctx_per_row)
    context_frames = vcat(
        *[
            hcat(
                *context_frames[row * n_ctx_per_row: (row + 1) * n_ctx_per_row],
                gap=24,
            )
            for row in range(num_rows)
        ]
    )
    return context_frames, n_ctx_per_row


def _av2_prepare_images(target_dict, pred_dict):
    """Prepare target and predicted images for AV2 visualization."""
    def denormalize(x, already_channel_last=False):
        if not already_channel_last:
            x = rearrange(x, "t v c h w -> t v h w c")
        return rearrange(x, "t v h w c -> t v c h w")

    target_images = target_dict["target_image"][0]
    target_images = denormalize(target_images)
    render_results = pred_dict["render_results"]
    pred_images = render_results[render_results["rgb_key"]][0]
    pred_images = denormalize(pred_images, already_channel_last=True)
    return target_images, pred_images, render_results


def _av2_build_rgb_frame(images, frame_info: FrameInfo, target_v, av2_resize: AV2ResizeInfo, label):
    t = frame_info.t
    current_frame_idx = frame_info.current_frame_idx
    w = av2_resize.w
    h = av2_resize.h
    return add_label(
        hcat(
            *[
                (
                    images[t][v_id]
                    if v_id != target_v // 2
                    else _av2_resize(images[t][v_id], (w, h))
                )
                for v_id in range(target_v)
            ],
            align="bottom",
        ),
        f"{label} (t={current_frame_idx})",
        font_size=24,
        align="center",
    )


def _av2_assemble_frame(frame_list, n_ctx_per_row, context_frames, input_dict):
    num_rows = len(frame_list) // n_ctx_per_row
    frame = vcat(
        context_frames,
        vcat(
            *[
                hcat(
                    *frame_list[row * n_ctx_per_row: (row + 1) * n_ctx_per_row],
                    gap=24,
                )
                for row in range(num_rows)
            ]
        ),
    )
    if len(frame_list) % n_ctx_per_row != 0:
        frame = vcat(
            frame,
            hcat(
                *frame_list[num_rows * n_ctx_per_row:],
                gap=24,
            ),
        )
    frame = add_border(
        add_label(
            frame,
            f"Scene{input_dict['scene_id']:03d}-{input_dict['scene_name'][:15]}",
            font_size=24,
            align="center",
        )
    )
    return prep_image(frame)


def _av2_process_frame(
        frame_info: FrameInfo, render_data: AV2FrameRenderData, tgt_size: TargetSize,
        av2_resize: AV2ResizeInfo, skip_flags: SkipFlags):
    """Process a single frame for AV2 video visualization."""
    t = frame_info.t
    target_dict = render_data.target_dict
    pred_images = render_data.pred_images
    target_images = render_data.target_images
    render_results = render_data.render_results
    target_v = render_data.target_v
    n_ctx_per_row = render_data.n_ctx_per_row
    context_frames = render_data.context_frames
    input_dict = render_data.input_dict
    current_frame_idx = frame_info.current_frame_idx
    skip_plot_gt_depth_and_flow = skip_flags.skip_plot_gt_depth_and_flow
    w = av2_resize.w
    h = av2_resize.h
    frame_list = [_av2_build_rgb_frame(
        pred_images, frame_info, target_v, av2_resize, "Predicted RGB")]
    frame_list.append(_av2_build_rgb_frame(
        target_images, frame_info, target_v, av2_resize, "Target GT RGB"))
    frame_list.extend(_av2_process_depth_decoder(
        render_results, frame_info, target_v, av2_resize))
    depth_gs_frames, alpha_image = _av2_process_depth_gs(
        render_results, frame_info, target_v, tgt_size, av2_resize)
    frame_list.extend(depth_gs_frames)
    vis_options = VisOptions(
        skip_flags=skip_flags, tgt_size=tgt_size, av2_resize=av2_resize)
    frame_list.extend(_av2_process_gt_depth(
        target_dict, frame_info, target_v, vis_options, av2_resize))
    frame_list.extend(_av2_process_flow(
        render_results, target_dict, frame_info, target_v, vis_options))
    if render_results["flow_key"] is not None and render_results["depth_key"] is not None:
        frame_list.extend(_av2_process_opacity(
            alpha_image, frame_info, target_v, av2_resize))
    if render_results["flow_key"] is not None:
        frame_list.extend(_av2_process_sky_mask(
            target_dict, frame_info, target_v, av2_resize))
    return _av2_assemble_frame(frame_list, n_ctx_per_row, context_frames, input_dict)


@torch.no_grad()
def make_video_av2(
        video_inputs: VideoInputs,
        output_filename,
        skip_plot_gt_depth_and_flow: bool = False,
):
    dataset = video_inputs.dataset
    model = video_inputs.model
    device = video_inputs.device
    scene_id = video_inputs.scene_id
    if scene_id is None:
        scene_id = np.random.randint(0, len(dataset))
    data_dict = dataset.__getitem__(scene_id, 10, return_all=True)
    data_dict = to_batch_tensor(data_dict)
    input_dict, target_dict = prepare_inputs_and_targets(
        data_dict,
        device,
    )

    with torch.no_grad():
        pred_dict = model(input_dict)
    b, context_t, context_v, _, h, w = input_dict["context_image"].shape
    _, target_t, target_v, _, h_tgt, w_tgt = target_dict["target_image"].shape

    device = input_dict["context_image"].device
    mean = torch.tensor([[MEAN]], device=device)
    std = torch.tensor([[STD]], device=device)

    reduct_mat = None
    context_frames, n_ctx_per_row = _av2_build_context_frames(
        input_dict, context_t, context_v, w, h)
    target_images, pred_images, render_results = _av2_prepare_images(target_dict, pred_dict)
    tgt_size = TargetSize(h_tgt=h_tgt, w_tgt=w_tgt)

    video_frames = []
    for t in range(target_t):
        current_frame_idx = int(target_dict["target_frame_idx"][0][t].item())
        frame_info = FrameInfo(t=t, current_frame_idx=current_frame_idx)
        av2_resize = AV2ResizeInfo(resize=_av2_resize, w=w, h=h)
        skip_flags = SkipFlags(skip_plot_gt_depth_and_flow=skip_plot_gt_depth_and_flow)
        vis_options = VisOptions(
            skip_flags=skip_flags, tgt_size=tgt_size, av2_resize=av2_resize)
        render_data = AV2FrameRenderData(
            target_dict=target_dict, pred_images=pred_images, target_images=target_images,
            render_results=render_results, target_v=target_v, n_ctx_per_row=n_ctx_per_row,
            context_frames=context_frames, input_dict=input_dict)
        video_frames.append(_av2_process_frame(
            frame_info, render_data, tgt_size, av2_resize, skip_flags))
    video_frame_reversed = video_frames[::-1][1:-1]
    video_frames.extend(video_frame_reversed)
    imageio.mimsave(output_filename, video_frames, fps=data_dict["fps"])
    return output_filename
