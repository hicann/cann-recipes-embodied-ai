# coding=utf-8
# Adapted from
# https://github.com/NVlabs/GaussianSTORM/blob/main/storm/dataset/data_utils.py

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) NVIDIA. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn.functional as F
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from src.dataset.constants import SEMANTIC_LABEL_LIST

if os.getenv("FEAT_DIST"):
    from tools.feats_tools import get_text_label_feats


def depth2xyz(depth, fxfycxcy, cam2world=None, return_pixel=False):
    h, w = depth.shape

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x = (x + 0.5) / w
    y = (y + 0.5) / h
    x = (x - fxfycxcy[2:3]) / fxfycxcy[0:1]
    y = (y - fxfycxcy[3:4]) / fxfycxcy[1:2]
    z = np.ones_like(x)
    ray_d = np.stack([x, y, z], axis=2)  # [b*v, h*w, 3]

    if cam2world is not None:
        ray_d = np.matmul(ray_d, cam2world[:3, :3].T)  # [b*v, h*w, 3]  a @ b.T
        ray_o = cam2world[:3, 3]

    xyz = ray_d * depth[..., None]

    if cam2world is not None:
        xyz = ray_o + ray_d * depth[..., None]

    if return_pixel:
        return xyz
    else:
        return xyz.reshape(-1, 3)


def xyz2depth(points, xyz2img_rt, height, width, rgb=None):
    """
    Convert 3D points to a depth map using the camera projection matrix.

    cam_intrinsic = torch.eye(4)
    cam_intrinsic[:3, :3] = Ks_batched[0, 0]
    xyz2img_rt = (cam_intrinsic @ camtoworlds_batched.float()[0, 0]
                  .inverse().to(cam_intrinsic.device))
    depth_map = xyz2depth(means_batched[0].to(cam_intrinsic.device),
                          xyz2img_rt, height=tgt_h, width=tgt_w)
    """
    pts_4d = torch.concat([points[:, :3], torch.ones((points.shape[0], 1))], dim=-1)
    pts_2d = pts_4d @ xyz2img_rt.T

    # NOTE: No need for 4 dimensions after matrix computation
    pts_2d = pts_2d[:, :3]

    # select points in front of the camera
    if rgb is not None:
        rgb = rgb[pts_2d[:, 2] > 0]
    pts_2d = pts_2d[pts_2d[:, 2] > 0]

    # normalize pixel points : (u,v,1)
    img_pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    # filter out points outside the image
    fov_inds = ((img_pts_2d[:, 0] < width)
                & (img_pts_2d[:, 0] >= 0)
                & (img_pts_2d[:, 1] < height)
                & (img_pts_2d[:, 1] >= 0))
    imgfov_pts_2d = img_pts_2d[fov_inds]
    pts_2d = pts_2d[fov_inds]
    if rgb is not None:
        rgb = rgb[fov_inds]
    # compute depth map
    depth_map = torch.zeros((height, width))
    depth_map[
        imgfov_pts_2d[:, 1].to(torch.int32),
        imgfov_pts_2d[:, 0].to(torch.int32),
    ] = pts_2d[:, 2]

    if rgb is not None:
        new_rgb = torch.zeros((height, width, 3))
        new_rgb[
            imgfov_pts_2d[:, 1].to(torch.int32),
            imgfov_pts_2d[:, 0].to(torch.int32),
        ] = rgb
        return depth_map, new_rgb
    else:
        return depth_map


def forward_pose(pose: torch.Tensor, transform: torch.Tensor,
                 inv=False) -> torch.Tensor:
    rotation = pose[..., :3, :3]
    translation = pose[..., :3, 3]
    if inv:
        return (torch.transpose(rotation, -1, -2)
                * (transform - translation).unsqueeze(-2)).sum(-1)
    else:
        return (rotation * transform.unsqueeze(-2)).sum(-1) + translation


def to_tensor(x: Union[np.ndarray, List, Tuple]) -> torch.Tensor:
    if isinstance(x, (list, tuple)):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    return x


def to_float_tensor(d):
    if isinstance(d, dict):
        return {k: to_float_tensor(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [to_float_tensor(v) for v in d]
    elif isinstance(d, torch.Tensor):
        return d.float()
    elif isinstance(d, np.ndarray):
        return torch.from_numpy(d).float()
    else:
        return d


def to_batch_tensor(d):
    if isinstance(d, dict):
        return {k: to_batch_tensor(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [to_batch_tensor(v) for v in d]
    elif isinstance(d, torch.Tensor):
        return d.unsqueeze(0)
    else:
        return d


def resize_depth(depth, target_size):
    height, width = depth.shape[-2:]
    if (height, width) == target_size:
        return depth
    if len(depth.shape) == 2:
        depth = depth[None, None, ...]
    elif len(depth.shape) == 3:
        depth = depth[None, ...]
    target_height, target_width = target_size
    kernel_size_h = height // target_height
    kernel_size_w = width // target_width

    if kernel_size_h > 0 and kernel_size_w > 0:
        depth = F.max_pool2d(
            depth,
            kernel_size=(kernel_size_h, kernel_size_w),
        )
    depth = F.interpolate(depth, size=target_size, mode="nearest")
    return depth.squeeze()


def resize_flow(flow, target_size):
    height, width = flow.shape[-3:-1]
    if (height, width) == target_size:
        return flow
    if len(flow.shape) == 3:
        flow = flow[None, ...]
    target_height, target_width = target_size
    kernel_size_h = height // target_height
    kernel_size_w = width // target_width
    # flows have direction, so we can't just use max_pool2d.
    # otherwise the direction will be wrong, e.g., max_pool2d([0, -1]) = [0]
    flow[torch.norm(flow, p=2, dim=-1) < 0.5] = -100000
    if kernel_size_h > 0 and kernel_size_w > 0:
        flow = F.max_pool2d(
            flow.permute(0, 3, 1, 2),
            kernel_size=(kernel_size_h, kernel_size_w),
        )
        flow = F.interpolate(flow, size=target_size, mode="nearest")
    else:
        flow = F.interpolate(
            flow.permute(0, 3, 1, 2), size=target_size, mode="nearest"
        )
    flow = flow.permute(0, 2, 3, 1)
    flow[torch.norm(flow, p=2, dim=-1) > 1000] = 0
    return flow.squeeze()


@dataclass
class DatasetShape:
    b: int
    context_t: int
    v: int
    target_t: int
    h: int
    w: int


@dataclass
class DatasetConfig:
    timespan: float
    feat_extractor: object
    is_vis: bool
    device: torch.device


def _build_input_dict(data_dict, shape: DatasetShape, config: DatasetConfig):
    """Build the input dict from raw data_dict."""
    input_dict = {
        "context_image": data_dict["context"]["image"].reshape(
            shape.b, shape.context_t, shape.v, 3, shape.h, shape.w
        ),
        "context_camtoworlds": data_dict["context"]["camtoworld"].reshape(
            shape.b, shape.context_t, shape.v, 4, 4
        ),
        "context_intrinsics": data_dict["context"]["intrinsics"].reshape(
            shape.b, shape.context_t, shape.v, 3, 3
        ),
        "target_camtoworlds": data_dict["target"]["camtoworld"].reshape(
            shape.b, shape.target_t, shape.v, 4, 4
        ),
        "target_intrinsics": data_dict["target"]["intrinsics"].reshape(
            shape.b, shape.target_t, shape.v, 3, 3
        ),
    }

    _add_context_optional_fields(input_dict, data_dict, shape.b, shape.context_t,
                                 shape.v, config.timespan, shape.h, shape.w)
    if "time" in data_dict["target"]:
        input_dict["target_time"] = (
                data_dict["target"]["time"].reshape(shape.b, shape.target_t, shape.v) / config.timespan
        )
    _add_context_semantic_feats(input_dict, data_dict, shape.b, shape.context_t,
                                shape.v, config.feat_extractor, config.is_vis,
                                config.device, shape.h, shape.w)

    # Move tensors to device
    input_dict = {k: v.to(config.device) for k, v in input_dict.items()}
    input_dict["timespan"] = config.timespan
    input_dict["fps"] = data_dict["fps"]
    input_dict["scene_id"] = data_dict["scene_id"]
    input_dict["scene_name"] = data_dict["scene_name"]
    input_dict["height"], input_dict["width"] = shape.h, shape.w
    if 'segment_to_ref' in data_dict:
        input_dict['segment_to_ref'] = data_dict['segment_to_ref']
    input_dict["context_frame_idx"] = torch.as_tensor(
        data_dict["context"]["frame_idx"], device=config.device
    )
    input_dict["target_frame_idx"] = torch.as_tensor(
        data_dict["target"]["frame_idx"], device=config.device
    )
    return input_dict


def _add_context_optional_fields(input_dict, data_dict, b, context_t, v,
                                 timespan, h, w):
    """Add optional context fields to input_dict."""
    ctx = data_dict["context"]
    tgt_shape_d = ctx.get("depth")
    tgt_shape_pts = ctx.get("pts3d")
    tgt_shape_mask = ctx.get("valid_masks")
    tgt_shape_flow = ctx.get("flow")
    tgt_shape_pseudo_d = ctx.get("pseudo_depth")
    tgt_shape_pseudo_dc = ctx.get("pseudo_depth_conf")

    if tgt_shape_d is not None:
        dh, dw = tgt_shape_d.shape[-2:]
        input_dict["context_depth"] = ctx["depth"].reshape(
            b, context_t, v, dh, dw
        )
    if tgt_shape_pts is not None:
        ph, pw = tgt_shape_pts.shape[-3:-1]
        input_dict['context_pts3d'] = ctx['pts3d'].reshape(
            b, context_t, v, ph, pw, 3
        )
    if tgt_shape_mask is not None:
        mh, mw = tgt_shape_mask.shape[-2:]
        input_dict['context_valid_masks'] = ctx['valid_masks'].reshape(
            b, context_t, v, mh, mw
        )
    if tgt_shape_flow is not None:
        fh, fw = tgt_shape_flow.shape[-3:-1]
        input_dict["context_flow"] = ctx["flow"].reshape(
            b, context_t, v, fh, fw, 3
        )
    if "time" in ctx:
        input_dict["context_time"] = (
                ctx["time"].reshape(b, context_t, v) / timespan
        )
    if "sky_masks" in ctx:
        input_dict["context_sky_masks"] = ctx["sky_masks"].reshape(
            b, context_t, v, h, w
        )
    if tgt_shape_pseudo_d is not None:
        pdh, pdw = tgt_shape_pseudo_d.shape[-2:]
        input_dict["context_pseudo_depth"] = ctx["pseudo_depth"].reshape(
            b, context_t, v, pdh, pdw
        )
    if tgt_shape_pseudo_dc is not None:
        pdc_h, pdc_w = tgt_shape_pseudo_dc.shape[-2:]
        input_dict['context_pseudo_depth_conf'] = (
            ctx['pseudo_depth_conf'].reshape(
                b, context_t, v, pdc_h, pdc_w
            )
        )


def _add_context_semantic_feats(input_dict, data_dict, b, context_t, v,
                                feat_extractor, is_vis, device, h, w):
    """Add context semantic features to input_dict."""
    ctx = data_dict["context"]

    if os.getenv("CONTEXT_FEAT") and "semantic_labels" in ctx:
        ctx_sem_labels = ctx["semantic_labels"].reshape(
            b, context_t, v, h, w
        ).to(torch.int32)
        ctx_sem_mask = ctx["semantic_labels_mask"].reshape(
            b, context_t, v
        )
        if ctx_sem_labels.max() > 0:
            input_dict["context_semantic_labels"] = ctx_sem_labels
            input_dict["context_semantic_labels_mask"] = ctx_sem_mask
        if is_vis:
            semantic_feats = torch.zeros(
                ctx_sem_labels.shape
                + (get_text_label_feats(SEMANTIC_LABEL_LIST).shape[-1],)
            )
            semantic_feats[:] = get_text_label_feats(SEMANTIC_LABEL_LIST)[
                ctx_sem_labels
            ]
            semantic_feats = einops.rearrange(
                semantic_feats, 'b t v h w c -> b t v c h w', b=b, v=v
            )
            input_dict["context_feat"] = semantic_feats

    if os.getenv("CONTEXT_FEAT") and "frame_images_to_extract_feat" in ctx:
        if feat_extractor is None:
            raise ValueError("feat_extractor must not be None")
        images_to_extract_feat = ctx["frame_images_to_extract_feat"]
        images_to_extract_feat = einops.rearrange(images_to_extract_feat, 'b tv c h w -> (b tv) c h w')
        with torch.no_grad():
            feat = feat_extractor.extract_lseg_feat_by_chunk(images_to_extract_feat.to(device), (h, w))
        feat = einops.rearrange(feat, '(b t v) c h w -> b t v c h w', b=b, v=v)
        input_dict["context_feat"] = feat


def _build_target_dict(data_dict, b, context_t, v, target_t,
                       timespan, feat_extractor, is_vis,
                       device, h, w):
    """Build the target dict from raw data_dict."""
    depth_h, depth_w = h, w
    if "depth" in data_dict["target"]:
        depth_h, depth_w = data_dict["target"]["depth"].shape[-2:]

    target_dict = {
        "target_image": data_dict["target"]["image"].reshape(
            b, target_t, v, 3, h, w
        ),
    }

    tgt = data_dict["target"]
    _add_target_rendering_fields(target_dict, data_dict, b, context_t, v,
                                 target_t, depth_h, depth_w)
    _add_target_semantic_feats(target_dict, data_dict, b, target_t, v,
                               feat_extractor, is_vis, device, h, w)

    target_dict["target_frame_idx"] = torch.as_tensor(
        data_dict["target"]["frame_idx"], device=device
    )
    target_dict = {k: v.to(device) for k, v in target_dict.items()}
    return target_dict


def _add_target_rendering_fields(target_dict, data_dict, b, context_t, v,
                                 target_t, depth_h, depth_w):
    """Add depth/flow/mask fields to target_dict."""
    tgt = data_dict["target"]
    ctx = data_dict["context"]

    if "depth" in tgt:
        target_dict["target_depth"] = tgt["depth"].reshape(
            b, target_t, v, depth_h, depth_w
        )
    if "flow" in tgt:
        target_dict["target_flow"] = tgt["flow"].reshape(
            b, target_t, v, depth_h, depth_w, 3
        )
    if "flow" in ctx:
        target_dict["context_flow"] = ctx["flow"].reshape(
            b, context_t, v, depth_h, depth_w, 3
        )
    if "sky_masks" in tgt:
        target_dict["target_sky_masks"] = tgt["sky_masks"].reshape(
            b, target_t, v, depth_h, depth_w
        )
    if "sky_masks" in ctx:
        target_dict["context_sky_masks"] = ctx["sky_masks"].reshape(
            b, context_t, v, depth_h, depth_w
        )
    if "dynamic_masks" in tgt:
        target_dict["target_dynamic_masks"] = tgt["dynamic_masks"].reshape(
            b, target_t, v, depth_h, depth_w
        )
    if "ground_masks" in tgt:
        target_dict["target_ground_masks"] = tgt["ground_masks"].reshape(
            b, target_t, v, depth_h, depth_w
        )
    _add_target_pseudo_depth(target_dict, data_dict, b, context_t, v,
                             target_t)


def _add_target_pseudo_depth(target_dict, data_dict, b, context_t, v,
                             target_t):
    """Add pseudo-depth fields if present."""
    tgt = data_dict["target"]
    if "pseudo_depth" in tgt:
        pdh, pdw = tgt["pseudo_depth"].shape[-2:]
        target_dict["target_pseudo_depth"] = tgt["pseudo_depth"].reshape(
            b, target_t, v, pdh, pdw
        )
    if "pseudo_depth_conf" in tgt:
        pdc_h, pdc_w = tgt["pseudo_depth_conf"].shape[-2:]
        target_dict['target_pseudo_depth_conf'] = (
            tgt['pseudo_depth_conf'].reshape(
                b, target_t, v, pdc_h, pdc_w
            )
        )


def _add_target_semantic_feats(target_dict, data_dict, b, target_t, v,
                               feat_extractor, is_vis, device, h, w):
    """Add target semantic/feature fields to target_dict."""
    tgt = data_dict["target"]

    if "semantic_labels" in tgt:
        tgt_sem_labels = tgt["semantic_labels"].reshape(
            b, target_t, v, h, w
        ).to(torch.int32)
        tgt_sem_mask = tgt["semantic_labels_mask"].reshape(
            b, target_t, v
        )
        if tgt_sem_labels.max() > 0:
            target_dict["target_semantic_labels"] = tgt_sem_labels
            target_dict["target_semantic_labels_mask"] = tgt_sem_mask
        if is_vis:
            semantic_feats = torch.zeros(
                tgt_sem_labels.shape
                + (get_text_label_feats(SEMANTIC_LABEL_LIST).shape[-1],)
            )
            semantic_feats[:] = get_text_label_feats(SEMANTIC_LABEL_LIST)[
                tgt_sem_labels
            ]
            semantic_feats = einops.rearrange(
                semantic_feats, 'b t v h w c -> b t v c h w', b=b, v=v
            )
            target_dict["target_feat"] = semantic_feats

    if not os.getenv("CONTEXT_FEAT") and "frame_images_to_extract_feat" in tgt:
        if feat_extractor is None:
            raise ValueError("feat_extractor must not be None")
        images_to_extract_feat = tgt["frame_images_to_extract_feat"]
        images_to_extract_feat = einops.rearrange(images_to_extract_feat, 'b tv c h w -> (b tv) c h w')
        with torch.no_grad():
            feat = feat_extractor.extract_lseg_feat_by_chunk(images_to_extract_feat.to(device), (h, w))
        feat = einops.rearrange(feat, '(b t v) c h w -> b t v c h w', b=b, v=v)
        target_dict["target_feat"] = feat


# ---------------------------------------------------------------------------
# Main data preparation entry point
# ---------------------------------------------------------------------------

def prepare_inputs_and_targets(
        data_dict,
        device=torch.device("cuda"),
        v=3,
        timespan=2.0,
        feat_extractor=None,
        is_vis=False,
):
    """Build input and target dicts from raw data, moved to *device*."""
    if data_dict["context"]["image"].dim() != 5:
        raise ValueError("need to be b, tv, c, h, w")
    b, tv, c, h, w = data_dict["context"]["image"].shape
    context_t = tv // v
    target_t = data_dict["target"]["image"].shape[1] // v

    input_dict = _build_input_dict(
        data_dict,
        DatasetShape(b=b, context_t=context_t, v=v, target_t=target_t, h=h, w=w),
        DatasetConfig(timespan=timespan, feat_extractor=feat_extractor,
                      is_vis=is_vis, device=device),
    )

    target_dict = _build_target_dict(
        data_dict, b, context_t, v, target_t, timespan,
        feat_extractor, is_vis, device, h, w,
    )

    return input_dict, target_dict


# ---------------------------------------------------------------------------
# Legacy / experimental helpers (kept for reference, not used in main flow)
# ---------------------------------------------------------------------------

def prepare_inputs_and_targets_novel_view(data_dict,
                                          device=torch.device("cpu")):
    """Build inputs/targets for novel-view rendering (experimental).

    NOTE: This function is kept for reference but is not currently called
    in the main evaluation / inference pipelines.
    """
    raise NotImplementedError(
        "Novel-view preparation is not currently integrated. "
        "Use prepare_inputs_and_targets instead."
    )
