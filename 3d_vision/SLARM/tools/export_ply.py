# coding=utf-8

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass
from typing import Optional

from plyfile import PlyData, PlyElement
import numpy as np
import matplotlib
import torch

C0 = 0.28209479177387814


def sh_to_rgb(sh):
    return sh * C0 + 0.5


def rgb_to_sh(rgb):
    return (rgb - 0.5) / C0


def construct_dtypes_with_deal_not_fp16(features_dic, enable_gs_viewer, sh_degree):
    features_dc = features_dic["features_dc"]
    features_rest = features_dic["features_rest"]
    scale = features_dic["scale"]
    rotation = features_dic["rotation"]
    l = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    # All channels except the 3 DC
    if sh_degree > 0:
        for i in range(features_dc.shape[1] * features_dc.shape[2]):
            l.append((f"f_dc_{i}", "f4"))
    else:
        for i in range(features_dc.shape[1]):
            l.append((f"f_dc_{i}", "f4"))

    if enable_gs_viewer:
        if sh_degree > 3:
            raise ValueError("GS viewer only supports SH up to degree 3")
        if sh_degree > 0:
            sh_degree = 3
            for i in range(((sh_degree + 1) ** 2 - 1) * 3):
                l.append((f"f_rest_{i}", "f4"))
    else:
        if sh_degree > 0:
            for i in range(
                    features_rest.shape[1] * features_rest.shape[2]
            ):
                l.append((f"f_rest_{i}", "f4"))

    l.append(("opacity", "f4"))
    for i in range(scale.shape[1]):
        l.append((f"scale_{i}", "f4"))
    for i in range(rotation.shape[1]):
        l.append((f"rot_{i}", "f4"))

    return l


def construct_dtypes_with_deal_fp16(features_dic, enable_gs_viewer, sh_degree):
    features_dc = features_dic["features_dc"]
    features_rest = features_dic["features_rest"]
    scale = features_dic["scale"]
    rotation = features_dic["rotation"]
    l = [
        ("x", "f2"),
        ("y", "f2"),
        ("z", "f2"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    # All channels except the 3 DC
    for i in range(features_dc.shape[1] * features_dc.shape[2]):
        l.append((f"f_dc_{i}", "f2"))

    if sh_degree > 0:
        for i in range(
                features_rest.shape[1] * features_rest.shape[2]
        ):
            l.append((f"f_rest_{i}", "f2"))
    l.append(("opacity", "f2"))
    for i in range(scale.shape[1]):
        l.append((f"scale_{i}", "f2"))
    for i in range(rotation.shape[1]):
        l.append((f"rot_{i}", "f2"))

    return l


def construct_dtypes(features_dic, use_fp16=False, enable_gs_viewer=True, sh_degree=0):
    if not use_fp16:
        return construct_dtypes_with_deal_not_fp16(features_dic, enable_gs_viewer, sh_degree)
    else:
        return construct_dtypes_with_deal_fp16(features_dic, enable_gs_viewer, sh_degree)


def _extract_base_attributes(gaussians):
    xyz = gaussians[0, :, 0:3].contiguous().float().detach().cpu().numpy()
    rgb = gaussians[0, :, 3:6].contiguous().float()
    opacities = gaussians[0, :, 6:7].contiguous().float().detach().cpu().numpy()
    scale = gaussians[0, :, 7:10].contiguous().float().detach().cpu().numpy()
    rotation = gaussians[0, :, 10:14].contiguous().float().detach().cpu().numpy()

    return {
        "xyz": xyz,
        "rgb": rgb,
        "opacities": opacities,
        "scale": scale,
        "rotation": rotation,
    }


def _extract_optional_attributes(gaussians, semantic_start_idx, flow_start_idx, mask_indices_start_idx):
    extras = {}

    if semantic_start_idx is not None:
        extras["semantic"] = (
            gaussians[
                0,
                :,
                semantic_start_idx:semantic_start_idx + 1,
            ]
            .contiguous()
            .float()
            .detach()
            .cpu()
            .numpy()
        )

    if flow_start_idx is not None:
        extras["flow"] = (
            gaussians[
                0,
                :,
                flow_start_idx:flow_start_idx + 3,
            ]
            .contiguous()
            .float()
            .detach()
            .cpu()
            .numpy()
        )

    if mask_indices_start_idx is not None:
        extras["mask_indices"] = (
            gaussians[
                0,
                :,
                mask_indices_start_idx:mask_indices_start_idx + 1,
            ]
            .contiguous()
            .float()
            .detach()
            .cpu()
            .numpy()
        )

    return extras


def _build_sh_features(rgb, sh_degree):
    features_dc = rgb_to_sh(rgb)

    features_rest = (
        features_dc[:, 1:, :].contiguous()
        if sh_degree > 0
        else None
    )

    if sh_degree > 0:
        f_dc = (
            features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
    else:
        f_dc = features_dc.detach().contiguous().cpu().numpy()

    return features_dc, features_rest, f_dc


def _build_rgb(xyz, f_dc, color_code):
    if not color_code:
        return (
                sh_to_rgb(f_dc) * 255.0
        ).clip(0.0, 255.0).astype(np.uint8)

    index = np.linspace(0, 1, xyz.shape[0])
    rgb = matplotlib.colormaps["viridis"](index)[..., :3]
    return (rgb * 255.0).clip(0.0, 255.0).astype(np.uint8)


def _build_f_rest(features_rest, xyz, sh_degree, enable_gs_viewer):
    if sh_degree <= 0:
        return None

    f_rest = (
        features_rest.detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )

    if not enable_gs_viewer:
        return f_rest

    viewer_degree = 3
    target_dim = 3 * ((viewer_degree + 1) ** 2 - 1)

    if f_rest.shape[1] >= target_dim:
        return f_rest

    f_rest_pad = np.zeros(
        (xyz.shape[0], target_dim),
        dtype=np.float32,
    )
    f_rest_pad[:, :f_rest.shape[1]] = f_rest
    return f_rest_pad


def _build_attributes(data):
    xyz = data["xyz"]
    rgb = data["rgb"]
    f_dc = data["f_dc"]
    f_rest = data["f_rest"]
    opacities = data["opacities"]
    scale = data["scale"]
    rotation = data["rotation"]
    parts = [xyz, rgb, f_dc]

    if f_rest is not None:
        parts.append(f_rest)

    parts.extend([opacities, scale, rotation])

    return np.concatenate(parts, axis=1)


def _append_optional_attributes(attributes, extras):
    if "semantic" in extras:
        attributes = np.concatenate(
            (attributes, extras["semantic"]),
            axis=1,
        )

    if "flow" in extras:
        attributes = np.concatenate(
            (attributes, extras["flow"]),
            axis=1,
        )

    if "mask_indices" in extras:
        attributes = np.concatenate(
            (attributes, extras["mask_indices"]),
            axis=1,
        )

    return attributes


@dataclass
class PlyExportConfig:
    """Configuration for PLY file export."""
    use_fp16: bool = False
    enable_gs_viewer: bool = True
    color_code: bool = False
    sh_degree: int = 0
    semantic_start_idx: Optional[int] = None
    flow_start_idx: Optional[int] = None
    mask_indices_start_idx: Optional[int] = None


def save_ply(gaussians, path, config: Optional[PlyExportConfig] = None):
    if gaussians.shape[0] != 1:
        raise ValueError('only support batch size 1')

    if config is None:
        config = PlyExportConfig()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    base_attrs = _extract_base_attributes(gaussians)
    extras = _extract_optional_attributes(
        gaussians,
        config.semantic_start_idx,
        config.flow_start_idx,
        config.mask_indices_start_idx
    )
    features_dc, features_rest, f_dc = _build_sh_features(base_attrs["rgb"], config.sh_degree)
    rgb = _build_rgb(base_attrs["xyz"], f_dc, config.color_code)
    scale = np.log(base_attrs["scale"])
    opacities = np.log(base_attrs["opacities"] / (1 - base_attrs["opacities"]))

    features_dic = {
        "features_dc": features_dc,
        "features_rest": features_rest,
        "scale": scale,
        "rotation": base_attrs["rotation"]
    }
    dtype_full = construct_dtypes(features_dic,
                                  config.use_fp16, config.enable_gs_viewer, sh_degree=config.sh_degree)
    if config.semantic_start_idx is not None:
        dtype_full.append(('semantic', 'u1'))
    if config.flow_start_idx is not None:
        dtype_full.extend([('flow_x', 'f4'), ('flow_y', 'f4'), ('flow_z', 'f4')])
    if config.mask_indices_start_idx is not None:
        dtype_full.append(('mask_indices', 'u4'))
    elements = np.empty(base_attrs["xyz"].shape[0], dtype=dtype_full)

    f_rest = _build_f_rest(features_rest, base_attrs["xyz"], config.sh_degree, config.enable_gs_viewer)
    attributes = _build_attributes({
        "xyz": base_attrs["xyz"],
        "rgb": rgb,
        "f_dc": f_dc,
        "f_rest": f_rest,
        "opacities": opacities,
        "scale": scale,
        "rotation": base_attrs["rotation"]
    })
    attributes = _append_optional_attributes(attributes, extras)

    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)
