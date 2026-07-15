# coding=utf-8
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Shared dataclass types for 3D Gaussian rendering."""

from dataclasses import dataclass
from typing import Optional

from torch import Tensor
from typing_extensions import Literal


@dataclass
class ProjectionInput:
    means: Tensor  # [..., N, 3]
    covars: Tensor  # [..., N, 3, 3]


@dataclass
class CameraParams:
    viewmats: Tensor  # [..., C, 4, 4]
    ks: Tensor  # [..., C, 3, 3]
    camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"] = "pinhole"


@dataclass
class ProjectionConfig:
    width: int
    height: int
    eps2d: float = 0.3
    near_plane: float = 0.01
    far_plane: float = 1e10
    calc_compensations: bool = False


@dataclass
class ProjectionOutput:
    radii: Tensor
    means2d: Tensor
    depths: Tensor
    conics: Tensor
    compensations: Optional[Tensor]
    covars2d: Tensor


@dataclass
class GaussianData:
    means: Tensor  # [N, 3]
    quats: Tensor  # [N, 4]
    scales: Tensor  # [N, 3]
    opacities: Tensor  # [N]
    colors: Tensor  # [N, D]


@dataclass
class RenderConfig:
    width: int
    height: int
    near_plane: float = 0.01
    far_plane: float = 1e10
    eps2d: float = 0.3
    sh_degree: Optional[int] = None
    tile_size: int = 64
    backgrounds: Optional[Tensor] = None
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB"
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    channel_chunk: int = 32
    batch_per_iter: int = 100
    packed: bool = False
    radius_clip: float = 0.0


@dataclass
class ProjectionResult:
    """Output of projection_three_dims_gaussian_fused for tile build & render."""
    means2d: Tensor
    opacities: Tensor
    conics: Tensor
    covars2d: Tensor
    depths: Tensor
    cnt: Tensor
    colors: Tensor


@dataclass
class SortData:
    """Sorted gaussian indices and scheduling metadata."""
    sorted_gs_ids: Tensor
    sorted_offset: Tensor
    render_sched_tensor: Tensor
