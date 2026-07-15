# coding=utf-8

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import importlib
from typing import NamedTuple

import torch
from torch.autograd import Function

META_GAUSS_RENDER_OPS = importlib.import_module("meta_gauss_render._C")


@dataclass
class ProjectionInputs:
    means: torch.Tensor
    colors: torch.Tensor
    covars: torch.Tensor = None
    quat: torch.Tensor = None
    scales: torch.Tensor = None
    opacities: torch.Tensor = None
    viewmats: torch.Tensor = None
    ks: torch.Tensor = None
    width: int = 0
    height: int = 0
    eps: float = 0.3
    near_plane: float = 0.01
    far_plane: float = 1e10
    calc_compensations: bool = False
    camera_model: str = "pinhole"

    @classmethod
    def from_args(cls, args):
        if len(args) < 2:
            raise ValueError("'means' and 'colors' are required.")
        defaults = [
            None, None, None, None, None, None, 0, 0, 0.3, 0.01, 1e10,
            False, "pinhole",
        ]
        values = list(args[:2]) + list(args[2:]) + defaults[len(args[2:]):]
        if len(values) != len(cls.__dataclass_fields__):
            raise ValueError(f"Expected at most 15 arguments, got {len(args)}.")
        return cls(*values)


class ProjectionOutputs(NamedTuple):
    means2d: torch.Tensor
    depths: torch.Tensor
    conics: torch.Tensor
    opacities: torch.Tensor
    radius: torch.Tensor
    covars2d: torch.Tensor
    colors: torch.Tensor
    cnt: torch.Tensor


def decode_uint8_bitmask(filter_mask: torch.Tensor, high_order=True) -> torch.Tensor:
    """
    Decode uint8 single-bit mask (shape: 1,1,N/8) to bool mask (shape: 1,1,N)

    Args:
        filter_mask: uint8 Tensor, shape=(1,1,N/8)

    Returns:
        bool Tensor, shape=(1,1,N)
    """
    # 1. Squeeze redundant dimensions: (1,1,N/8) -> (N/8,)
    mask_squeezed = filter_mask.squeeze()  # shape: (N/8,)

    # 2. Generate bit masks: extract 8 bits from each uint8
    if high_order:
        # Mask example: [128, 64, 32, 16, 8, 4, 2, 1] -> corresponds to bits 7 to 0 (high -> low)
        bit_masks = (1 << torch.arange(7, -1, -1)).to(filter_mask.device, dtype=torch.uint8)  # shape: (8,)
    else:
        bit_masks = (1 << torch.arange(0, 8)).to(filter_mask.device, dtype=torch.uint8)  # shape: (8,)

    # 3. Extract 8 bits from each byte: broadcast computation (N/8,) x (8,) -> (N/8, 8)
    # bit_values: each element is 0 or 1, shape=(N/8,8)
    bit_values = (mask_squeezed.unsqueeze(1) & bit_masks) != 0

    # 4. Flatten and reshape to target shape: (N/8x8,) -> (1,1,N)
    bool_mask = bit_values.flatten().reshape(1, 1, -1)  # shape: (1,1,N)

    return bool_mask


class ProjectionThreeDimsGaussianFused(Function):
    @staticmethod
    def forward(ctx, *args):
        inputs = ProjectionInputs.from_args(args)
        prepared = _prepare_projection_inputs(inputs)
        projection = _run_projection(prepared)
        filtered = _run_gaussian_filter(prepared, projection)

        ctx.save_for_backward(
            prepared.means, projection.conics, prepared.viewmats,
            prepared.quat, prepared.scales, prepared.ks, filtered.filter_mask,
            projection.compensations,
        )
        ctx.width = prepared.width
        ctx.height = prepared.height
        return ProjectionOutputs(
            filtered.means2d, filtered.depths, filtered.conics,
            filtered.opacities, filtered.radius, filtered.covars2d,
            filtered.colors, filtered.cnt,
        )

    @staticmethod
    def backward(
        ctx, *v_args
    ):
        means, conics, viewmats, quats, scales, ks, filter_mask, compensations = ctx.saved_tensors

        width = ctx.width
        height = ctx.height
        v_means2d, v_depths, v_conics, v_opacities_culling, v_radii, v_covars2d, v_colors_culling, v_cnt = v_args
        v_pw, v_quats, v_scales, _, v_colors, v_opacities = META_GAUSS_RENDER_OPS.fully_fused_projection_bwd(
                means,
                quats,
                scales,
                conics,
                viewmats,
                ks,
                v_means2d,
                v_depths,
                v_conics,
                v_colors_culling,
                v_opacities_culling,
                filter_mask,
                compensations,
                width,
                height
        )

        filter_mask = decode_uint8_bitmask(filter_mask, high_order=False)

        v_pw = torch.where(~filter_mask.unsqueeze(-1), torch.tensor(0.0, device=v_pw.device), v_pw)
        v_quats = torch.where(~filter_mask.unsqueeze(-1), torch.tensor(0.0, device=v_quats.device), v_quats)
        v_scales = torch.where(~filter_mask.unsqueeze(-1), torch.tensor(0.0, device=v_scales.device), v_scales)
        v_colors = torch.where(~filter_mask, torch.tensor(0.0, device=v_colors.device), v_colors)
        v_opacities = torch.where(~filter_mask.squeeze(1), torch.tensor(0.0, device=v_opacities.device), v_opacities)

        gradients = (
            v_pw, v_colors, None, v_quats, v_scales, v_opacities,
            None, None, None, None, None, None, None, None, None,
        )
        return gradients


@dataclass
class PreparedProjectionInputs:
    means: torch.Tensor
    colors: torch.Tensor
    covars: torch.Tensor
    quat: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    viewmats: torch.Tensor
    ks: torch.Tensor
    width: int
    height: int
    eps: float
    near_plane: float
    far_plane: float
    calc_compensations: bool
    camera_model: str


@dataclass
class ProjectionIntermediate:
    means2d: torch.Tensor
    depths: torch.Tensor
    conics: torch.Tensor
    compensations: torch.Tensor
    det: torch.Tensor
    radius: torch.Tensor
    covars2d: torch.Tensor


@dataclass
class FilteredGaussians:
    colors: torch.Tensor
    means2d: torch.Tensor
    depths: torch.Tensor
    radius: torch.Tensor
    covars2d: torch.Tensor
    conics: torch.Tensor
    opacities: torch.Tensor
    filter_mask: torch.Tensor
    cnt: torch.Tensor


def _prepare_projection_inputs(inputs):
    quat = inputs.quat
    scales = inputs.scales
    covars = inputs.covars

    if quat is not None:
        if scales is None:
            raise ValueError("'quat' and 'scales' are required together.")
        if covars is not None:
            raise ValueError(
                "Invalid parameter combination: 'covars' and pair are "
                "mutually exclusive."
            )
        quat = quat.permute((0, 2, 1)).contiguous()
        scales = scales.permute((0, 2, 1)).contiguous()
        covars = META_GAUSS_RENDER_OPS.quat_scales_to_covars(quat, scales)
    else:
        covars = covars.permute((0, 2, 3, 1)).contiguous()

    return PreparedProjectionInputs(
        means=inputs.means.permute((0, 2, 1)).contiguous(),
        colors=inputs.colors,
        covars=covars,
        quat=quat,
        scales=scales,
        opacities=inputs.opacities,
        viewmats=inputs.viewmats,
        ks=inputs.ks,
        width=inputs.width,
        height=inputs.height,
        eps=inputs.eps,
        near_plane=inputs.near_plane,
        far_plane=inputs.far_plane,
        calc_compensations=inputs.calc_compensations,
        camera_model=inputs.camera_model,
    )


def _run_projection(inputs):
    means2d, depths, conics, compensations, det, radius, covars2d = (
        META_GAUSS_RENDER_OPS.projection_three_dims_gaussian_forward(
            inputs.means, inputs.covars, inputs.opacities, inputs.viewmats,
            inputs.ks, inputs.width, inputs.height, inputs.eps,
            inputs.calc_compensations, inputs.camera_model,
        )
    )

    compensations = (
        compensations.squeeze(-2) if inputs.calc_compensations else None
    )
    return ProjectionIntermediate(
        means2d=means2d,
        depths=depths.squeeze(-2),
        conics=conics,
        compensations=compensations,
        det=det.squeeze(-2),
        radius=radius,
        covars2d=covars2d,
    )


def _run_gaussian_filter(inputs, projection):
    _, colors, means2d, depths, radius, covars2d, conics, opacities, \
        filter_mask, cnt = META_GAUSS_RENDER_OPS.gaussian_filter(
            inputs.means, inputs.colors, projection.det, inputs.opacities,
            projection.means2d, projection.depths, projection.radius,
            projection.conics, projection.covars2d, projection.compensations,
            inputs.width, inputs.height, inputs.near_plane, inputs.far_plane,
        )
    return FilteredGaussians(
        colors=colors,
        means2d=means2d,
        depths=depths,
        radius=radius,
        covars2d=covars2d,
        conics=conics,
        opacities=opacities,
        filter_mask=filter_mask,
        cnt=cnt,
    )

projection_three_dims_gaussian_fused = ProjectionThreeDimsGaussianFused.apply
