# coding=utf-8
# Adapted from
# https://github.com/NVlabs/GaussianSTORM/blob/main/storm/visualization/annotation.py

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) NVIDIA. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
From https://github.com/dcharatan/pixelsplat/blob/main/src/visualization/annotation.py
"""

from pathlib import Path
from string import ascii_letters, digits, punctuation

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from .layout import vcat

EXPECTED_CHARACTERS = digits + punctuation + ascii_letters


def draw_label(
    text: str,
    font: Path,
    font_size: int,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "3 height width"]:
    """Draw a black label on a white background with no border."""
    try:
        font = ImageFont.truetype(str(font), font_size)
    except OSError:
        font = ImageFont.load_default()
    left, _, right, _ = font.getbbox(text)
    width = right - left
    _, top, _, bottom = font.getbbox(EXPECTED_CHARACTERS)
    height = bottom - top
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font, fill="black")
    image = torch.tensor(np.array(image) / 255, dtype=torch.float32, device=device)
    return rearrange(image, "h w c -> c h w")


def add_label(
    image: Float[Tensor, "3 width height"],
    label: str,
    font: Path = Path("assets/Inter-Regular.otf"),
    font_size: int = 24,
    align: str = "center",
) -> Float[Tensor, "3 width_with_label height_with_label"]:
    return vcat(
        draw_label(label, font, font_size, image.device),
        image,
        align=align,
        gap=4,
    )
