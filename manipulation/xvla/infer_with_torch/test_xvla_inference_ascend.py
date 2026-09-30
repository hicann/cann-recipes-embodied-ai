#!/usr/bin/env python3
#
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
"""XVLA NPU checkpoint inference script."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.utils.constants import IMAGENET_STATS, OBS_LANGUAGE_TOKENS, OBS_STATE


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)


@dataclass
class LiberoHdf5Sample:
    hdf5_path: str
    demo_key: str | None
    frame_index: int
    task: str
    domain_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XVLA inference on Ascend NPU")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--domain_id", type=int, default=0)
    parser.add_argument("--num_warmup", type=int, default=1)
    parser.add_argument("--num_inference", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--libero_hdf5", type=str, default=None, help="Optional LIBERO HDF5 demo file")
    parser.add_argument("--demo_key", type=str, default=None, help="HDF5 demo key, for example demo_0")
    parser.add_argument("--frame_index", type=int, default=0, help="Frame index to read from the selected demo")
    parser.add_argument(
        "--task",
        type=str,
        default="put the black bowl in the bottom drawer of the cabinet and close it",
        help="Language instruction used when tokenizing the LIBERO sample",
    )
    parser.add_argument("--save_action", type=str, default=None, help="Optional path to save the final action tensor")
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def ensure_torch_npu(device: torch.device) -> None:
    if device.type != "npu":
        return
    try:
        importlib.import_module("torch_npu")
    except ImportError as exc:
        raise RuntimeError("torch_npu is required when --device uses npu") from exc


def load_policy(args: argparse.Namespace, device: torch.device) -> XVLAPolicy:
    config = PreTrainedConfig.from_pretrained(args.pretrained_model_name_or_path)
    config.device = str(device)
    config.dtype = args.dtype
    policy = XVLAPolicy.from_pretrained(args.pretrained_model_name_or_path, config=config)

    policy.to(device)
    policy.eval()
    return policy


def make_mock_batch(
    policy: XVLAPolicy,
    device: torch.device,
    dtype: torch.dtype,
    domain_id: int,
) -> dict[str, torch.Tensor]:
    state_dim = policy.model.dim_proprio
    state = torch.randn(1, state_dim, device=device, dtype=dtype)
    tokens = torch.tensor([[2, 4, 5, 6, 7, 3, 0, 0]], dtype=torch.long, device=device)
    batch = {
        OBS_LANGUAGE_TOKENS: tokens,
        OBS_STATE: state,
        "domain_id": torch.full((1,), int(domain_id), dtype=torch.long, device=device),
    }
    for image_key, image_feature in policy.config.image_features.items():
        batch[image_key] = torch.rand(1, *image_feature.shape, device=device, dtype=dtype)
    return batch


def tokenize_task(policy: XVLAPolicy, task: str, device: torch.device) -> torch.Tensor:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(policy.config.tokenizer_name, local_files_only=True)
        tokenizer.padding_side = policy.config.tokenizer_padding_side
        tokenized = tokenizer(
            task,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=policy.config.tokenizer_max_length,
        )
        return tokenized["input_ids"].to(device=device)
    except ImportError as exc:
        logger.warning("Falling back to fixed language tokens because tokenizer loading failed: %s", exc)
        return torch.tensor([[2, 4, 5, 6, 7, 3, 0, 0]], dtype=torch.long, device=device)
    except OSError as exc:
        if not isinstance(exc, FileNotFoundError):
            raise
        logger.warning("Falling back to fixed language tokens because tokenizer files were not found: %s", exc)
        return torch.tensor([[2, 4, 5, 6, 7, 3, 0, 0]], dtype=torch.long, device=device)


def image_to_tensor(image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    image = torch.as_tensor(image, device=device)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {tuple(image.shape)}")

    image = image.permute(2, 0, 1).contiguous().to(dtype=dtype) / 255.0
    mean = torch.tensor(IMAGENET_STATS["mean"], device=device, dtype=dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STATS["std"], device=device, dtype=dtype).view(3, 1, 1)
    return ((image - mean) / std).unsqueeze(0)


def make_libero_hdf5_batch(
    policy: XVLAPolicy,
    sample: LiberoHdf5Sample,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    import h5py

    image_keys = list(policy.config.image_features)
    if len(image_keys) < 2:
        raise ValueError(f"Expected at least 2 image features, got {image_keys}")

    path = Path(sample.hdf5_path)
    if not path.is_file():
        raise FileNotFoundError(f"LIBERO HDF5 file not found: {path}")

    with h5py.File(path, "r") as h5_file:
        demos = sorted(h5_file["data"].keys())
        selected_demo = sample.demo_key or demos[0]
        demo = h5_file[f"data/{selected_demo}"]
        num_frames = demo["actions"].shape[0]
        frame_index = sample.frame_index
        if frame_index < 0:
            frame_index = num_frames + frame_index
        if frame_index < 0 or frame_index >= num_frames:
            raise IndexError(f"frame_index {frame_index} out of range for {selected_demo} with {num_frames} frames")

        agentview = demo["obs/agentview_rgb"][frame_index]
        wrist = demo["obs/eye_in_hand_rgb"][frame_index]
        state = torch.as_tensor(demo["robot_states"][frame_index, :8], device=device, dtype=dtype).unsqueeze(0)

    return {
        OBS_LANGUAGE_TOKENS: tokenize_task(policy, sample.task, device),
        image_keys[0]: image_to_tensor(agentview, device, dtype),
        image_keys[1]: image_to_tensor(wrist, device, dtype),
        OBS_STATE: state,
        "domain_id": torch.full((1,), int(sample.domain_id), dtype=torch.long, device=device),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def run_once(policy: XVLAPolicy, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    policy.reset()
    with torch.inference_mode():
        action = policy.select_action(batch)
    synchronize(device)
    return action


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    ensure_torch_npu(device)
    if device.type == "npu":
        torch.npu.set_device(device)
    dtype = resolve_dtype(args.dtype)

    logger.info("Loading XVLA policy")
    policy = load_policy(args, device)
    if args.libero_hdf5:
        sample = LiberoHdf5Sample(
            hdf5_path=args.libero_hdf5,
            demo_key=args.demo_key,
            frame_index=args.frame_index,
            task=args.task,
            domain_id=args.domain_id,
        )
        batch = make_libero_hdf5_batch(
            policy,
            sample,
            device,
            dtype,
        )
    else:
        batch = make_mock_batch(policy, device, dtype, args.domain_id)

    action = None
    for _ in range(args.num_warmup):
        action = run_once(policy, batch, device)

    total_time = 0.0
    for _ in range(args.num_inference):
        start = time.perf_counter()
        action = run_once(policy, batch, device)
        total_time += time.perf_counter() - start

    assert action is not None
    avg_ms = total_time / max(args.num_inference, 1) * 1000
    logger.info("----------------------------------------")
    logger.info("Device: %s, Dtype: %s", device, dtype)
    logger.info("NPU fusion attention: %s", os.environ.get("LEROBOT_XVLA_USE_NPU_FUSION_ATTENTION", "0"))
    logger.info("NPU baddbmm domain linear: %s", os.environ.get("LEROBOT_XVLA_USE_NPU_BADD_BMM", "0"))
    logger.info("Action shape: %s", action.shape)
    logger.info("Action dtype/device: %s/%s", action.dtype, action.device)
    logger.info("Average latency: %.4f ms", avg_ms)
    logger.info("Throughput: %.2f FPS", 1000.0 / avg_ms if avg_ms > 0 else 0.0)
    logger.info("----------------------------------------")
    if args.save_action:
        save_path = Path(args.save_action)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(action.detach().cpu(), save_path)
        logger.info("Saved action tensor to %s", save_path)


if __name__ == "__main__":
    main()
