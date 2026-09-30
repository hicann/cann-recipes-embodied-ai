#!/usr/bin/env python3
#
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
"""Verify XVLA NPU optimized path against the fallback PyTorch path."""

from __future__ import annotations

import argparse
import logging
import os

import torch

from test_xvla_inference_ascend import (
    ensure_torch_npu,
    LiberoHdf5Sample,
    load_policy,
    make_libero_hdf5_batch,
    make_mock_batch,
    resolve_dtype,
    synchronize,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify XVLA NPU fusion attention consistency")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--domain_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--libero_hdf5", type=str, default=None)
    parser.add_argument("--demo_key", type=str, default=None)
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument(
        "--task",
        type=str,
        default="put the black bowl in the bottom drawer of the cabinet and close it",
    )
    return parser.parse_args()


def make_batch(args: argparse.Namespace, policy, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    if args.libero_hdf5:
        sample = LiberoHdf5Sample(
            hdf5_path=args.libero_hdf5,
            demo_key=args.demo_key,
            frame_index=args.frame_index,
            task=args.task,
            domain_id=args.domain_id,
        )
        return make_libero_hdf5_batch(
            policy,
            sample,
            device,
            dtype,
        )
    return make_mock_batch(policy, device, dtype, args.domain_id)


def run_action_chunk(policy, batch: dict[str, torch.Tensor], device: torch.device, seed: int, optimized: bool):
    os.environ["LEROBOT_XVLA_USE_NPU_BADD_BMM"] = "1" if optimized else "0"
    os.environ["LEROBOT_XVLA_USE_NPU_FUSION_ATTENTION"] = "1" if optimized else "0"
    torch.manual_seed(seed)
    policy.reset()
    with torch.inference_mode():
        action = policy.predict_action_chunk(batch)
    synchronize(device)
    return action.detach().float().cpu()


def compare_actions(reference: torch.Tensor, optimized: torch.Tensor, threshold: float) -> bool:
    if not torch.isfinite(reference).all():
        logger.error("Reference output contains NaN/Inf.")
        return False
    if not torch.isfinite(optimized).all():
        logger.error("Optimized output contains NaN/Inf.")
        return False

    ref_flat = reference.flatten()
    opt_flat = optimized.flatten()
    cosine = torch.nn.functional.cosine_similarity(ref_flat.unsqueeze(0), opt_flat.unsqueeze(0), dim=1).item()
    mse = torch.nn.functional.mse_loss(reference, optimized).item()
    max_abs = (reference - optimized).abs().max().item()

    logger.info("Global Cosine Similarity: %.6f", cosine)
    logger.info("MSE Loss: %.8f", mse)
    logger.info("Max Abs Error: %.8f", max_abs)
    if cosine >= threshold:
        logger.info("Verification SUCCESS: similarity %.6f >= %.6f", cosine, threshold)
        return True

    logger.error("Verification FAILED: similarity %.6f < %.6f", cosine, threshold)
    return False


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ensure_torch_npu(device)
    if device.type == "npu":
        torch.npu.set_device(device)
    dtype = resolve_dtype(args.dtype)

    torch.manual_seed(args.seed)
    policy = load_policy(args, device)
    batch = make_batch(args, policy, device, dtype)

    logger.info("Running fallback path with NPU optimizations disabled")
    reference = run_action_chunk(policy, batch, device, args.seed, optimized=False)
    logger.info("Running optimized path with NPU fusion attention and baddbmm enabled")
    optimized = run_action_chunk(policy, batch, device, args.seed, optimized=True)

    ok = compare_actions(reference, optimized, args.threshold)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
