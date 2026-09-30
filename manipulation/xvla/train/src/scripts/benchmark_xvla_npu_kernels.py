#!/usr/bin/env python3
#
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Mulan PSL v2.
# You may obtain a copy of the License at:
#     http://license.coscl.org.cn/MulanPSL2
#
"""Benchmark XVLA NPU kernel paths without downloading weights or datasets."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import time
from dataclasses import dataclass

import torch

from lerobot.policies.xvla.soft_transformer import Attention, DomainAwareLinear, SoftPromptedTransformer


logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


@dataclass
class BenchResult:
    name: str
    fusion: bool
    avg_ms: float
    throughput: float
    output_shape: tuple[int, ...]


@dataclass
class DomainLinearCase:
    name: str
    input_size: int
    output_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark XVLA NPU fusion attention paths")
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--action_dim", type=int, default=20)
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--num_warmup", type=int, default=5)
    parser.add_argument("--num_iters", type=int, default=20)
    parser.add_argument("--train_step", action="store_true", help="Benchmark forward+backward instead of forward only")
    return parser.parse_args()


def resolve_dtype(dtype: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]


def ensure_torch_npu(device: torch.device) -> None:
    if device.type != "npu":
        return
    try:
        importlib.import_module("torch_npu")
    except ImportError as exc:
        raise RuntimeError("torch_npu is required when --device uses npu") from exc


def synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def run_attention(args: argparse.Namespace, device: torch.device, dtype: torch.dtype, fusion: bool) -> BenchResult:
    os.environ["LEROBOT_XVLA_USE_NPU_FUSION_ATTENTION"] = "1" if fusion else "0"
    torch.manual_seed(1000)
    model = Attention(dim=args.hidden_size, num_heads=args.num_heads).to(device=device, dtype=dtype)
    model.train(args.train_step)
    x = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device=device, dtype=dtype)

    def step() -> torch.Tensor:
        inp = x.detach().requires_grad_(args.train_step)
        out = model(inp)
        if args.train_step:
            model.zero_grad(set_to_none=True)
            out.float().square().mean().backward()
        return out

    return time_loop("attention", step, args, device, fusion)


def run_transformer(args: argparse.Namespace, device: torch.device, dtype: torch.dtype, fusion: bool) -> BenchResult:
    os.environ["LEROBOT_XVLA_USE_NPU_FUSION_ATTENTION"] = "1" if fusion else "0"
    torch.manual_seed(1000)
    model = SoftPromptedTransformer(
        hidden_size=args.hidden_size,
        multi_modal_input_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=4.0,
        num_domains=30,
        dim_action=args.action_dim,
        dim_propio=args.action_dim,
        len_soft_prompts=32,
        dim_time=32,
        max_len_seq=max(512, args.seq_len + args.chunk_size + 64),
    ).to(device=device, dtype=dtype)
    model.train(args.train_step)
    batch = args.batch_size
    vlm_tokens = max(1, args.seq_len // 2)
    aux_tokens = max(1, args.seq_len - vlm_tokens)
    inputs = {
        "domain_id": torch.zeros(batch, dtype=torch.long, device=device),
        "vlm_features": torch.randn(batch, vlm_tokens, args.hidden_size, device=device, dtype=dtype),
        "aux_visual_inputs": torch.randn(batch, aux_tokens, args.hidden_size, device=device, dtype=dtype),
        "action_with_noise": torch.randn(batch, args.chunk_size, args.action_dim, device=device, dtype=dtype),
        "t": torch.rand(batch, device=device, dtype=dtype),
        "proprio": torch.randn(batch, args.action_dim, device=device, dtype=dtype),
    }

    def step() -> torch.Tensor:
        call_inputs = {
            key: value.detach().requires_grad_(args.train_step) if value.is_floating_point() else value
            for key, value in inputs.items()
        }
        out = model(**call_inputs)
        if args.train_step:
            model.zero_grad(set_to_none=True)
            out.float().square().mean().backward()
        return out

    return time_loop("transformer", step, args, device, fusion)


def run_domain_linear(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    baddbmm_enabled: bool,
    case: DomainLinearCase,
) -> BenchResult:
    os.environ["LEROBOT_XVLA_USE_NPU_BADD_BMM"] = "1" if baddbmm_enabled else "0"
    torch.manual_seed(1000)
    model = DomainAwareLinear(case.input_size, case.output_size, num_domains=30).to(device=device, dtype=dtype)
    model.train(args.train_step)
    x = torch.randn(args.batch_size, args.chunk_size, case.input_size, device=device, dtype=dtype)
    domain_id = torch.arange(args.batch_size, dtype=torch.long, device=device) % 30

    def step() -> torch.Tensor:
        inp = x.detach().requires_grad_(args.train_step)
        out = model(inp, domain_id)
        if args.train_step:
            model.zero_grad(set_to_none=True)
            out.float().square().mean().backward()
        return out

    return time_loop(case.name, step, args, device, baddbmm_enabled)


def time_loop(name: str, step, args: argparse.Namespace, device: torch.device, fusion: bool) -> BenchResult:
    out = None
    for _ in range(args.num_warmup):
        out = step()
    synchronize(device)

    start = time.perf_counter()
    for _ in range(args.num_iters):
        out = step()
    synchronize(device)
    total = time.perf_counter() - start

    if out is None:
        raise RuntimeError("Benchmark did not run any iteration")
    avg_ms = total / max(args.num_iters, 1) * 1000
    return BenchResult(
        name=name,
        fusion=fusion,
        avg_ms=avg_ms,
        throughput=1000.0 / avg_ms if avg_ms > 0 else 0.0,
        output_shape=tuple(out.shape),
    )


def log_result(result: BenchResult) -> None:
    switch_name = "baddbmm" if result.name.startswith("domain_") else "fusion"
    logger.info(
        f"{result.name:12s} {switch_name}={int(result.fusion)} "
        f"avg_ms={result.avg_ms:.4f} throughput={result.throughput:.2f}/s output_shape={result.output_shape}"
    )


def log_speedup(name: str, baseline: BenchResult, optimized: BenchResult) -> None:
    speedup = baseline.avg_ms / optimized.avg_ms if optimized.avg_ms > 0 else 0.0
    delta = (baseline.avg_ms - optimized.avg_ms) / baseline.avg_ms * 100 if baseline.avg_ms > 0 else 0.0
    logger.info(f"{name:12s} fusion speedup={speedup:.4f}x latency_delta={delta:.2f}%")


def log_baddbmm_speedup(name: str, baseline: BenchResult, optimized: BenchResult) -> None:
    speedup = baseline.avg_ms / optimized.avg_ms if optimized.avg_ms > 0 else 0.0
    delta = (baseline.avg_ms - optimized.avg_ms) / baseline.avg_ms * 100 if baseline.avg_ms > 0 else 0.0
    logger.info(f"{name:12s} baddbmm speedup={speedup:.4f}x latency_delta={delta:.2f}%")


def setup_device(args: argparse.Namespace) -> tuple[torch.device, torch.dtype]:
    device = torch.device(args.device)
    ensure_torch_npu(device)
    if device.type == "npu":
        torch.npu.set_device(device)
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch_npu is installed, but no NPU is visible")
    return device, resolve_dtype(args.dtype)


def log_config(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> None:
    logger.info(f"device={device} dtype={dtype} train_step={args.train_step}")
    logger.info(
        "config="
        f"batch_size={args.batch_size}, seq_len={args.seq_len}, hidden_size={args.hidden_size}, "
        f"num_heads={args.num_heads}, depth={args.depth}, chunk_size={args.chunk_size}"
    )


def run_domain_cases(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[BenchResult, BenchResult, BenchResult, BenchResult]:
    attn_base = run_attention(args, device, dtype, fusion=False)
    attn_opt = run_attention(args, device, dtype, fusion=True)
    encoder_case = DomainLinearCase(
        name="domain_enc",
        input_size=args.action_dim + args.action_dim + 32,
        output_size=args.hidden_size,
    )
    decoder_case = DomainLinearCase(
        name="domain_dec",
        input_size=args.hidden_size,
        output_size=args.action_dim,
    )
    domain_encoder_base = run_domain_linear(
        args, device, dtype, baddbmm_enabled=False, case=encoder_case
    )
    domain_encoder_opt = run_domain_linear(
        args, device, dtype, baddbmm_enabled=True, case=encoder_case
    )
    domain_decoder_base = run_domain_linear(
        args, device, dtype, baddbmm_enabled=False, case=decoder_case
    )
    domain_decoder_opt = run_domain_linear(
        args, device, dtype, baddbmm_enabled=True, case=decoder_case
    )
    log_result(attn_base)
    log_result(attn_opt)
    log_speedup("attention", attn_base, attn_opt)
    return domain_encoder_base, domain_encoder_opt, domain_decoder_base, domain_decoder_opt


def main() -> None:
    args = parse_args()
    device, dtype = setup_device(args)
    log_config(args, device, dtype)
    domain_encoder_base, domain_encoder_opt, domain_decoder_base, domain_decoder_opt = run_domain_cases(
        args, device, dtype
    )
    trans_base = run_transformer(args, device, dtype, fusion=False)
    trans_opt = run_transformer(args, device, dtype, fusion=True)

    log_result(domain_encoder_base)
    log_result(domain_encoder_opt)
    log_baddbmm_speedup("domain_enc", domain_encoder_base, domain_encoder_opt)
    log_result(domain_decoder_base)
    log_result(domain_decoder_opt)
    log_baddbmm_speedup("domain_dec", domain_decoder_base, domain_decoder_opt)
    log_result(trans_base)
    log_result(trans_opt)
    log_speedup("transformer", trans_base, trans_opt)


if __name__ == "__main__":
    main()
