# coding=utf-8
# Adapted from
# https://github.com/NVlabs/GaussianSTORM/blob/main/storm/utils/misc.py

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) NVIDIA. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections.abc
import datetime
import logging
import math
import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob
from itertools import repeat

import numpy as np
import torch
from torch import inf

logger = logging.getLogger("PerceptualModel")


def fix_random_seeds(seed=31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _ntuple(n):
    """
    Creates a parser that converts an input to a tuple of length n.

    Args:
        n (int): Length of the tuple.

    Returns:
        Callable: A function that parses the input into a tuple of length n.
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def cleanup_checkpoints(ckpt_dir, keep_num=1):
    """
    Clean up old checkpoints, keeping only the latest 'keep_num' checkpoints.

    Args:
        ckpt_dir (str): Directory containing the checkpoints.
        keep_num (int): Number of recent checkpoints to keep.
    """
    ckpts = glob(f"{ckpt_dir}/*.pth")
    ckpts = [
        ckpt for ckpt in ckpts
        if "latest" not in ckpt and "best" not in ckpt
    ]
    ckpts = sorted(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Remove older checkpoints
    for ckpt in ckpts[:-keep_num]:
        os.remove(ckpt)
        logger.info(f"Removed checkpoint: {ckpt}")

    # Create or update latest symlink
    if ckpts:
        latest_symlink = f"{ckpt_dir}/latest.pth"
        try:
            os.remove(latest_symlink)
        except FileNotFoundError:
            pass
        os.symlink(os.path.abspath(ckpts[-1]), latest_symlink)
        logger.info(f"Created symlink: {latest_symlink} -> {ckpts[-1]}")


# ---------------------------------------------------------------------------
# Model loading helpers (extracted from load_model to reduce complexity)
# ---------------------------------------------------------------------------

def _resolve_resume_checkpoint(args):
    """Determine the checkpoint path for resume/auto-resume.

    Returns the resolved path or None.
    """
    if args.resume_from:
        return args.resume_from

    if args.auto_resume:
        checkpoints = [
            ckpt for ckpt in glob(f"{args.ckpt_dir}/*.pth")
            if "latest" not in ckpt
        ]
        checkpoints = sorted(checkpoints, key=os.path.getmtime)
        if checkpoints:
            return checkpoints[-1]

    return None


def _load_optimizer_metadata(checkpoint, model_without_ddp, optimizer,
                             loss_scaler, args):
    """Load optimizer, loss scaler, and training metadata from checkpoint."""
    vis_slice_id = 0
    msg = model_without_ddp.load_state_dict(checkpoint["model"], strict=True)
    logger.info(f"[Model-resume] Loaded model: {msg}")

    if (
        "optimizer" in checkpoint
        and "latest_step" in checkpoint
        and optimizer is not None
    ):
        msg = optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info(f"[Model-resume] Loaded optimizer: {msg}")
        args.start_iteration = checkpoint["latest_step"] + 1
        if "loss_scaler" in checkpoint and loss_scaler is not None:
            msg = loss_scaler.load_state_dict(checkpoint["loss_scaler"])
            logger.info(f"[Model-resume] Loaded loss_scaler: {msg}")
        if "vis_slice_id" in checkpoint:
            vis_slice_id = checkpoint["vis_slice_id"] + 1

    if "latest_step" in checkpoint:
        args.prev_num_iterations = checkpoint["latest_step"]
        args.start_iteration = checkpoint["latest_step"] + 1

    if "total_elapsed_time" in checkpoint:
        args.total_elapsed_time = float(checkpoint["total_elapsed_time"])
        elapsed_str = str(datetime.timedelta(
            seconds=int(args.total_elapsed_time)
        ))
        logger.info(f"Loaded elapsed_time: {elapsed_str}")

    return vis_slice_id


def _affine_token_shape_adapt(checkpoint, model_without_ddp):
    """Adapt affine_token shape when checkpoint/model camera counts differ."""
    ckpt_affine = checkpoint["model"].get("aggregator.affine_token")
    if ckpt_affine is None:
        return
    model_affine = model_without_ddp.aggregator.affine_token
    if ckpt_affine.shape == model_affine.shape:
        return
    if ckpt_affine.shape > model_affine.shape:
        # e.g. 3 cam → 1 cam
        checkpoint["model"]["aggregator.affine_token"] = ckpt_affine[
            :, 1:2, ...
        ]
    else:
        # e.g. 3 cam → 6 cam
        checkpoint["model"]["aggregator.affine_token"] = ckpt_affine.repeat(
            1, 2, 1
        )


def _load_state_dict_fallback(checkpoint, model_without_ddp, load_path):
    """Fallback: load only matching parameters, skip shape mismatches."""
    model_state_dict = model_without_ddp.state_dict()
    filtered_dict = OrderedDict()
    for k, v in checkpoint.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                filtered_dict[k] = v
            else:
                logger.info(
                    f"Skipping parameter due to shape mismatch: {k} "
                    f"({v.shape} vs {model_state_dict[k].shape})"
                )
        else:
            logger.info(f"Skipping unexpected key: {k}")

    msg = model_without_ddp.load_state_dict(filtered_dict, strict=False)
    logger.info(f"Load status: {msg}")


def _load_from_checkpoint_path(args, model_without_ddp):
    """Load model weights from --load_from path (no optimizer/metadata).

    Returns True if a checkpoint was successfully loaded.
    """
    if not args.load_from or not os.path.exists(args.load_from):
        return False

    logger.info(f"Loading checkpoint from: {args.load_from}")
    checkpoint = torch.load(
        args.load_from, map_location='cpu', weights_only=False
    )

    # Adapt affine_token shape when cam count differs
    if "model" in checkpoint:
        _affine_token_shape_adapt(checkpoint, model_without_ddp)

    if "model" in checkpoint:
        checkpoint = checkpoint["model"]

    try:
        msg = model_without_ddp.load_state_dict(checkpoint, strict=False)
        logger.info(f"[Model-init] Loaded model: {msg}")
    except Exception as e:
        logger.error(e)
        logger.info(
            f"[Model-init] Loading model from {args.load_from} failed. "
            f"Error: {e}"
        )
        _load_state_dict_fallback(checkpoint, model_without_ddp, args.load_from)

    del checkpoint
    return True


def load_model(args, model_without_ddp, optimizer=None, loss_scaler=None):
    """
    Load model, optimizer, and loss scaler states from a checkpoint.

    Args:
        args: Arguments containing checkpoint paths and loading configurations.
        model_without_ddp (torch.nn.Module): Model to load the state into.
        optimizer (torch.optim.Optimizer, optional): Optimizer for loading states.
        loss_scaler (torch.cuda.amp.GradScaler, optional): Loss scaler for AMP.

    Returns:
        int: Visualization slice ID if available.
    """
    vis_slice_id, checkpoint_loaded = 0, False

    # Priority 1: resume from checkpoint (includes optimizer / metadata)
    resume_path = _resolve_resume_checkpoint(args)
    if resume_path and os.path.exists(resume_path):
        logger.info(f"[Model-resume] Resuming from: {resume_path}")
        checkpoint = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        vis_slice_id = _load_optimizer_metadata(
            checkpoint, model_without_ddp, optimizer, loss_scaler, args
        )
        del checkpoint
        checkpoint_loaded = True

    # Priority 2: load weights only (no optimizer)
    if not checkpoint_loaded:
        checkpoint_loaded = _load_from_checkpoint_path(
            args, model_without_ddp
        )

    if not checkpoint_loaded:
        logger.info(f"Training from scratch. No checkpoint found.")
    return vis_slice_id


# ---------------------------------------------------------------------------
# Learning rate & model utilities
# ---------------------------------------------------------------------------

def adjust_learning_rate(optimizer, iteration, args):
    """
    Adjust the learning rate using a cosine decay schedule with warmup.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to update learning rate.
        iteration (int): Current training iteration.
        args: Arguments defining the learning rate schedule.

    Returns:
        float: Updated learning rate.
    """
    if iteration < args.warmup_iters:
        lr = args.lr * iteration / args.warmup_iters
    else:
        if args.lr_sched == "constant":
            lr = args.lr
        elif args.lr_sched == "cosine":
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (iteration - args.warmup_iters)
                    / (args.num_iterations - args.warmup_iters)
                )
            )
        else:
            raise ValueError(f"Unknown lr_sched: {args.lr_sched}")

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group.get("lr_scale", 1.0)

    return lr


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def get_grad_norm_(parameters, norm_type=2.0):
    """
    Compute gradient norm for a set of parameters.

    Args:
        parameters (Iterable): Parameters to compute gradients for.
        norm_type (float): Norm type for gradient computation.

    Returns:
        torch.Tensor: Gradient norm.
    """
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(
            p.grad.detach().abs().max().to(device) for p in parameters
        )
    else:
        total_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), norm_type).to(device)
                for p in parameters
            ]),
            norm_type,
        )
    return total_norm


# ---------------------------------------------------------------------------
# NativeScalerWithGradNormCount
# ---------------------------------------------------------------------------

@dataclass
class GradStepConfig:
    loss: torch.Tensor
    optimizer: torch.optim.Optimizer
    parameters: object
    clip_grad: float = None
    create_graph: bool = False
    update_grad: bool = True

    @classmethod
    def from_call(cls, call_args, call_kwargs):
        field_names = [
            "loss", "optimizer", "parameters", "clip_grad",
            "create_graph", "update_grad",
        ]
        values = dict(zip(field_names, call_args))
        values.update(call_kwargs)
        missing = [name for name in field_names[:3] if name not in values]
        if missing:
            raise TypeError(f"Missing required arguments: {', '.join(missing)}")
        return cls(**values)


class NativeScalerWithGradNormCount:
    """
    A wrapper for torch.cuda.amp.GradScaler with gradient norm tracking.

    Args:
        enabled (bool): Whether to enable automatic mixed precision.
    """

    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True):
        self._scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    def __call__(self, *args, **kwargs):
        grad_step = GradStepConfig.from_call(args, kwargs)
        if os.getenv('LOSS_SCALING'):  # need in FP16
            return self._step_with_scaling(grad_step)
        else:  # FP32, BF16 do not need loss scaling
            return self._step_without_scaling(grad_step)

    def state_dict(self):
        """Save state dictionary for the scaler."""
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dictionary for the scaler."""
        self._scaler.load_state_dict(state_dict)

    def _step_with_scaling(self, grad_step):
        self._scaler.scale(grad_step.loss).backward(
            create_graph=grad_step.create_graph
        )
        if not grad_step.update_grad:
            return None
        self._scaler.unscale_(grad_step.optimizer)
        norm = self._compute_grad_norm(
            grad_step.parameters, grad_step.clip_grad
        )
        self._scaler.step(grad_step.optimizer)
        self._scaler.update()
        return norm

    def _step_without_scaling(self, grad_step):
        grad_step.loss.backward(create_graph=grad_step.create_graph)
        if not grad_step.update_grad:
            return None
        norm = self._compute_grad_norm(
            grad_step.parameters, grad_step.clip_grad
        )
        grad_step.optimizer.step()
        return norm

    @staticmethod
    def _compute_grad_norm(parameters, clip_grad):
        if clip_grad is not None and clip_grad > 0.0:
            return torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        else:
            return get_grad_norm_(parameters)
