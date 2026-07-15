# coding=utf-8
# Adapted from
# https://github.com/NVlabs/GaussianSTORM/blob/main/engine_storm.py

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) NVIDIA. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime, timezone
import logging
import os
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import einops

import src.models as models
import src.utils.distributed as distributed
import src.utils.misc as misc
from src.dataset.constants import MEAN, STD, SEMANTIC_LABEL_LIST
from src.dataset.data_utils import prepare_inputs_and_targets
from src.utils.losses import compute_scene_flow_metrics, compute_semantic_metrics
from src.visualization.video_maker import make_video

if os.getenv("FEAT_DIST"):
    from tools.feats_tools import get_text_label_feats, feat2class

LOGGER = logging.getLogger("PerceptualModel")

# Depth evaluation thresholds (in meters)
DEPTH_MIN_THRESHOLD = 0.01  # Minimum valid depth to exclude noise near camera
DEPTH_MAX_THRESHOLD = 200.0  # Maximum valid depth for evaluation
DEPTH_MID_THRESHOLD = 100.0  # Split threshold for near/far range evaluation


def build_model(args):
    if args.model in models.STORM_models:
        model = models.STORM_models[args.model](
            img_size=args.input_size, gs_dim=args.gs_dim,
            decoder_type=args.decoder_type,
            grad_checkpointing=not args.disable_grad_checkpointing,
            use_sky_token=args.use_sky_token,
            use_affine_token=args.use_affine_token,
            num_motion_tokens=args.num_motion_tokens,
            num_cams=args.num_max_cameras,
            sigmoid_rgb=args.sigmoid_rgb, gs_marbles=args.gs_marbles,
            max_scale=args.max_scale, render_context_view=args.render_context_view,
            render_context_frame_contribution=args.render_context_frame_contribution,
            pred_gs_conf=args.pred_gs_conf, voxelize=args.voxelize,
            voxel_size=args.voxel_size, use_ms3_motion=args.use_ms3_motion,
            add_angular_velocity=args.add_angular_velocity,
            use_render_novel_view=args.use_render_novel_view,
        )
    elif args.model == 'slarm':
        model = models.SLARM(img_size=args.input_size, gs_dim=args.gs_dim, decoder_type=args.decoder_type,
                             embed_dim=args.embed_dim, patch_size=args.patch_size, depth=args.depth,
                             patch_embed=args.patch_embed, grad_checkpointing=not args.disable_grad_checkpointing,
                             vggt_pretrained_weight_filepath=args.vggt_pretrained_weight_filepath,
                             enable_depth_head=args.enable_depth_head, enable_camera_head=args.enable_camera_head,
                             enable_point_head=args.enable_point_head, shortcut_rgb=args.shortcut_rgb,
                             use_ms3_motion=args.use_ms3_motion,
                             gs_marbles=args.gs_marbles, render_context_view=args.render_context_view,
                             render_context_frame_contribution=args.render_context_frame_contribution,
                             use_2dgs=args.use_2dgs, pesudo_3dgs=args.pesudo_3dgs, save_gaussian=args.save_gaussian,
                             gaussian_save_path=args.gaussian_save_path, save_rendered_pc=args.save_rendered_pc,
                             rendered_pc_save_path=args.rendered_pc_save_path, use_time_token=args.use_time_token,
                             use_sky_token=args.use_sky_token, use_affine_token=args.use_affine_token,
                             num_motion_tokens=args.num_motion_tokens, pred_gs_conf=args.pred_gs_conf,
                             enable_lifespan=args.enable_lifespan, voxelize=args.voxelize, voxel_size=args.voxel_size,
                             num_cams=args.num_max_cameras, sigmoid_rgb=args.sigmoid_rgb,
                             use_pred_camera_pose=args.use_pred_camera_pose, use_pred_depth=args.use_pred_depth,
                             add_patch_plucker_embed=args.add_patch_plucker_embed,
                             add_camera_embed=args.add_camera_embed,
                             concat_plucker_embed=args.concat_plucker_embed, use_last_token=args.use_last_token,
                             use_render_novel_view=args.use_render_novel_view,
                             add_angular_velocity=args.add_angular_velocity,
                             with_feat=not args.without_feat,
                             similarity_probs_threshold=args.similarity_probs_threshold,
                             mode=args.mode)
    else:
        raise ValueError(f"Invalid model name: {args.model}")
    return model


def load_model(args, device, model):
    LOGGER.info(f"Model = {str(model)}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(f"{args.model} Parameters: {n_params / 1e6:.2f}M ({n_params:,})")

    model.to(device)
    misc.load_model(args, model)

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(f"{args.model} Trainable Parameters: {num_trainable_params / 1e6:.2f}M")

    model.eval()
    return model


@dataclass
class VisualizeContext:
    args: Any
    model: Any
    dset_train: Any
    step: int
    train_vis_id: Any
    device: Any
    dset_val: Optional[Any] = None
    val_vis_id: Optional[Any] = None
    log_writer: Optional[Any] = None
    feat_extractor: Optional[Any] = None


@torch.no_grad()
def visualize(ctx: VisualizeContext):
    ctx.model.eval()
    global_rank = distributed.get_global_rank()
    split = "train"
    for vis_id, dataset in zip([ctx.train_vis_id, ctx.val_vis_id], [ctx.dset_train, ctx.dset_val]):
        if vis_id is None or dataset is None:  # sometimes there is no validation set
            continue

        sample_id = global_rank * 80 + vis_id
        out_pth = f"{ctx.args.video_dir}/step{ctx.step}-rank{global_rank}-sample{sample_id}-{split}.mp4"

        LOGGER.info(f"saving video to {out_pth}")
        video = make_video(
            dataset,
            ctx.model,
            ctx.device,
            output_filename=out_pth,
            scene_id=sample_id,
            skip_plot_gt_depth_and_flow=False,
            feat_extractor=ctx.feat_extractor,
        )
        video = np.array(video)
        video = einops.rearrange(video, '(b t) h w c -> b t c h w', b=1)  # batch size 1 in visualize
        if ctx.log_writer is not None:
            ctx.log_writer.add_video(f"video/{split}", video)

        LOGGER.info(f"saved video to {out_pth}")
        split = "val"

    torch.cuda.empty_cache()
    return ctx.train_vis_id + 1, ctx.val_vis_id + 1 if ctx.val_vis_id is not None else None


@dataclass
class ImageData:
    """Container for image-related data."""
    pred_rgb: torch.Tensor
    gt_rgb: torch.Tensor
    pred_depth: torch.Tensor
    gt_depth: torch.Tensor
    occupied_mask: torch.Tensor
    dynamic_mask: torch.Tensor
    valid_depth_mask: torch.Tensor
    valid_depth_mask_0_100: torch.Tensor
    valid_depth_mask_100_200: torch.Tensor
    index: int = 0


@dataclass
class BatchData:
    """Container for batch data."""
    pred_rgb: torch.Tensor
    gt_rgb: torch.Tensor
    pred_depth: torch.Tensor
    gt_depth: torch.Tensor
    occupied_mask: torch.Tensor
    dynamic_mask: torch.Tensor
    valid_depth_mask: torch.Tensor
    valid_depth_mask_0_100: torch.Tensor
    valid_depth_mask_100_200: torch.Tensor


@dataclass
class BatchMetrics:
    """Container for batch metrics."""
    psnr: List[float] = None
    ssim: List[float] = None
    depth_rmse: List[float] = None
    occupied_psnr: List[float] = None
    occupied_ssim: List[float] = None
    dynamic_psnr: List[float] = None
    dynamic_ssim: List[float] = None
    dynamic_depth_rmse: List[float] = None
    depth_rmse_0_100: List[float] = None
    depth_rmse_100_200: List[float] = None
    dynamic_depth_rmse_0_100: List[float] = None
    dynamic_depth_rmse_100_200: List[float] = None
    dynamic_count: int = 0
    valid_dynamic_depth_count: int = 0
    valid_dynamic_depth_0_100_count: int = 0
    valid_dynamic_depth_100_200_count: int = 0

    def __post_init__(self):
        """Initialize all lists."""
        if self.psnr is None:
            self.psnr = []
        if self.ssim is None:
            self.ssim = []
        if self.depth_rmse is None:
            self.depth_rmse = []
        if self.occupied_psnr is None:
            self.occupied_psnr = []
        if self.occupied_ssim is None:
            self.occupied_ssim = []
        if self.dynamic_psnr is None:
            self.dynamic_psnr = []
        if self.dynamic_ssim is None:
            self.dynamic_ssim = []
        if self.dynamic_depth_rmse is None:
            self.dynamic_depth_rmse = []
        if self.depth_rmse_0_100 is None:
            self.depth_rmse_0_100 = []
        if self.depth_rmse_100_200 is None:
            self.depth_rmse_100_200 = []
        if self.dynamic_depth_rmse_0_100 is None:
            self.dynamic_depth_rmse_0_100 = []
        if self.dynamic_depth_rmse_100_200 is None:
            self.dynamic_depth_rmse_100_200 = []


@dataclass
class EvaluationResults:
    """Container for all evaluation metrics results."""
    avg_psnr: float
    avg_ssim: float
    avg_depth_rmse: float
    avg_occupied_psnr: float
    avg_occupied_ssim: float
    avg_dynamic_psnr: float
    avg_dynamic_ssim: float
    avg_dynamic_rmse: float
    avg_depth_rmse_0_100: float
    avg_depth_rmse_100_200: float
    avg_dynamic_depth_rmse_0_100: float
    avg_dynamic_depth_rmse_100_200: float
    total_samples: int
    valid_depth_samples_0_100: int
    valid_depth_samples_100_200: int
    valid_dynamic_depth_samples_0_100: int
    valid_dynamic_depth_samples_100_200: int


@torch.no_grad()
def evaluate(dataloader, model, args, name_str=None):
    """Evaluate model on the given dataloader."""
    torch.cuda.empty_cache()
    model.eval()
    device = next(model.parameters()).device
    mean = torch.tensor(MEAN).to(device)
    std = torch.tensor(STD).to(device)

    eval_result_dir = _setup_eval_directory(args)
    name_str = _get_eval_name(name_str)

    # Initialize metric accumulators
    metric_accum = _init_metric_accumulators()
    pbar = tqdm(dataloader, desc="Evaluating")

    # Main evaluation loop
    for data_dict in pbar:
        batch_metrics = _evaluate_batch(data_dict, model, device, args)
        metric_accum = _update_metric_accumulators(metric_accum, batch_metrics)

        pbar.set_postfix(
            avg_psnr=metric_accum["total_psnr"] / metric_accum["total_samples"],
            avg_depth_rmse=metric_accum["total_depth_rmse"] / metric_accum["total_samples"],
            avg_dynamic_psnr=metric_accum["total_dynamic_psnr"] / metric_accum["total_dynamic_samples"],
            avg_dynamic_depth_rmse=metric_accum["total_dynamic_rmse"] / metric_accum[
                "total_valid_dynamic_depth_samples"],
        )

    # Aggregate results across distributed processes
    agg_tensors = _aggregate_metrics_tensors(metric_accum, device)

    # Compute final metrics and save results
    result = _compute_and_save_final_metrics(agg_tensors, eval_result_dir, name_str)

    torch.cuda.empty_cache()
    return result


def _setup_eval_directory(args):
    """Create and setup evaluation results directory."""
    eval_result_dir = os.path.join(args.log_dir, "eval_results")
    os.makedirs(eval_result_dir, exist_ok=True)
    LOGGER.info(f"Saving evaluation results to {eval_result_dir}")
    return eval_result_dir


def _get_eval_name(name_str):
    """Generate evaluation name if not provided."""
    if name_str is None:
        return datetime.now(tz=timezone.utc).strftime("%y-%m-%d-%H-%M")
    return name_str


def _init_metric_accumulators():
    """Initialize all metric accumulator variables."""
    return {
        "total_samples": 0,
        "total_dynamic_samples": 0,
        "total_valid_dynamic_depth_samples": 0,
        "total_psnr": 0.0,
        "total_ssim": 0.0,
        "total_depth_rmse": 0.0,
        "total_occupied_psnr": 0.0,
        "total_occupied_ssim": 0.0,
        "total_dynamic_psnr": 0.0,
        "total_dynamic_ssim": 0.0,
        "total_dynamic_rmse": 0.0,
        "total_depth_rmse_0_100": 0.0,
        "total_depth_rmse_100_200": 0.0,
        "total_valid_depth_samples_0_100": 0,
        "total_valid_depth_samples_100_200": 0,
        "total_dynamic_depth_rmse_0_100": 0.0,
        "total_dynamic_depth_rmse_100_200": 0.0,
        "total_valid_dynamic_depth_samples_0_100": 0,
        "total_valid_dynamic_depth_samples_100_200": 0,
    }


def _evaluate_batch(data_dict, model, device, args):
    """Evaluate a single batch and return metrics."""
    input_dict, target_dict = prepare_inputs_and_targets(
        data_dict, device, v=args.num_max_cameras
    )

    input_indices, test_indices = _extract_indices(input_dict, target_dict)
    pred_dict = model(input_dict)

    # Extract ground truth and prediction tensors
    batch_data = _extract_batch_data(target_dict, pred_dict, test_indices)

    # Compute metrics for each image
    batch_metrics = _compute_batch_metrics(batch_data)

    return batch_metrics


def _extract_indices(input_dict, target_dict):
    """Extract input and test indices from dictionaries."""
    input_indices = input_dict["context_frame_idx"][0].cpu().numpy().tolist()
    input_indice_start = input_indices[0]
    input_indices = [idx - input_indice_start for idx in input_indices]

    target_indices = target_dict["target_frame_idx"][0].cpu().numpy().tolist()
    target_indices = [idx - input_indice_start for idx in target_indices]
    test_indices = [idx for idx in target_indices if idx not in input_indices]

    LOGGER.info(f"Input indices: {input_indices}")
    LOGGER.info(f"Test indices: {test_indices}")
    return input_indices, test_indices


def _extract_batch_data(target_dict, pred_dict, test_indices):
    """Extract and organize all batch data."""
    # Extract RGB
    gt_rgb = target_dict["target_image"][:, test_indices]
    gt_rgb = gt_rgb.permute(0, 1, 2, 4, 5, 3)
    height, width = gt_rgb.shape[-3], gt_rgb.shape[-2]
    gt_rgb = gt_rgb.reshape(-1, height, width, 3)

    # Extract depth
    gt_depth = target_dict["target_depth"][:, test_indices].view(-1, height, width)
    gt_sky_mask = target_dict["target_sky_masks"][:, test_indices].view(-1, height, width)

    # Extract dynamic mask
    if "target_dynamic_masks" in target_dict:
        gt_dynamic_mask = target_dict["target_dynamic_masks"][:, test_indices]
        gt_dynamic_mask = gt_dynamic_mask.view(-1, height, width)
    else:
        gt_dynamic_mask = None

    # Extract predictions
    rendered_results = pred_dict["render_results"]
    pred_rgb = rendered_results[rendered_results["rgb_key"]][:, test_indices]
    pred_rgb = pred_rgb.reshape(-1, height, width, 3).detach()
    pred_rgb = torch.clamp(pred_rgb, 0, 1)

    if rendered_results["decoder_depth_key"] is None:
        pred_depth = rendered_results[rendered_results["depth_key"]][:, test_indices]
        pred_depth = pred_depth.view(-1, height, width)
    else:
        pred_depth = rendered_results[rendered_results["decoder_depth_key"]][:, test_indices]
        pred_depth = pred_depth.view(-1, height, width)

    # Create masks
    occupied_mask = (gt_sky_mask == 0).bool()
    dynamic_mask = _get_dynamic_mask(gt_dynamic_mask, occupied_mask)
    valid_depth_mask = (gt_depth > DEPTH_MIN_THRESHOLD) & (gt_depth < DEPTH_MAX_THRESHOLD)
    valid_depth_mask_0_100 = valid_depth_mask & (gt_depth <= DEPTH_MID_THRESHOLD)
    valid_depth_mask_100_200 = valid_depth_mask & (gt_depth > DEPTH_MID_THRESHOLD)

    return BatchData(
        pred_rgb=pred_rgb,
        gt_rgb=gt_rgb,
        pred_depth=pred_depth,
        gt_depth=gt_depth,
        occupied_mask=occupied_mask,
        dynamic_mask=dynamic_mask,
        valid_depth_mask=valid_depth_mask,
        valid_depth_mask_0_100=valid_depth_mask_0_100,
        valid_depth_mask_100_200=valid_depth_mask_100_200,
    )


def _get_dynamic_mask(gt_dynamic_mask, occupied_mask):
    """Get dynamic mask, defaulting to all occupied area if not available."""
    if gt_dynamic_mask is not None:
        return gt_dynamic_mask.bool()
    return torch.ones_like(occupied_mask)


def _compute_batch_metrics(batch_data: BatchData):
    """Compute all metrics for a batch."""
    batch_metrics = BatchMetrics()

    batch_items = zip(
        batch_data.pred_rgb,
        batch_data.gt_rgb,
        batch_data.pred_depth,
        batch_data.gt_depth,
        batch_data.occupied_mask,
        batch_data.dynamic_mask,
        batch_data.valid_depth_mask,
        batch_data.valid_depth_mask_0_100,
        batch_data.valid_depth_mask_100_200,
    )
    for i, item in enumerate(batch_items):
        (
            pred_rgb,
            gt_rgb,
            pred_depth,
            gt_depth,
            occupied_mask,
            dynamic_mask,
            valid_depth_mask,
            valid_depth_mask_0_100,
            valid_depth_mask_100_200,
        ) = item
        image_data = ImageData(
            pred_rgb=pred_rgb,
            gt_rgb=gt_rgb,
            pred_depth=pred_depth,
            gt_depth=gt_depth,
            occupied_mask=occupied_mask,
            dynamic_mask=dynamic_mask,
            valid_depth_mask=valid_depth_mask,
            valid_depth_mask_0_100=valid_depth_mask_0_100,
            valid_depth_mask_100_200=valid_depth_mask_100_200,
            index=i,
        )
        _compute_single_image_metrics(image_data, batch_metrics)

    return batch_metrics


def _compute_single_image_metrics(image_data: ImageData, batch_metrics: BatchMetrics):
    """Compute metrics for a single image."""
    # RGB metrics
    ssim_score = _compute_ssim(image_data)
    occupied_ssim_score = _compute_occupied_ssim(image_data)
    psnr_score = _compute_psnr(image_data)
    occupied_psnr_score = _compute_occupied_psnr(image_data)

    batch_metrics.ssim.append(ssim_score)
    batch_metrics.occupied_ssim.append(occupied_ssim_score)
    batch_metrics.psnr.append(psnr_score)
    batch_metrics.occupied_psnr.append(occupied_psnr_score)

    # Depth metrics
    depth_rmse = _compute_depth_rmse(image_data)
    batch_metrics.depth_rmse.append(depth_rmse)

    _compute_depth_rmse_by_range(image_data, batch_metrics)

    # Dynamic region metrics
    if image_data.dynamic_mask.sum() > 0:
        _compute_dynamic_metrics(image_data, batch_metrics)


def _compute_ssim(image_data: ImageData):
    """Compute SSIM between prediction and ground truth."""
    return ssim(
        get_numpy(image_data.pred_rgb),
        get_numpy(image_data.gt_rgb),
        data_range=1.0,
        channel_axis=-1,
    )


def _compute_occupied_ssim(image_data: ImageData):
    """Compute SSIM on occupied regions."""
    return ssim(
        get_numpy(image_data.pred_rgb),
        get_numpy(image_data.gt_rgb),
        data_range=1.0,
        channel_axis=-1,
        full=True,
    )[1][get_numpy(image_data.occupied_mask)].mean()


def _compute_psnr(image_data: ImageData):
    """Compute PSNR between prediction and ground truth."""
    return -10 * torch.log10(F.mse_loss(image_data.pred_rgb, image_data.gt_rgb)).item()


def _compute_occupied_psnr(image_data: ImageData):
    """Compute PSNR on occupied regions."""
    return -10 * torch.log10(
        F.mse_loss(
            image_data.pred_rgb[image_data.occupied_mask],
            image_data.gt_rgb[image_data.occupied_mask]
        )
    ).item()


def _compute_depth_rmse(image_data: ImageData):
    """Compute RMSE for depth prediction."""
    return torch.sqrt(
        F.mse_loss(
            image_data.pred_depth[image_data.valid_depth_mask],
            image_data.gt_depth[image_data.valid_depth_mask]
        )
    ).item()


def _compute_depth_rmse_by_range(image_data: ImageData, batch_metrics: BatchMetrics):
    """Compute depth RMSE for different depth ranges."""
    if image_data.valid_depth_mask_0_100.sum() > 0:
        rmse_0_100 = torch.sqrt(
            F.mse_loss(
                image_data.pred_depth[image_data.valid_depth_mask_0_100],
                image_data.gt_depth[image_data.valid_depth_mask_0_100]
            )
        ).item()
        batch_metrics.depth_rmse_0_100.append(rmse_0_100)

    if image_data.valid_depth_mask_100_200.sum() > 0:
        rmse_100_200 = torch.sqrt(
            F.mse_loss(
                image_data.pred_depth[image_data.valid_depth_mask_100_200],
                image_data.gt_depth[image_data.valid_depth_mask_100_200]
            )
        ).item()
        batch_metrics.depth_rmse_100_200.append(rmse_100_200)


def _compute_dynamic_metrics(image_data: ImageData, batch_metrics: BatchMetrics):
    """Compute metrics for dynamic regions."""
    batch_metrics.dynamic_count += 1

    dynamic_ssim = _compute_dynamic_ssim(image_data)
    dynamic_psnr = _compute_dynamic_psnr(image_data)
    batch_metrics.dynamic_ssim.append(dynamic_ssim)
    batch_metrics.dynamic_psnr.append(dynamic_psnr)

    _compute_dynamic_depth_metrics(image_data, batch_metrics)


def _compute_dynamic_ssim(image_data: ImageData):
    """Compute SSIM on dynamic regions."""
    return ssim(
        get_numpy(image_data.pred_rgb),
        get_numpy(image_data.gt_rgb),
        data_range=1.0,
        channel_axis=-1,
        full=True,
    )[1][get_numpy(image_data.dynamic_mask)].mean()


def _compute_dynamic_psnr(image_data: ImageData):
    """Compute PSNR on dynamic regions."""
    return -10 * torch.log10(
        F.mse_loss(
            image_data.pred_rgb[image_data.dynamic_mask],
            image_data.gt_rgb[image_data.dynamic_mask]
        )
    ).item()


def _compute_dynamic_depth_metrics(image_data: ImageData, batch_metrics: BatchMetrics):
    """Compute depth metrics for dynamic regions."""
    _valid_depth_mask = image_data.dynamic_mask & image_data.valid_depth_mask
    if _valid_depth_mask.sum() == 0:
        return

    batch_metrics.valid_dynamic_depth_count += 1
    dynamic_depth_rmse = torch.sqrt(
        F.mse_loss(
            image_data.pred_depth[_valid_depth_mask],
            image_data.gt_depth[_valid_depth_mask]
        )
    ).item()
    batch_metrics.dynamic_depth_rmse.append(dynamic_depth_rmse)

    # Dynamic region depth RMSE by range
    _valid_depth_0_100 = image_data.dynamic_mask & image_data.valid_depth_mask_0_100
    if _valid_depth_0_100.sum() > 0:
        rmse_0_100 = torch.sqrt(
            F.mse_loss(
                image_data.pred_depth[_valid_depth_0_100],
                image_data.gt_depth[_valid_depth_0_100]
            )
        ).item()
        batch_metrics.dynamic_depth_rmse_0_100.append(rmse_0_100)
        batch_metrics.valid_dynamic_depth_0_100_count += 1

    _valid_depth_100_200 = image_data.dynamic_mask & image_data.valid_depth_mask_100_200
    if _valid_depth_100_200.sum() > 0:
        rmse_100_200 = torch.sqrt(
            F.mse_loss(
                image_data.pred_depth[_valid_depth_100_200],
                image_data.gt_depth[_valid_depth_100_200]
            )
        ).item()
        batch_metrics.dynamic_depth_rmse_100_200.append(rmse_100_200)
        batch_metrics.valid_dynamic_depth_100_200_count += 1


def _update_metric_accumulators(metric_accum: Dict, batch_metrics: BatchMetrics):
    """Update global metric accumulators with batch results."""
    metric_accum["total_samples"] += len(batch_metrics.psnr)
    metric_accum["total_psnr"] += np.sum(batch_metrics.psnr)
    metric_accum["total_ssim"] += np.sum(batch_metrics.ssim)
    metric_accum["total_depth_rmse"] += np.sum(batch_metrics.depth_rmse)
    metric_accum["total_occupied_psnr"] += np.sum(batch_metrics.occupied_psnr)
    metric_accum["total_occupied_ssim"] += np.sum(batch_metrics.occupied_ssim)

    if batch_metrics.dynamic_count > 0:
        metric_accum["total_dynamic_samples"] += batch_metrics.dynamic_count
        metric_accum["total_dynamic_psnr"] += np.sum(batch_metrics.dynamic_psnr)
        metric_accum["total_dynamic_ssim"] += np.sum(batch_metrics.dynamic_ssim)

    if batch_metrics.valid_dynamic_depth_count > 0:
        metric_accum["total_valid_dynamic_depth_samples"] += batch_metrics.valid_dynamic_depth_count
        metric_accum["total_dynamic_rmse"] += np.sum(batch_metrics.dynamic_depth_rmse)

    metric_accum["total_depth_rmse_0_100"] += np.sum(batch_metrics.depth_rmse_0_100)
    metric_accum["total_depth_rmse_100_200"] += np.sum(batch_metrics.depth_rmse_100_200)
    metric_accum["total_valid_depth_samples_0_100"] += len(batch_metrics.depth_rmse_0_100)
    metric_accum["total_valid_depth_samples_100_200"] += len(batch_metrics.depth_rmse_100_200)

    if batch_metrics.dynamic_depth_rmse_0_100:
        metric_accum["total_dynamic_depth_rmse_0_100"] += np.sum(batch_metrics.dynamic_depth_rmse_0_100)
        metric_accum["total_valid_dynamic_depth_samples_0_100"] += batch_metrics.valid_dynamic_depth_0_100_count

    if batch_metrics.dynamic_depth_rmse_100_200:
        metric_accum["total_dynamic_depth_rmse_100_200"] += np.sum(batch_metrics.dynamic_depth_rmse_100_200)
        metric_accum["total_valid_dynamic_depth_samples_100_200"] += batch_metrics.valid_dynamic_depth_100_200_count

    return metric_accum


def _aggregate_metrics_tensors(metric_accum: Dict, device):
    """Convert metrics to tensors and aggregate across distributed processes."""
    tensors = {
        "total_psnr": torch.tensor(metric_accum["total_psnr"], dtype=torch.float32, device=device),
        "total_ssim": torch.tensor(metric_accum["total_ssim"], dtype=torch.float32, device=device),
        "total_depth_rmse": torch.tensor(metric_accum["total_depth_rmse"], dtype=torch.float32, device=device),
        "total_occupied_psnr": torch.tensor(metric_accum["total_occupied_psnr"], dtype=torch.float32, device=device),
        "total_occupied_ssim": torch.tensor(metric_accum["total_occupied_ssim"], dtype=torch.float32, device=device),
        "total_dynamic_psnr": torch.tensor(metric_accum["total_dynamic_psnr"], dtype=torch.float32, device=device),
        "total_dynamic_ssim": torch.tensor(metric_accum["total_dynamic_ssim"], dtype=torch.float32, device=device),
        "total_dynamic_rmse": torch.tensor(metric_accum["total_dynamic_rmse"], dtype=torch.float32, device=device),
        "total_depth_rmse_0_100": torch.tensor(metric_accum["total_depth_rmse_0_100"], dtype=torch.float32,
                                               device=device),
        "total_depth_rmse_100_200": torch.tensor(metric_accum["total_depth_rmse_100_200"], dtype=torch.float32,
                                                 device=device),
        "total_valid_depth_samples_0_100": torch.tensor(metric_accum["total_valid_depth_samples_0_100"], device=device),
        "total_valid_depth_samples_100_200": torch.tensor(metric_accum["total_valid_depth_samples_100_200"],
                                                          device=device),
        "total_dynamic_depth_rmse_0_100": torch.tensor(metric_accum["total_dynamic_depth_rmse_0_100"],
                                                       dtype=torch.float32, device=device),
        "total_dynamic_depth_rmse_100_200": torch.tensor(metric_accum["total_dynamic_depth_rmse_100_200"],
                                                         dtype=torch.float32, device=device),
        "total_valid_dynamic_depth_samples_0_100": torch.tensor(metric_accum["total_valid_dynamic_depth_samples_0_100"],
                                                                device=device),
        "total_valid_dynamic_depth_samples_100_200": torch.tensor(
            metric_accum["total_valid_dynamic_depth_samples_100_200"], device=device),
        "total_samples": torch.tensor(metric_accum["total_samples"], device=device),
        "total_dynamic_samples": torch.tensor(metric_accum["total_dynamic_samples"], device=device),
        "total_valid_dynamic_depth_samples": torch.tensor(metric_accum["total_valid_dynamic_depth_samples"],
                                                          device=device),
    }

    torch.cuda.synchronize()

    if distributed.is_enabled():
        for tensor in tensors.values():
            torch.distributed.all_reduce(tensor)

    return tensors


def _compute_and_save_final_metrics(tensors: Dict, eval_result_dir: str, name_str: str):
    """Compute final metrics, save results, and return result dictionary."""
    result = None

    if distributed.is_main_process():
        # Compute all averages
        eval_results = _compute_evaluation_results(tensors)

        # Save results to file
        _save_evaluation_results(eval_results, eval_result_dir, name_str)

        # Log results
        _log_evaluation_results(eval_results)

        # Build result dictionary
        result = {
            "psnr": eval_results.avg_psnr,
            "ssim": eval_results.avg_ssim,
            "depth_rmse_0_100m": eval_results.avg_depth_rmse_0_100,
            "depth_rmse_100_200m": eval_results.avg_depth_rmse_100_200,
            "depth_rmse": eval_results.avg_depth_rmse,
            "occupied_psnr": eval_results.avg_occupied_psnr,
            "occupied_ssim": eval_results.avg_occupied_ssim,
            "dynamic_psnr": eval_results.avg_dynamic_psnr,
            "dynamic_ssim": eval_results.avg_dynamic_ssim,
            "dynamic_depth_rmse_0_100m": eval_results.avg_dynamic_depth_rmse_0_100,
            "dynamic_depth_rmse_100_200m": eval_results.avg_dynamic_depth_rmse_100_200,
            "dynamic_depth_rmse": eval_results.avg_dynamic_rmse,
            "valid_depth_samples_0_100m": eval_results.valid_depth_samples_0_100,
            "valid_depth_samples_100_200m": eval_results.valid_depth_samples_100_200,
            "valid_dynamic_depth_samples_0_100m": eval_results.valid_dynamic_depth_samples_0_100,
            "valid_dynamic_depth_samples_100_200m": eval_results.valid_dynamic_depth_samples_100_200,
        }

    return result


def _compute_evaluation_results(tensors: Dict) -> EvaluationResults:
    """Compute all evaluation metrics from aggregated tensors."""
    # Batch convert all tensors to scalars to minimize CPU-GPU synchronization
    scalar_tensors = {k: v.item() for k, v in tensors.items()}

    # Compute basic metrics
    avg_psnr = scalar_tensors["total_psnr"] / scalar_tensors["total_samples"]
    avg_ssim = scalar_tensors["total_ssim"] / scalar_tensors["total_samples"]
    avg_depth_rmse = scalar_tensors["total_depth_rmse"] / scalar_tensors["total_samples"]
    avg_occupied_psnr = scalar_tensors["total_occupied_psnr"] / scalar_tensors["total_samples"]
    avg_occupied_ssim = scalar_tensors["total_occupied_ssim"] / scalar_tensors["total_samples"]
    avg_dynamic_psnr = scalar_tensors["total_dynamic_psnr"] / scalar_tensors["total_dynamic_samples"]
    avg_dynamic_ssim = scalar_tensors["total_dynamic_ssim"] / scalar_tensors["total_dynamic_samples"]
    avg_dynamic_rmse = scalar_tensors["total_dynamic_rmse"] / scalar_tensors["total_valid_dynamic_depth_samples"]

    # Compute range-specific depth metrics
    avg_depth_rmse_0_100 = _safe_divide(tensors["total_depth_rmse_0_100"], tensors["total_valid_depth_samples_0_100"])
    avg_depth_rmse_100_200 = _safe_divide(tensors["total_depth_rmse_100_200"],
                                          tensors["total_valid_depth_samples_100_200"])
    avg_dynamic_depth_rmse_0_100 = _safe_divide(
        tensors["total_dynamic_depth_rmse_0_100"],
        tensors["total_valid_dynamic_depth_samples_0_100"]
    )
    avg_dynamic_depth_rmse_100_200 = _safe_divide(
        tensors["total_dynamic_depth_rmse_100_200"],
        tensors["total_valid_dynamic_depth_samples_100_200"]
    )

    return EvaluationResults(
        avg_psnr=avg_psnr,
        avg_ssim=avg_ssim,
        avg_depth_rmse=avg_depth_rmse,
        avg_occupied_psnr=avg_occupied_psnr,
        avg_occupied_ssim=avg_occupied_ssim,
        avg_dynamic_psnr=avg_dynamic_psnr,
        avg_dynamic_ssim=avg_dynamic_ssim,
        avg_dynamic_rmse=avg_dynamic_rmse,
        avg_depth_rmse_0_100=avg_depth_rmse_0_100,
        avg_depth_rmse_100_200=avg_depth_rmse_100_200,
        avg_dynamic_depth_rmse_0_100=avg_dynamic_depth_rmse_0_100,
        avg_dynamic_depth_rmse_100_200=avg_dynamic_depth_rmse_100_200,
        total_samples=int(scalar_tensors["total_samples"]),
        valid_depth_samples_0_100=int(scalar_tensors["total_valid_depth_samples_0_100"]),
        valid_depth_samples_100_200=int(scalar_tensors["total_valid_depth_samples_100_200"]),
        valid_dynamic_depth_samples_0_100=int(scalar_tensors["total_valid_dynamic_depth_samples_0_100"]),
        valid_dynamic_depth_samples_100_200=int(scalar_tensors["total_valid_dynamic_depth_samples_100_200"]),
    )


def _save_evaluation_results(eval_results: EvaluationResults, eval_result_dir: str, name_str: str):
    """Save evaluation results to file."""
    filepath = os.path.join(eval_result_dir, f"eval_{name_str}.txt")

    with open(filepath, "w") as f:
        f.write(f"Average PSNR: {eval_results.avg_psnr:.4f}\n")
        f.write(f"Average SSIM: {eval_results.avg_ssim:.4f}\n")
        f.write(f"Average Depth RMSE (0.01-100m): {_format_metric(eval_results.avg_depth_rmse_0_100)}\n")
        f.write(f"Average Depth RMSE (100-200m): {_format_metric(eval_results.avg_depth_rmse_100_200)}\n")
        f.write(f"Average Depth RMSE (0.01-200m): {eval_results.avg_depth_rmse:.4f}\n")
        f.write(f"Average Occupied PSNR: {eval_results.avg_occupied_psnr:.4f}\n")
        f.write(f"Average Occupied SSIM: {eval_results.avg_occupied_ssim:.4f}\n")
        f.write(f"Average Dynamic PSNR: {eval_results.avg_dynamic_psnr:.4f}\n")
        f.write(f"Average Dynamic SSIM: {eval_results.avg_dynamic_ssim:.4f}\n")
        f.write(
            f"Average Dynamic Depth RMSE (0.01-100m): {_format_metric(eval_results.avg_dynamic_depth_rmse_0_100)}\n")
        f.write(
            f"Average Dynamic Depth RMSE (100-200m): {_format_metric(eval_results.avg_dynamic_depth_rmse_100_200)}\n")
        f.write(f"Average Dynamic Depth RMSE (0.01-200m): {eval_results.avg_dynamic_rmse:.4f}\n")
        f.write(f"Evaluated on {eval_results.total_samples} samples.\n")
        f.write(f"Valid depth samples (0.01-100m): {eval_results.valid_depth_samples_0_100}\n")
        f.write(f"Valid depth samples (100-200m): {eval_results.valid_depth_samples_100_200}\n")
        f.write(f"Valid dynamic depth samples (0.01-100m): {eval_results.valid_dynamic_depth_samples_0_100}\n")
        f.write(f"Valid dynamic depth samples (100-200m): {eval_results.valid_dynamic_depth_samples_100_200}\n")

    LOGGER.info(f"Evaluation results saved to {filepath}")


def _log_evaluation_results(eval_results: EvaluationResults):
    """Log evaluation results to console."""
    LOGGER.info(f"Evaluated on {eval_results.total_samples} samples.")
    LOGGER.info(f"Average PSNR: {eval_results.avg_psnr:.4f}, Average SSIM: {eval_results.avg_ssim:.4f}")
    LOGGER.info(
        f"Average Depth RMSE (0.01-100m): {_format_metric(eval_results.avg_depth_rmse_0_100)}, "
        f"Average Depth RMSE (100-200m): {_format_metric(eval_results.avg_depth_rmse_100_200)}"
    )
    LOGGER.info(f"Average Depth RMSE (0.01-200m): {eval_results.avg_depth_rmse:.4f}")
    LOGGER.info(
        f"Average Occupied PSNR: {eval_results.avg_occupied_psnr:.4f}, "
        f"Average Occupied SSIM: {eval_results.avg_occupied_ssim:.4f}"
    )
    LOGGER.info(
        f"Average Dynamic PSNR: {eval_results.avg_dynamic_psnr:.4f}, "
        f"Average Dynamic SSIM: {eval_results.avg_dynamic_ssim:.4f}"
    )
    LOGGER.info(
        f"Average Dynamic Depth RMSE (0.01-100m): {_format_metric(eval_results.avg_dynamic_depth_rmse_0_100)}, "
        f"Average Dynamic Depth RMSE (100-200m): {_format_metric(eval_results.avg_dynamic_depth_rmse_100_200)}"
    )
    LOGGER.info(f"Average Dynamic Depth RMSE (0.01-200m): {eval_results.avg_dynamic_rmse:.4f}")


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor):
    """Safe division, returns -1.0 if denominator is 0."""
    if denominator.item() > 0:
        return numerator.item() / denominator.item()
    return -1.0


def _format_metric(val):
    """Format metric value for logging."""
    if isinstance(val, (int, float)):
        return f"{val:.4f}"
    return str(val)


def get_numpy(tensor):
    """Convert tensor to numpy array."""
    return tensor.squeeze().detach().cpu().numpy()


def _init_eval_environment(model, args, name_str=None):
    torch.cuda.empty_cache()
    model.eval()
    device = next(model.parameters()).device

    eval_result_dir = os.path.join(args.log_dir, "eval_results")
    os.makedirs(eval_result_dir, exist_ok=True)
    LOGGER.info(f"Saving evaluation results to {eval_result_dir}")

    if name_str is None:
        name_str = datetime.now(tz=timezone.utc).strftime("%y-%m-%d-%H-%M")

    totals = {
        "flow_epes": 0.0,
        "flow_accs_strict": 0.0,
        "flow_accs_relax": 0.0,
        "flow_angles": 0.0,
        "flow_rmse": 0.0,
        "num_samples": 0.0,
    }
    return device, eval_result_dir, name_str, totals


def _prepare_eval_data(target_dict, pred_dict, args):
    b, t, v, height, width = target_dict["target_depth"].shape

    gt_depth = target_dict["target_depth"].view(b * t, -1, height, width)
    valid_depth_mask = (gt_depth > 0.01) & (gt_depth < 200)

    gt_ground_mask = None
    if args.load_ground and "target_ground_masks" in target_dict:
        gt_ground_mask = target_dict["target_ground_masks"].view(
            b * t, -1, height, width
        ).bool()

    return {
        "num_imgs": gt_depth.shape[0],
        "valid_depth_mask": valid_depth_mask,
        "gt_ground_mask": gt_ground_mask,
        "rendered_results": pred_dict["render_results"],
        "height": height,
        "width": width,
        "b": b,
        "t": t,
    }


def _should_eval_flow(eval_data, target_dict, args):
    return (
            "rendered_flow" in eval_data["rendered_results"]
            and args.decoder_type == "dummy"
            and args.load_flow
            and "target_flow" in target_dict
    )


def _get_flow_tensors(target_dict, eval_data):
    b, t = eval_data["b"], eval_data["t"]
    height, width = eval_data["height"], eval_data["width"]

    gt_flow = target_dict["target_flow"].view(b * t, -1, height, width, 3)
    pred_flow = eval_data["rendered_results"]["rendered_flow"].view(
        b * t, -1, height, width, 3
    )
    return gt_flow, pred_flow


def _get_valid_flow_points(
        i,
        gt_flow,
        pred_flow,
        valid_depth_mask,
        gt_ground_mask,
):
    if gt_ground_mask is not None:
        mask = ~gt_ground_mask[i] & valid_depth_mask[i]
    else:
        mask = valid_depth_mask[i]

    return gt_flow[i][mask], pred_flow[i][mask]


def _compute_single_flow_metrics(
        i,
        gt_flow,
        pred_flow,
        valid_depth_mask,
        gt_ground_mask,
):
    non_ground_gt_flow, non_ground_pred_flow = _get_valid_flow_points(
        i,
        gt_flow,
        pred_flow,
        valid_depth_mask,
        gt_ground_mask,
    )

    flow_metrics = compute_scene_flow_metrics(
        non_ground_pred_flow,
        non_ground_gt_flow,
    )

    rmse = torch.sqrt(
        F.mse_loss(
            pred_flow[i][valid_depth_mask[i]],
            gt_flow[i][valid_depth_mask[i]],
        )
    ).item()

    return {
        "epe": flow_metrics["EPE3D"],
        "acc_strict": flow_metrics["acc3d_strict"] * 100,
        "acc_relax": flow_metrics["acc3d_relax"] * 100,
        "angle": flow_metrics["angle_error"],
        "rmse": rmse,
    }


def _evaluate_flow_samples(
        gt_flow,
        pred_flow,
        valid_depth_mask,
        gt_ground_mask,
        num_imgs,
):
    metrics = {
        "epe": [],
        "acc_strict": [],
        "acc_relax": [],
        "angle": [],
        "rmse": [],
    }

    num_valid_samples = 0

    for i in range(1, num_imgs - 1):
        if torch.max(gt_flow[i].norm(dim=-1)) <= 1.0:
            continue

        sample_metrics = _compute_single_flow_metrics(
            i,
            gt_flow,
            pred_flow,
            valid_depth_mask,
            gt_ground_mask,
        )

        metrics["epe"].append(sample_metrics["epe"])
        metrics["acc_strict"].append(sample_metrics["acc_strict"])
        metrics["acc_relax"].append(sample_metrics["acc_relax"])
        metrics["angle"].append(sample_metrics["angle"])
        metrics["rmse"].append(sample_metrics["rmse"])
        num_valid_samples += 1

    return metrics, num_valid_samples


def _update_flow_totals(totals, metrics, num_valid_samples):
    totals["flow_epes"] += np.sum(metrics["epe"])
    totals["flow_accs_strict"] += np.sum(metrics["acc_strict"])
    totals["flow_accs_relax"] += np.sum(metrics["acc_relax"])
    totals["flow_angles"] += np.sum(metrics["angle"])
    totals["flow_rmse"] += np.sum(metrics["rmse"])
    totals["num_samples"] += num_valid_samples


def _update_progress_bar(pbar, totals):
    if totals["num_samples"] <= 0:
        return

    pbar.set_postfix(
        avg_flow_epe=totals["flow_epes"] / totals["num_samples"],
        avg_flow_acc_relax=totals["flow_accs_relax"] / totals["num_samples"],
        avg_flow_acc_strict=totals["flow_accs_strict"] / totals["num_samples"],
        avg_flow_angle=totals["flow_angles"] / totals["num_samples"],
        avg_flow_rmse=totals["flow_rmse"] / totals["num_samples"],
    )


def _create_metric_tensors(totals, device):
    return {
        "flow_epes": torch.tensor(
            totals["flow_epes"], dtype=torch.float32, device=device
        ),
        "flow_accs_strict": torch.tensor(
            totals["flow_accs_strict"], dtype=torch.float32, device=device
        ),
        "flow_accs_relax": torch.tensor(
            totals["flow_accs_relax"], dtype=torch.float32, device=device
        ),
        "flow_angles": torch.tensor(
            totals["flow_angles"], dtype=torch.float32, device=device
        ),
        "flow_rmse": torch.tensor(
            totals["flow_rmse"], dtype=torch.float32, device=device
        ),
        "num_samples": torch.tensor(
            totals["num_samples"], device=device
        ),
    }


def _reduce_metric_tensors(metric_tensors):
    if not distributed.is_enabled():
        return

    for tensor in metric_tensors.values():
        torch.distributed.all_reduce(tensor)


def _save_and_log_results(
        metric_tensors,
        eval_result_dir,
        name_str,
):
    sample_count = metric_tensors["num_samples"].item()
    if not (distributed.is_main_process() and sample_count > 0):
        return None

    avg_flow_epe = metric_tensors["flow_epes"].item() / sample_count
    avg_flow_acc_strict = metric_tensors["flow_accs_strict"].item() / sample_count
    avg_flow_acc_relax = metric_tensors["flow_accs_relax"].item() / sample_count
    avg_flow_angle = metric_tensors["flow_angles"].item() / sample_count
    avg_flow_rmse = metric_tensors["flow_rmse"].item() / sample_count

    with open(os.path.join(eval_result_dir, f"eval_{name_str}_flow.txt"), "w") as f:
        f.write(f"Average Flow EPE: {avg_flow_epe:.4f}\n")
        f.write(f"Average Flow Acc Strict: {avg_flow_acc_strict:.4f}\n")
        f.write(f"Average Flow Acc Relax: {avg_flow_acc_relax:.4f}\n")
        f.write(f"Average Flow Angle: {avg_flow_angle:.4f}\n")
        f.write(f"Average Flow RMSE: {avg_flow_rmse:.4f}\n")
        f.write(f"Evaluated on {sample_count} samples.\n")

    LOGGER.info("Evaluation results saved.")
    LOGGER.info(f"Evaluated on {sample_count} samples.")
    LOGGER.info(
        f"Average Flow EPE: {avg_flow_epe:.4f}, "
        f"Average Flow Acc Strict: {avg_flow_acc_strict:.4f}, "
        f"Average Flow Acc Relax: {avg_flow_acc_relax:.4f}, "
        f"Average Flow Angle: {avg_flow_angle:.4f}"
    )
    LOGGER.info(f"Average Flow RMSE: {avg_flow_rmse:.4f}")

    return {
        "flow_epe": avg_flow_epe,
        "flow_acc_strict": avg_flow_acc_strict,
        "flow_acc_relax": avg_flow_acc_relax,
        "flow_angle": avg_flow_angle,
        "flow_rmse": avg_flow_rmse,
    }


@torch.no_grad()
def evaluate_flow(dataloader, model, args, name_str=None):
    device, eval_result_dir, name_str, totals = _init_eval_environment(
        model,
        args,
        name_str,
    )
    pbar = tqdm(dataloader, desc="Evaluating Flow")
    for data_dict in pbar:
        input_dict, target_dict = prepare_inputs_and_targets(
            data_dict,
            device,
            v=args.num_max_cameras,
        )

        pred_dict = model(input_dict)
        eval_data = _prepare_eval_data(target_dict, pred_dict, args)

        if _should_eval_flow(eval_data, target_dict, args):
            gt_flow, pred_flow = _get_flow_tensors(target_dict, eval_data)

            metrics, num_valid_samples = _evaluate_flow_samples(
                gt_flow,
                pred_flow,
                eval_data["valid_depth_mask"],
                eval_data["gt_ground_mask"],
                eval_data["num_imgs"],
            )

            _update_flow_totals(
                totals,
                metrics,
                num_valid_samples,
            )

        _update_progress_bar(pbar, totals)

    metric_tensors = _create_metric_tensors(totals, device)

    torch.cuda.synchronize()
    _reduce_metric_tensors(metric_tensors)

    result = _save_and_log_results(
        metric_tensors,
        eval_result_dir,
        name_str,
    )

    torch.cuda.empty_cache()
    return result


def _init_semantic_eval(model, args, name_str=None):
    torch.cuda.empty_cache()
    model.eval()

    device = next(model.parameters()).device
    eval_result_dir = os.path.join(args.log_dir, "eval_results")
    os.makedirs(eval_result_dir, exist_ok=True)

    LOGGER.info(f"Saving evaluation results to {eval_result_dir}")

    if name_str is None:
        name_str = datetime.now(tz=timezone.utc).strftime("%y-%m-%d-%H-%M")

    totals = {
        "semantic_miou": 0.0,
        "semantic_acc": 0.0,
        "num_samples": 0.0,
    }

    return device, eval_result_dir, name_str, totals


def _prepare_semantic_eval_data(target_dict, pred_dict):
    b, t, v, height, width = target_dict["target_depth"].shape

    gt_depth = target_dict["target_depth"].view(b * t, -1, height, width)

    return {
        "b": b,
        "t": t,
        "height": height,
        "width": width,
        "num_imgs": gt_depth.shape[0],
        "rendered_results": pred_dict["render_results"],
    }


def _should_eval_semantic(eval_data, target_dict):
    rendered_results = eval_data["rendered_results"]

    has_prediction = (
        "rendered_semantic" in rendered_results
        if os.getenv("CONTEXT_FEAT")
        else "rendered_feat" in rendered_results
    )

    return has_prediction and "target_semantic_labels" in target_dict


def _build_pred_semantic(rendered_results, size_dict, args):
    b = size_dict["b"]
    t = size_dict["t"]
    height = size_dict["height"]
    width = size_dict["width"]
    if os.getenv("CONTEXT_FEAT"):
        return rendered_results["rendered_semantic"].view(
            b * t,
            -1,
            height,
            width,
        )

    pred_feat = rendered_results["rendered_feat"]
    pred_feat = einops.rearrange(
        pred_feat,
        "b t v h w c -> (b t v h w) c",
    )

    pred_semantic = feat2class(
        pred_feat,
        get_text_label_feats(SEMANTIC_LABEL_LIST),
        args.similarity_probs_threshold,
    )

    return pred_semantic.view(
        b * t,
        -1,
        height,
        width,
    )


def _get_semantic_tensors(target_dict, eval_data, args):
    b = eval_data["b"]
    t = eval_data["t"]
    height = eval_data["height"]
    width = eval_data["width"]

    gt_semantic = target_dict["target_semantic_labels"].view(
        b * t,
        -1,
        height,
        width,
    ).long()

    pred_semantic = _build_pred_semantic(
        eval_data["rendered_results"],
        {
            "b": b,
            "t": t,
            "height": height,
            "width": width,
        },
        args,
    )

    return gt_semantic, pred_semantic


def _evaluate_semantic_samples(
        gt_semantic,
        pred_semantic,
        num_imgs,
):
    semantic_miou = []
    semantic_acc = []
    num_valid_samples = 0

    for i in range(num_imgs):
        if gt_semantic[i].max() <= 0:
            continue

        metrics = compute_semantic_metrics(
            pred_semantic[i],
            gt_semantic[i],
        )

        semantic_miou.append(metrics["MIOU"])
        semantic_acc.append(metrics["ACC"])
        num_valid_samples += 1

    return {
        "miou": semantic_miou,
        "acc": semantic_acc,
        "num_samples": num_valid_samples,
    }


def _update_semantic_totals(totals, metrics):
    totals["semantic_miou"] += np.sum(metrics["miou"])
    totals["semantic_acc"] += np.sum(metrics["acc"])
    totals["num_samples"] += metrics["num_samples"]


def _update_semantic_progress(pbar, totals):
    if totals["num_samples"] <= 0:
        return

    pbar.set_postfix(
        avg_semantic_miou=totals["semantic_miou"] / totals["num_samples"],
        avg_semantic_acc=totals["semantic_acc"] / totals["num_samples"],
    )


def _create_semantic_tensors(totals, device):
    return {
        "semantic_miou": torch.tensor(
            totals["semantic_miou"],
            dtype=torch.float32,
            device=device,
        ),
        "semantic_acc": torch.tensor(
            totals["semantic_acc"],
            dtype=torch.float32,
            device=device,
        ),
        "num_samples": torch.tensor(
            totals["num_samples"],
            device=device,
        ),
    }


def _reduce_semantic_tensors(metric_tensors):
    if not distributed.is_enabled():
        return

    for tensor in metric_tensors.values():
        torch.distributed.all_reduce(tensor)


def _save_semantic_results(
        metric_tensors,
        eval_result_dir,
        name_str,
):
    sample_count = metric_tensors["num_samples"].item()

    if not (distributed.is_main_process() and sample_count > 0):
        return None

    avg_semantic_miou = (
            metric_tensors["semantic_miou"].item() / sample_count
    )
    avg_semantic_acc = (
            metric_tensors["semantic_acc"].item() / sample_count
    )

    with open(
            os.path.join(eval_result_dir, f"eval_{name_str}_semantic.txt"),
            "w",
    ) as f:
        f.write(f"Average Semantic mIOU: {avg_semantic_miou:.4f}\n")
        f.write(f"Average Semantic Accuracy: {avg_semantic_acc:.4f}\n")
        f.write(f"Evaluated on {sample_count} samples.\n")

    LOGGER.info("Evaluation results saved.")
    LOGGER.info(f"Evaluated on {sample_count} samples.")
    LOGGER.info(
        f"Average Semantic mIOU: {avg_semantic_miou:.4f}, "
        f"Average Semantic Accuracy: {avg_semantic_acc:.4f}"
    )

    return {
        "semantic_miou": avg_semantic_miou,
        "semantic_acc": avg_semantic_acc,
    }


@torch.no_grad()
def evaluate_semantic(dataloader, model, args, name_str=None, feat_extractor=None):
    device, eval_result_dir, name_str, totals = _init_semantic_eval(model, args, name_str, )

    pbar = tqdm(dataloader, desc="Evaluating")

    for data_dict in pbar:
        input_dict, target_dict = prepare_inputs_and_targets(
            data_dict,
            device,
            v=args.num_max_cameras,
            feat_extractor=feat_extractor,
        )

        pred_dict = model(input_dict)
        eval_data = _prepare_semantic_eval_data(
            target_dict,
            pred_dict,
        )

        if _should_eval_semantic(eval_data, target_dict):
            gt_semantic, pred_semantic = _get_semantic_tensors(
                target_dict,
                eval_data,
                args,
            )

            metrics = _evaluate_semantic_samples(
                gt_semantic,
                pred_semantic,
                eval_data["num_imgs"],
            )

            _update_semantic_totals(
                totals,
                metrics,
            )

        _update_semantic_progress(
            pbar,
            totals,
        )

    metric_tensors = _create_semantic_tensors(
        totals,
        device,
    )

    torch.cuda.synchronize()
    _reduce_semantic_tensors(metric_tensors)

    result = _save_semantic_results(
        metric_tensors,
        eval_result_dir,
        name_str,
    )

    torch.cuda.empty_cache()
    return result
