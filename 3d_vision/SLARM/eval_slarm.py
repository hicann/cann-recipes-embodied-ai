# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) NVIDIA. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os

import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    logging.info("Running on Ascend NPU.")
except ImportError:
    logging.info("Import torch_npu failed.")
import torch.backends.cudnn as cudnn
import torch.distributed
import torch.utils.data

import src.utils.distributed as distributed
import src.utils.misc as misc
from engine_tools import build_model, evaluate, evaluate_flow, evaluate_semantic
from src.utils.parser import get_args_parser
from src.dataset.constants import DATASET_DICT
from src.dataset.datasets import PerceptualModelDatasetEval, CustomConcatDataset
from src.dataset.samplers import NoPaddingDistributedSampler
from src.utils.logging import WandbLogger, TensorboardLogger, setup_logging

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
cudnn.benchmark = True

logger = None


def build_eval_dataloaders(args, world_size, global_rank):
    """Build the reconstruction/semantic and scene-flow evaluation dataloaders.

    Returns (data_loader_eval, data_loader_eval_flow) or (None, None) when none
    of the requested datasets provide a validation annotation.
    """
    eval_datasets = []
    eval_flow_datasets = []
    dataset_names_list = []

    for dataset_name in args.dataset:
        val_annotation = DATASET_DICT[dataset_name]["annotation_txt_file_val"]
        if val_annotation is None:
            continue
        if dataset_name == "nuscenes":
            val_annotation = f"{args.data_root}/dataset_scene_list/nuscenes_val.txt"
        else:
            val_annotation = f"{args.data_root}/{val_annotation}"
        if not os.path.exists(val_annotation):
            logger.info(f"Skipping dataset {dataset_name}: annotation not found at {val_annotation}")
            continue

        common_kwargs = dict(
            data_root=args.data_root,
            annotation_txt_file_list=val_annotation,
            target_size=args.input_size,
            num_context_timesteps=args.num_context_timesteps,
            num_target_timesteps=args.num_target_timesteps,
            timespan=args.timespan,
            num_max_cams=args.num_max_cameras,
            load_depth=args.load_depth,
            load_flow=args.load_flow,
            load_ground_label=args.load_ground,
            load_semantic_label=args.load_semantic_label,
            skip_sky_mask=args.skip_sky_mask,
        )
        # dataloader for evaluation (reconstruction and semantic)
        eval_datasets.append(PerceptualModelDatasetEval(load_dynamic_mask=True, **common_kwargs))
        # dataloader for evaluation (scene flow)
        eval_flow_datasets.append(PerceptualModelDatasetEval(
            load_dynamic_mask=False, return_context_as_target=False, **common_kwargs))
        dataset_names_list.append(dataset_name)
        logger.info(f"Processing dataset: {dataset_name}, val: {val_annotation}")

    if not eval_datasets:
        return None, None

    def concat(dsets):
        if len(dsets) > 1:
            return CustomConcatDataset(dsets, dataset_names=dataset_names_list)
        return dsets[0]

    def make_loader(dset):
        sampler = NoPaddingDistributedSampler(
            dset, num_replicas=world_size, rank=global_rank, shuffle=False)
        return torch.utils.data.DataLoader(
            dset,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            sampler=sampler,
            pin_memory=False,
            persistent_workers=True,
            shuffle=False,
            drop_last=False,
        )

    return make_loader(concat(eval_datasets)), make_loader(concat(eval_flow_datasets))


def run_evaluation(args, model, data_loader_eval, data_loader_eval_flow, log_writer):
    """Run the same evaluation suite as the training script and report metrics."""
    eval_result = evaluate(data_loader_eval, model, args)
    if log_writer is not None and eval_result is not None:
        log_writer.update({f"eval/{k}": v for k, v in eval_result.items()})

    if "waymo" in args.dataset or "driving_sim" in args.dataset:
        if args.decoder_type != "conv":
            flow_eval_result = evaluate_flow(data_loader_eval_flow, model, args)
            if log_writer is not None and flow_eval_result is not None:
                log_writer.update({f"eval/{k}": v for k, v in flow_eval_result.items()})

    if args.load_semantic_label:
        if "waymo" in args.dataset or "b2d" in args.dataset or "driving_sim" in args.dataset:
            semantic_eval_result = evaluate_semantic(data_loader_eval, model, args)
            if log_writer is not None and semantic_eval_result is not None:
                log_writer.update({f"eval/{k}": v for k, v in semantic_eval_result.items()})

    logger.info("Evaluation done, exiting.")


def _init_distributed_and_device(args):
    """Enable distributed runtime, resolve the device, and return world metadata."""
    distributed.enable(overwrite=True)
    device = torch.device(args.device)
    world_size = distributed.get_world_size()
    global_rank = distributed.get_global_rank()
    misc.fix_random_seeds(args.seed + global_rank)
    return device, world_size, global_rank


def _resolve_output_paths(args):
    """Compute and attach log / checkpoint / video directories to args."""
    args.exp_name = args.model.replace("/", "-") if args.exp_name is None else args.exp_name
    log_dir = os.path.join(args.output_dir, args.project, args.exp_name)
    args.log_dir = log_dir
    args.ckpt_dir = os.path.join(log_dir, "checkpoints")
    args.video_dir = os.path.join(log_dir, "videos")
    return log_dir


def _create_output_dirs(log_dir, args):
    """Ensure output directories exist (rank-0 only)."""
    for d in [log_dir, args.ckpt_dir, args.video_dir]:
        os.makedirs(d, exist_ok=True)


def _build_log_writer(args, log_dir):
    """Create a Wandb or Tensorboard log writer (rank-0 only, mutually exclusive)."""
    if args.enable_wandb:
        run_id_path = os.path.join(log_dir, "wandb_run_id.txt")
        run_id = None
        if os.path.exists(run_id_path) and not args.overwrite_wandb:
            with open(run_id_path, "r") as f:
                run_id = f.readlines()[-1].strip()
        log_writer = WandbLogger(args=args, resume="must", id=run_id)
        if run_id is None:
            with open(run_id_path, "a") as f:
                f.write(log_writer.run_id + "\n")
        return log_writer
    if args.enable_tensorboard:
        return TensorboardLogger(args=args)
    return None


def _setup_logger_and_save_args(args, log_dir, global_rank):
    """Initialise file + console logging, dump runtime args, and return the logger."""
    setup_logging(output=log_dir, level=logging.INFO)
    _logger = logging.getLogger("SLARM")
    _logger.info(f"hostname: {os.uname().nodename}\n")
    _logger.info(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    _logger.info(f"Logging to {log_dir}")
    _logger.info(json.dumps(args.__dict__, indent=4, sort_keys=True))

    if global_rank == 0:
        with open(os.path.join(log_dir, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=4)

    return _logger


def _build_and_load_model(args, device):
    """Instantiate the model, report parameter count, move to device, and load weights."""
    model = build_model(args)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{args.model} Parameters: {n_params / 1e6:.2f}M ({n_params:,})")
    model.to(device)
    misc.load_model(args, model)
    return model


def main(args):
    global logger
    device, world_size, global_rank = _init_distributed_and_device(args)
    log_dir = _resolve_output_paths(args)
    log_writer = None
    if global_rank == 0:
        _create_output_dirs(log_dir, args)
        log_writer = _build_log_writer(args, log_dir)
    logger = _setup_logger_and_save_args(args, log_dir, global_rank)
    data_loader_eval, data_loader_eval_flow = build_eval_dataloaders(args, world_size, global_rank)
    if data_loader_eval is None:
        logger.error("No validation annotation found for the requested datasets; nothing to evaluate.")
        return
    model = _build_and_load_model(args, device)
    run_evaluation(args, model, data_loader_eval, data_loader_eval_flow, log_writer)
    if log_writer is not None:
        log_writer.finish()


if __name__ == "__main__":
    if os.getenv("DETERMINISTIC"):
        torch.use_deterministic_algorithms(True)
    args = get_args_parser().parse_args()
    if not args.evaluate:
        logging.info("[eval_slarm_v2] forcing --evaluate; this entrypoint only runs evaluation.")
        args.evaluate = True
    main(args)