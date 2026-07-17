# coding=utf-8
# Adapted from
# https://github.com/facebookresearch/vggt/blob/evaluation/evaluation/test_co3d.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
import gc
import random
import logging

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from torch_npu.contrib import transfer_to_npu

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.cast_weight import cast_model_weight
from general_utils import fix_random_seed, get_pose_evaluation_opts, sync_and_get_time, SPConfig, setup_distributed
from dataset_prepare.categories import SEEN_CATEGORIES
from utils import convert_pt3d_rt_to_opencv, calculate_auc_np, \
    se3_to_relative_pose_error, load_annotation, list_per_category_downloaded_seq_names
from quant.vggt_utils import replace_linear_in_vggt, set_ignore_quantize
from quant.vggt_linear import LinearW8A8

logging.basicConfig(level=logging.INFO)


def model_inference(model, images, dtype):
    """Inference function"""
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            imgs = torch.unsqueeze(images, 0)  # B S C H W

            num_images = imgs.shape[0] * imgs.shape[1]
            start_time = sync_and_get_time()
            predictions = model(imgs)
            exec_time = sync_and_get_time(start_time, log_result=True, num_images=num_images)
            
            batch_size, seq_len = imgs.shape[:2]
            ress = []
            for s in range(seq_len):
                res = {'camara_pose': predictions['pose_enc'][:, s, :]}
                ress.append(res)
            predictions = ress
            
        pose_enc = torch.stack([predictions[s]['camara_pose'].to("cpu") for s in range(len(predictions))], dim=1)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, imgs.shape[-2:])
        pred_extrinsic = extrinsic[0]
    return pred_extrinsic, exec_time


def get_testing_sequences(co3d_anno_dir, co3d_dir, category):
    """Get testing sequences for a category"""
    logging.info(f"Loading annotation for {category} test set")
    annotation = load_annotation(co3d_anno_dir, category)
    if annotation is None:
        return {}
    downloaded_seq_names = list_per_category_downloaded_seq_names(co3d_dir, category)
    annotation_seq_names = list(annotation.keys())

    seq_names = sorted(list(set(downloaded_seq_names) & set(annotation_seq_names)))

    if len(seq_names) >= 10:
        seq_names = random.sample(seq_names, 10)

    logging.info(f"Testing Sequences: {seq_names}")

    seqs = {seq_name: annotation[seq_name] for seq_name in seq_names}
    return seqs


def process_sequence(model, seq_data, args, device, dtype):
    """
    Process a single sequence and compute pose errors.

    Args:
        model: VGGT model
        seq_data: Sequence data
        args: argument list
        device: Device to run on
        dtype: Data type for model inference

    Returns:
        rError: Rotation errors
        tError: Translation errors
    """
    co3d_dir = args.co3d_dir
    min_num_images = args.min_num_images
    num_frames = args.num_frames

    if len(seq_data) < min_num_images:
        return None, None, None

    metadata = []
    for data in seq_data:
        # Make sure translations are not ridiculous
        if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
            return None, None, None
        extri_opencv = convert_pt3d_rt_to_opencv(data["R"], data["T"])
        metadata.append({
            "filepath": data["filepath"],
            "extri": extri_opencv,
        })

    # random sample num_frames images
    ids = np.random.choice(len(metadata), num_frames, replace=False)
    image_names = [os.path.join(co3d_dir, metadata[i]["filepath"]) for i in ids]
    
    gt_extri = [np.array(metadata[i]["extri"]) for i in ids]
    gt_extri = np.stack(gt_extri, axis=0)

    images = load_and_preprocess_images(image_names).to(device)
    
    pred_extrinsic, exec_time = model_inference(model, images, dtype)
    
    gt_extrinsic = torch.from_numpy(gt_extri)
    pred_extrinsic = pred_extrinsic.detach().cpu()
    
    add_row = torch.tensor([0, 0, 0, 1], device="cpu").expand(pred_extrinsic.size(0), 1, 4)

    pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
    gt_se3 = torch.cat((gt_extrinsic, add_row), dim=1)
  
    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, num_frames)

    return rel_rangle_deg.cpu().numpy(), rel_tangle_deg.cpu().numpy(), exec_time


def process_category_sequences(model, device, args, category, dtype):
    """Process all sequences for a category"""
    r_error = []
    t_error = []
    time_list = []

    seq = get_testing_sequences(args.co3d_anno_dir, args.co3d_dir, category)

    for seq_name in seq:
        seq_data = seq[seq_name]
        logging.info(f"Processing {seq_name} for {category} test set")
        if args.debug and not os.path.exists(os.path.join(args.co3d_dir, category, seq_name)):
            logging.warning(f"Skip Processing {seq_name} as it does not exist in the dataset")
            continue

        seq_r_error, seq_t_error, seq_time = process_sequence(model, seq_data, args, device, dtype)

        if seq_r_error is not None and seq_t_error is not None:
            r_error.extend(seq_r_error)
            t_error.extend(seq_t_error)
            time_list.append(seq_time)

    return r_error, t_error, time_list


def main(model, device, args, dtype, categories):
    """Main function to evaluate VGGT on CO3D dataset."""
    
    if args.debug:
        categories = ["apple"]

    per_category_results = {}
    all_times = []

    for category in categories:
        r_error, t_error, cat_time_list = process_category_sequences(model, device, args, category, dtype)
        all_times.extend(cat_time_list)
        if not r_error:
            logging.warning(f"No valid sequences available for {category}, skip processing")
            continue

        r_error = np.array(r_error)
        t_error = np.array(t_error)
        auc_30, _ = calculate_auc_np(r_error, t_error, max_threshold=30)

        per_category_results[category] = {
            "rError": r_error,
            "tError": t_error,
            "Auc_30": auc_30,
        }

        # Print results with colors
        green_color = "\033[92m"
        red_color = "\033[91m"
        blue_color = "\033[94m"
        bold_color = "\033[1m"
        reset_color = "\033[0m"

        logging.info(f"{bold_color}{blue_color}AUC of {category} test set: "
                    f"{reset_color} {green_color}{auc_30:.4f} (AUC@30){reset_color}")
        mean_auc_30_by_now = np.mean([per_category_results[cat]["Auc_30"] for cat in per_category_results])
        logging.info(f"{bold_color}{blue_color}Mean AUC of categories by now: "
                    f"{reset_color} {red_color}{mean_auc_30_by_now:.4f} (AUC@30){reset_color}")

    # Print summary results
    logging.info("\nSummary of AUC results:")
    for category in sorted(per_category_results.keys()):
        logging.info(f"{category:<15}: {per_category_results[category]['Auc_30']:.4f} (AUC@30)")

    if per_category_results:
        mean_auc_30 = np.mean([per_category_results[category]["Auc_30"] for category in per_category_results])
        logging.info(f"Mean AUC: {mean_auc_30:.4f} (AUC@30)")

    if all_times:
        avg_time = sum(all_times) / len(all_times)
        logging.info(f"Execution times (ms): {[t*1000 for t in all_times]}")
        logging.info(f"Average inference time: {avg_time*1000:.2f} ms ({avg_time:.4f} s)")


if __name__ == "__main__":
    # Set random seeds
    fix_random_seed(42)
    
    # Parse command-line arguments
    args = get_pose_evaluation_opts()
    
    # Add default values for SP parameters if not present
    if not hasattr(args, 'use_sp'):
        args.use_sp = False
    if not hasattr(args, 'ulysses_degree'):
        args.ulysses_degree = 1
    if not hasattr(args, 'ring_degree'):
        args.ring_degree = 1
    
    # Setup distributed environment if using sequence parallel
    if args.use_sp:
        rank, world_size, ulysses_pg, ring_pg, global_pg = setup_distributed(args)
        device = f"npu:{rank}"
        torch.npu.set_device(device)
        logging.info(f"Initialized distributed training: rank={rank}, world_size={world_size}")
    else:
        rank = 0
        world_size = 1
        ulysses_pg = None
        ring_pg = None
        global_pg = None
        device = "npu:0" if torch.cuda.is_available() else "cpu"
    
    dtype = torch.bfloat16

    # Load model
    checkpoint_path = args.ckpt
    
    if args.use_sp:
        # Sequence parallel mode
        sp_config = SPConfig(
            ulysses_degree=args.ulysses_degree,
            ring_degree=args.ring_degree,
            use_ring_overlap=True,
        )
        
        model = VGGT(
            sp_config=sp_config,
            sp_ulysses_group=ulysses_pg,
            sp_ring_group=ring_pg,
            sp_global_group=global_pg,
        )
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model = model.to(dtype)
        model.to(device).eval()
        model = cast_model_weight(model)
        logging.info("Loaded model with sequence parallel configuration")
    else:
        # Standard mode with optional quantization
        if args.enableW8A8:
            # Load quantized model
            model = torch.load(checkpoint_path, map_location=device)
            model.to(device).eval()
            logging.info("Loaded W8A8 quantized model")
        else:
            # Load standard model
            model = VGGT()
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint)
            model = model.to(dtype)
            model.to(device).eval()
            model = cast_model_weight(model)
            logging.info("Loaded standard model")
    
    torch.npu.set_compile_mode(jit_compile=False)

    # Run evaluation
    main(model, device, args, dtype, SEEN_CATEGORIES)
    
    # Synchronize if using distributed training
    if args.use_sp:
        dist.barrier()
        logging.info("All processes finished")
    
    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()