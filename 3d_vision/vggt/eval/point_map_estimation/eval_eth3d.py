# coding=utf-8
# Adapted from https://github.com/wzzheng/StreamVGGT/blob/main/src/eval/mv_recon/launch.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright [2025–present] StreamVGGT. All rights reserved.
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License.  
#
# --------------------------------------------------------
import os
import sys
import os.path as osp
import logging

from tqdm import tqdm

import torch
import torch.distributed as dist
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch.utils.data._utils.collate import default_collate


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from dataset_utils.criterion import Regr3D_t_ScaleShiftInv, L21
from dataset_utils.eth3d import ETH3D
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.cast_weight import cast_model_weight
from general_utils import fix_random_seed, get_point_map_estimation_opts, sync_and_get_time, SPConfig, setup_distributed
from perf_metric import calc_performance
from utils import transfer_data_between_devices, \
    denormalize_image, extract_pts3d, projection_alignment, get_pcd, write_pcd
    
logging.basicConfig(level=logging.INFO)


def model_inference(model, data, dtype, use_proj):
    with torch.cuda.amp.autocast(dtype=dtype):
        with torch.no_grad():
            images = torch.stack([view['img'] for view in data], dim=0).permute(1, 0, 2, 3, 4)
            num_images = images.shape[0] * images.shape[1]
            start_time = sync_and_get_time()
            predictions = model(images)
            exec_time = sync_and_get_time(start_time, log_result=True, num_images=num_images)

            batch_size, seq_len = images.shape[:2]
            ress = []

            for s in range(seq_len):
                res = {
                    'pts3d_in_other_view': predictions['world_points'][:, s],  # [B, H, W, 3]
                    'conf': predictions['world_points_conf'][:, s],  # [B, H, W]

                    'depth': predictions['depth'][:, s],  # [B, H, W, 1]
                    'depth_conf': predictions['depth_conf'][:, s],  # [B, H, W]
                    'camera_pose': predictions['pose_enc'][:, s, :],  # [B, 9]
                }
                ress.append(res)
            
            preds, data = ress, data

            transfer_data_between_devices(preds, "preds", "cpu")

            if use_proj:
                pose_enc = torch.stack([preds[s]["camera_pose"] for s in range(len(preds))], dim=1)
                depth_map = torch.stack([preds[s]["depth"] for s in range(len(preds))], dim=1)
                depth_conf = torch.stack([preds[s]["depth_conf"] for s in range(len(preds))], dim=1)
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc,
                                                                    data[0]["img"].shape[-2:])
                point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0),
                                                                                extrinsic.squeeze(0),
                                                                                intrinsic.squeeze(0))
                return preds, data, depth_conf, point_map_by_unprojection, exec_time
            return preds, data, None, None, exec_time


def main(model, dataset, device, args, dtype):
    name_data = "ETH3D"

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    with torch.no_grad():
        save_path = osp.join(args.output_dir, name_data)
        acc_all = 0
        comp_all = 0
        nc1_all = 0
        nc2_all = 0
        exec_time_list = []

        for data_idx in tqdm(range(len(dataset))):
            batch = default_collate([dataset[data_idx]])
            scene_id = batch[0]['label'][0].rsplit("/", 1)[0]
            transfer_data_between_devices(batch, "input", device)
            denormalize_image(batch, dtype)
            preds, batch, depth_conf, point_map_by_unprojection, exec_time = \
                model_inference(model, batch, dtype, args.use_proj)
            exec_time_list.append(exec_time)
            logging.info(f"Evaluation for {name_data} {data_idx+1}/{len(dataset)}")

            transfer_data_between_devices(batch, "input", "cpu")

            pred_result = [preds, point_map_by_unprojection, depth_conf]

            pts_all_masked, pts_gt_all_masked, images_all_masked = \
               extract_pts3d(criterion, batch, pred_result, args.use_proj)
            
            if args.use_proj:
                pts_all_masked, pts_gt_all_masked, pts_all_aligned = \
                    projection_alignment(pts_all_masked, pts_gt_all_masked)

            pcd, pcd_gt = get_pcd(pts_all_masked, pts_gt_all_masked, images_all_masked)

            write_pcd(pcd, pcd_gt, save_path, scene_id)

            acc, nc1, comp, nc2 = calc_performance(pcd, pcd_gt)
            logging.info(f"Accuracy: {acc}, NC1: {nc1}, "
                        f"Completeness: {comp}, NC2: {nc2}")
            
            acc_all += acc
            comp_all += comp
            nc1_all += nc1
            nc2_all += nc2


            torch.cuda.empty_cache()
        acc_mean = acc_all / len(dataset)
        comp_mean = comp_all / len(dataset)
        overall_mean = (acc_mean + comp_mean) / 2
        logging.info(f"Final Results--Accuracy:{acc_mean} | "
                    f"Completeness: {comp_mean} | Overall: {overall_mean}")

        if exec_time_list:
            avg_time = sum(exec_time_list) / len(exec_time_list)
            logging.info(f"Execution times (ms): {[t*1000 for t in exec_time_list]}")
            logging.info(f"Average inference time: {avg_time*1000:.2f} ms ({avg_time:.4f} s)")
        

if __name__ == "__main__":
    # Set random seeds
    fix_random_seed(42)
    
    # Parse command-line arguments
    args = get_point_map_estimation_opts()
    
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
    else:
        # Standard mode with optional quantization
        if args.enableW8A8:
            model = torch.load(checkpoint_path, map_location=device)
            model.to(device).eval()
        else:
            model = VGGT()
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint)
            model = model.to(dtype)
            model.to(device).eval()
            model = cast_model_weight(model)
    
    # Load dataset
    resolution = (518, 392)
    test_dataset = ETH3D(
        root_dir=args.dataset_dir,
        resolution=resolution,
        num_seq=1,
    )
    
    torch.npu.set_compile_mode(jit_compile=False)
    
    # Run evaluation
    main(model, test_dataset, device, args, dtype)
    
    # Synchronize if using distributed training
    if args.use_sp:
        dist.barrier()
    
    # Cleanup
    torch.cuda.empty_cache()