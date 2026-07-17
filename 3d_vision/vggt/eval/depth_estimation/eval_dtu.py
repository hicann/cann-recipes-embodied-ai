# coding=utf-8
# Adapted from https://github.com/alibaba/cascade-stereo/blob/master/CasMVSNet/test.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) 2019 Alibaba. All rights reserved.
# Licensed under MIT.

import os
import sys
import gc
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn.parallel
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from vggt.models.vggt import VGGT
from vggt.utils.cast_weight import cast_model_weight
from dataset_utils.data_io import tocuda, write_cam, save_pfm
from dataset_utils.dtu import DTUDataset
from general_utils import fix_random_seed, get_depth_estimation_opts, sync_and_get_time, SPConfig, setup_distributed

logging.basicConfig(level=logging.INFO)


def model_inference(model, data, dtype):
    with torch.cuda.amp.autocast(dtype=dtype):
        with torch.no_grad():
            images = torch.stack([view for view in data['imgs']], dim=0).permute(1, 0, 2, 3, 4)
            num_images = images.shape[0] * images.shape[1]
            start_time = sync_and_get_time()
            predictions = model(images)
            exec_time = sync_and_get_time(start_time, log_result=True, num_images=num_images)
            pred_depth = predictions['depth']
            pred_depth_conf = predictions['depth_conf']
            return pred_depth, pred_depth_conf, exec_time


def main(model, test_img_loader, device, args, dtype):
    exec_time_list = []
    for _, sample in enumerate(test_img_loader):
        sample_cuda = tocuda(sample)
        pred_depth, pred_depth_conf, exec_time = model_inference(model, sample_cuda, dtype)
        exec_time_list.append(exec_time)
        del sample_cuda
        filenames = sample["filename"]
        cams = sample["proj_matrices"].numpy()
        imgs = sample["imgs"][0]
        gt_depths = sample['depth']
        for filename, cam, img, gt_depth, depth_est, depth_conf in \
            zip(filenames, cams, imgs, gt_depths, pred_depth, pred_depth_conf):
            img = img.numpy()   
            cam = cam[0]            
            depth_est = depth_est.to("cpu")[0]
            depth_conf = depth_conf.to("cpu")[0]


            depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
            gt_depth_filename = os.path.join(args.outdir, filename.format('depth_gt', '.pfm'))
            confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
            cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
            img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
            os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
            os.makedirs(gt_depth_filename.rsplit('/', 1)[0], exist_ok=True)
            os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
            os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
            os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
            
            save_pfm(depth_filename, depth_est.to(torch.float))
            save_pfm(gt_depth_filename, gt_depth.to(torch.float))
            save_pfm(confidence_filename, depth_conf.to(torch.float)) 
            write_cam(cam_filename, cam)
            img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_filename, img_bgr)

    if exec_time_list:
        avg_time = sum(exec_time_list) / len(exec_time_list)
        logging.info(f"Execution times (ms): {[t*1000 for t in exec_time_list]}")
        logging.info(f"Average inference time: {avg_time*1000:.2f} ms ({avg_time:.4f} s)")

    return len(test_img_loader)


if __name__ == '__main__':
    # Set random seeds
    fix_random_seed(42)
    
    # Parse command-line arguments
    args = get_depth_estimation_opts()
    
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
    test_dataset = DTUDataset(args.testpath, args.testlist, args.n_views, max_wh=(518, 518))
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    
    torch.npu.set_compile_mode(jit_compile=False)
    
    # Run evaluation
    main(model, TestImgLoader, device, args, dtype)
    
    # Synchronize if using distributed training
    if args.use_sp:
        dist.barrier()
    
    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()