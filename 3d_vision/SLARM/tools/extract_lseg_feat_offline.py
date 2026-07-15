# coding=utf-8

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) NVIDIA. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from glob import glob
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from modules.lseg_module import LSegModule
from modules.models.lseg_blocks import forward_vit

FEAT_H, FEAT_W = 168, 252  # Saved feature height and width, corresponding to training image dimensions


def load_model():
    torch.manual_seed(1)  # seed = 1, consistent with open source demo

    model = LSegModule.load_from_checkpoint(checkpoint_path='checkpoints/demo_e200.ckpt', data_path='../datasets/',
                                            dataset='ade20k', backbone='clip_vitl16_384', aux=False, num_features=256,
                                            aux_weight=0, se_loss=False, se_weight=0, base_lr=0, batch_size=1,
                                            max_epochs=0, ignore_index=255, dropout=0.0, scale_inv=False, augment=False,
                                            no_batchnorm=False, widehead=True, widehead_hr=False, map_locatin="cpu",
                                            arch_option=0, block_depth=0, activation='lrelu', )

    model.eval()

    model.mean = [0.5, 0.5, 0.5]
    model.std = [0.5, 0.5, 0.5]
    return model.net


def extract_lseg_feat(lseg_model_, x, save_path_):
    layer_1, layer_2, layer_3, layer_4 = forward_vit(lseg_model_.pretrained, x)

    layer_1_rn = lseg_model_.scratch.layer1_rn(layer_1)
    layer_2_rn = lseg_model_.scratch.layer2_rn(layer_2)
    layer_3_rn = lseg_model_.scratch.layer3_rn(layer_3)
    layer_4_rn = lseg_model_.scratch.layer4_rn(layer_4)

    path_4 = lseg_model_.scratch.refinenet4(layer_4_rn)
    path_3 = lseg_model_.scratch.refinenet3(path_4, layer_3_rn)
    path_2 = lseg_model_.scratch.refinenet2(path_3, layer_2_rn)
    path_1 = lseg_model_.scratch.refinenet1(path_2, layer_1_rn)

    image_features = lseg_model_.scratch.head1(path_1)  # torch.Size([1, 512, 640, 960])

    image_features = image_features / image_features.norm(dim=1, keepdim=True)

    image_features = F.interpolate(
        image_features,
        size=(FEAT_H, FEAT_W),
        mode='nearest'
    )

    np.savez(save_path_, x=image_features.squeeze(0).cpu().numpy())  # 512, 168, 252


lseg_model = load_model().cuda()

# use 089 scene currently
image_paths = glob('xxx/SLARM_data/datasets/waymo/training/089/images/*_0.jpg')
image_paths.sort()
logging.info(len(image_paths))

SAVE_FEAT_PATH = 'xxx/SLARM_data/datasets/waymo/training/089/features/lseg'

for image_path in tqdm(image_paths):
    with Image.open(image_path) as image:
        to_tensor = transforms.ToTensor()
        img_tensor = to_tensor(image).cuda()
        img_tensor = img_tensor.unsqueeze(0)  # torch.Size([1, 3, 1280, 1920])
        feat_name = image_path.split('/')[-1].strip().replace('jpg', 'npz')
        save_path = os.path.join(SAVE_FEAT_PATH, feat_name)
        with torch.no_grad():
            extract_lseg_feat(lseg_model, img_tensor, save_path)

        pass
