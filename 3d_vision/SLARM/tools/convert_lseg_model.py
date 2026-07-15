# coding=utf-8

# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) NVIDIA. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


'''
LSeg model conversion script, requires LSeg open source code setup: https://github.com/isl-org/lang-seg
Place this script in the LSeg code root directory and run it
Converts the open source pytorch_lightning-based model to a pytorch model
and saves weights needed for LSeg feature extraction
Subsequent model loading does not require extra code or open source environment
'''
import logging
import os
import torch
from third_party.lang_seg.modules.lseg_module import LSegModule

SAVE_PATH = "ckpts/lseg"  # path to save converted model
os.makedirs(SAVE_PATH, exist_ok=True)

torch.manual_seed(1)  # seed = 1, consistent with open source demo

model = LSegModule.load_from_checkpoint(
    checkpoint_path='ckpts/demo_e200.ckpt',
    data_path='../datasets/',
    dataset='ade20k',
    backbone='clip_vitl16_384',
    aux=False,
    num_features=256,
    aux_weight=0,
    se_loss=False,
    se_weight=0,
    base_lr=0,
    batch_size=1,
    max_epochs=0,
    ignore_index=255,
    dropout=0.0,
    scale_inv=False,
    augment=False,
    no_batchnorm=False,
    widehead=True,
    widehead_hr=False,
    map_locatin="cpu",
    arch_option=0,
    block_depth=0,
    activation='lrelu',
    weights_only=False,
)

model.eval()

model.mean = [0.5, 0.5, 0.5]
model.std = [0.5, 0.5, 0.5]

logging.info('pytorch_lightning model loaded.')

logging.info(f'saving lseg_model.scratch to {os.path.join(SAVE_PATH, "lseg_model_scratch.pth")}.')
torch.save(model.net.scratch.state_dict(), os.path.join(SAVE_PATH, "lseg_model_scratch.pth"))
logging.info('lseg_model.scratch saved.')

# Convert 1x1 conv to linear for better torch.compile support
logging.info('converting 1x1 conv to linear in lseg_model.pretrained.')
lseg_model_pretrained_state_dict = model.net.pretrained.state_dict()
lseg_model_pretrained_state_dict["act_postprocess1.1.weight"] = lseg_model_pretrained_state_dict[
    "act_postprocess1.3.weight"].squeeze()
lseg_model_pretrained_state_dict["act_postprocess1.1.bias"] = lseg_model_pretrained_state_dict[
    "act_postprocess1.3.bias"]
lseg_model_pretrained_state_dict["act_postprocess2.1.weight"] = lseg_model_pretrained_state_dict[
    "act_postprocess2.3.weight"].squeeze()
lseg_model_pretrained_state_dict["act_postprocess2.1.bias"] = lseg_model_pretrained_state_dict[
    "act_postprocess2.3.bias"]
lseg_model_pretrained_state_dict["act_postprocess3.1.weight"] = lseg_model_pretrained_state_dict[
    "act_postprocess3.3.weight"].squeeze()
del lseg_model_pretrained_state_dict["act_postprocess3.3.weight"]
lseg_model_pretrained_state_dict["act_postprocess3.1.bias"] = lseg_model_pretrained_state_dict[
    "act_postprocess3.3.bias"]
del lseg_model_pretrained_state_dict["act_postprocess3.3.bias"]
lseg_model_pretrained_state_dict["act_postprocess4.1.weight"] = lseg_model_pretrained_state_dict[
    "act_postprocess4.3.weight"].squeeze()
lseg_model_pretrained_state_dict["act_postprocess4.1.bias"] = lseg_model_pretrained_state_dict[
    "act_postprocess4.3.bias"]

lseg_model_pretrained_state_dict["act_postprocess1.3.weight"] = lseg_model_pretrained_state_dict[
    "act_postprocess1.4.weight"]
del lseg_model_pretrained_state_dict["act_postprocess1.4.weight"]
lseg_model_pretrained_state_dict["act_postprocess1.3.bias"] = lseg_model_pretrained_state_dict[
    "act_postprocess1.4.bias"]
del lseg_model_pretrained_state_dict["act_postprocess1.4.bias"]
lseg_model_pretrained_state_dict["act_postprocess2.3.weight"] = lseg_model_pretrained_state_dict[
    "act_postprocess2.4.weight"]
del lseg_model_pretrained_state_dict["act_postprocess2.4.weight"]
lseg_model_pretrained_state_dict["act_postprocess2.3.bias"] = lseg_model_pretrained_state_dict[
    "act_postprocess2.4.bias"]
del lseg_model_pretrained_state_dict["act_postprocess2.4.bias"]
lseg_model_pretrained_state_dict["act_postprocess4.3.weight"] = lseg_model_pretrained_state_dict[
    "act_postprocess4.4.weight"]
del lseg_model_pretrained_state_dict["act_postprocess4.4.weight"]
lseg_model_pretrained_state_dict["act_postprocess4.3.bias"] = lseg_model_pretrained_state_dict[
    "act_postprocess4.4.bias"]
del lseg_model_pretrained_state_dict["act_postprocess4.4.bias"]

lseg_model_path = os.path.join(SAVE_PATH, "lseg_model_pretrained_replace_1x1conv_with_linear.pth")
logging.info(
    f'saving converted lseg_model.pretrained to {lseg_model_path}.')
torch.save(lseg_model_pretrained_state_dict,
           os.path.join(SAVE_PATH, "lseg_model_pretrained_replace_1x1conv_with_linear.pth"))
logging.info('converted lseg_model.pretrained saved.')
