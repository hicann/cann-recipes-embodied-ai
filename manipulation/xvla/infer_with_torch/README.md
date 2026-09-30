# XVLA Torch 推理昇腾使用指南

本目录提供 XVLA 在昇腾 NPU 上的 PyTorch 推理样例。当前样例复用 `manipulation/xvla/train` 中的 LeRobot 环境和 Ascend patch。

## 1. 环境准备

推理脚本复用训练 recipe 准备的外部 `lerobot` 仓库。无需手动 `pip install lerobot`；执行训练侧 `setup.sh` 后，会在 recipe 同级目录准备 `lerobot`，checkout 到固定 commit，应用 XVLA Ascend patch，并执行 editable install。

```bash
conda activate lerobot-xvla
source /usr/local/Ascend/cann-9.0.0/set_env.sh

# 如未准备 LeRobot，请先执行训练 recipe 的 setup
./manipulation/xvla/train/src/scripts/setup.sh
```

## 2. 真实 checkpoint 推理

准备 `lerobot/xvla-base` 或微调后的 XVLA checkpoint 到本地目录，例如：

```text
../models/lerobot/xvla-base
```


推理脚本会使用 `facebook/bart-large` tokenizer。离线环境需提前缓存 tokenizer 小文件：

```bash
hf download facebook/bart-large \
  config.json tokenizer_config.json vocab.json merges.txt tokenizer.json special_tokens_map.json
```

然后运行：

```bash
python manipulation/xvla/infer_with_torch/test_xvla_inference_ascend.py \
  --pretrained_model_name_or_path ../models/lerobot/xvla-base \
  --device npu:0 \
  --dtype bfloat16 \
  --domain_id 3 \
  --num_warmup 1 \
  --num_inference 3
```

期望输出包含：

```text
Action shape: torch.Size([1, 20])
Action dtype/device: torch.float32/npu:0
```

## 3. LIBERO HDF5 样本推理

脚本支持直接从 LIBERO HDF5 demo 中读取一帧数据构造 XVLA batch：

```bash
./manipulation/xvla/infer_with_torch/run_xvla_inference.sh \
  --pretrained_model_name_or_path ../models/lerobot/xvla-base \
  --device npu:0 \
  --dtype float32 \
  --num_warmup 0 \
  --num_inference 1 \
  --libero_hdf5 ../dataset/libero/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5 \
  --demo_key demo_0 \
  --frame_index 0
```

默认映射：

- `obs/agentview_rgb` -> `observation.images.image`
- `obs/eye_in_hand_rgb` -> `observation.images.image2`
- `robot_states[..., :8]` -> `observation.state`
- 未提供的第三路图像由 XVLA 内部按 `num_image_views` 补零并置无效 mask

已验证结果，Ascend 910B/NPU，`xvla-base`：

```text
float32: Action shape: torch.Size([1, 20]), Action dtype/device: torch.float32/npu:0, Average latency: 277.0563 ms  # warmup=1, inference=3
float16: Action shape: torch.Size([1, 20]), Action dtype/device: torch.float16/npu:0, Average latency: 1259.3652 ms
```

## 4. 精度验证与优化开关

本样例不仅验证模型可运行，也提供 NPU 优化路径与 fallback 路径的一致性检查。

当前优化项：

- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`：降低 NPU 显存碎片风险
- `ACLNN_CACHE_LIMIT=100000`、`HOST_CACHE_CAPACITY=20`：增加算子/host cache 容量
- `LEROBOT_XVLA_USE_NPU_FUSION_ATTENTION=1`：在 XVLA action transformer 中启用 `torch_npu.npu_fusion_attention`
- `LEROBOT_XVLA_USE_NPU_BADD_BMM=1`：在 domain-aware action encoder / decoder 中使用 `torch.baddbmm` 合并 batch matmul 与 bias add
- 推理主循环使用 `torch.inference_mode()`，减少 autograd 开销
- 支持 `--save_action` 保存输出，便于跨版本或跨设备比对

真实 `xvla-base + LIBERO HDF5` 的优化路径一致性已验证通过：fusion attention 与 baddbmm 同时开启后，cosine `1.000000`，MSE `0.00000000`，Max Abs Error `0.00000000`。

验证优化开关前后的输出一致性：

```bash
cd ../lerobot
python ../cann-recipes-embodied-ai/manipulation/xvla/infer_with_torch/verify_xvla_accuracy_ascend.py \
  --pretrained_model_name_or_path ../models/lerobot/xvla-base \
  --device npu:0 \
  --dtype float32 \
  --libero_hdf5 ../dataset/libero/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5 \
  --demo_key demo_0 \
  --frame_index 0
```
