# XVLA 模型训练昇腾迁移与性能优化说明

## 背景介绍

本案例在昇腾 Atlas A2 / 910B 上适配 LeRobot 社区的 `xvla` policy，并面向 LIBERO 数据集提供训练、推理、kernel benchmark、真实 checkpoint 推理和短步数训练验证。XVLA 已在 LeRobot 主线提供模型实现，因此本 recipe 采用外部 LeRobot 固定 commit + Ascend patch 的方式维护，避免复制大段上游代码。

## 模型训练性能分析与优化

XVLA 的主要计算热点集中在两处：

1. Florence2/VLM 图像编码与语言融合，真实 `xvla-base` 推理中占比较高；
2. action transformer 的多步 denoising，每个 denoising step 都会重复调用 soft-prompted transformer。

当前 recipe 先处理更稳定、风险更低且容易复现的 action transformer 路径。该路径不改变模型结构和权重语义，主要优化 NPU 上的 attention 执行方式和 domain-aware linear 执行方式，并补齐 NPU device、dtype、缓存和评测工具链。

### 优化项

| 优化项 | 默认状态 | 生效位置 | 作用原理 | 当前实测结论 |
| --- | --- | --- | --- | --- |
| `LEROBOT_XVLA_USE_NPU_FUSION_ATTENTION=1` | 默认开启 | `xvla/soft_transformer.py` | 在 action transformer attention 中使用 `torch_npu.npu_fusion_attention` 替换 eager attention，减少 attention 内部算子碎片化 | 真实 `xvla-base + HDF5` 一致性验证 cosine 为 `1.000000`；稳定 benchmark 中整体 transformer forward 提升 `1.83%`，train-step 提升 `1.95%` |
| `LEROBOT_XVLA_USE_NPU_BADD_BMM=1` | 默认开启 | `xvla/soft_transformer.py` | 在 `DomainAwareLinear` 中使用 `torch.baddbmm` 合并 batch matmul 与 bias add，覆盖 action encoder / decoder 路径 | 单独算子 benchmark 中 action encoder 约 `3.57%` 提升，action decoder 约 `5.81%` 提升；数值一致性 max abs `0.0` |
| `LEROBOT_XVLA_DISABLE_SDPA=1` | 默认关闭 | `xvla/soft_transformer.py` | 保留 PyTorch SDPA 和手写 attention 的 fallback 对照路径，便于定位 torch_npu fusion attention 版本兼容问题 | 用于问题定位，不推荐常态开启 |
| `policy.dtype: bfloat16/float16` | 配置控制 | `xvla/configuration_xvla.py`、`modeling_xvla.py` | 支持 Ascend 常用低精度推理/训练；当 BF16 栈异常时可用 `LEROBOT_XVLA_BF16_FALLBACK_FP16=1` 回退 FP16 | `xvla-base` HDF5 推理已验证 `float32` 与 `float16` 均可在 NPU 输出 action |
| `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` | 默认开启 | `run_train.sh`、`run_profiling.sh`、推理脚本 | 降低长时间训练/评测中的 NPU 显存碎片风险 | 保留为默认运行参数 |
| `ACLNN_CACHE_LIMIT=100000`、`HOST_CACHE_CAPACITY=20` | 默认开启 | 训练、profiling、推理脚本 | 增大算子与 host cache，减少重复编译/调度抖动 | 保留为默认运行参数 |
| `tokenizer_max_length: 64` / 推理不做 `max_length` padding | 默认开启 | 训练配置、推理脚本 | 避免语言 token padding 过长导致 XVLA 序列超过 `max_len_seq`，同时减少无效 token 计算 | 真实 tokenizer 曾触发 `1204 > max_len_seq=512`，去掉 max padding 后真实 HDF5 推理通过 |
| `policy.action_mode: auto` | 默认开启 | XVLA action space | 自动按数据集动作维度做 padding / trimming，避免 LIBERO 动作维度与预训练 action head 不一致 | 真实推理验证通过 |
| `policy.empty_cameras: 1` | LIBERO 配置开启 | XVLA image view 处理 | LIBERO 常见两路图像输入，XVLA base 配置为三视角；缺失视角由模型内部补零并置无效 mask | 真实 HDF5 推理使用两路图像，第三路由模型补齐 |

XVLA 的 action transformer 中 `q/k/v` 已经由单个 `nn.Linear(dim, 3 * dim)` 生成。当前优化重点是把这一路径接到 NPU fusion attention，减少 attention 主干的 kernel 数量；同时将 domain-conditioned action encoder / decoder 中的 `matmul + bias` 合并为 `baddbmm` 路径，减少小算子调度开销。两个优化都保留 fallback 路径用于版本兼容和精度验证。

### 推荐训练与 profiling 方式

无权重 NPU kernel benchmark：

```bash
cd cann-recipes-embodied-ai/manipulation/xvla/train/src/scripts
./run_profiling.sh --dtype float16 --num-iters 20
```

训练步 forward+backward benchmark：

```bash
cd cann-recipes-embodied-ai/manipulation/xvla/train/src/scripts
./run_profiling.sh --dtype float32 --train-step --num-iters 10
```

真实 checkpoint + LIBERO 训练：

```bash
./manipulation/xvla/train/src/scripts/run_train.sh xvla_libero --nproc 1 --port 29510
```

如果只验证训练链路，可先把训练配置中的 `steps` 和 `save_freq` 临时设置为 `10`。当前环境已确认真实 checkpoint 和 LeRobot 格式 LIBERO 数据集在 NPU 上完成 10 step 训练，并在 step 10 成功保存 checkpoint。

训练配置默认关闭 W&B，避免未配置 API key 的新环境在启动时失败。如需记录到 W&B，可在配置中设置 `wandb.enable: true` 并先执行 `wandb login`。

如果当前 `torch_npu` 版本的 fusion attention 异常，可直接回退：

```bash
LEROBOT_XVLA_USE_NPU_FUSION_ATTENTION=0 \
./manipulation/xvla/train/src/scripts/run_train.sh xvla_libero --nproc 1 --port 29510
```

### 从零复现推荐流程

1. 初始化代码与环境。

```bash
cd <your-workdir>
git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git
chmod +x cann-recipes-embodied-ai/manipulation/xvla/train/src/scripts/setup.sh
./cann-recipes-embodied-ai/manipulation/xvla/train/src/scripts/setup.sh
```

2. 激活环境并设置 CANN。

```bash
conda activate lerobot-xvla
source /usr/local/Ascend/cann-9.0.0/set_env.sh
```

3. 首次训练前，如果权重、数据集和 tokenizer 已缓存到本地，可启用离线模式，避免远端探测影响启动时间。

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

4. 先跑 kernel benchmark。

```bash
./manipulation/xvla/train/src/scripts/run_profiling.sh --dtype float16 --num-iters 20
```

5. 启动 LIBERO 训练。快速验证时可先使用 `--nproc 1`，并把配置中的 `steps`、`save_freq` 临时设为 `10`。

```bash
./manipulation/xvla/train/src/scripts/run_train.sh xvla_libero --nproc 1 --port 29510
```

6. 准备真实权重后跑真实 checkpoint/HDF5 推理验证；训练完成后，也可以把 `--pretrained_model_name_or_path` 指向训练输出目录下的 `checkpoints/<step>/pretrained_model`。

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

### 已完成测试与评测

当前环境中已完成以下验证：

| 测试项 | 命令/场景 | 结果 |
| --- | --- | --- |
| kernel benchmark forward | `run_profiling.sh --dtype float16 --num-warmup 5 --num-iters 20 --batch-size 2 --seq-len 128 --hidden-size 512 --num-heads 8 --depth 4 --chunk-size 30` | attention latency `0.5112 -> 0.6219 ms`，该独立 kernel shape 下无收益；transformer latency `5.5702 -> 5.4680 ms`，提升 `1.0187x` |
| kernel benchmark train-step | `run_profiling.sh --dtype float32 --train-step --num-warmup 3 --num-iters 10 --batch-size 1 --seq-len 128 --hidden-size 256 --num-heads 8 --depth 2 --chunk-size 16` | attention latency `1.6746 -> 1.5786 ms`，提升 `1.0608x`；transformer latency `10.0322 -> 9.8368 ms`，提升 `1.0199x` |
| domain linear benchmark | action encoder / decoder，`LEROBOT_XVLA_USE_NPU_BADD_BMM=0/1` | encoder latency `0.2332 -> 0.2249 ms`，提升 `1.0371x`；decoder latency `0.2418 -> 0.2277 ms`，提升 `1.0617x` |
| 真实训练链路 | `xvla-base + LeRobotDataset`，`steps=10`，`save_freq=10`，`--nproc 1` | 10 step 训练完成，`checkpoints/000010` 下成功写出 `pretrained_model/model.safetensors` 与 `training_state/training_step.json` |
| 训练后 checkpoint 推理 | step 10 输出的 `checkpoints/000010/pretrained_model`，`float32` | action shape `[1, 7]`，平均延迟约 `279.9 ms` |
| 真实 `xvla-base` mock 推理 | 3 路 mock 图像，`float32` | action shape `[1, 20]`，约 `1091.5 ms` |
| 真实 LIBERO HDF5 推理 | `demo_0/frame_0`，`float32`，warmup `1`，inference `3` | action shape `[1, 20]`，平均延迟约 `277.1 ms`，action 已保存到 `${XVLA_STORAGE_ROOT}/ckpt/actions/` |
| 真实 LIBERO HDF5 推理 | `demo_0/frame_0`，`float16` | action shape `[1, 20]`，约 `1259.4 ms` |
| 真实 `xvla-base` 优化一致性 | `verify_xvla_accuracy_ascend.py --pretrained_model_name_or_path ... --libero_hdf5 ...` | fusion attention + baddbmm 同时开启后，cosine `1.000000`，MSE `0.00000000`，Max Abs Error `0.00000000` |

### 输出位置与结果解读

- 训练日志位于 `${XVLA_STORAGE_ROOT:-<workspace>}/ckpt/logs/train_<model>_<timestamp>.log`。
- kernel benchmark 日志位于 `${XVLA_STORAGE_ROOT:-<workspace>}/ckpt/logs/profiling_xvla_kernels_<timestamp>.log`。
- `run_profiling.sh` 会分别打印 attention 和 transformer 在 `fusion=0/1` 下的平均延迟、吞吐与 speedup。
- 如果 fusion path 和 fallback path 出现明显数值差异，优先运行 `infer_with_torch/verify_xvla_accuracy_ascend.py` 缩小问题范围。

## 仿真实时渲染

NPU 不支持 OpenGL 渲染。如果后续使用 LIBERO/MuJoCo 做在线仿真评测，需要切换为 CPU 离屏渲染：

```bash
Xvfb :1 -screen 0 1024x768x24 >/tmp/xvfb.log 2>&1 &
export DISPLAY=:1
export LIBGL_ALWAYS_SOFTWARE=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libOSMesa.so
export LD_LIBRARY_PATH=/lib/aarch64-linux-gnu/:$LD_LIBRARY_PATH
export MUJOCO_GL=osmesa
```

## 已知风险

- `xvla-base` 约 0.9B 参数，正式训练前建议先用 `batch_size=1`、`--nproc 1` 确认显存、数据路径和模型路径。
- `torch_npu.npu_fusion_attention` 接口随 torch_npu 版本可能变化；如遇到算子参数错误，设置 `LEROBOT_XVLA_USE_NPU_FUSION_ATTENTION=0` 回退。
- 当前真实推理测试使用的是单帧 HDF5 样本，不代表 LIBERO 在线闭环成功率。在线评测仍需在 MuJoCo/offscreen 环境中执行。

## Citation

```bibtex
@article{zheng2025x,
  title   = {X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model},
  author  = {Zheng, Jinliang and Li, Jianxiong and Wang, Zhihao and Liu, Dongxiu and Kang, Xirui and Feng, Yuchun and Zheng, Yinan and Zou, Jiayin and Chen, Yilun and Zeng, Jia and others},
  journal = {arXiv preprint arXiv:2510.10274},
  year    = {2025}
}
```
