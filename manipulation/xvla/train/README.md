# XVLA 在昇腾 Atlas A2 上的训练样例

本目录提供 XVLA 基于 LeRobot 的昇腾训练与评测样例，面向 Atlas A2 / 910B。样例只保存 recipe、配置、脚本和 patch；LeRobot、数据集和模型权重均作为外部依赖准备。

## 1. 适用场景

- 硬件：昇腾 Atlas A2 / 910B
- CANN：建议 8.3.0 及以上
- 外部框架：`huggingface/lerobot`
- 模型：`lerobot/xvla-base`
- 数据集：`HuggingFaceVLA/libero`

## 2. 外部依赖与固定版本

当前样例固定 LeRobot commit：

```text
3f2179f3b69708b6ad009b2e7685dd9d05269ee1
```

该版本包含 XVLA policy：`src/lerobot/policies/xvla`。

## 3. 目录说明

```text
manipulation/xvla/train/
├── README.md
├── doc/
│   └── README.md
└── src/
    ├── configs/
    │   └── xvla_libero.yaml
    ├── patches/
    │   └── lerobot_xvla_ascend.patch
    └── scripts/
        ├── benchmark_xvla_npu_kernels.py
        ├── run_profiling.sh
        ├── run_eval.sh
        ├── run_train.sh
        └── setup.sh
```

## 4. 环境准备

### 4.1 基础软件要求

`setup.sh` 会准备并安装外部 `lerobot`，但不会自动下载 Ascend 平台相关 wheel。

| 组件 | 要求 |
| --- | --- |
| OS / 架构 | Linux，`aarch64` 或与 Ascend PyTorch wheel 匹配的架构 |
| 硬件 | 昇腾 Atlas A2 / 910B，至少 1 张 NPU |
| CANN | 建议 `8.3.0` 及以上；当前验证环境为 `9.0.0` |
| Python | 建议 `3.12` |
| PyTorch | 需与 `torch_npu`、CANN 版本匹配 |
| torch_npu | 需能 `import torch_npu` 且 `torch.npu.is_available()` 为 true |
| git | 用于拉取外部 `lerobot` 并 checkout 固定 commit |
| conda | 可选；仅在使用 `--create-conda` 时需要 |

当前已验证的一组平台栈：

```text
Python 3.12
CANN 9.0.0
torch 2.10.0+cpu
torch_npu 2.10.0.post2
torchvision 0.25.0
LeRobot commit 3f2179f3b69708b6ad009b2e7685dd9d05269ee1
```

每次打开新终端后，需要先设置 CANN 环境，例如：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
```

### 4.2 获取 recipe 并准备 LeRobot

从空工作目录复现时，推荐目录关系为：

```text
<workspace>/
├── cann-recipes-embodied-ai/
└── lerobot/                  # setup.sh 自动 clone / checkout / patch
```

执行：

```bash
cd <workspace>
git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git
cd cann-recipes-embodied-ai

conda activate lerobot-xvla
source /usr/local/Ascend/cann-9.0.0/set_env.sh

chmod +x manipulation/xvla/train/src/scripts/setup.sh
./manipulation/xvla/train/src/scripts/setup.sh
```

该脚本会：

- 在 `cann-recipes` 同级目录下准备外部 `lerobot` 仓库；
- checkout 到固定 commit `3f2179f3b69708b6ad009b2e7685dd9d05269ee1`；
- 应用 XVLA Ascend patch；
- 安装 LeRobot / LIBERO / XVLA 所需 Python 依赖；
- 默认复用当前环境中的 `torch`、`torchvision`、`torch_npu`，不会自动下载平台 wheel；
- 末尾检查 `torch`、`torch_npu`、`transformers`、`safetensors` 是否可导入。

如需由脚本新建 conda 环境并使用本地平台 wheel：

```bash
./manipulation/xvla/train/src/scripts/setup.sh \
  --create-conda \
  --env-name lerobot-xvla \
  --python-version 3.12 \
  --torch-wheel /path/to/torch.whl \
  --torchvision-wheel /path/to/torchvision.whl \
  --torch-npu-wheel /path/to/torch_npu.whl
```

如果已经手动准备好 `lerobot`，可通过 `LEROBOT_ROOT` 指定路径：

```bash
export LEROBOT_ROOT=/path/to/lerobot
./manipulation/xvla/train/src/scripts/setup.sh
```

`setup.sh` 会把该仓库切到固定 commit 并应用 `lerobot_xvla_ascend.patch`。如果已有本地修改导致 patch 无法干净应用，脚本会报错退出，避免后续训练处于未适配状态。

### 4.3 模型、数据与大文件目录

推荐工作区布局：

```text
<workspace>/
├── cann-recipes-embodied-ai/
├── lerobot/
├── dataset/
│   └── HuggingFaceVLA/
│       └── libero/
├── models/
│   └── lerobot/
│       ├── xvla-base/
│       └── xvla-libero/
└── ckpt/
```


本 recipe 的 train / eval / profiling / infer 脚本会优先使用 `/data/docker/xvla_storage` 作为 `XVLA_STORAGE_ROOT`，缓存与日志默认写入该目录；也可手动覆盖：

```bash
export XVLA_STORAGE_ROOT=/path/to/large/disk/xvla_storage
```

本仓库不随 recipe 一起提交模型、数据集或 checkpoint。正式训练或真实推理前，需要自行准备：

- `dataset/HuggingFaceVLA/libero`：LeRobot 格式 LIBERO 数据集，根目录应包含 `data/` 和 `meta/`；
- `models/lerobot/xvla-base`：本地可直接加载的 XVLA base checkpoint；
- `models/lerobot/xvla-libero`：可选，已微调 checkpoint，用于评测。
- `facebook/bart-large` tokenizer：XVLA 配置中的语言 tokenizer，离线环境需提前缓存。



缓存 tokenizer 小文件：

```bash
hf download facebook/bart-large \
  config.json tokenizer_config.json vocab.json merges.txt tokenizer.json special_tokens_map.json
```

如果网络环境无法访问 Hugging Face，可在有网络机器上下载完整模型目录后拷贝到 `models/lerobot/xvla-base`。至少需要包含：

```text
config.json
model.safetensors
policy_preprocessor.json
policy_postprocessor.json
```

### 4.4 环境验证

完成 setup 后，在 `lerobot` 根目录或 recipe 根目录执行：

```bash
python - <<'PY'
import torch
import torch_npu
import transformers
import safetensors

print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("npu available", torch.npu.is_available())
print("npu count", torch.npu.device_count() if torch.npu.is_available() else 0)
print("transformers", transformers.__version__)
print("safetensors", safetensors.__version__)
PY
```

期望 `npu available` 为 `True`。

### 4.5 完整复现流程

下面给出从空工作目录到完成真实 checkpoint 推理和 LeRobotDataset 训练验证的最短流程。需要先准备好 Ascend 平台可用的 `torch` / `torchvision` / `torch_npu` wheel 或已有环境。

```bash
# 1. 准备工作目录和 recipe
cd <workspace>
git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git
cd cann-recipes-embodied-ai

# 2. 激活 Ascend PyTorch 环境并设置 CANN
conda activate lerobot-xvla
source /usr/local/Ascend/cann-9.0.0/set_env.sh

# 3. 准备外部 lerobot、应用 Ascend patch、安装 Python 依赖
./manipulation/xvla/train/src/scripts/setup.sh

# 4. 准备模型和数据
mkdir -p ../models/lerobot ../dataset/HuggingFaceVLA
hf download lerobot/xvla-base --local-dir ../models/lerobot/xvla-base
hf download facebook/bart-large \
  config.json tokenizer_config.json vocab.json merges.txt tokenizer.json special_tokens_map.json
# 将 LeRobot 格式 LIBERO 数据集放到 ../dataset/HuggingFaceVLA/libero

# 5. 验证真实 checkpoint 可在 NPU 上推理
./manipulation/xvla/infer_with_torch/run_xvla_inference.sh \
  --pretrained_model_name_or_path ../models/lerobot/xvla-base \
  --device npu:0 \
  --dtype float32 \
  --num_warmup 1 \
  --num_inference 3

# 6. 启动 LIBERO 训练
./manipulation/xvla/train/src/scripts/run_train.sh xvla_libero --nproc 1 --port 29510

# 7. 使用训练保存的 checkpoint 推理
./manipulation/xvla/infer_with_torch/run_xvla_inference.sh \
  --pretrained_model_name_or_path ../ckpt/xvla_libero_<timestamp>/checkpoints/000010/pretrained_model \
  --device npu:0 \
  --dtype float32 \
  --num_warmup 1 \
  --num_inference 1
```

如果需要从本地 wheel 新建环境，将第 2、3 步替换为：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
./manipulation/xvla/train/src/scripts/setup.sh \
  --create-conda \
  --env-name lerobot-xvla \
  --python-version 3.12 \
  --torch-wheel /path/to/torch.whl \
  --torchvision-wheel /path/to/torchvision.whl \
  --torch-npu-wheel /path/to/torch_npu.whl
```

## 6. 训练

### 6.1 YAML 格式约束

`run_train.sh` 会读取配置中的 `output_dir` 和 `job_name` 并自动追加时间戳。为保证脚本解析稳定，这两个字段需要保持顶层、单行、简单键值格式，例如：

```yaml
output_dir: ../ckpt/xvla_libero
job_name: xvla_libero
```

不要把这两个字段写成多行块、嵌套字段或 YAML 锚点引用。

### 6.2 启动命令

正式训练：

```bash
./manipulation/xvla/train/src/scripts/run_train.sh xvla_libero --nproc 8 --port 29510
```

快速链路验证可把 `steps` 和 `save_freq` 临时设置为 `10`，并使用 `--nproc 1`。当前环境已用真实 `xvla-base` checkpoint 和 LeRobot 格式 LIBERO 数据集完成 10 step 训练，step 10 成功写出 checkpoint。

默认开启：

- `policy.dtype: bfloat16`
- `policy.action_mode: auto`
- `LEROBOT_XVLA_USE_NPU_FUSION_ATTENTION=1`
- `LEROBOT_XVLA_USE_NPU_BADD_BMM=1`
- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`

默认关闭 W&B，避免未配置 API key 的环境阻塞开箱训练。如需上传实验记录，可在配置中设置 `wandb.enable: true` 并先执行 `wandb login`。

如模型、数据集和 tokenizer 已缓存到本地，可启用离线模式减少远端探测带来的启动等待：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

## 7. Profiling 与优化验证

无权重 kernel benchmark，可对比 action transformer 中 fusion attention 开/关的延迟：

```bash
./manipulation/xvla/train/src/scripts/run_profiling.sh \
  --dtype float16 \
  --num-iters 20
```

训练步 forward+backward benchmark：

```bash
./manipulation/xvla/train/src/scripts/run_profiling.sh \
  --dtype float32 \
  --train-step \
  --num-iters 10
```

更完整的性能优化说明见 `manipulation/xvla/train/doc/README.md`。

若当前 torch_npu 版本的融合注意力不可用，可临时关闭：

```bash
LEROBOT_XVLA_USE_NPU_FUSION_ATTENTION=0 \
./manipulation/xvla/train/src/scripts/run_train.sh xvla_libero --nproc 1
```

## 8. 评测

在线 LIBERO 评测建议将仿真与渲染放在 CPU 侧执行，policy 推理放在 NPU 侧执行。`run_eval.sh` 默认设置 `MUJOCO_GL=osmesa`，并在传入 `--policy.device=npu` / `npu:x` 时同步设置评测进程的 NPU 设备。

LIBERO 示例：

```bash
./manipulation/xvla/train/src/scripts/run_eval.sh \
  --policy.path=../models/lerobot/xvla-libero \
  --policy.device=npu \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.control_mode=absolute \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --env.episode_length=800 \
  --output_dir=../evals/xvla_libero
```

## 9. 已验证结果摘要

| 测试项 | 场景 | 结果 |
| --- | --- | --- |
| kernel benchmark forward | action transformer，fusion attention 开/关 | 整体 transformer forward 约 `1.83%` 提升 |
| kernel benchmark train-step | forward+backward，fusion attention 开/关 | 整体 transformer train-step 约 `1.95%` 提升 |
| domain linear benchmark | action encoder / decoder，baddbmm 开/关 | encoder 约 `3.57%` 提升，decoder 约 `5.81%` 提升 |
| 真实训练链路 | `xvla-base + LeRobotDataset`，`steps=10`，`save_freq=10` | step 10 训练完成并成功保存 checkpoint |
| 训练后 checkpoint 推理 | `checkpoints/000010/pretrained_model`，`float32` | action shape `[1, 7]`，平均延迟约 `279.9 ms` |
| 真实 checkpoint 推理 | `xvla-base + LIBERO HDF5`，`float32` | action shape `[1, 20]`，平均延迟约 `277.1 ms` |
| 优化一致性验证 | 真实 checkpoint fallback vs NPU fusion + baddbmm | cosine `1.000000`，MSE `0.00000000` |

## 10. 常见问题

- `torch_npu.npu_fusion_attention` 参数或版本不兼容：设置 `LEROBOT_XVLA_USE_NPU_FUSION_ATTENTION=0` 回退。
- `torch.baddbmm` 在特定版本上表现异常：设置 `LEROBOT_XVLA_USE_NPU_BADD_BMM=0` 回退到 `matmul + bias` 路径。
- 在线评测渲染失败：确认 `MUJOCO_GL=osmesa`，必要时按 `doc/README.md` 配置 Xvfb / OSMesa。

## 11. 说明

- XVLA 官方建议新 embodiment 微调使用 `policy.action_mode=auto`，自动按数据集动作维度做 padding / trimming。
- `lerobot/xvla-base` 约 0.9B 参数，首次训练建议 `batch_size=1`、`--nproc 1`，确认可跑后再扩展到多卡。
- 本 recipe 不自动下载 Ascend 平台 wheel。

## Citation

```bibtex
@article{zheng2025x,
  title   = {X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model},
  author  = {Zheng, Jinliang and Li, Jianxiong and Wang, Zhihao and Liu, Dongxiu and Kang, Xirui and Feng, Yuchun and Zheng, Yinan and Zou, Jiayin and Chen, Yilun and Zeng, Jia and others},
  journal = {arXiv preprint arXiv:2510.10274},
  year    = {2025}
}
```
