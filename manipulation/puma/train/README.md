# PUMA 在昇腾 Atlas A2 上的训练样例

本目录提供 PUMA（DOMINO 动态操作 VLA 模型）在 RoboTwin/DOMINO 数据集上的昇腾训练样例，包含环境初始化脚本、8 卡 DeepSpeed ZeRO-2 训练启动脚本以及优化文档。

当前样例基于以下原则整理：

- `cann-recipes` 仓库仅保存 recipe 脚本与文档；
- `DOMINO` 作为外部依赖仓单独 clone，并固定到已验证 commit；
- PUMA 的昇腾适配代码已合入 DOMINO 上游主干，本样例无需任何补丁；
- 上游发布的 PUMA / Qwen3-VL 预训练权重可在昇腾上直接加载，无需转换。

## 1. 适用场景

- 硬件：昇腾 Atlas 800T A2（910B3 × 8，已验证拓扑）
- CANN：8.5.2
- 任务：DOMINO 动态操作任务（RoboTwin 仿真）离线训练
- 数据集：[DOMINO](https://huggingface.co/datasets/h-embodvis/DOMINO)（LeRobot 格式）
- 外部代码仓：[H-EmbodVis/DOMINO](https://github.com/H-EmbodVis/DOMINO)

已验证的训练拓扑为 8 卡 DeepSpeed ZeRO-2（bf16）。单卡全参训练会在 optimizer step 时 OOM，属预期行为。

## 2. 外部依赖与固定版本

本样例不内嵌 DOMINO 源码，默认使用如下 commit（已包含全部昇腾训练与推理适配）：

```text
9c94f2d3a700fff3b65f041df4038d497139ed1f
```

已验证软件栈：

| 组件 | 版本 |
| --- | --- |
| NPU | Atlas 910B3 × 8 |
| CANN | 8.5.2 |
| torch / torch-npu | 2.5.1 / 2.5.1.post1 |
| transformers | 4.57.0 |
| deepspeed | 0.16.9 |
| Python | 3.10 |

相邻的 CANN / torch-npu 版本组合大概率可用，但只有上述软件栈经过端到端验证。请确保 CANN 版本与 `torch-npu` 构建的要求匹配（参见 [torch-npu 版本配套表](https://gitee.com/ascend/pytorch#安装)）。

## 3. 目录说明

```text
manipulation/puma/train/
├── README.md
└── src/
    └── scripts/
        ├── setup.sh               # clone DOMINO、固定 commit、安装昇腾运行时
        └── run_train.sh           # 8 卡 ZeRO-2 训练启动封装
```

训练侧昇腾适配与优化文档位于 [docs/manipulation/puma/train/README.md](../../../docs/manipulation/puma/train/README.md)。

训练配置与 DeepSpeed 配置位于上游 DOMINO 仓库内，无需在本样例中重复维护：

- Accelerate + DeepSpeed 配置：`policy/PUMA/PUMA/config/deepseeds/deepspeed_zero2_ascend.yaml`、`ds_config_ascend_zero2.json`
- 训练超参：`policy/PUMA/examples/Robotwin/train_files/puma_train_robotwin_ascend.yaml`
- 底层启动器：`policy/PUMA/scripts/run_scripts/run_lerobot_robotwin_puma_ascend.sh`

## 4. 环境准备

### 4.1 clone 代码

```bash
git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git
cd cann-recipes-embodied-ai
```

### 4.2 准备 DOMINO 与运行时

```bash
# 先加载 CANN 工具链环境（路径按实际安装位置调整）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

chmod +x manipulation/puma/train/src/scripts/setup.sh
./manipulation/puma/train/src/scripts/setup.sh --create-conda
```

该脚本会：

- 在 `cann-recipes` 同级目录下准备 `DOMINO` 代码仓；
- checkout 到固定 commit `9c94f2d3a700fff3b65f041df4038d497139ed1f`；
- 先安装 `requirements-ascend.txt`（固定 torch==2.5.1 / torch-npu==2.5.1.post1），避免后续依赖解析拉取 GPU 版 torch；
- 以 `--no-build-isolation` 可编辑方式安装 PUMA；
- 校验 torch / torch_npu / transformers / deepspeed 可正常导入。

注意事项：

- 不要安装 `flash-attn`、`decord`、`eva-decord`，它们会拉取 GPU 专用依赖并破坏环境；昇腾侧使用 `sdpa` 注意力与 `torchvision_av` 视频后端；
- 保持 `numpy==1.26.4`。后续安装 `supervision`、`opencv-python` 等包可能将 NumPy 静默升级到 2.x，导致 torch-npu 运行时崩溃，如发生请重新固定；
- 若处于受限网络环境无法直接访问 GitHub，可提前在工作区同级目录手动准备 `DOMINO/`（checkout 到固定 commit），`setup.sh` 会直接复用；也可通过 `DOMINO_REPO_URL` 指定镜像地址。

### 4.3 推荐工作区布局

```text
<workspace>/
├── cann-recipes-embodied-ai/
└── DOMINO/
    └── policy/PUMA/
        ├── playground/Pretrained_models/
        │   ├── Qwen3-VL-4B-Instruct/          # 基座 VLM 权重
        │   └── grounded_sam2/                 # 世界模型监督所需 Grounded-SAM-2 权重
        └── results/Checkpoints/               # 默认训练输出目录
```

数据与权重准备与上游原有流程完全一致，参照 [DOMINO 上游 README 训练章节](https://github.com/H-EmbodVis/DOMINO/tree/main/policy/PUMA#-2-training)：

- 基座 VLM 放到 `playground/Pretrained_models/`（默认 `Qwen3-VL-4B-Instruct`，可用 `BASE_VLM` 覆盖）；
- 世界模型监督需要 Grounded-SAM-2 权重，放到 `playground/Pretrained_models/grounded_sam2/`；
- `DATA_ROOT_DIR` 指向 LeRobot 格式数据集根目录，且每个任务的 `meta/` 目录中已复制 `modality.json`。

数据集位于只读挂载时无需额外处理：启动器通过 `PUMA_*_CACHE_DIR` 环境变量把重建产物（数据集统计、step 索引、光流与 grounding 缓存）全部重定向到本次运行的输出目录。

## 5. 启动训练

```bash
# 先 DRY_RUN 预览完整命令，不占用 NPU
DRY_RUN=1 DATA_ROOT_DIR=/path/to/lerobot_dataset \
  ./manipulation/puma/train/src/scripts/run_train.sh

# 8 卡 ZeRO-2 正式训练
DATA_ROOT_DIR=/path/to/lerobot_dataset \
  ./manipulation/puma/train/src/scripts/run_train.sh
```

启动器会自动完成：加载 CANN 环境、清理残留的 GPU 侧设备与通信环境变量、设置 HCCL 超时，然后以 Accelerate + DeepSpeed ZeRO-2 启动训练。常用可覆盖变量：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `DATA_ROOT_DIR` | —（必填） | LeRobot 格式数据集根目录 |
| `BASE_VLM` | `playground/Pretrained_models/Qwen3-VL-4B-Instruct` | 基座 VLM 权重 |
| `NUM_GPUS` | `8` | NPU 数量 |
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | 指定本次训练可见的 NPU 卡 |
| `ASCEND_SET_ENV` | `/usr/local/Ascend/ascend-toolkit/set_env.sh` | CANN `set_env.sh` 路径 |
| `WORLD_MODEL_ENABLED` | `true` | 世界模型监督开关 |
| `PER_DEVICE_BATCH_SIZE` | `4` | 单卡 batch size |
| `MAX_TRAIN_STEPS` | `200000` | 训练步数 |
| `ENABLE_WANDB` | `0` | 置 `1` 记录到 Weights & Biases |
| `RUN_ROOT_DIR` | `results/Checkpoints` | 输出根目录，checkpoint 存于 `RUN_ROOT_DIR/RUN_ID` |

其余超参（学习率、优化器、保存间隔等）与上游原有配方含义一致，完整列表见上游 `examples/Robotwin/train_files/run_robotwin_train_ascend.sh`。

## 6. 已验证结果摘要

- 已在 Atlas 910B3 × 8 上完成 8 卡 DeepSpeed ZeRO-2（bf16）训练链路端到端验证，含世界模型监督（`WORLD_MODEL_ENABLED=true`），训练可正确收敛并达到正常性能水平；
- 上游发布的 PUMA / Qwen3-VL 预训练权重直接加载，无需权重转换；
- 训练关键配置：`PER_DEVICE_BATCH_SIZE=4`、`sdpa` 注意力、bf16、action head 保持 FP32；
- 训练所得 checkpoint 可通过[在线推理样例](../infer_with_torch/README.md)在昇腾上直接服务，评测流程与上游原有流程完全一致。

## 7. 常见问题

- **启动阶段疑似卡住**：通常是数据集统计与 step 索引构建、光流缓存预热等首次运行才有的准备工作，启动器已提高 HCCL 超时来容忍这一阶段；建议先用 `DRY_RUN=1` 检查路径配置。
- **与其他 NPU 任务共享机器**：请为每个任务分配独立的 HCCL 端口区间（`HCCL_IF_BASE_PORT`、`HCCL_HOST_SOCKET_PORT_RANGE`、`HCCL_NPU_SOCKET_PORT_RANGE`），否则第二个任务会因端口占用启动失败。
- **loss 出现非有限值**：训练器默认每步做非有限 loss 检查并立即报错终止（可通过 `NON_FINITE_CHECK_INTERVAL` 调整频率），便于第一时间定位问题步。

## 8. 相关说明

- 当前样例目录不包含 DOMINO 源码，也不包含补丁；
- 昇腾适配的实现要点、优化动机与收益见：[docs/manipulation/puma/train/README.md](../../../docs/manipulation/puma/train/README.md)；
- 上游英文训练指南：[docs/ascend_training.md](https://github.com/H-EmbodVis/DOMINO/blob/main/policy/PUMA/docs/ascend_training.md)。
