# PUMA 在昇腾 Atlas 910 上的在线推理样例

本目录提供 PUMA（DOMINO 动态操作 VLA 模型）基于 PyTorch 的昇腾在线推理样例：将已有的 PUMA checkpoint 原样加载到昇腾 NPU 上，以 policy server 形式对 RoboTwin 仿真评测端提供动作推理服务。

几点说明：

- 权重可在昇腾直接加载，checkpoint 内容与 key 不做任何改动；
- 服务端只需加 `--device npu`，仿真评测端不感知设备差异；
- 环境不满足要求（缺 torch_npu、NPU 不可用）时，服务在启动阶段就会报错退出；
- 昇腾适配代码已合入 DOMINO 上游主干，本样例只保存脚本与文档，不携带补丁。

## 1. 适用场景

- 硬件：昇腾 Atlas 910 系列
- CANN：8.5.2
- 任务：PUMA policy server 在线推理 + RoboTwin/DOMINO 仿真评测
- 外部代码仓：[H-EmbodVis/DOMINO](https://github.com/H-EmbodVis/DOMINO)
- 模型权重：[H-EmbodVis/PUMA](https://huggingface.co/H-EmbodVis/PUMA)

## 2. 外部依赖与固定版本

本样例不内嵌 DOMINO 源码，默认使用如下 commit（已包含全部昇腾适配）：

```text
9c94f2d3a700fff3b65f041df4038d497139ed1f
```

已验证软件栈：

| 组件 | 版本 |
| --- | --- |
| NPU | Atlas 910 系列 |
| CANN | 8.5.2 |
| torch / torch-npu | 2.5.1 / 2.5.1.post1 |
| transformers | 4.57.0 |
| Python | 3.10 |

相邻的 CANN / torch-npu 版本组合大概率可用，但只有上述软件栈经过端到端验证。请确保 CANN 版本与 `torch-npu` 构建的要求匹配（参见 [torch-npu 版本配套表](https://gitee.com/ascend/pytorch#安装)）。

## 3. 目录说明

```text
manipulation/puma/infer_with_torch/
├── README.md
└── src/
    └── scripts/
        ├── setup.sh               # 复用训练 setup 并追加 RoboTwin 评测通信依赖
        └── run_server.sh          # policy server 启动封装（--device npu）
```

推理侧昇腾适配与优化文档位于 [docs/manipulation/puma/infer_with_torch/README.md](../../../docs/manipulation/puma/infer_with_torch/README.md)。

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

chmod +x manipulation/puma/infer_with_torch/src/scripts/setup.sh
./manipulation/puma/infer_with_torch/src/scripts/setup.sh --create-conda
```

该脚本复用[训练样例](../train/README.md)的 setup 逻辑（clone DOMINO、固定 commit、优先安装 `requirements-ascend.txt`、可编辑安装 PUMA），并额外安装 RoboTwin 评测通信协议依赖（`examples/Robotwin/eval_files/requirements.txt`，与上游一致）。

注意事项：

- 不要安装 `flash-attn`、`decord`、`eva-decord`，它们会拉取 GPU 专用依赖并破坏环境，`requirements-ascend.txt` 已刻意排除；
- 保持 `numpy==1.26.4`，被其他包升级到 2.x 时请重新固定；
- 模型权重准备方式与上游原有流程完全一致：按 [DOMINO 上游权重下载说明](https://github.com/H-EmbodVis/DOMINO/tree/main/policy/PUMA#12-download-pre-trained-weights) 下载后，放置或软链到 `DOMINO/policy/PUMA/playground/Pretrained_models` 下。

## 5. 启动 policy server

```bash
./manipulation/puma/infer_with_torch/src/scripts/run_server.sh \
  --ckpt /absolute/path/to/checkpoints/steps_100000_pytorch_model.pt \
  --port 9001 \
  --npu 0
```

等价的直接调用方式（与上游文档一致）：

```bash
cd ../DOMINO/policy/PUMA
ASCEND_RT_VISIBLE_DEVICES=0 python deployment/model_server/server_policy.py \
  --ckpt_path /absolute/path/to/checkpoints/steps_100000_pytorch_model.pt \
  --port 9001 \
  --device npu \
  --use_bf16
```

说明：

- `ASCEND_RT_VISIBLE_DEVICES` 用于指定服务使用的 NPU 卡；
- 首个请求会慢于稳态请求（设备与内存池初始化、权重首次搬运等一次性开销），建议压测或评测前先发一个预热请求，再统计稳态时延；
- 服务端加载权重时自动应用昇腾配置覆盖（FlashAttention→SDPA、Conv3d patch embed 线性化等），细节见 [docs/manipulation/puma/infer_with_torch/README.md](../../../docs/manipulation/puma/infer_with_torch/README.md)。

## 6. RoboTwin 评测

仿真评测端流程与上游原有流程完全一致：按 [DOMINO 上游评测章节](https://github.com/H-EmbodVis/DOMINO/tree/main/policy/PUMA#-3-evaluation) 配置 RoboTwin 环境，将 `deploy_policy.yml` 指向本 policy server 的 host 与端口即可。仿真在主机侧执行，policy 在 NPU 侧执行，二者通过 WebSocket 通信。

## 7. 已验证结果摘要

- 已在 Atlas 910（CANN 8.5.2）上完成 policy server 端到端验证：PUMA checkpoint 直接加载（bf16），RoboTwin 评测链路与上游原有流程一致；
- 环境不满足要求时（无 torch_npu / NPU 不可用），服务在启动阶段直接报错退出，不会带着错误配置继续服务。

## 8. 相关说明

- 昇腾适配的实现要点、优化动机与收益见：[docs/manipulation/puma/infer_with_torch/README.md](../../../docs/manipulation/puma/infer_with_torch/README.md)；
- 训练样例见：[../train/README.md](../train/README.md)；
- 上游英文推理指南：[docs/ascend_inference.md](https://github.com/H-EmbodVis/DOMINO/blob/main/policy/PUMA/docs/ascend_inference.md)。
