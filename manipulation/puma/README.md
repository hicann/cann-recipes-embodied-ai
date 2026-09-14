# PUMA（DOMINO）昇腾样例

PUMA 是 DOMINO（ECCV 2026）提出的动态操作 VLA 模型：以 Qwen3-VL-4B 为骨干，结合场景级历史光流与世界查询（world queries）对物体未来状态做短时预测，从而在动态环境中更及时地做出反应。DOMINO 同时提供 35 个层次化动态操作任务、超过 110K 条专家轨迹的数据集与多维评测套件（基于 RoboTwin 仿真）。

- 论文：[Towards Generalizable Robotic Manipulation in Dynamic Environments](https://arxiv.org/abs/2603.15620)
- 代码：[H-EmbodVis/DOMINO](https://github.com/H-EmbodVis/DOMINO)
- 模型权重：[H-EmbodVis/PUMA](https://huggingface.co/H-EmbodVis/PUMA)

PUMA 的昇腾适配已合入 DOMINO 上游主干：上游发布的 PUMA 权重可在昇腾上直接加载，无需任何转换；昇腾相关逻辑只在 NPU 上运行时启用，上游原有工作流不受影响。因此本样例目录不携带补丁，只包含 recipe 脚本、文档与配置说明，上游 DOMINO 仓库由 setup 脚本单独 clone 并固定到已验证 commit。

## 样例入口

| 样例 | 平台 | 说明 |
| --- | --- | --- |
| [训练](train/README.md) | Atlas 800T A2（910B3 × 8） | 8 卡 DeepSpeed ZeRO-2（bf16）训练，支持世界模型监督 |
| [在线推理](infer_with_torch/README.md) | Atlas 910 系列 | PyTorch 在线推理，policy server 直接服务已有 checkpoint |

## 适配与优化文档

| 文档 | 说明 |
| --- | --- |
| [训练侧适配与优化](../../docs/manipulation/puma/train/README.md) | 训练期补丁的动机、实现方法与收益 |
| [推理侧适配与优化](../../docs/manipulation/puma/infer_with_torch/README.md) | 推理期适配的动机、实现方法与收益 |

## 已验证环境

| 组件 | 版本 |
| --- | --- |
| NPU | Atlas 910 系列（训练在 910B3 × 8 上验证） |
| CANN | 8.5.2 |
| torch / torch-npu | 2.5.1 / 2.5.1.post1 |
| transformers | 4.57.0 |
| Python | 3.10 |

## 外部依赖与固定版本

本样例不内嵌 DOMINO 源码，默认固定到以下 commit（已包含全部昇腾适配）：

```text
9c94f2d3a700fff3b65f041df4038d497139ed1f
```

## 目录说明

```text
manipulation/puma/
├── README.md                          # 本文件
├── train/                             # 训练样例
│   ├── README.md
│   └── src/scripts/
│       ├── setup.sh                   # 准备 DOMINO 仓库与昇腾运行时
│       └── run_train.sh               # 8 卡 ZeRO-2 训练启动封装
└── infer_with_torch/                  # 在线推理样例
    ├── README.md
    └── src/scripts/
        ├── setup.sh                   # 复用训练 setup 并追加评测通信依赖
        └── run_server.sh              # policy server 启动封装（--device npu）

docs/manipulation/puma/                # 适配与优化文档统一存放位置
├── train/README.md                    # 训练侧优化文档
└── infer_with_torch/README.md         # 推理侧优化文档
```
