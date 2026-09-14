# PUMA 昇腾训练适配与优化说明

本文档说明 PUMA 昇腾训练适配的设计原则、各优化点的动机、实现方法与收益。全部适配代码已合入 [DOMINO 上游仓库](https://github.com/H-EmbodVis/DOMINO)，本样例通过固定 commit 引用，文中路径均相对 `DOMINO/policy/PUMA/`。推理侧适配（设备抽象、SDPA、patch embed 线性化等）见[推理优化文档](../infer_with_torch/README.md)，训练复用这些基础设施。

## 1. 总体设计原则

- **NPU 门控**：所有训练期补丁集中在 `PUMA/training/ascend/`，仅当加速器设备为 NPU 时安装（`setup_ascend_runtime`），且每项补丁可通过环境变量单独关闭（`PUMA_ASCEND_PATCH_*`）。
- **上游行为不变**：非 NPU 设备上不安装任何补丁，包括保留 DeepSpeed 原生的 float64 梯度范数路径，保证已有实验可复现。
- **权重不做转换**：上游发布的 PUMA / Qwen3-VL 预训练权重直接加载继续训练。

## 2. 优化点明细

### 2.1 DeepSpeed ZeRO 梯度范数 float32 化

- **为什么**：DeepSpeed ZeRO-1/2 的 `get_grad_norm_direct` 会把每个梯度 cast 到 float64（double）再求范数。昇腾 NPU 对 float64 支持差：要么走极慢的模拟路径，要么直接报错，导致 ZeRO-2 训练无法进行。
- **怎么做**：`training/ascend/runtime.py` 的 `patch_deepspeed_zero_grad_norm_for_ascend` 以 monkey-patch 方式重写该方法：全程使用 float32 累加（L2 范数按平方和累加，inf 范数按逐张量 max），再做 `all_reduce`，同时保留 pipeline 复制参数与模型并行参数的原有过滤语义。另外重写 `scaled_global_norm`：当 `clip_grad <= 0`（未启用梯度裁剪）时直接返回零张量，跳过整套范数计算与集合通信。
- **收益**：解除 ZeRO-2 在昇腾上的 float64 阻塞；未开启梯度裁剪时每步省去一次全参数范数计算和一次 all_reduce。

### 2.2 Qwen RMSNorm 融合算子替换

- **为什么**：transformers 中 `Qwen3RMSNorm` / `Qwen3VLTextRMSNorm` 的逐算子实现（pow/mean/rsqrt/mul）在 NPU 上产生多次 kernel 发射与中间张量。
- **怎么做**：`patch_qwen_rms_norm_for_ascend` 将两个类的 `forward` 替换为 `torch_npu.npu_rms_norm` 融合算子，并带三重守卫：输入必须位于 NPU、dtype 属于 {fp16, bf16, fp32}、权重与输入 dtype 一致，否则回退原实现。
- **收益**：RMSNorm 是 Qwen3 骨干中的高频算子（每层两次），融合后显著减少 kernel 发射与访存开销。

### 2.3 混合精度策略：bf16 主干 + FP32 action head

- **为什么**：全参模型在 8 卡 ZeRO-2 下需要 bf16 才能放下；但 action head 输出连续控制量，对数值精度敏感。
- **怎么做**：主干 bf16（`MODEL_DTYPE=bfloat16`）、注意力使用 `sdpa`（flash-attn 为 GPU 专用），action head 在 NPU 上保持 FP32。
- **收益**：显存可行的同时保住动作回归精度，训练损失口径与上游 bf16 配方对齐。

### 2.4 梯度检查点：非重入模式

- **为什么**：PyTorch 重入式（reentrant）梯度检查点在 NPU 上与 autograd 钩子存在兼容性问题。
- **怎么做**：`maybe_enable_ascend_gradient_checkpointing` 仅在 NPU 上将 Qwen 骨干切到 `use_reentrant=False` 的检查点实现，并同步关闭 `use_cache`。
- **收益**：模型在 910B3 单卡 64GB 显存下的激活开销可控，8 卡 ZeRO-2 训练可稳定运行。

### 2.5 训练指标延迟物化（去 host-device 同步）

- **为什么**：训练循环每步对标量指标调用 `.item()` 会触发 host-device 同步，NPU 上该同步开销明显，成为步内隐藏瓶颈。
- **怎么做**：`training/ascend/metrics.py` 在 NPU 上仅 `detach()` 保留设备端张量（`collect_training_metrics`），只在真正打日志的步（默认每 100 步）才 `materialize_training_metrics` 转成 Python 标量；非有限 loss 检查保留但支持配置检查间隔与 warmup（`NON_FINITE_CHECK_INTERVAL` 等）。
- **收益**：消除每步一次以上的强制同步，保住流水线并发；同时保留快速发现 NaN/Inf 的能力。

### 2.6 世界模型模块（GroundingDINO / SAM2）NPU 兼容化

- **为什么**：PUMA 的世界模型监督依赖 GroundingDINO 与 SAM2。GroundingDINO 的多尺度可变形注意力原生依赖 GPU 自定义算子，纯 PyTorch 回退路径的张量布局对 NPU 不友好；Swin backbone 与 SAM2 prompt encoder 亦有零星 NPU 不兼容算子。
- **怎么做**：`ms_deform_attn.py` 在纯 PyTorch 采样路径上按设备选择聚合布局（`_merge_sampling_values(..., use_npu_layout=value.device.type == "npu")`）；`swin_transformer.py`、`transformer.py`、`prompt_encoder.py` 等做等价算子改写；`world_feature.py` 配合 grounding 结果缓存减少重复前向。
- **收益**：世界模型监督（`WORLD_MODEL_ENABLED=true`）在 NPU 上端到端可用，与上游数值口径一致。

### 2.7 数据管线：只读数据集与缓存重定向

- **为什么**：训练首次运行需要重建数据集统计、step 索引、历史光流与 grounding 缓存。共享集群上数据集常为只读挂载，默认写回数据集目录会直接失败。
- **怎么做**：dataloader 支持通过 `PUMA_DATASET_STATS_CACHE_DIR` / `PUMA_STEPS_CACHE_DIR` / `PUMA_HISTORY_FLOW_CACHE_DIR` / `PUMA_GROUNDING_CACHE_DIR` 将全部重建产物重定向到本次运行的输出目录，启动器默认设置好这些变量；历史光流在 CPU 侧多进程计算（`HISTORY_FLOW_CPU_WORKERS`），不占 NPU。
- **收益**：数据集只读时无需额外配置即可训练；缓存跨 rank 复用，重复启动时跳过重建。

### 2.8 启动脚本整合（HCCL 超时与环境变量清理）

- **为什么**：昇腾分布式使用 HCCL；环境中残留的 GPU 侧设备与通信变量会误导 Accelerate/DeepSpeed；首次运行的数据集统计与索引构建耗时较长，默认通信超时容易误杀。
- **怎么做**：`scripts/run_scripts/run_lerobot_robotwin_puma_ascend.sh` 统一完成：source CANN `set_env.sh`、unset 残留的 GPU 侧设备与通信环境变量、提高 `HCCL_CONNECT_TIMEOUT` / `HCCL_EXEC_TIMEOUT`、限制 OMP/MKL 线程数、路径预检查（`BASE_VLM` / `DATA_ROOT_DIR` 等缺失即退出），并支持 `DRY_RUN=1` 预览完整命令。
- **收益**：一条命令可复现启动 8 卡训练，常见环境类故障在启动前暴露。

## 3. 已验证环境与拓扑

| 项目 | 值 |
| --- | --- |
| NPU | Atlas 910B3 × 8 |
| CANN | 8.5.2 |
| torch / torch-npu | 2.5.1 / 2.5.1.post1 |
| deepspeed | 0.16.9 |
| 并行策略 | 8 卡 DeepSpeed ZeRO-2，bf16 |
| 单卡 batch size | 4 |

说明：单卡全参训练会在 optimizer step 时 OOM，8 卡 ZeRO-2 为已验证的最小可行拓扑。

## 4. 后续工作

- 评估 `npu_fusion_attention` 等更多融合算子在训练路径的收益；
- 探索 ZeRO-3 与更大 batch 的扩展性。
