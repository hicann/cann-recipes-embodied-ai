# PUMA 昇腾在线推理适配与优化说明

本文档说明 PUMA 昇腾推理适配的设计原则、各优化点的动机、实现方法与收益。全部适配代码已合入 [DOMINO 上游仓库](https://github.com/H-EmbodVis/DOMINO)，本样例通过固定 commit 引用，文中路径均相对 `DOMINO/policy/PUMA/`。

## 1. 总体设计原则

- **权重不做转换**：上游发布的 checkpoint 在 NPU 上原样加载，不修改任何 checkpoint key；所有结构性替换均在加载时完成且数值等价。
- **后端中立**：通用代码（VLM 封装、部署入口）不出现 `if npu` 分支；昇腾专用逻辑集中在 `PUMA/model/modules/vlm/ascend/` 包内，通过 `--device npu` 一个开关注入。
- **启动即校验**：`PUMA/util/device.py` 的 `resolve_device` 在 torch_npu 缺失或 NPU 不可用时于启动阶段直接报错，避免以错误配置继续提供服务。
- **上游行为不变**：所有适配仅在 NPU 上激活，上游原有推理路径逐比特保持原状。

## 2. 优化点明细

### 2.1 加载期配置覆盖：FlashAttention → SDPA、bf16

- **为什么**：PUMA 上游配方默认使用 FlashAttention，该库为 GPU 专用，昇腾上不可用也不应安装。
- **怎么做**：服务端检测到 `--device npu` 后，通过 `ascend_inference_config_overrides()` 对 checkpoint 携带的配置做内存内覆盖：`attn_implementation=sdpa`、`model_dtype=bfloat16`、`linearize_vision_patch_embed=true`、`enable_ascend_inference_adapter=true`。checkpoint 文件本身不做任何修改。
- **收益**：同一份 checkpoint 无需转换即可在昇腾上服务；SDPA 走 torch_npu 的融合注意力实现，bf16 与训练精度口径一致。

### 2.2 Conv3d 视觉 patch embed 线性化

- **为什么**：Qwen3-VL 视觉塔的 patch embed 是 `kernel_size == stride` 的 Conv3d。该形态的 Conv3d 在 NPU 上不支持/性能差，是视觉编码的入口热点。
- **怎么做**：`ascend/patch_embed.py` 在加载时将 Conv3d 投影替换为数学等价的 `F.linear`（`LinearizedConv3dPatchEmbed`）：当 kernel 与 stride 相等时，Conv3d 严格等价于对展平 patch 的一次线性投影，权重直接 reshape 复用，无需重训或转换。替换带守卫（仅当 `proj` 确为 Conv3d 且 kernel==stride），失败时报错而非静默降级。
- **收益**：绕开 NPU 上的 Conv3d 劣化路径，改走高度优化的矩阵乘；数值与原 Conv3d 完全等价。

### 2.3 Qwen3 推理请求计划（CPU 预计算控制流）

- **为什么**：Qwen3-VL 的视觉前处理包含大量小规模、数据依赖的控制流计算（position embedding 双线性插值索引、`cu_seqlens`、rotary 位置 id、按 grid 切分长度等）。这些计算在 NPU 上会带来形态多变的小算子下发与频繁的 host-device 同步，是时延与稳定性的主要来源。
- **怎么做**：`ascend/qwen3_inference.py` 引入"推理请求计划"（`Qwen3AscendInferencePlan`）：每个请求先在 CPU 上基于 `grid_thw` 元数据构建完整计划（强制校验 `grid_thw` 必须位于 CPU），一次性生成全部索引/权重/切分信息，再通过 `ContextVar` 注入到运行时被覆盖的模型类（`Qwen3VLModel` / `Qwen3VLVisionModel` 等）中执行；在可安全省略处直接跳过 attention mask 构建。类覆盖不改动 checkpoint key，也不影响上游原有路径。
- **收益**：NPU 侧只执行静态、规整的张量运算，消除形态波动与逐步同步；对相同分辨率输入，vision 计划签名可复用。

### 2.4 首请求预热

- **为什么**：稳态请求时延正常，但服务启动后的首个请求明显更慢。CANN 的二进制算子库已覆盖绝大多数常见算子形态，通常不涉及即时编译；首请求的额外耗时主要来自一次性开销：ACL/设备初始化、显存池首次扩张、权重首次搬运到设备，以及 torch_npu 侧首次算子下发。
- **怎么做**：不改变默认行为，仅在启动脚本与文档中提示该现象；同时通过 2.3 的请求计划把视觉前处理的动态 shape 收敛为静态规整形态，减少形态波动带来的首包抖动。
- **收益**：避免把首请求耗时误判为性能问题；压测与评测按"先预热一次、再统计稳态时延"的口径进行。

### 2.5 依赖治理：requirements-ascend.txt

- **为什么**：`flash-attn`、`decord`、`eva-decord` 均会拉取 GPU 专用 wheel 或在昇腾 aarch64 上不可用；依赖解析顺序不当会导致 pip 先装 GPU 版 torch。
- **怎么做**：独立的 `requirements-ascend.txt` 置顶固定 torch 三件套（torch==2.5.1 / torchvision==0.20.1 / torch-npu==2.5.1.post1），刻意排除上述 GPU 专用包，视频解码改用 `torchvision_av` 后端；`numpy` 固定 1.26.4 以匹配 torch-npu 运行时。
- **收益**：一条命令得到可复现的昇腾环境，避免 GPU 版 wheel 混入。

## 3. 已验证环境

| 组件 | 版本 |
| --- | --- |
| NPU | Atlas 910 系列 |
| CANN | 8.5.2 |
| torch / torch-npu | 2.5.1 / 2.5.1.post1 |
| transformers | 4.57.0 |
| Python | 3.10 |

## 4. 后续工作

- 探索 torchair 图模式进一步降低稳态时延；
- 上游 transformers 版本升级后回归验证 Qwen3 类覆盖的兼容性（当前针对 4.57.0 验证）。
