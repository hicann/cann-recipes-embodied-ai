## YAML 参数说明

VGGT 推理参数通过 `config/*.yaml` 文件管理。通过 `--config` 参数指定配置文件。

默认配置：

- 单卡基准配置：`single.yaml`
- 2卡序列并行：`sp2.yaml`
- 4卡序列并行：`sp4.yaml`
- 8卡序列并行：`sp8.yaml`
- 单卡 W8A8 量化：`single_w8a8.yaml`

```yaml
model_args:
  ckpt: "ckpt/model.pt"              # 模型权重路径（必填）
  images-path: "examples/kitchen/images"  # 输入图像目录（必填）
  enable-profiling: false            # 是否启用性能分析
  profile-dir: "prof_sp"             # 性能分析输出目录
  num-runs: 6                        # 推理运行次数

# 优化项配置
optimization:
  # 计算冗余消除优化：通过缓存机制避免重复计算
  computation-redundancy-elimination:
    rope-cache: true                        # 旋转编码缓存
    dpt-pos-embed-cache: true               # DPT头位置编码缓存
    cos-sin-dtype-optimization: true        # Cos/Sin数据类型优化

  # 并行计算优化：利用多卡并行处理，仅多卡场景生效
  parallel-computation:
    enable: false                           # 并行计算总开关
    ulysses-degree: 2                       # Ulysses序列并行度
    ring-degree: 2                           # Ring Attention并行度

  # 量化优化：降低数据精度，减少内存占用和计算开销
  quantization:
    dtype: "bf16"                           # 模型数据类型：fp32 或 bf16
    int8-w8a8:
      enable: false                         # INT8量化开关
      build: false                          # INT8量化构建开关

  # 内存与数据格式优化：提前转换数据格式，避免推理时的转换开销
  memory-and-data-format:
    conv-weight-layout-preconvert: true     # 卷积核布局提前转换

model_name: "vggt"                   # 模型名称
world_size: 1                        # 启动进程数
master_port: 29600                   # torchrun主节点端口
entry_script: "demo_infer.py"        # 入口脚本
```

### 优化项四大类说明

| 优化类别          | 优化项                             | 类型   | 说明                                                         |
| ------------- | ------------------------------- | ---- | ---------------------------------------------------------- |
| **计算冗余消除**    | rope-cache                      | bool | 旋转编码三层缓存，关闭时每次重新计算                                         |
| <br />        | dpt-pos-embed-cache             | bool | DPT头位置编码缓存，关闭时不使用缓存                                        |
| <br />        | cos-sin-dtype-optimization      | bool | Cos/Sin使用bfloat16，关闭时使用float64                             |
| **并行计算（仅多卡）** | enable                          | bool | 启用序列并行，单卡场景必须为false                                        |
| <br />        | ulysses-degree                  | int  | Ulysses并行度，需满足 `ulysses-degree × ring-degree = world_size` |
| <br />        | ring-degree                     | int  | Ring并行度，需满足 `ulysses-degree × ring-degree = world_size`    |
| **量化优化**      | dtype                           | str  | 模型数据类型：`"fp32"`（原始）或 `"bf16"`（半精度）                         |
| <br />        | int8-w8a8.enable                | bool | 启用INT8量化推理（W8A8）                                           |
| <br />        | int8-w8a8.build                 | bool | 构建INT8量化模型，仅首次运行时使用                                        |
| **内存与数据格式**   | conv-weight-layout-preconvert   | bool | 卷积核提前转为Fractal\_Z格式，关闭时使用默认格式                              |

### 参数约束说明

**量化参数关系**：

- `dtype` 和 `int8-w8a8.enable` 可以同时为 true（模型整体BF16 + Linear层INT8）
- `int8-w8a8.build` 和 `int8-w8a8.enable` 不应同时为 true（build用于生成模型，enable用于使用模型）

**并行计算参数约束**：

- `ulysses-degree × ring-degree = world_size`
- 单卡场景（world\_size=1）时，`parallel-computation.enable` 必须为 `false`

**有效配置示例**：

| 场景     | world\_size | parallel.enable | ulysses-degree | ring-degree | dtype | int8-w8a8.enable |
| ------ | ----------- | --------------- | -------------- | ----------- | ----- | ---------------- |
| 单卡推理   | 1           | false           | 1              | 1           | bf16  | false            |
| 单卡量化推理 | 1           | false           | 1              | 1           | bf16  | true             |
| 2卡并行   | 2           | true            | 2              | 1           | bf16  | false            |
| 4卡并行   | 4           | true            | 2              | 2           | bf16  | false            |
| 8卡并行   | 8           | true            | 4              | 2           | bf16  | false            |

### 多卡推理配置说明

多卡推理通过 YAML 配置文件控制，关键参数如下：

| 参数                                                 | 说明                                                      |
| -------------------------------------------------- | ------------------------------------------------------- |
| `world_size`                                       | NPU 卡数，需等于 `ulysses-degree × ring-degree`               |
| `optimization.parallel-computation.enable`         | 是否启用序列并行，多卡推理时设为 `true`                                 |
| `optimization.parallel-computation.ulysses-degree` | Ulysses 并行度，`num_attention_heads` 必须能被其整除               |
| `optimization.parallel-computation.ring-degree`    | Ring 并行度，约束 `ulysses-degree × ring-degree = world_size` |

### 量化参数说明

| 参数                              | 说明                            | 约束            |
| ------------------------------- | ----------------------------- | ------------- |
| `quantization.dtype`            | 模型数据类型：fp32（原始）、bf16（优化，内存减半） | 可与int8-w8a8并存 |
| `quantization.int8-w8a8.enable` | 是否启用INT8量化推理                  | 需配合dtype使用    |
| `quantization.int8-w8a8.build`  | 是否构建INT8量化模型                  | 仅构建阶段使用       |

**约束条件**：

- `dtype` 和 `int8-w8a8.enable` 可以同时为 true（模型整体BF16 + Linear层INT8）
- `build` 和 `enable` 不应同时为 true（build用于生成模型，enable用于使用模型）

### 注意事项

- 多卡配置需满足 `world_size = ulysses-degree × ring-degree`。
- 单卡场景（world\_size=1）时，`parallel-computation.enable` 必须为 `false`。
- INT8量化首次运行需设置 `int8-w8a8.build: true` 构建量化模型。
- 性能分析结果保存在 `profile-dir` 目录。
- `images-path` 可使用相对路径或绝对路径。

