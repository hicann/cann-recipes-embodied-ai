## YAML Parameter Reference

VGGT inference parameters are managed through `config/*.yaml` files. Use the `--config` parameter to specify the configuration file.

Default configurations:

- Single NPU baseline: `single.yaml`
- 2-NPU sequence parallel: `sp2.yaml`
- 4-NPU sequence parallel: `sp4.yaml`
- 8-NPU sequence parallel: `sp8.yaml`
- Single NPU W8A8 quantization: `single_w8a8.yaml`

```yaml
model_args:
  ckpt: "ckpt/model.pt"              # Model weights path (required)
  images-path: "examples/kitchen/images"  # Input image directory (required)
  enable-profiling: false            # Enable performance profiling
  profile-dir: "prof_sp"             # Profiling output directory
  num-runs: 6                        # Number of inference runs

# Optimization configuration
optimization:
  # Computation redundancy elimination optimization: avoid redundant computation through caching mechanism
  computation-redundancy-elimination:
    rope-cache: true                        # Rotary encoding cache
    dpt-pos-embed-cache: true               # DPT head position encoding cache
    cos-sin-dtype-optimization: true        # Cos/Sin data type optimization

  # Parallel computation optimization: utilize multi-NPU parallel processing, only effective in multi-NPU scenarios
  parallel-computation:
    enable: false                           # Parallel computation master switch
    ulysses-degree: 2                       # Ulysses sequence parallelism degree
    ring-degree: 2                           # Ring Attention parallelism degree

  # Quantization optimization: reduce data precision to decrease memory usage and computation overhead
  quantization:
    dtype: "bf16"                           # Model data type: fp32 or bf16
    int8-w8a8:
      enable: false                         # INT8 quantization switch
      build: false                          # INT8 quantization build switch

  # Memory and data format optimization: convert data format in advance to avoid runtime conversion overhead
  memory-and-data-format:
    conv-weight-layout-preconvert: true     # Convolution kernel layout pre-conversion

model_name: "vggt"                   # Model name
world_size: 1                        # Number of processes to start
master_port: 29600                   # torchrun master node port
entry_script: "demo_infer.py"        # Entry script
```

### Four Major Optimization Categories

| Category | Parameter | Type | Description |
| ------------- | ------------------------------- | ---- | ---------------------------------------------------------- |
| **Computation Redundancy Elimination** | rope-cache | bool | Rotary encoding three-layer cache, recalculates each time when disabled |
| <br /> | dpt-pos-embed-cache | bool | DPT head position encoding cache, no caching when disabled |
| <br /> | cos-sin-dtype-optimization | bool | Cos/Sin uses bfloat16, uses float64 when disabled |
| **Parallel Computation (Multi-NPU only)** | enable | bool | Enable sequence parallelism, must be false for single NPU scenario |
| <br /> | ulysses-degree | int | Ulysses parallelism degree, must satisfy `ulysses-degree × ring-degree = world_size` |
| <br /> | ring-degree | int | Ring parallelism degree, must satisfy `ulysses-degree × ring-degree = world_size` |
| **Quantization** | dtype | str | Model data type: `"fp32"` (original) or `"bf16"` (half precision) |
| <br /> | int8-w8a8.enable | bool | Enable INT8 quantization inference (W8A8) |
| <br /> | int8-w8a8.build | bool | Build INT8 quantization model, only used during first run |
| **Memory and Data Format** | conv-weight-layout-preconvert | bool | Convolution kernel pre-converted to Fractal_Z format, uses default format when disabled |

### Parameter Constraints

**Quantization Parameter Relationships**:

- `dtype` and `int8-w8a8.enable` can both be true (model overall BF16 + Linear layer INT8)
- `int8-w8a8.build` and `int8-w8a8.enable` should not both be true (build is for generating model, enable is for using model)

**Parallel Computation Parameter Constraints**:

- `ulysses-degree × ring-degree = world_size`
- For single NPU scenario (world_size=1), `parallel-computation.enable` must be `false`

**Valid Configuration Examples**:

| Scenario | world_size | parallel.enable | ulysses-degree | ring-degree | dtype | int8-w8a8.enable |
| -------- | ----------- | --------------- | -------------- | ----------- | ----- | ---------------- |
| Single NPU inference | 1 | false | 1 | 1 | bf16 | false |
| Single NPU quantization inference | 1 | false | 1 | 1 | bf16 | true |
| 2-NPU parallel | 2 | true | 2 | 1 | bf16 | false |
| 4-NPU parallel | 4 | true | 2 | 2 | bf16 | false |
| 8-NPU parallel | 8 | true | 4 | 2 | bf16 | false |

### Multi-NPU Inference Configuration

Multi-NPU inference is controlled through YAML configuration files, with key parameters as follows:

| Parameter | Description |
| -------------------------------------------------- | ------------------------------------------------------- |
| `world_size` | Number of NPU cards, must equal `ulysses-degree × ring-degree` |
| `optimization.parallel-computation.enable` | Whether to enable sequence parallelism, set to `true` for multi-NPU inference |
| `optimization.parallel-computation.ulysses-degree` | Ulysses parallelism degree, `num_attention_heads` must be divisible by it |
| `optimization.parallel-computation.ring-degree` | Ring parallelism degree, constraint `ulysses-degree × ring-degree = world_size` |

### Quantization Parameters

| Parameter | Description | Constraint |
| ------------------------------- | ----------------------------- | ------------- |
| `quantization.dtype` | Model data type: fp32 (original), bf16 (optimized, half memory) | Compatible with int8-w8a8 |
| `quantization.int8-w8a8.enable` | Whether to enable INT8 quantization inference | Use with dtype |
| `quantization.int8-w8a8.build` | Whether to build INT8 quantization model | Only used during build phase |

**Constraints**:

- `dtype` and `int8-w8a8.enable` can both be true (model overall BF16 + Linear layer INT8)
- `build` and `enable` should not both be true (build is for generating model, enable is for using model)

### Notes

- Multi-NPU configurations must satisfy `world_size = ulysses-degree × ring-degree`.
- For single NPU scenario (world_size=1), `parallel-computation.enable` must be `false`.
- For first-time INT8 quantization run, set `int8-w8a8.build: true` to build quantization model.
- Performance profiling results are saved in `profile-dir` directory.
- `images-path` can use relative or absolute path.