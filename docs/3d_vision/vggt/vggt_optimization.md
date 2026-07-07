# NPU VGGT 模型推理优化实践
本文主要介绍VGGT模型基于NPU的推理优化策略，其中包括以下优化点：
- Cos/Sin算子输入优化
- 旋转编码计算优化
  - 支持NPU npu_rotary_mul融合算子
  - 重复计算消除
- DPT头位置编码计算优化
- 支持NPU npu_add_layer_norm融合算子
- 权重格式转为BF16
- 二维卷积核私有格式提前转换
- 支持NPU  npu_fused_infer_attention_score融合算子
- 支持Ulysses与Ringattention序列并行推理
---
## 性能优化介绍
### Cos/Sin算子优化
**优化原因：** 原网络结构代码中Cos算子和Sin算子的输入数据类型为double，导致这两算子会下发到AI CPU上执行，导致算子性能低。

**优化方式：** 将`vggt/heads/utils.py`文件中make_sincos_pos_embed函数的omega变量数据类型从torch.double修改为torch.bfloat16。
```python
'''替换部分
def _make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(embed_dim // 2, dtype=torch.float32 if device.type == "mps" else torch.double, device=device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.float()
'''
#替换后
def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(embed_dim // 2, dtype=torch.bfloat16, device=device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb
```

### 旋转编码优化
#### 融合算子npu_rotary_mul使能
- **优化原因：** 原网络代码中通过`(token * cos) + (self._rotate_features(tokens) * sin)` 实现rotary操作，Host侧需要下发多个小算子。
- **优化方式：** 修改`vggt/layers/rope.py`文件中_apply_1d_rope函数，使用`npu_rotary_mul`替换原来的算子实现。
```python
'''替换部分
def __apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
  # Embed positions with frequency components
  cos = F.embedding(positions, cos_comp)[:, None, :, :]
  sin = F.embedding(positions, sin_comp)[:, None, :, :]

  # Apply rotation
  return (tokens * cos) + (self._rotate_features(tokens) * sin)
'''
#替换后
def _apply_1d_rope(
        self, tokens: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
  # Apply rotation
  return torch_npu.npu_rotary_mul(tokens, cos, sin, rotary_mode='half')
```

#### 冗余操作去除
- **优化原因：** 在vggt的Attention网络rope模块实现中，每次计算都需要重新计算cos和sin值，存在冗余计算，具体如下： 
    - `vggt/layers/rope.py`文件中_apply_1d_rope函数通过`cos = F.embedding(positions, cos_comp)[:, None, :, :]` 和`sin = F.embedding(positions, sin_comp)[:, None, :, :]`分别计算cos和sin变量。
    - `vggt/layers/rope.py`文件中forward函数需要通过max函数计算输入变量positions的最大值。
    - 每次对q变量和k变量进行rope计算时，都需要分别在垂直维度和水平维度计算cos和sin，并且计算positions的最大值。
- **优化方式：** 旋转编码计算的输入依赖positions变量，positions变量与输入图片的高度与宽度相关。因此通过建立key为(height, width)的字典，缓存cos变量和sin变量的计算结果和positions的最大值，针对不同输入下同样大小的图片，能够减少重复的计算。
```python
'''替换部分
def __compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
  cache_key = (dim, seq_len, device, dtype)
  if cache_key not in self.frequency_cache:
      # Compute frequency bands
      exponents = torch.arange(0, dim, 2, device=device).float() / dim
      inv_freq = 1.0 / (self.base_frequency**exponents)

      # Generate position-dependent frequencies
      positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
      angles = torch.einsum("i,j->ij", positions, inv_freq)

      # Compute and cache frequency components
      angles = angles.to(dtype)
      angles = torch.cat((angles, angles), dim=-1)
      cos_components = angles.cos().to(dtype)
      sin_components = angles.sin().to(dtype)
      self.frequency_cache[cache_key] = (cos_components, sin_components)

  return self.frequency_cache[cache_key]
'''
# 替换后
def _compute_frequency_components(
        self, dim: int,  input_positions: torch.tensor, device: torch.device, dtype: torch.dtype,
        height: None, width: None, batch_size: None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
  if height == None or width == None:
      seq_len = int(input_positions.max()) + 1
  else:
      hw_key = (height, width)
      if hw_key not in self.max_position_cache:
        self.max_position_cache[hw_key] = int(input_positions.max()) + 1
      seq_len = self.max_position_cache[hw_key]
  cache_key = (dim, seq_len, device, dtype)
  if cache_key not in self.frequency_cache:
      # Compute frequency bands
      exponents = torch.arange(0, dim, 2, device=device).float() / dim
      inv_freq = 1.0 / (self.base_frequency**exponents)

      # Generate position-dependent frequencies
      positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
      angles = torch.einsum("i,j->ij", positions, inv_freq)

      # Compute and cache frequency components
      angles = angles.to(dtype)
      angles = torch.cat((angles, angles), dim=-1)
      cos_components = angles.cos().to(dtype)
      sin_components = angles.sin().to(dtype)
      self.frequency_cache[cache_key] = (cos_components, sin_components)

  cos_components, sin_components = self.frequency_cache[cache_key]

  if height == None or width == None or batch_size == None:
      vertical_cos = F.embedding(input_positions[...,0], cos_components)[:, None, :, :]
      vertical_sin = F.embedding(input_positions[...,0], sin_components)[:, None, :, :]
      horizontal_cos = F.embedding(input_positions[...,1], cos_components)[:, None, :, :]
      horizontal_sin = F.embedding(input_positions[...,1], sin_components)[:, None, :, :]
      return vertical_cos, vertical_sin, horizontal_cos, horizontal_sin

  sub_cache_key = (height, width)
  if sub_cache_key not in self.cos_sin_cache:
      # Embed positions with frequency components
      vertical_cos = F.embedding(input_positions[...,0], cos_components)[:, None, :, :]
      vertical_sin = F.embedding(input_positions[...,0], sin_components)[:, None, :, :]
      horizontal_cos = F.embedding(input_positions[...,1], cos_components)[:, None, :, :]
      horizontal_sin = F.embedding(input_positions[...,1], sin_components)[:, None, :, :]
      self.cos_sin_cache[sub_cache_key] = (vertical_cos, vertical_sin, horizontal_cos, horizontal_sin)
  vertical_cos, vertical_sin, horizontal_cos, horizontal_sin = self.cos_sin_cache[sub_cache_key]
  return vertical_cos, vertical_sin, horizontal_cos, horizontal_sin

```
### DPT头位置编码计算优化
- **优化原因：** VGGT网络在DPT头的实现中，每次计算都需要对输入重新进行位置编码的计算，存在冗余计算。
-  **优化方式：** 由于位置编码的结果取决于输入图片的大小和token的长度，因此通过建立字典将结果进行缓存的方式把位置编码的结果提前保存，避免冗余计算，减少计算量。
```python
'''替换部分
def __apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
  patch_w = x.shape[-1]
  patch_h = x.shape[-2]
  pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
  pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
  pos_embed = pos_embed * ratio
  pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
  return x + pos_embed
'''
#替换后
def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
   if (W, H, x.shape) not in self.pos_embed_cache:
       patch_w = x.shape[-1]
       patch_h = x.shape[-2]
       pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
       pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
       pos_embed = pos_embed * ratio
       pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
       self.pos_embed_cache[(W, H, x.shape)] = pos_embed
   pos_embed = self.pos_embed_cache[(W, H, x.shape)]
   return x + pos_embed
```
### Add+LayerNorm融合算子
- **优化原因：** 将Add和LayerNorm两个小算子替换为融合大算子，减少算子下发次数和中间结果内存读写，提升性能。
- **优化方式：** 使用融合算子`npu_add_layer_norm`替换原来的Add+LayerNorm分开执行的方式。
```python
#替换前（两个算子分开执行）
def original_layernorm_forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
    if residual is None:
        return torch_npu.npu_layer_norm_eval(x, self.normalized_shape, self.weight, self.bias, self.eps)
    else:
        residual = residual + x  # 算子1: Add操作（单独执行）
        y = torch_npu.npu_layer_norm_eval(residual, self.normalized_shape, self.weight, self.bias, self.eps)  # 算子2: LayerNorm（单独执行）
        return y, residual

#替换后（融合算子，Add+LayerNorm合并为1个算子）
def vggt_layernorm_forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
    if residual is None:
        return torch_npu.npu_layer_norm_eval(x, self.normalized_shape, self.weight, self.bias, self.eps)
    else:
        y, _, _, residual = torch_npu.npu_add_layer_norm(residual, x, self.weight, self.bias, self.eps, additional_output = True)
        return y, residual

nn.LayerNorm.forward = vggt_layernorm_forward
```
> **说明：** 优化前后都使用NPU兼容的`npu_layer_norm_eval`作为底层LayerNorm实现，区别仅在于Add操作是否融合（2个算子 → 1个融合算子）。
### 使用BF16权重
- **优化原因：** 目前vggt网络的权重使用float数据类型，考虑将vggt网络权重转为bfloat16。
- **优化方式：** 在加载完模型后，将模型权重转为bfloat16。使用该方案后，取得6.62%的性能收益，相机位姿估算任务精度相比fp32，精度从0.919下降至0.911，精度损失在0.5%以内。
<!-- - 量化后相比原模型的性能收益和对应的精度损失 -->

### 使用INT8权重
- **优化原因：** 目前vggt网络的Linear层使用W8A8量化精度可控(下降1%以内)，考虑将vggt网络权重离线转为int8。
- **优化方式：** 将部分Linear层的激活使用动态per-token量化，权重使用静态per-channel量化。fp32(原始)模型大小为4.9GB，bf16模型大小为2.46GB，int8模型大小为2.16G. 相机位姿估算任务精度相比bf16，精度从0.911下降至0.907，精度损失在0.5%以内。。
- **使能方式：** 默认关闭，使能需将enableW8A8设为True。

### 卷积核私有格式提前转换
- **优化原因：** NPU上进行二维卷积操作时需要提前通过Transdata算子将卷积核转为`Fractal_Z`格式，在推理过程中存在数据格式的转换开销。
- **优化方式：** 在加载完模型权重后，提前将二维卷积核的数据格式转为`Fractal_Z`，进而避免转换开销的引入。
```python
#替换后
def cast_model_weight(model):
    def __format_cast(module, class_name):
        if issubclass(class_name, torch.nn.Conv2d):
            if module.groups > 1:
                return
            if hasattr(module, "weight") and module.weight is not None and \
                "weight" in dict(module.named_parameters()):
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 4)
        return module
    def cast_weight(module):
        current_class = module.__class__
        module = __format_cast(module, current_class)

        if not module.children:
            return
        for sub_module in module.children():
            if isinstance(sub_module, torch.nn.Module):
                sub_module = cast_weight(sub_module)
        return module
    for _, module in model.named_modules():
        module = cast_weight(module)
    return model
# 推理时
model = VGGT()
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
model = model.to(dtype)
model.to(device).eval()
model = cast_model_weight(model)#调用cast_model_weight函数，在推理前提前转换卷积核数据格式
predictions = model(images)
```

### NPU fused_infer_attention_score算子适配
- **优化原因：** npu_fused_infer_attention_score融合算子是Ascend Extension for PyTorch提供的适配增量和全量推理场景的FlashAttention算子，在昇腾环境上，相比SDPA有更好的推理性能表现。
- **优化方式：** 针对Frame-Global双层架构实现算子分离。Frame Attention处理帧内短序列，保持使用PyTorch SDPA；Global Attention处理跨帧长序列，切换到NPU FIA融合算子。
```python
# 原始：统一使用SDPA
x = F.scaled_dot_product_attention(q, k, v, dropout_p=...)

# 优化：根据is_global_attention分流
if self.is_global_attention:
    # Global用FIA
    x = torch_npu.npu_fused_infer_attention_score(
        q, k, v, num_heads=num_heads, scale=scale,
        input_layout="BNSD", pre_tokens=65535, 
        next_tokens=65535, inner_precise=0
    )[0]
else:
    # Frame用SDPA
    x = F.scaled_dot_product_attention(q, k, v, ...)
```

### 序列并行的推理适配
- **优化原因：** VGGT网络原生仅支持单卡推理，所有计算在一张卡上完成，未能有效利用Atlas A2的多卡优势；多卡并行将序列切分到各rank，可以有效提升整网的推理性能。
- **优化方式：**

  1.Ulysses并行将num_heads维度切分到多卡，通过all-to-all通信将序列维度聚合，使每个rank能看到完整序列但只处理部分attention heads。
  
  2.Ring并行将序列切分到多卡，通过overlap策略隐藏通信开销：利用NPU FIA算子返回的LSE（log-sum-exp）信息，支持分块attention结果的数值稳定合并。
```python
# Ulysses并行（头维度切分）
# 通过all-to-all通信实现head和sequence维度的互换
# 输入: [B, shard_s, H, D] - 每个rank持有部分序列、完整头
# 输出: [B, S, shard_hc, D] - 每个rank持有完整序列、部分头
def _all_to_all_head_to_seq(self, input_):
    input_t = input_.reshape(bs, shard_s, world_size, shard_hc, d)
    input_t = input_t.permute(2, 1, 0, 3, 4).contiguous()
    dist.all_to_all_single(output, input_t, group=self.ulysses_pg)
```
```python
# Ring并行（序列维度切分）
# Ring Attention with Overlap: 边通信边计算
def _ring_attention_overlap(self, params):
    # Step 1: 异步启动K/V的allgather
    k_handle = dist.all_gather_into_tensor(k_gathered, k, async_op=True)
    v_handle = dist.all_gather_into_tensor(v_gathered, v, async_op=True)
    # Step 2: 计算local attention (Q @ local_K/V)
    out_local, lse_local = self._compute_attention_with_lse(q, k, v)
    # Step 3: 等待通信完成，计算cross attention (Q @ other_K/V)
    k_handle.wait()
    out_others, lse_others = self._compute_attention_with_lse(q, k_others, v_others)
    # Step 4: 使用LSE合并结果
    out_merged = self._merge_two_outputs(out_local, lse_local, out_others, lse_others)
```
- **额外影响：** 对VGGT网络适配序列并行后，以下是整网精度情况
  |并行度|AUC@30|
  |:---:|:---:|
  |单卡|91.10|
  |Ulysses=2|91.12|
  |Ulysses=4|91.10|
  |Ulysses=8|90.86|
  |Ring=2|91.12|
  |Ring=4|91.09|
  |Ring=8|90.88|

## 性能优化指标
本方案使用8卡Atlas 800I A2推理产品，输入vggt提供的样例数据(`examples/kitchen`)，包含25张图片，性能指标如下
|使能方法|推理耗时（ms）|
|:---:|:---:|
|Cos\Sin算子优化|1324.83|
|旋转编码优化|1239.55|
|DPT头计算优化|1211.26|
|npu_add_layer_norm融合算子|1208.17|
|权重格式转BF16|1128.18|
|私有格式提前转换|1121.09|

采用序列并行的多卡推理后，性能提升如下
|并行度|推理提升|
|:---:|:---:|
|Ulysses=2|1.75x|
|Ulysses=4|3.43x|
|Ulysses=8|6.47x|
|Ring=2|1.82x|
|Ring=4|3.48x|
|Ring=8|6.42x|
