# OpenVLA on 310P具身智能VLA大模型昇腾使用指南

本目录介绍在 Ascend 310P 上如何对 OpenVLA 模型进行离线模型转换及推理，附带精度验证及仿真步骤。

## OpenVLA整体介绍

在《OpenVLA: An Open-Source Vision-Language-Action Model》论文中提出了 OpenVLA 模型，论文地址为：  
https://arxiv.org/abs/2406.09246

OpenVLA 官方仓库地址为：  
https://github.com/openvla/openvla

### 功能介绍

OpenVLA 是一种典型的视觉-语言-动作（Vision-Language-Action, VLA）通用控制模型，其核心思想是将视觉观测与语言指令统一编码到同一序列表示中，并利用自回归（autoregressive）生成的方式输出动作表示（如动作 token 或离散化动作序列），再解码为可执行的连续控制量。通过在大规模多任务机器人示范数据上学习“从感知与语义到动作”的统一映射，OpenVLA 旨在提升跨任务、跨场景的泛化能力，并降低为每个任务单独训练策略的成本

## OpenVLA的相关代码仓拉取、仿真测试集和模型下载

本样例使用的示例模型为： https://huggingface.co/openvla/openvla-7b-finetuned-libero-object 

这是OpenVLA官方发布的在libero_object数据集上微调后的模型

### 模型输入输出

> 说明：OpenVLA 的输入由 **文本指令（token）** 与 **图像张量（pixel_values）** 共同构成。  
> 当启用 *fused vision backbone* 时，`pixel_values` 的通道数为 **6（3+3）**，表示同一帧图像经过两套视觉预处理后在通道维拼接。

#### 输入（Inputs）

| 输入名 | 含义 | dtype | shape（示例） | 备注 |
| --- | --- | --- | --- | --- |
| `input_ids` | 指令/提示词的 token 序列 | `int64` | `[B, T]` | `T` 为文本 token 长度（包含特殊 token）；`B` 为 batch size（常见为 1） |
| `attention_mask` | 文本 token 的有效位掩码 | `bool`（或 `int64/int32`，依实现而定） | `[B, T]` | 1/True 表示有效 token，0/False 表示 padding |
| `pixel_values` | 摄像头 RGB 图像经 processor 预处理后的张量 | `float16`（常见） | `[B, C, H, W]` | 若 `use_fused_vision_backbone=True`，则 `C=6（3+3）`；否则 `C=3` |

#### 输出（Outputs）

| 输出名 | 含义 | dtype | shape（示例） | 备注 |
| --- | --- | --- | --- | --- |
| `actions` / `generated_ids` | 动作 token（或离散动作序列的 token id） | `int64`（常见） | `[B, A]` | `A` 为动作维度/动作 token 个数（通常由 `action_dim` 决定）；后续需用 `bin_centers` + `action_norm_stats` 反归一化得到连续动作 |

**参数符号说明：**

- `B`：batch size（离线验证通常为 1）。
- `T`：文本 token 序列长度（由 prompt 长度与 tokenizer 规则决定，含特殊 token）。
- `H, W`：processor 输出的视觉输入分辨率（常见为 224×224，具体以 processor 配置为准）。
- `C`：图像通道数；启用 fused backbone 时为 `6=3+3`（两套视觉塔输入拼接），否则为 `3`。
- `A`：动作序列长度/动作维度（通常等于 `action_dim`，与机器人自由度、动作表示方式有关）。


## OpenVLA在昇腾310P上的运行配置

### 与昇腾平台相关的环境配置

.om 模型转化及运行需要安装 CANN 软件包。

本样例的编译执行依赖 CANN 开发套件包（cann-toolkit）与 CANN 二进制算子包（cann-kernels），支持的 CANN 软件版本为 `CANN 8.0.0-8.2.RC1`。请从软件包下载地址下载对应架构软件包，并参考 CANN 安装文档依次进行安装。

```bash
# ${cann_install_path}为CANN包的实际安装目录，注意每次新建终端时，首先source一下set_env.sh。
# 方式1：默认路径安装，以root用户为例
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 方式2：指定路径进行安装
source ${cann_install_path}/ascend-toolkit/set_env.sh
```

### 与昇腾服务器无关的环境配置

```bash
# 创建运行环境
conda create -y -n openvla python=3.10
conda activate openvla

# 拉取openvla仓库并安装（示例）
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

```
### 仿真渲染（MuJoCo）无头模式

如果服务器/容器缺少显示环境或 OpenGL 渲染后端，MuJoCo 可能无法正常渲染。
可以在运行仿真/评测前指定 EGL 无头渲染：

```bash
export MUJOCO_GL=egl
```

## OpenVLA在昇腾310P上的推理步骤

本节介绍离线推理模式（通过昇腾亲和的 OM 文件）的部署参考，更多使用参数可参考[atc工具使用文档](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/devaids/atctool/atlasatc_16_0003.html)。
<img src="https://raw.gitcode.com/user-images/assets/7380116/c583e4bd-fddf-4d44-bc65-7ad69d84ab02/om_compile_workflow.png" style="zoom:50%;" />
下面给出一条推荐的单机器链路：

1) 在 310P 宿主机导出 ONNX（使用Host CPU）
2) 使用 ATC 将 ONNX 转为 OM（在 310P 上）
3) 使用 OM-backend sim-evaluator 在仿真环境评测（在 310P 上）

需要在转化onnx的机器上额外安装onnx runtime依赖:
```bash
pip install onnx
# 基于Host CPU转换onnx请安装(310P 宿主机执行):
pip install onnxruntime
# 基于Host GPU转换onnx请安装：
pip install onnxruntime-gpu
```

#### 1) 导出 ONNX
导出前，需要先对环境中的transformers库进行算子的转化修复，确保OM模型转换时能匹配昇腾亲和的算子
```bash
cd /path/to/conda/envs/openvla/lib/python3.10/site-packages/transformers/models/llama
git apply --check -p1 /path/to/openvla/modeling_llama.patch
git apply -p1 /path/to/openvla/modeling_llama.patch
```

在 Host（CPU 或 GPU）上执行：

```bash
# 以本地目录为例：models/ 里包含 config.json 等文件
# 也可以先用 huggingface-cli 下载到 models/:
#   pip install -U huggingface_hub
#   huggingface-cli download openvla/openvla-7b-finetuned-libero-object --local-dir models

python3 convert_and_verify_onnx.py \
  --model-path models/openvla-7b-finetuned-libero-object \
  --vision-export-dir outputs/onnx/vision \
  --llama-prefill-export-dir outputs/onnx/llama_prefill \
  --llama-decoder-export-dir outputs/onnx/llama_decoder \
  --unnorm-key libero_object
```

说明：
- 默认会用 ONNXRuntime CPU 对比 PyTorch 输出，打印 max/mean diff；如需跳过可加 `--no-validate`。

样例输出如下：
```
============================================================
Validating Full Inference Pipeline
============================================================

[1/2] Running PyTorch inference...
[2/2] Running ONNX inference...
Loading ONNX models with provider: CPUExecutionProvider...
ONNX models loaded successfully.

[3/3] Comparing results...
full_pipeline_action: max abs diff = 0.000000e+00
full_pipeline_action: mean abs diff = 0.000000e+00
full_pipeline_action: ✓ MATCH (rtol=0.001, atol=0.001, mean_diff_threshold=1e-2)

PyTorch action:
[ 1.43156521e-01  2.43907466e-02  9.26470588e-01 -3.15118654e-05
  7.75504180e-02 -3.35294148e-02  0.00000000e+00]

ONNX action:
[ 1.43156521e-01  2.43907466e-02  9.26470588e-01 -3.15118654e-05
  7.75504180e-02 -3.35294148e-02  0.00000000e+00]

✅ Full pipeline validation passed!

============================================================
```

#### 2) ATC 将 ONNX 转为 OM

在 310P 上（已安装并 `source` CANN 环境）执行转换脚本：

```bash
./convert_onnx_to_om.sh \
    --vision-onnx-dir outputs/onnx/vision \
    --llama-prefill-onnx-dir outputs/onnx/llama_prefill \
    --llama-decoder-onnx-dir outputs/onnx/llama_decoder \
    --vision-om-dir outputs/om/vision \
    --llama-prefill-om-dir outputs/om/llama_prefill \
    --llama-decoder-om-dir outputs/om/llama_decoder \
    --soc-version Ascend310P3
```
当模型转换完成后，各个模型转换出的.om模型应在相应的各个指定output目录中，在终端中有输出“ATC run success, welcome to the next use”。

#### 3) 使用 OM-backend sim-evaluator 进行仿真评测

在 310P 上执行（需要 ACL/ACLLite Python 依赖可用）。
可参考：[ACLLite安装教程](https://gitee.com/ascend/ACLLite#%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B)

仿真评测是在openvla官方给出的libero仿真评测的基础上修改而来，可以通过应用仓库中给出的仿真适配patch获取OM-backend仿真评测代码环境，仿真相关patch位于仓库`sim/`目录下，包含`robot_utils.patch` `openvla_utils.patch` `run_libero_eval.patch`三个patch，以及一个需要新增的文件`openvla_om_utils.py`
```bash
#确保处于openvla仓库根目录
cd openvla
git apply --check /path/to/xxx.patch
git apply xxx.patch

#新增文件需置于experiments/robot/目录下
cp /path/to/openvla_om_utils.py ./experiments/robot/
```
准备好代码环境后，可以运行以下命令进行仿真评测
```bash
python3 -m experiments.robot.libero.run_libero_eval \
    --model_family openvla \
    --pretrained_checkpoint models/openvla-7b-finetuned-libero-object/ \
    --task_suite_name libero_object \
    --center_crop True \
    --vision_backbone_om outputs/om/vision/vision_backbone.om \
    --projector_om outputs/om/vision/projector.om \
    --embedding_om outputs/om/vision/embedding.om \
    --prefill_om outputs/om/llama_prefill/vla_prefill.om \
    --decode_om outputs/om/llama_decoder/vla_decoder.om
```

输出：
- 评测结果日志会写入 `experiments/logs`，包含成功率等信息
- 仿真结果视频位于 `rollout/date`目录下，`date`为日期

## OpenVLA在昇腾上的精度验证步骤

接下来介绍两种验证转换后的 .om 模型在 NPU 上运行的验证方法。

### 1）基于mock的数据输入，CPU/GPU与原始Pytorch输出相似度对比

构造固定输入（如全0图像 + 固定指令 token），测试 PyTorch CPU/GPU 和 OM 310P NPU 的输出精度对比：

```bash
# 在 310P 上执行（需要 ACL/AclLite Python 依赖）
python3 verify_om_onnx.py \
   --model-path models/openvla-7b-finetuned-libero-object \
   --unnorm-key libero_object \
   --vision-backbone-om outputs/om/vision/vision_backbone.om \
   --projector-om outputs/om/vision/projector.om \
   --embedding-om outputs/om/vision/embedding.om \
   --prefill-om outputs/om/llama_prefill/vla_prefill.om \
   --decode-om outputs/om/llama_decoder/vla_decoder.om
```

### 2）基于仿真模拟器的功能测试（MuJoCo / LIBERO）

使用`libero`仿真环境数据在 NPU 进行推理，在 Host CPU 上进行仿真渲染（或控制循环）：

```bash
python3 -m experiments.robot.libero.run_libero_eval \
    --model_family openvla \
    --pretrained_checkpoint models/openvla-7b-finetuned-libero-object/ \
    --task_suite_name libero_object \
    --center_crop True \
    --vision_backbone_om outputs/om/vision/vision_backbone.om \
    --projector_om outputs/om/vision/projector.om \
    --embedding_om outputs/om/vision/embedding.om \
    --prefill_om outputs/om/llama_prefill/vla_prefill.om \
    --decode_om outputs/om/llama_decoder/vla_decoder.om
```

示例效果: \
<img src="https://raw.gitcode.com/user-images/assets/7380116/7b2551e1-efab-4540-9bcd-77b3eda0b6e7/libero.gif " style="zoom:60%;" />

## Citation

```bibtex
@article{kim24openvla,
    title={OpenVLA: An Open-Source Vision-Language-Action Model},
    author={{Moo Jin} Kim and Karl Pertsch and Siddharth Karamcheti and Ted Xiao and Ashwin Balakrishna and Suraj Nair and Rafael Rafailov and Ethan Foster and Grace Lam and Pannag Sanketi and Quan Vuong and Thomas Kollar and Benjamin Burchfiel and Russ Tedrake and Dorsa Sadigh and Sergey Levine and Percy Liang and Chelsea Finn},
    journal = {arXiv preprint arXiv:2406.09246},
    year={2024}
} 
```

## 附录：openvla 根目录相关代码目录树（示例）

检查整体代码目录树，经过上述的操作，OpenVLA 适配昇腾的工程目录树示例如下所示：

```text
格式
├── README.md                       # 本文件
├── README_en.md                    # 英文使用指南
├── models                          # Huggingface或其他来源下载的模型
├── openvla/                        
|   ├── convert_and_verify_onnx.py  # PyTorch -> ONNX 转化脚本
|   ├── verify_om_onnx.py           # PyTorch(CPU) vs OM(NPU) 误差对比
|   ├── vla_validation_utils.py     # 精度验证辅助方法
|   ├── convert_onnx_to_om.sh       # ONNX -> OM 转化脚本
|   ├── lib
|   |   └── modeling_llama.patch    # 对transformers lib的适配patch
|   |
|   └── sim
|       ├── robot_utils.patch       # 仿真文件robot_utils.py补丁
|       ├── openvla_utils.patch     # 仿真文件openvla_utils.py补丁
|       ├── run_libero_eval.patch   # 仿真文件run_libero_eval.py补丁
|       └── openvla_om_utils.py     # 仿真新增OM-Backend支持文件
|   
└── outputs
    ├── onnx/                       # 输出的onnx格式模型
    └── om/                         # 输出的om格式模型

```
