# Pi0.5 具身智能 VLA 大模型昇腾使用指南

本目录用于在昇腾 310P NPU 上完成 Pi0.5 的完整部署链路：

1. PyTorch 导出 ONNX（Part1 / Part2）
2. 使用 ATC 将 ONNX 编译为 OM
3. CPU 侧 ONNX 端到端推理验证
4. NPU 侧 OM 端到端推理验证

文档同时给出与 Libero 仿真环境结合的建议方式，便于后续评测与集成。

### Pi0.5 整体介绍

在 $π_{0.5}$: a Vision-Language-Action Model with Open-World Generalization 论文中提出了 Pi0 模型，原仓库地址为：https://github.com/Physical-Intelligence/openpi

LeRobot 基于 OpenPI 的代码进行了迁移，并在 v0.4.0 后重构了 pi0 和 pi0.5 的代码结构。基于旧版 LeRobot 训练的模型在新版代码推理时可能出现配置类报错。

因此，本样例主要适配 `58f70b6bd370864139a3795ac3497a9eae8c42d5` 对应版本的 LeRobot 代码。

为适配昇腾 NPU 的显存和计算特性，我们在推理时将 Pi0.5 解耦为两部分进行部署：
1. **Part 1 (VLM 模块)**：主要由 PaliGemma 构成，负责处理视觉图像（Image）、本体状态（State）与语言指令（Language Tokens），提取多模态特征，输出缓存的 `past_kv_tensor` 与 `prefix_pad_masks`。
2. **Part 2 (Action Expert 模块)**：主要由 Gemma 构成，利用 Part 1 提取的特征，在循环内部进行 10 步去噪迭代，最终输出机器人执行动作（Action）。

#### 功能介绍

Pi0.5 是基于 OpenPI/LeRobot 框架的升级版 Vision-Language-Action (VLA) 模型。与原版 Pi0 类似，Pi0.5 同样基于条件去噪扩散过程（Diffusion）来生成动作，但其针对视觉输入、Token长度以及网络结构进行了优化升级。

### Pi05 的相关代码仓、仿真测试集和模型下载

本样例使用的示例模型为：https://huggingface.co/lerobot/pi05_libero_finetuned_v044

该模型在 MuJoCo 仿真环境中完成微调，测试集为 LIBERO。

> 建议：使用与训练/导出阶段一致的 LeRobot 代码版本、配置文件和 tokenizer，避免输入特征定义不一致导致的推理错误。

---

### 模型输入输出

由于模型被拆分为两部分，各自的输入输出如下：

#### Part 1 (VLM)
| **输入数据名** | **数据类型(dtype)** | **数据大小(shape)** |
| :--- | :--- | :--- |
| observation.state （机器人位姿状态） | Float32 | $[1, 8]$ |
| observation.images.image （第一视角摄像头图像） | Float32 | $[1, 3, 256, 256]$ |
| observation.images.image2 （第二视角摄像头图像） | Float32 | $[1, 3, 256, 256]$ |
| observation.language.tokens （语言指令 Token） | Int64 | $[1, 200]$ |
| observation.language.attention_mask （语言注意力掩码） | Bool | $[1, 200]$ |
| prefix_att_masks （图像与语言拼接 1D 掩码） | Bool | $[1, 968]$ |
| prefix_att_2d_masks_4d （图像与语言拼接 4D 注意力掩码） | Float32 | $[1, 1, 968, 968]$ |

| **输出数据名** | **数据类型(dtype)** | **数据大小(shape)** |
| :--- | :--- | :--- |
| past_kv_tensor （视觉与语言特征 KV 缓存） | Float16 | 动态 (由模型层数和注意力头数决定) |
| prefix_pad_masks （特征 Padding 掩码） | Bool | $[1, 968]$ |

#### Part 2 (Action Expert)
| **输入数据名** | **数据类型(dtype)** | **数据大小(shape)** |
| :--- | :--- | :--- |
| past_kv_tensor （来自 Part 1 的特征 KV 缓存） | Float16 | 与 Part 1 输出的 Shape 保持一致 |
| prefix_pad_masks （来自 Part 1 的掩码） | Bool | $[1, 968]$ |
| time （扩散过程时间步参数） | Float16 | $[1]$ |
| noise （初始或中间扩散动作噪声） | Float16 | $[1, 50, 32]$ |

| **输出数据名** | **数据类型(dtype)** | **数据大小(shape)** |
| :--- | :--- | :--- |
| action （单步去噪后的动作块 Chunk） | Float16 / Float32 | $[1, 50, 32]$ |

**参数符号与数值补充说明：**
* **$256$**: 图像输入的分辨率（$H=256, W=256$），Pi0.5 默认将图像处理为该尺寸。
* **$200$**: 文本指令的最大 Token 长度。
* **$968$**: 拼接后的序列总长度（包含 3 路图像 Token $3 \times 256 = 768$ 以及 200 长度的文本 Token）。
* **$50$**: 动作执行步数（$chunk\_size$）。
* **$32$**: Pi0.5 预设的最大动作维度（$max\_action\_dim$），最终输出后通常会根据实际机器人的自由度进行截断。

输入输出格式定义在模型文件中的config.json中，样例示例为二路摄像头，内容如下：
```json
"input_features": {
        "observation.images.image": {
            "type": "VISUAL",
            "shape": [
                3,
                256,
                256
            ]
        },
        "observation.images.image2": {
            "type": "VISUAL",
            "shape": [
                3,
                256,
                256
            ]
        },
        "observation.state": {
            "type": "STATE",
            "shape": [
                8
            ]
        }
    },
    "output_features": {
        "action": {
            "type": "ACTION",
            "shape": [
                7
            ]
        }
    },
"chunk_size": 50,
```
---

### Pi05在昇腾310P上的运行配置

#### 与昇腾平台相关的环境配置

.om模型转化及运行需要安装CANN软件包。

本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.0.0-8.2.RC1`。 请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1)下载对应架构软件包，例如` Ascend-cann-toolkit_8.2.RC1_linux-x86_64.run`与` Ascend-cann-kernels-310p_8.2.RC1_linux-x86_64.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)依次进行安装。

```bash
# ${cann_install_path}为CANN包的实际安装目录，注意每次新建终端时，首先source一下set_env.sh。
# 方式1：默认路径安装，以root用户为例
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 方式2：指定路径进行安装
source ${cann_install_path}/ascend-toolkit/set_env.sh

# 导入 ACLLite 环境变量 (请替换为您实际的 acllite_utils.py 所在目录的上一级)
# 有可能路径为 /path/to/Ascend/ascend-toolkit/latest/thirdpart/python/
export PYTHONPATH=/path/to/Ascend/ascend-toolkit/latest_owner/thirdpart/python/:$PYTHONPATH
# 使用 HuggingFace 国内镜像网址
export HF_ENDPOINT=https://hf-mirror.com
```

---

#### 与昇腾服务器无关的环境配置（导出 ONNX 与 CPU 验证）
```bash
# 1) 拉取 GitCode 本 PI05 使用仓库并切换指定分支
git clone https://gitcode.com/CANN/cann-recipes-embodied-ai.git

# 2) 拉取 lerobot 仓库并切到指定 **commit**
cd cann-recipes-embodied-ai/manipulation
git clone https://github.com/huggingface/lerobot.git  # **如果contrib下没有lerobot目录**
cd lerobot
git checkout 58f70b6bd370864139a3795ac3497a9eae8c42d5

# 3) 创建并激活 Conda 环境
conda create -y -n lerobot-pi05 python=3.10
conda activate lerobot-pi05

# 4) 安装 lerobot（本地可编辑安装）
pip install -e . # **如果使用不同版本的 lerobot，请重新执行pip install -e .进行更新**

# 5) Pi0.5 依赖安装
pip install -e ".[pi]"

# 6) 仿真依赖（LIBERO 仿真依赖）
# 推荐用 extra 一次性安装：
pip install -e ".[libero]"

# 7) 固定 numpy 版本（如你的环境需要）
pip install numpy==1.26.0 # 使用atc工具时推荐使用该版本numpy以避免潜在兼容性问题
```

#### 本地补丁应用

在开始 ONNX 导出前，建议先把 LeRobot 适配文件同步到本地仓，并对当前环境中的 `transformers` 应用补丁。

```bash
# 1) 将 Part1,2 适配代码复制到 LeRobot 源码目录
cd pi05/infer_with_om/lerobot_modified
bash ./cp_to_lerobot.sh ./modeling_pi05_part1.py policies/pi05
bash ./cp_to_lerobot.sh ./modeling_pi05_part2.py policies/pi05
# 若提示未找到 lerobot 根目录，检查是否真正运行 pip install -e . 进行本地可编辑安装
# 支持第三个参数输入 lerobot 根目录保底: $bash $SCRIPT_NAME <源文件路径> <目标子目录> [lerobot根目录(可选)]

# 2) 给 transformers 打 Gemma 相关补丁
cd pi05/infer_with_om/lib_modified
bash ./apply_gemma_patch.sh

# 3) 给 transformers 打 ONNX symbolic(opset14) 相关补丁
bash ./apply_symbolic_opset14_patch.sh
# 针对 torch==2.7.1 中的 torch/onnx/symbolic_opset14.py 进行修改，若版本不同，可自行查看 patch 文件进行修改
```

---

### 步骤一：导出 ONNX (PyTorch -> ONNX)
在宿主机（CPU/GPU 环境）上，使用提供的 convert_pi05_part1.py 和 convert_pi05_part2.py 将模型导出为 ONNX 格式。此步骤会自动执行精度验证。

```bash
mkdir -p output/onnx_models/pi05
mkdir -p runtime_save
```

#### 请提前将测试用例数据 start_obs_0.pt 放入 input_data/ 目录中

#### 1. 转换 Part 1 (VLM)
##### 脚本内部会读取 input_data/start_obs_0.pt 作为 mock 数据，并将产生的中间张量存入 runtime_save/
```bash
python convert_pi05_part1.py \
    --pretrained-path /path/to/models/pi05-libero \
    --output-dir output/onnx_models/pi05 \
    --onnx-filename pi05-part1.onnx
```
#### 2. 转换 Part 2 (Action Expert)
##### 脚本会加载上一步在 runtime_save/ 中保存的特征张量，验证去噪网络单步推理
```bash
python convert_pi05_part2.py \
    --pretrained-path /path/to/models/pi05-libero \
    --output-dir output/onnx_models/pi05 \
    --onnx-filename pi05-part2.onnx
```

执行成功后，会在 output/onnx_models/pi05/ 目录下生成 pi05-part1.onnx 和 pi05-part2.onnx。终端将打印出两者的余弦相似度及最大差异，用以确认导出成功。

### 步骤二：使用 ATC 转换 OM (ONNX -> OM)
在 310P 昇腾 NPU 环境下，利用 ATC 工具将生成的 ONNX 模型转换为 Ascend 亲和的 OM 格式。

```bash
mkdir -p output/om_models/pi05
```

#### 1. 转换 Part 1
```bash
atc --model=output/onnx_models/pi05/pi05-part1.onnx \
    --framework=5 \
    --output=output/om_models/pi05/pi05-part1 \
    --soc_version=Ascend310P1 \
    --precision_mode_v2=origin
```
#### 2. 转换 Part 2
```bash
atc --model=output/onnx_models/pi05/pi05-part2.onnx \
    --framework=5 \
    --output=output/om_models/pi05/pi05-part2 \
    --soc_version=Ascend310P1 \
    --precision_mode_v2=origin
```
(注意：--soc_version 请根据 npu-smi info 查看到的芯片型号进行替换，如 Ascend310P1 或 Ascend310P3。若报错 `[ERROR] Execute model failed for acl.mdl.execute error 507011`，检查 `npu-smi info` 显示的信息是否与 ATC 转化时使用的 `--soc_version` 相同)
![npu-smi info](https://raw.gitcode.com/user-images/assets/7380116/91e63da3-7c42-4b70-b7d5-47fa358beeed/npu.jpg)

### 步骤三：端到端 ONNX 推理验证（CPU）
为确保整个去噪扩散 Pipeline 的计算逻辑无误，建议先使用 pi05_onnx_e2e.py 在 CPU 上跑通完整流程。

```bash
# 执行前，请确保脚本内的 ONNX_PART1_PATH 和 ONNX_PART2_PATH 路径正确指向您的 ONNX 文件
python pi05_onnx_e2e.py
```
该脚本会完整串联 Part 1 的特征提取和 Part 2 的 10 次去噪循环，并将最终的动作张量保存为 onnx_baseline_action.pt。

### 步骤四：昇腾 OM 端到端推理与 ONNX 结果比较
使用 NPU 执行最终的 OM 模型推理。由于代码依赖昇腾媒体处理引擎，需确保已正确安装及引入 acllite。

```bash
# 执行 OM 端到端推理测试
# 执行前，请确保 pi05_om_e2e.py 内的 paligemma_model_path 和 gemma_model_path 指向刚才生成的 OM 文件
python pi05_om_e2e.py
```

> 运行前请同步检查：
> - `pi05_om_e2e.py` 中的 `paligemma_model_path`、`gemma_model_path` 路径
> - `start_obs_0.pt` 的绝对路径是否存在
> - `PYTHONPATH` 是否能正确导入 `acllite_utils`

执行完毕后，终端将输出 Inference completed 及最终动作的 Shape 信息，并将结果保存至 om_baseline_action.pt。您可以通过`compare_action.py`比较 NPU 产出的 .pt 文件与 ONNX CPU 产出的 .pt 文件来验证模型精度。

```bash
python compare_action.py  --file1 om_baseline_action.pt  --file2 onnx_baseline_action.pt
```

### 步骤五：昇腾 OM 推理 LIBERO 仿真测试
```bash
cd pi05/infer_with_om/lerobot_modified
python lerobot_eval_om.py \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.batch_size=1 \
    --eval.n_episodes=50 \
    --policy.path=/path/to/models/pi05-libero \
    --output_dir=/path/to/output_dir
```
---

### OM 推理性能
#### 1. Part 1/2 推理速度

| 模块 | 平均耗时 (ms) |
| :--- | ---: |
| PaliGemma (Part1) | 223.08 |
| Gemma (Part2) | 15.58 |
| End-to-End | 410.2 |
- Part1 为 VLM Part，每次推理仅允许一次
- Part2 为 Action Expert，每次推理按照 Config 中的 num_inference_steps 推理 10 次
- E2E 包含 Part1 + 10 * Part2 以及 KV-Cache拷贝等时间

#### 2. OM 推理正确性验证 (Action 相似度比较)
- Tips：使用 input_data/start_obs_0.pt 输入，比较 OM 和 Pytorch 推理动作

| 指标 | 数值 |
| :--- | ---: |
| 张量形状 | torch.Size([1, 50, 7]) |
| 平均余弦相似度 | 0.999984 |
| 最小余弦相似度 | 0.999850 |
| 最大余弦相似度 | 1.000000 |
| 余弦相似度标准差 | 0.000029 |

---

### 常见问题（FAQ）

1. `ModuleNotFoundError: acllite_*`
    - 说明 ACLLite 路径未加入 `PYTHONPATH`。

2. ATC 编译失败
    - 检查 `source ${cann_install_path}/ascend-toolkit/set_env.sh` 是否已执行。
    - 检查 `--soc_version` 与实际芯片型号是否一致。

3. 输入 shape 不匹配
    - 确保 `start_obs_0.pt` 中的 key 与脚本读取 key 完全一致。
    - 确保图像分辨率、`chunk_size`、动作维度与模型配置一致。

4. 运行lerobot/scripts/eval.py时，若使用网络环境下载`google/paligemma-3b-pt-224`模型，需提前取模型对应的 huggingface 页面请求访问 Access，参考详见`https://huggingface.co/docs/huggingface_hub/main/cn/quick-start`和`https://huggingface.co/docs/hub/models-gated`

5. 运行 `lerobot_eval_om.py` 时提示 `Failed to load library ('libOSMesa.so.0')`
    - 这是仿真环境依赖的 OSMesa 动态库缺失导致的。
    - 可以先执行 `ldconfig -p | grep OSMesa` 确认系统是否已经安装。
    - Ubuntu / Debian：`sudo apt-get update && sudo apt-get install -y libosmesa6`
    - CentOS / RHEL / openEuler / EulerOS：`sudo yum install -y mesa-libOSMesa`

```
@misc{intelligence2025pi05visionlanguageactionmodelopenworld,
      title={$\pi_{0.5}$: a Vision-Language-Action Model with Open-World Generalization}, 
      author={Physical Intelligence and Kevin Black and Noah Brown and James Darpinian and Karan Dhabalia and Danny Driess and Adnan Esmail and Michael Equi and Chelsea Finn and Niccolo Fusai and Manuel Y. Galliker and Dibya Ghosh and Lachy Groom and Karol Hausman and Brian Ichter and Szymon Jakubczak and Tim Jones and Liyiming Ke and Devin LeBlanc and Sergey Levine and Adrian Li-Bell and Mohith Mothukuri and Suraj Nair and Karl Pertsch and Allen Z. Ren and Lucy Xiaoyang Shi and Laura Smith and Jost Tobias Springenberg and Kyle Stachowicz and James Tanner and Quan Vuong and Homer Walke and Anna Walling and Haohuan Wang and Lili Yu and Ury Zhilinsky},
      year={2025},
      eprint={2504.16054},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.16054}, 
}
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```
### 附录

#### lerobot 根目录相关代码目录树

- 检查整体代码目录树，经过上述的操作，pi05适配昇腾的lerobot根目录中的最终相关代码目录树如下所示：

```plaintext
|-- pi05/infer_with_om
├── README.md                            # 本使用指南
├── convert_pi05_part1.py                # Part 1: PyTorch -> ONNX 转换
├── convert_pi05_part2.py                # Part 2: PyTorch -> ONNX 转换
├── pi05_onnx_e2e.py                     # ONNX 端到端 10 步去噪推理基准验证
├── pi05_om_e2e.py                       # OM 端到端 10 步去噪推理与 NPU 部署类
├── input_data/                          
│   └── start_obs_0.pt                   # 测试用例输入张量数据
├── lerobot_modified/                    # 对 LeRobot 源码的修改适配文件夹
│   ├── cp_to_lerobot.sh                 # 将本目录改动拷贝到 LeRobot 对应源码路径脚本
│   ├── lerobot_eval_om.py               # 基于 OM 推理后端的 Libero 评测入口示例
│   ├── pi05_e2e.py                      # OM 推理后端基础脚本 （lerobot-eval 头文件）
│   ├── modeling_pi05_part1.py           # Pi0.5 Part1（VLM）昇腾导出/推理适配版实现
│   └── modeling_pi05_part2.py           # Pi0.5 Part2（Action Expert）昇腾导出/推理适配版实现
├── lib_modified/                        # 第三方库补丁与打补丁脚本
│   ├── apply_gemma_patch.sh             # 根据下方 Symbolic Opset14 文件生成 Patch 补丁并自动运用到库
│   ├── apply_symbolic_opset14_patch.sh  # 根据下方 Gemma 文件生成 Patch 补丁并自动运用到库
│   ├── modeling_gemma.py                # Modified 完成的 Gemma 文件
│   ├── symbolic_opset14.py              # Modified 完成的 Symbolic Opset14 文件
│   ├── gemma.patch                      # Sh 脚本生成的 Gemma 兼容性补丁
│   └── symbolic_opset14.patch           # Sh 脚本生成的 ONNX opset14 symbolic 修复补丁
├── runtime_save/                        # 运行 Part 1 时保存的中间态张量
│   ├── past_kv_tensor.pth               
│   └── prefix_pad_masks.pth             
└── output/
    ├── onnx_models/pi05/                # 生成的 ONNX 存放路径
    │   ├── pi05-part1.onnx 
    │   ├── pi05-part1.onnx.data 
    │   ├── pi05-part2.onnx 
    │   └── pi05-part2.onnx.data                    
    └── om_models/pi05/                  # 生成的 OM 存放路径
        ├── pi05-part1.om 
        └── pi05-part2_linux_x86_64.om                      
```