# ACT 具身智能VLA大模型昇腾使用指南

本目录介绍在 310P 上如何对 ACT 模型进行离线模型转换及推理，附带精度验证及仿真步骤。

### ACT整体介绍

在 Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware 论文中提出了ACT: Action Chunking with Transformers 模型，仓库地址为：https://github.com/tonyzhaozh/act

#### 功能介绍

ACT 是一种典型的基于 Transformer 的模仿学习控制策略，其核心思想是以“动作分块（Action Chunking）”的方式一次性预测一段连续动作序列，从而有效缓解逐步预测带来的误差累积问题，并在多个双臂精细操作任务中验证了其有效性。

### ACT的相关代码仓拉取、仿真测试集和模型下载

本样例使用的示例模型为：https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human/tree/main

该模型是一个在ALOHA仿真环境中经过微调的模型。

#### 模型输入输出

| **输入数据名**                        | **数据类型(dtype)** | **数据大小(shape)** |
| ------------------------------------- | ------------------- | ------------------- |
| observation.images_xxx （摄像头图像） | Float32             | $[1, N_c, 3, H, W]$ |
| observation.state （机器人位姿）      | Float32             | $[1, N_k]$          |

| **输出数据名** | **数据大小(shape)**     |
| -------------- | ----------------------- |
| actions        | $[1, chunk\_size, N_k]$ |

**参数符号说明：**

- $N_c$ ：摄像头个数（图片个数），训练时决定，通常为一路摄像头～三路摄像头。
- $N_k$ ：自由度，机器人构型决定。
- $H, W$： 摄像头拍摄rgb图像分辨率。
- $chunk\_size$：机器人执行步数个数。

输入输出格式定义在模型文件中的config.json中，样例示例为一路摄像头，内容如下：

```
"input_features": {
    "observation.images.top": {
        "type": "VISUAL",
        "shape": [
            3,
            480,
            640
        ]
    },
    "observation.state": {
        "type": "STATE",
        "shape": [
            14
        ]
    }
},
"output_features": {
    "action": {
        "type": "ACTION",
        "shape": [
            14
        ]
    }
},
"chunk_size": 100,
```

### ACT在昇腾310P上的运行配置

#### 与昇腾平台相关的环境配置

.om模型转化及运行需要安装CANN软件包。

本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.0.0-8.2.RC1`。 请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1)下载对应架构软件包，例如` Ascend-cann-toolkit_8.2.RC1_linux-x86_64.run`与` Ascend-cann-kernels-310b_8.2.RC1_linux-x86_64.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)依次进行安装。

```
# ${cann_install_path}为CANN包的实际安装目录，注意每次新建终端时，首先source一下set_env.sh。
# 方式1：默认路径安装，以root用户为例
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 方式2：指定路径进行安装
source ${cann_install_path}/ascend-toolkit/set_env.sh
```

#### 与昇腾服务器无关的环境配置

```bash
# 1) 拉取 lerobot 仓库并切到指定 commit
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout d9e74a9d374a8f26582ad326c699740a227b483c

# 2) 创建运行环境
conda create -y -n lerobot python=3.10
conda activate lerobot

# 3) 安装 lerobot（本地可编辑安装）
pip install -e .

# 4) 仿真依赖（Aloha 仿真需要 gym_aloha / gym-aloha）
# 推荐用 extra 一次性安装：
pip install -e ".[aloha]"

# 5) 固定 numpy 版本（如你的环境需要）
pip install numpy==1.26.0
```

##### 仿真渲染（MuJoCo）无头模式

如果服务器/容器缺少显示环境或 OpenGL 渲染后端，MuJoCo 可能无法正常渲染。
可以在运行仿真/评测前指定 EGL 无头渲染：

```bash
export MUJOCO_GL=egl
```

#### ACT在昇腾310P上的推理步骤

本节介绍离线推理模式（通过昇腾亲和的OM文件）的部署参考，更多使用参数可参考[atc工具使用文档](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/devaids/atctool/atlasatc_16_0003.html)。

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

在 Host（CPU 或 GPU）上执行：

```bash
# 以本地目录为例：act_model/ 里包含 config.json 等文件
# 也可以先用 huggingface-cli 下载到 act_model/:
#   pip install -U huggingface_hub
#   huggingface-cli download lerobot/act_aloha_sim_transfer_cube_human --local-dir act_model

python3 act/convert_and_verify_onnx.py \
    --pretrained-policy-path act_model \
    --output outputs/onnx/act.onnx \
    --device cpu \
    --opset 14
```

说明：
- 输入 schema（包含多个摄像头 key 时也支持）严格来自 `config.json.input_features`。
- 默认会用 ONNXRuntime CPU 对比 PyTorch 输出，打印 max/mean abs diff；如需跳过可加 `--no-validate`。

样例输出如下：
```
INFO: ONNX export finished
INFO: Validating ONNX output vs PyTorch...
INFO: ONNXRuntime validation uses CPUExecutionProvider; diff computed on CPU
INFO: ONNX inference time: 0.084086 sec
INFO: PyTorch output shape: (1, 100, 14)
INFO: ONNX output shape: (1, 100, 14)
INFO: Max abs diff: 0.000371844
INFO: Mean abs diff: 0.000106432
```

#### 2) ATC 将 ONNX 转为 OM

在 310P 上（已安装并 `source` CANN 环境）执行 ATC：

```bash
# 推荐直接用 atc（路径按你的实际文件位置修改）
atc --model=outputs/onnx/act.onnx \
        --framework=5 \
        --output=outputs/om/act \
        --soc_version=Ascend310P1
```

当模型转换完成后，当前目录应当存在一个名为 `act.om` 的模型（或者 `--output` 参数指定目录下）,在终端中有输出“ATC run success, welcome to the next use”。

#### 3) 使用 OM-backend sim-evaluator 进行仿真评测

在 310P 上执行（需要 ACL/ACLLite Python 依赖可用）。
可参考：[ACLLite安装教程](https://gitee.com/ascend/ACLLite#%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B)

```bash
python3 eval_act_ascend.py \
    --policy.path=act_model \
    --om_model_path=outputs/om/act.om \
    --env.type=aloha \
    --env.task=AlohaTransferCube-v0 \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --env.episode_length=600 \
    --seed=4412
```

输出：
- 评测结果会写入 `outputs/eval/.../eval_info.json`
- 日志会打印 `Aggregated eval metrics`（包含 success_rate 等聚合指标）

#### ACT在昇腾上的精度验证步骤

接下来介绍两种验证转换后的.om模型在NPU上运行验证方法。

##### 基于mock的数据输入，CPU/GPU与原始Pytorch输出相似度对比

构造输入数据测试Pytorch CPU/GPU和OM 310P NPU的输出余弦相似度对比：

```bash
# 在 310P 上执行（需要 ACL/AclLite Python 依赖）
python3 act/verify_om_onnx.py \
    --pretrained-policy-path act_model \
    --onnx-model-path outputs/onnx/act.onnx \
    --om-model-path outputs/om/act.om \
    --seed 42
```
该脚本会根据 config.json 生成确定性的 dummy 输入（支持多摄像头输入），并对比 ONNXRuntime(CPU) vs OM(NPU)

##### 基于Aloha仿真模拟器的功能测试示例

使用Aloha数据集在NPU进行推理，在Host CPU上进行仿真渲染:

```bash
python3 eval_act_ascend.py \
    --policy.path=act_model \
    --om_model_path=outputs/om/act.om \
    --env.type=aloha \
    --env.task=AlohaTransferCube-v0 \
    --eval.batch_size=1 \
    --eval.n_episodes=1 \
    --env.episode_length=600 \
    --seed=4412
```

示例效果：
![act sim-demo](https://raw.gitcode.com/user-images/assets/7380116/82c69770-5005-4eec-ac6c-66cb07a2bb29/act.gif)


##### 可能遇到的问题

关于 policy_preprocessor / policy_postprocessor 的归一化参数丢失。ACT 在训练时通常对 observation / action 做了归一化（normalization）。为了让 **OM 输出的 action** 能正确落回环境动作空间，并且 **OM 输入的 observation** 与训练时一致，评测需要两份处理器产物：

- `policy_preprocessor.json` + 对应的 `*.safetensors`：负责推理前对 observation 做与训练一致的预处理/归一化
- `policy_postprocessor.json` + 对应的 `*.safetensors`：负责把模型输出 action 做反归一化（unnormalize），再送入环境 step

获取方式有两种：

1) **模型目录里已经自带**（部分模型仓库支持）：直接检查 `act_model/` 是否包含上述两个 json 文件及其 safetensors。

2) **用迁移脚本生成**：如果你下载的模型只有 `model.safetensors + config.json`，没有上述 processor 文件，先在 Host 侧运行一次迁移脚本生成 `<模型目录>_migrated/`：

```bash
# 在lerobot根目录下执行：
python3 src/lerobot/processor/migrate_policy_normalization.py \
    --pretrained-path act_model
```

默认会生成如下内容：

```text
act_model_migrated/
    ├── policy_preprocessor.json
    ├── policy_postprocessor.json
    ├── policy_preprocessor_step_*.safetensors
    ├── policy_postprocessor_step_*.safetensors
    └── ...
```

说明：
- [eval_act_ascend.py](eval_act_ascend.py) 会优先在 `--policy.path` 指向的目录查找 processors；如果没有，会自动尝试同级的 `<policy_dir>_migrated/`。
- 也可以直接把 `--policy.path` 指向 `act_model_migrated/`（前提是该目录包含 `config.json`）。


### Citation

```
@misc{zhao2023learningfinegrainedbimanualmanipulation,
      title={Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
      author={Tony Z. Zhao and Vikash Kumar and Sergey Levine and Chelsea Finn},
      year={2023},
      eprint={2304.13705},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2304.13705},
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

- 检查整体代码目录树，经过上述的操作，ACT适配昇腾的lerobot根目录中的最终相关代码目录树如下所示：

```text
├── readme.md                           # 本文件
├── act_model                           # Huggingface或其他来源下载的模型
├── act/
|   ├── eval_act_ascend.py              # Ascend OM-backend sim-evaluator
|   ├── convert_and_verify_onnx.py      # PyTorch -> ONNX 转化脚本
|   ├── verify_om_onnx.py               # ONNXRuntime(CPU) vs OM(NPU) 误差对比
|   ├── convert_om.sh                   # ATC ONNX -> OM 示例脚本
|   ├── convert_utils.py                # PyTorch -> ONNX 转化辅助函数
|   ├── policy_input_schema.py          # 自动获取并读取模型文件config.json
|   └── figure/                         # README所用示意图
└── outputs/                            # 转换产生文件
    ├── onnx/
    ├── om/
    └── eval/
```
