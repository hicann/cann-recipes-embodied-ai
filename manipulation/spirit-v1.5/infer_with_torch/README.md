# Spirit-v1.5具身大模型适配昇腾310P

## Spirit-v1.5介绍

Spirit v1.5是由千寻智能自研的具身智能模型，在2026.1.12的RoboChallenge评测中取得综合排名第一，在多项任务中保持较高成功率，尤其在多任务连续执行、复杂指令拆解以及跨构型迁移等维度中表现稳定。

本样例展示了如何在昇腾310P平台上适配Spirit v1.5模型并进行推理。

## 支持的产品型号

昇腾310P系列

## 代码与权重准备

1. 拉取本代码仓：`git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git`


2. 从千寻智能官方开源的代码仓拉取源码：`git clone https://github.com/Spirit-AI-Team/spirit-v1.5.git`


3. 将本代码仓内的下列文件拷贝至Spirit v1.5的代码仓（若有同名文件，则进行替换）

    | 文件在本代码仓内的位置| Spirit v1.5代码仓内的位置 |
    | ---- | ---- |
    | `manipulation/spirit-v1.5/infer_with_torch/pyproject.toml` | `pyproject.toml` |
    | `manipulation/spirit-v1.5/infer_with_torch/requirements.txt` | `requirements.txt` |
    | `manipulation/spirit-v1.5/infer_with_torch/modeling_spirit_vla.py` | `model/modeling_spirit_vla.py` |
    | `manipulation/spirit-v1.5/infer_with_torch/attention_processor_patch.py` | `model/attention_processor_patch.py` |
    | `manipulation/spirit-v1.5/infer_with_torch/infer_mozrobot_ascend.py` | `scripts/infer_mozrobot_ascend.py` |


4. 下载千寻智能提供的模型权重
    | Model | Type |
    |----------|-------------|
    | [Spirit-v1.5](https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5) | Base Model |
    | [Spirit-v1.5-move-objects-into-box](https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5-for-RoboChallenge-move-objects-into-box) | Fine-tuned Model |

5. 下载[Qwen/Qwen3VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)的权重（可选）


## 运行环境准备

### 安装CANN软件包

本样例的执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.3.RC1`。

请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1)下载`Ascend-cann-toolkit_8.3.RC1_linux-${arch}.run`与`Atlas-cann-kernels_310p_8.3.RC1_linux-${arch}.run`，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=netconda&OS=openEuler&Software=cannToolKit)进行安装。

- `${arch}`表示CPU架构，根据host机器的架构选择aarch64或x86_64。

### 配置Python环境

与千寻智能官方保持一致，使用`uv`工具进行环境管理，首先检查本地是否已安装了uv，若没有，可以使用`pip install uv`进行安装。

进入Spirit v1.5的代码路径，执行`uv sync`，这个过程将自动解析各种包的依赖关系并进行安装。

在代码仓根目录下会生成`.venv`路径，执行`source .venv/bin/activate`以激活虚拟环境，此时可以执行`uv pip list`检查是否所有依赖包都已安装完成。

## 推理

按照Spirit v1.5代码仓内的README文档指导，执行推理脚本。
