# 在昇腾Atlas A2/A3环境上适配VGGT模型的推理
本样例基于[VGGT开源模型](https://github.com/facebookresearch/vggt)完成其在NPU上的推理适配，并提供其在相机位姿估计、点云重建、深度估计三个任务上的精度评测脚本。详细内容可至[精度评测章节](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/docs/3d_vision/vggt/vggt_accurancy_evaluation.md)查看。

此外，本样例基于VGGT模型在NPU进行了性能优化，目前VGGT模型在25张图片输入下，推理时间下降至1.12秒。详细内容可至[性能优化章节](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/docs/3d_vision/vggt/vggt_optimization.md)查看。

本样例支持昇腾Atlas A2/A3环境的单机单卡推理与单机多卡序列并行推理。
> 使用一站式平台的用户可直接跳转 [「一站式平台的快速启动」](#一站式平台的快速启动)章节。
---
## 执行样例
### CANN环境准备
1. 本样例的执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），目前使用CANN软件版本为`CANN.8.5.0`。
请从[CANN软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-${chip_type}-ops_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/quickstart/instg_quick.html)进行安装。

2. 本样例依赖的torch以及torch_npu版本为2.7.1。
请从[Ascend Extension for PyTorch插件](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)下载torch与torch_npu安装包，本样例依赖的torch与torch_npu版本分别为2.7.1和2.7.1.post2。
    ```shell
    conda create -n vggt python==3.11.13
    conda activate vggt
    pip3 install torch==2.7.1
    pip3 install torch_npu==2.7.1.post2
    ```
### 网络模型代码准备
- 本仓库依赖[VGGT](https://github.com/facebookresearch/vggt/tree/main)的开源仓库代码。
- 进入VGGT的官方仓库，下载VGGT模型网络结构代码：
  ```shell
  git clone https://github.com/facebookresearch/vggt.git
  ```
- 下载本仓库代码：
  ```shell
  git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git
  ```
- VGGT 模型权重下载：[VGGT model checkpoint](https://huggingface.co/spaces/facebook/vggt)，并将权重文件`model.pt`复制到ckpt目录下：
  ```shell
  pip install -U huggingface_hub
  export HF_ENDPOINT=https://hf-mirror.com
  hf download facebook/VGGT-1B --local-dir vggt
  ```
- 将VGGT仓库的网络模型文件以**非覆盖模式**复制到本项目目录下：
  ```shell
  cp vggt/visual_util.py cann-recipes-embodied-ai/3d_vision/vggt/
  cp -r vggt/examples cann-recipes-embodied-ai/3d_vision/vggt/
  cp -rn vggt/vggt/dependency cann-recipes-embodied-ai/3d_vision/vggt/vggt/dependency
  cp -rn vggt/vggt/heads cann-recipes-embodied-ai/3d_vision/vggt/vggt/
  cp -rn vggt/vggt/layers cann-recipes-embodied-ai/3d_vision/vggt/vggt/
  cp -rn vggt/vggt/utils cann-recipes-embodied-ai/3d_vision/vggt/vggt/ 
  ```
- 安装Python依赖：
  ```shell
  cd cann-recipes-embodied-ai/3d_vision/vggt/
  pip3 install -r requirements.txt
  ```
- 模型权重与模型结构在文件目录中罗列如下：
  ```
  VGGT
    +--- examples
    +--- demo_infer.py
    +--- eval
    +--- ckpt
          +--- model.pt
    +--- config
    +--- quant
    +--- vggt
          +--- dependency
          +--- heads
          +--- layers
          +--- models
          +--- utils
          +--- sp
  ```

### 快速启动

本样例使用 YAML 配置文件管理参数，支持多配置场景切换，详细的参数说明和约束条件请参考 [YAML配置文件说明](config/README.md)。

执行脚本前，请参考[Ascend社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/quickstart/instg_quick.html)中的CANN安装软件教程，配置环境变量：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
```

#### **单卡/多卡推理**

- 启动方式一：Python 直接启动（仅单卡推理）

```bash
# 默认使用 single.yaml 配置（单卡推理）
python demo_infer.py

# 指定配置文件（单卡推理）
python demo_infer.py --config config/single.yaml
```

- 启动方式二：Shell 脚本启动（支持单卡和多卡）

```bash
# 单卡推理（默认使用 single.yaml）
bash infer_test.sh

# 多卡推理（2卡并行，内部调用torchrun）
bash infer_test.sh sp2.yaml

# 多卡推理（4卡并行）
bash infer_test.sh sp4.yaml

# 多卡推理（8卡并行）
bash infer_test.sh sp8.yaml
```

#### **int8量化模型推理**

需要先生成 int8 模型（当前实现中，只将 VGGT 模型中 K=4096 的 Linear 层进行了 8bit 量化）：

**步骤1：生成 int8 量化模型**

修改build配置为true以构建量化模型：

```yaml
optimization:
  quantization:
    int8-w8a8:
      enable: false
      build: true
```

然后运行（yaml文件名需替换为实际配置文件）：

```bash
python demo_infer.py --config config/xxx.yaml
```

int8 模型会生成在当前路径（文件名：`VGGT_model_W8A8.pt`）。

**步骤2：使用 int8 量化模型推理**

使用 `config/single_w8a8.yaml` 配置文件：

```bash
python demo_infer.py --config config/single_w8a8.yaml
```

## 一站式平台的快速启动

本章节面向使用一站式平台的用户，平台已预置完整的 CANN 环境，按以下步骤即可在单卡上完成 VGGT 的三维重建推理。

> 使用一站式平台的用户请选择A2/A3上的python3.11相关的实例进行创建。

### 修改文件中变量

修改 infer\_platform\_env\_prepare.sh 中 WORKSPACE\_DIR 变量指代的路径，如 `cann_recipes`

### 代码与权重准备

运行下列命令，一键拉起脚本，进行代码与权重的准备：

```bash
cd cann-recipes-embodied-ai/3d_vision/vggt
bash infer_platform_env_prepare.sh
```

### 推理脚本运行

推理 bf16 模型单卡运行：

```bash
python demo_infer.py --config config/single.yaml
```

***

## Citation

```bibtex
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```