# pi0机器人VLA大模型昇腾使用指南
<br>


## pi0整体介绍
论文题目：π0: A Vision-Language-Action Flow Model for General Robot Control

中文译文：π0: 一个用于通用机器人控制的视觉-语言-动作流匹配模型

### 功能介绍

pi0是一个视觉-语言-动作（VLA）模型，专为通用机器人控制而设计。它基于预训练的视觉语言模型（VLM），结合流匹配（flow matching）机制，能够生成高频连续动作，实现对复杂、灵巧机器人任务的精准控制。整合OXE开源数据集和自有数据集，总计超过10,000小时机器人操作数据。在叠衣服、桌面清理、装盒等复杂任务上表现优异，零样本和微调设置下均显著优于现有基线方法（OpenVLA、Octo、ACT等）。成功完成5-20分钟的长时序多阶段任务，展现出强大的鲁棒性和泛化能力。

<br>


## pi0的相关代码仓拉取、数据集和模型下载
```bash
# 进入需要放置代码仓的本地目录下，执行下面的指令：
git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git

chmod +x cann-recipes-embodied-ai/manipulation/pi0/infer_with_torch/download_code_and_data.sh

./cann-recipes-embodied-ai/manipulation/pi0/infer_with_torch/download_code_and_data.sh
```
完成上述操作之后，最终lerobot根目录中相关代码目录树详见[附录：lerobot根目录相关代码目录树](#lerobot根目录相关代码目录树)。

<br>


## pi0在昇腾A2上的运行环境配置
### 与昇腾服务器无关的环境配置
```bash
# 创建运行环境
conda create -y -n lerobot python=3.10
conda activate lerobot

# 回到lerobot根目录，安装lerobot。
cd lerobot
pip install -e .
```

### 与昇腾平台相关的环境配置
安装CANN软件包。本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.3.RC1`。
请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1)下载`Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run`与`Ascend-cann-kernels-910b_8.3.RC1_linux-aarch64.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)依次进行安装。

```bash
# ${cann_install_path}为CANN包的实际安装目录，注意每次新建终端时，首先source一下set_env.sh。
# 方式1：默认路径安装，以root用户为例
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 方式2：指定路径进行安装
source ${cann_install_path}/ascend-toolkit/set_env.sh

# 在上述运行环境中继续安装对应版本torch-npu
pip install torch-npu==2.1.0.post12
```

<br>


### pi0在昇腾上的推理步骤
运行下面的代码，即可自动加载koch机械臂数据集，进行pi0模型推理，打印推理性能及机器人动作。
```bash
# 进入lerobot代码仓根目录
cd lerobot
conda activate lerobot
chmod +x run_pi0_inference.sh
./run_pi0_inference.sh koch_test pi0_model 10 100
```

基于上述运行过程，得到pi0的单次推理时间及结果如下所示（详细的优化过程介绍见 [pi0 优化说明文档](../../../docs/manipulation/pi0/infer_with_torch/README.md)）：
- 推理性能：单次推理时间下降至80 ms，达到了预期的推理时间性能优化目标。
- 推理结果：单次推理结果为50组机械臂关节角度序列，shape为[50,6]。
<br>


## pi0在昇腾上的精度验证步骤
### 基于koch机械臂末端位姿的ATE(绝对误差)来验证昇腾的推理精度
- 为了能够以固定变量法进行昇腾平台的推理精度测试，需将pi0推理中action_expert中的高斯噪声采样部分进行固定噪声文件加载(即使用同样的高斯噪声采样数据)。
- 基于pi0模型推理得到的整段轨迹六关节角度序列(维度:50x6)，通过获取koch机械臂的物理DH参数，执行koch机械臂正运动学运算，得到koch机械臂末端执行器中心的实际位姿（位置x-y-z + 姿态r-p-y），然后通过ATE(absolute error)方法进行二范数计算，得到昇腾平台上koch机械臂末端位姿的误差参数，误差参考范围如下所示：
    - 位置ATE误差参考范围：[0, +0.03]m
    - 姿态ATE误差参考范围：[0, +0.2 ]rad

<br>


## gitcode云开发环境一站式平台的快速启动
本章节面向使用一站式平台的用户，平台已预置完整的 昇腾CANN 环境，按以下步骤即可在单卡上完成 pi0 的torch版本昇腾推理。
> 使用一站式平台的用户请选择A2/A3上的python3.12相关的实例进行创建。


### 代码与权重准备
```bash
# 进入需要放置代码仓的本地目录下，如/mnt/workspace/gitCode/cann，可根据个人情况进行修改：
cd /mnt/workspace/gitCode/cann

git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git

chmod +x cann-recipes-embodied-ai/manipulation/pi0/infer_with_torch/download_code_and_data.sh

./cann-recipes-embodied-ai/manipulation/pi0/infer_with_torch/download_code_and_data.sh
```

### 运行环境配置
```bash
# 创建运行环境
conda create -y -n lerobot python=3.10
conda activate lerobot

# 回到lerobot根目录，安装lerobot。
cd lerobot
pip install -e .

# 在lerobot运行环境中继续安装对应版本torch-npu
pip install torch-npu==2.1.0.post12
```

### 推理脚本运行
```bash
# 确保lerobot代码仓根目录下，并已激活lerobot这个conda环境
chmod +x run_pi0_inference.sh
./run_pi0_inference.sh koch_test pi0_model 10 100
```

<br>


## Citation
```
@misc{black2024pi0,
      title={$\pi$0: A Vision-Language-Action Flow Model for General Robot Control}, 
      author={Kevin Black and Noah Brown and Danny Driess and Adnan Esmail and Michael Equi and Chelsea Finn and Niccolo Fusai and Lachy Groom and Karol Hausman and Brian Ichter and Szymon Jakubczak and Tim Jones and Liyiming Ke and Sergey Levine and Adrian Li-Bell and Mohith Mothukuri and Suraj Nair and Karl Pertsch and Lucy Xiaoyang Shi and James Tanner and Quan Vuong and Anna Walling and Haohuan Wang and Ury Zhilinsky},
      year={2024},
      eprint={2410.24164},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2410.24164}
}
```

<br>


## 附录
### lerobot根目录相关代码目录树
- 检查整体代码目录树，经过上述的复制及替换操作，pi0适配昇腾的lerobot根目录中的最终相关代码目录树如下所示：
```bash
├── koch_test                                 # koch机械臂抓取任务数据集,符合lerobot数据集格式
├── lerobot                                   # pi0模型训练及推理框架
|   ├── common
|   |   ├── policies
|   |   |   ├── pi0
|   |   |   |   ├── modeling_pi0.py           # pi0的模型训练及推理代码
|   |   |   |   ├── paligemma_with_expert.py  # pi0的模型训练及推理代码
├── pi0_model                                 # koch机械臂抓取任务预训练pi0模型
├── pyproject.toml                            # 运行环境第三方包的安装版本
├── README_en.md                              # 昇腾上运行pi0推理的环境配置及操作指导 (英文版)
├── README.md                                 # 昇腾上运行pi0推理的环境配置及操作指导 (中文版)
├── run_pi0_inference.sh                      # 昇腾上运行pi0推理过程一键启动脚本
└── test_pi0_on_ascend.py                     # 昇腾上运行pi0推理主代码
```