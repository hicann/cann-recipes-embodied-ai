# pi0.5机器人VLA大模型昇腾使用指南
<br>


## pi0.5整体介绍
论文题目：π0.5: a Vision-Language-Action Model with Open-World Generalization

中文译文：π0.5: 一种具备开放世界泛化能力的视觉–语言–动作模型

### 功能介绍

pi0.5，一种基于pi0的新模型；它通过在异构任务上进行协同训练（co-training），实现更广泛的泛化能力。pi0.5利用来自多种机器人平台的数据、高层语义预测、网络数据以及其他来源，使其能够在真实世界的机器人操作任务中实现更强的通用性。它结合了协同训练与混合多模态样例：这些样例将图像观测、语言指令、目标检测、语义子任务预测以及底层动作整合在一起，通过知识迁移实现有效泛化。pi0.5首次展示：一个端到端、由学习驱动的机器人系统，能够在全新的家庭环境中执行长时序且高灵巧度的操作技能，例如在完全陌生的住宅里完成厨房或卧室清洁等任务。

<br>


## pi0.5的相关代码仓拉取、数据集和模型下载
```bash
# 进入需要放置代码仓的本地xxx目录下：
cd xxx
git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git
chmod +x cann-recipes-embodied-ai/manipulation/pi05/infer_with_torch/download_code_and_data.sh
./cann-recipes-embodied-ai/manipulation/pi05/infer_with_torch/download_code_and_data.sh
```
完成上述操作之后，最终lerobot根目录中相关代码目录树详见[附录：lerobot根目录相关代码目录树](#lerobot根目录相关代码目录树)。

<br>


## pi0.5在昇腾上的运行环境配置
### 与昇腾服务器无关的环境配置
```bash
# 创建运行环境
conda create -y -n lerobot python=3.10
conda activate lerobot

# 回到lerobot根目录，安装lerobot。
cd lerobot
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
pip install "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
pip install numpy==1.26.4
```

### 与昇腾平台相关的环境配置
1. 安装CANN软件包。

  本样例的编译执行依赖CANN开发套件包（toolkit）与CANN二进制算子包（kernels/ops）。

  310P请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1)下载`Ascend-cann-toolkit_8.2.RC1_linux-${arch}.run`与`Ascend-cann-kernels-310p_8.2.RC1_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)依次进行安装。

  A2请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0)下载`Ascend-cann-toolkit_9.0.0_linux-${arch}.run`与`Ascend-cann-910b-ops_9.0.0_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/90RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)依次进行安装。

  - `${arch}`表示服务器CPU架构，ARM架构对应`aarch64`，x86架构对应`x86_64`。可通过`uname -m`查询当前服务器的CPU架构，并下载与之匹配的软件包。

  ```bash
  # ${cann_install_path}为CANN包的实际安装目录，注意每次新建终端时，首先source一下set_env.sh。
  # 方式1：默认路径安装，以root用户为例
  source /usr/local/Ascend/ascend-toolkit/set_env.sh

  # 方式2：指定路径进行安装
  source ${cann_install_path}/ascend-toolkit/set_env.sh
  ```
2. 安装Ascend Extension for PyTorch（torch_npu）。

  Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件，本样例支持的Ascend Extension for PyTorch版本为`7.1.0`，PyTorch版本为`2.5.1`。
  请根据服务器CPU架构下载对应的[x86_64安装包](https://gitee.com/ascend/pytorch/releases/download/v7.1.0-pytorch2.5.1/torch_npu-2.5.1.post1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl)或[aarch64安装包](https://gitee.com/ascend/pytorch/releases/download/v7.1.0-pytorch2.5.1/torch_npu-2.5.1.post1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl)，参考[torch_npu安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0004.html)进行安装。
<br>


### pi0.5在昇腾上的推理步骤
运行下面的代码，即可自动构造mock输入，进行pi05模型推理，打印推理性能及机器人动作。
```bash
# 进入lerobot代码仓根目录
cd lerobot
conda activate lerobot
chmod +x run_pi05_inference.sh
./run_pi05_inference.sh pi05_model float16 1 3 npu
```

基于上述运行过程，得到pi05的单次推理时间及结果如下所示：
- 推理性能：单次推理时间约860ms，示例输出如下：
```
INFO - Starting inference timing (3 iterations)...
INFO - ----------------------------------------
INFO - Inference Results for pi05_model
INFO - Device: npu:0, Dtype: torch.float16
INFO - Action shape: torch.Size([1, 32])
INFO - Total time for 3 runs: 2.5864 s
INFO - Average latency: 862.1430 ms
INFO - Throughput: 1.16 FPS
INFO - ----------------------------------------
```
- 推理结果：单次推理结果为50组机械臂关节角度序列，shape为[50,32]，每次推理后保存在queue中，action输出一组。
<br>


## pi05在昇腾上的精度验证步骤
### 基于mock的数据输入，NPU与原始CPU/GPU Pytorch输出相似度对比
构造固定输入（如全0图像 + 固定指令 token），测试 PyTorch CPU/GPU 和 310P NPU 的输出精度对比：
```bash
python verify_pi05_accuracy_ascend.py \
    --pretrained_model_name_or_path pi05_model \
    --device npu:0
```
示例输出如下
```
Global Cosine Similarity: 1.000000
Per-timestep Cosine Similarity:
  Step 0: 1.000000
  ...
  Step 49: 0.999999
  Minimum Per-step Similarity: 0.999999
  Average Per-step Similarity: 0.999999
MSE Loss: 0.000000
Verification SUCCESS: All similarities > 0.99
```
<br>

## 可能遇到的问题
1. 运行推理时，若使用网络环境下载`google/paligemma-3b-pt-224`模型，需提前取模型对应的 huggingface 页面请求访问 Access，参考详见`https://huggingface.co/docs/huggingface_hub/main/cn/quick-start`和`https://huggingface.co/docs/hub/models-gated` 
2. 若网络环境下载huggingface模型较慢，遇到下载`google/paligemma-3b-pt-224`卡顿，可手动下载模型到本地路径，再修改`lerobot/src/lerobot/policies/pi05/processor_pi05.py`中对应145行处：`google/paligemma-3b-pt-224`为本地路径。

## Citation
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
```

<br>


## 附录
### lerobot根目录相关代码目录树
- 检查整体代码目录树，经过上述的复制及替换操作，pi05适配昇腾的lerobot根目录中的最终相关代码目录树如下所示：
```bash
├── src                                       # pi05模型训练及推理框架
|   ├── lerobot
|   |   ├── policies
|   |   |   ├── pi05
|   |   |   |   ├── modeling_pi05.py          # pi05的模型训练及推理代码
├── pi05_model                                # pi05 base模型
└── pyproject.toml                            # 运行环境第三方包的安装版本
└── README.md                                 # 昇腾上运行pi05推理的环境配置及操作指导
└── run_pi05_inference.sh                     # 昇腾上运行pi05推理过程一键启动脚本
└── run_pi05_example.py                       # 昇腾上运行pi05推理示例代码
└── verify_pi05_accuracy_ascend.py            # 昇腾上运行pi05推理结果精度验证代码
└── infer_utils.py                             # 推理与验证脚本共用工具函数
```
