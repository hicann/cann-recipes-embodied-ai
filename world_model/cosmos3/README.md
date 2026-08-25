# Cosmos3 昇腾 NPU 适配使用指南

<br>

## Cosmos3 整体介绍

### 功能介绍

Cosmos3是一个世界基础模型（World Foundation Models）框架，面向物理 AI、机器人、自动驾驶、视频生成与多模态理解等场景。Cosmos3-Nano 支持文生视频（T2V）、图生视频（I2V）、视频生视频（V2V）等推理任务，并可结合结构化输入完成世界模型生成与理解。本样例基于 Cosmos3 框架完成昇腾 NPU 适配，提供依赖配置、设备适配脚本、FIA attention 后端、本地权重路径适配和基础推理验证命令，帮助用户在昇腾 A3 环境上运行 Cosmos3-Nano 推理。

<br>

## 代码仓拉取与适配文件覆盖

本样例基于 Cosmos3 框架进行昇腾 NPU 适配。使用时先拉取 CANN 适配仓和 原始代码仓，再将 `world_model/cosmos3` 下的适配文件覆盖到 Cosmos3 仓根目录。

```bash
# 进入需要放置代码仓的本地目录，建议让 cann-recipes-embodied-ai 与 cosmos-framework 保持同级
git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git

git clone https://github.com/NVIDIA/cosmos-framework.git && cd cosmos-framework && git checkout a61b292

# 回到两个代码仓的共同上级目录
cd ../

# 将 CANN 仓中的 Cosmos3 适配文件覆盖到 Cosmos3 仓根目录
cp -rf cann-recipes-embodied-ai/world_model/cosmos3/* ./cosmos-framework
```

完成覆盖后，`cosmos-framework` 根目录下应包含 `npu_adapt.sh`、`pyproject.toml` 等适配文件。

## Cosmos3 在昇腾 A3 上的运行环境配置

### 与昇腾平台相关的环境配置

安装 CANN 软件包。本样例依赖 CANN 开发套件包（cann-toolkit）与 CANN 二进制算子包（cann-kernels），支持的 CANN 软件版本为 `CANN 9.0.0`，`torch_npu=2.10.0`,`python=3.13`。

请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0)下载 `Ascend-cann-toolkit_${version}_linux-${aarch}.run` 与 `Atlas-A3-cann-kernels_${version}_linux-${aarch}.run` 软件包，并参考 [CANN 安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit) 依次进行安装。

- `${version}` 表示 CANN 包版本号，如 9.0.0
- `${aarch}` 表示 CPU 架构，如 aarch64、x86_64

```bash
# ${cann_install_path} 为 CANN 包的实际安装目录，注意每次新建终端时，首先 source 一下 set_env.sh。
# 方式1：默认路径安装，以 root 用户为例
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 方式2：指定路径进行安装
source ${cann_install_path}/ascend-toolkit/set_env.sh
```

### uv 环境管理工具安装（可选，如果当前环境已经安装 uv，可以跳过）

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

### Python 运行环境安装

```bash
cd cosmos-framework
uv sync --python 3.13
```



## 模型权重下载

本样例使用 Cosmos3-Nano 权重进行推理验证。请从 [nvidia/Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano/tree/main) 下载模型权重，并将推理命令中的 `COSMOS_CHECKPOINT` 指向本地权重目录。

此外，视频生成还需要 Wan2.2 VAE 权重。请从 [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/tree/main) 下载 `Wan2.2_VAE.pth`，只需要该 VAE 权重文件，无需下载完整 Wan2.2-TI2V-5B 模型，并将其放置到 Cosmos3-Nano 本地权重目录下。

```bash
# 示例：将 Cosmos3-Nano 权重放置在本地目录 /mnt/workspace/cosmos3/cosmos3-nano
export COSMOS_CHECKPOINT=/mnt/workspace/cosmos3/cosmos3-nano

# Wan2.2 VAE 权重应放置为如下路径
ls ${COSMOS_CHECKPOINT}/Wan2.2_VAE.pth
```

如果部署环境无法直接访问 Hugging Face，可在可联网环境下载完整 Cosmos3-Nano 权重目录和 `Wan2.2_VAE.pth` 后拷贝到昇腾服务器，再使用本地路径作为 `--checkpoint-path`。

## 执行 NPU 适配脚本

```bash
cd cosmos-framework

bash npu_adapt.sh
```

## 推理验证示例

完成适配后，可在 `cosmos-framework` 根目录下执行以下命令进行基础场景验证。`COSMOS_CHECKPOINT` 用于指定本地权重目录；如不设置，可直接将命令中的 `--checkpoint-path` 替换为实际权重路径。

```bash
export COSMOS_CHECKPOINT=/mnt/workspace/cosmos3/cosmos3-nano
export COSMOS_RESOLUTION=480
export COSMOS_SEED=0
export COSMOS_NPUS=1
```

### 输入 JSON 配置说明

推理命令中的 `-i inputs/omni/t2v.json` 用于指定单条样例输入。常用字段如下：

- `model_mode`：任务类型，例如 `text2video`、`image2video`、`video2video`。
- `prompt`：文本提示词，T2V 只需要配置该字段即可。
- `vision_path`：I2V/V2V 的输入图片或视频路径，仅图生视频、视频生视频需要。

`vision_path` 可以写远程 URL，也可以写本地文件路径。若服务器无法访问 GitHub/Hugging Face，或遇到证书、代理、内网限制等网络问题，请先手动下载输入图片/视频到本地，然后在 JSON 中改成本地绝对路径，例如：

```json
{
  "model_mode": "image2video",
  "prompt": "A robot arm moves smoothly in a lab.",
  "vision_path": "/mnt/workspace/cosmos3/inputs/robot_153.jpg"
}
```

T2V 示例 JSON 可简化为：

```json
{
  "model_mode": "text2video",
  "name": "t2v",
  "prompt": "A realistic video of molten metal being poured in a steel mill."
}
```


### T2V 文生视频

```bash
torchrun --nproc-per-node=${COSMOS_NPUS} -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i inputs/omni/t2v.json \
    -o outputs/t2v \
    --checkpoint-path ${COSMOS_CHECKPOINT} \
    --resolution=${COSMOS_RESOLUTION} \
    --seed=${COSMOS_SEED} \
    --no-guardrails
```

### I2V 图生视频

```bash
torchrun --nproc-per-node=${COSMOS_NPUS} -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i inputs/omni/i2v.json \
    -o outputs/i2v \
    --checkpoint-path ${COSMOS_CHECKPOINT} \
    --resolution=${COSMOS_RESOLUTION} \
    --seed=${COSMOS_SEED} \
    --no-guardrails
```

### V2V 视频生视频

```bash
torchrun --nproc-per-node=${COSMOS_NPUS} -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i inputs/omni/v2v.json \
    -o outputs/v2v \
    --checkpoint-path ${COSMOS_CHECKPOINT} \
    --resolution=${COSMOS_RESOLUTION} \
    --seed=${COSMOS_SEED} \
    --no-guardrails
```

### 多卡并行

当前适配支持 CP、CFGP 和 FSDP 多卡推理。运行前请将 `COSMOS_NPUS` 设置为实际使用的 NPU 数量，并保证各并行度与进程数匹配。

| 并行方式 | 主要参数 | 适用目的与约束 |
| --- | --- | --- |
| CP（Context Parallel） | `--cp-size` | 沿 token 序列切分 Attention 计算，适合长序列并降低激活显存；当前 CP 范围为 1～32 |
| CFGP（Classifier-Free Guidance Parallel） | `--cfgp-size` | 将有条件与无条件 CFG 分支分配到不同设备；CFGP 仅支持 1 或 2，更大规模可与 CP/FSDP 组合 |
| FSDP | `--dp-shard-size` | 按进程数切分模型参数，优先降低单卡权重显存 |

FSDP 的 DP 通信组与 CP/CFGP 通信组相互独立，通信域大小满足：

```text
dp-shard-size × dp-replicate-size = WORLD_SIZE
WORLD_SIZE % (cp-size × cfgp-size) = 0
```


CP：

```bash
torchrun --nproc-per-node=${COSMOS_NPUS} -m cosmos_framework.scripts.inference \
    --parallelism-preset=throughput \
    --dp-shard-size=1 --cp-size=${COSMOS_NPUS} --cfgp-size=1 \
    -i inputs/omni/t2v.json \
    -o outputs/t2v_cp \
    --checkpoint-path ${COSMOS_CHECKPOINT} \
    --resolution=${COSMOS_RESOLUTION} \
    --seed=${COSMOS_SEED} \
    --no-guardrails
```

CFGP（单独启用时设置 `COSMOS_NPUS=2`）：

```bash
torchrun --nproc-per-node=${COSMOS_NPUS} -m cosmos_framework.scripts.inference \
    --parallelism-preset=throughput \
    --dp-shard-size=1 --cp-size=1 --cfgp-size=2 \
    -i inputs/omni/t2v.json \
    -o outputs/t2v_cfgp \
    --checkpoint-path ${COSMOS_CHECKPOINT} \
    --resolution=${COSMOS_RESOLUTION} \
    --seed=${COSMOS_SEED} \
    --no-guardrails
```

FSDP：

```bash
torchrun --nproc-per-node=${COSMOS_NPUS} -m cosmos_framework.scripts.inference \
    --parallelism-preset=throughput \
    --dp-shard-size=${COSMOS_NPUS} --cp-size=1 --cfgp-size=1 \
    -i inputs/omni/t2v.json \
    -o outputs/t2v_fsdp \
    --checkpoint-path ${COSMOS_CHECKPOINT} \
    --resolution=${COSMOS_RESOLUTION} \
    --seed=${COSMOS_SEED} \
    --no-guardrails
```


## 样例输出展示

以下为在昇腾 NPU 上运行上述基础场景得到的示例输出，可用于快速查看生成效果。

| 场景 | 示例输出 |
| --- | --- |
| T2V 文生视频 | ![t2v_vision.gif](https://raw.gitcode.com/user-images/assets/10199195/600da397-f263-4e82-96e9-4589de9ebc2a/t2v_vision.gif) |
| I2V 图生视频 | ![i2v_vision.gif](https://raw.gitcode.com/user-images/assets/10199195/a493e495-4dc5-4a83-9517-5d34f9d7e757/i2v_vision.gif) |
| V2V 视频生视频 | ![v2v_vision.gif](https://raw.gitcode.com/user-images/assets/10199195/bf6f811d-a96d-4511-af23-0a6ca65380a2/v2v_vision.gif) |

# citation

```bash
## Citation
@misc{nvidia2026cosmos3omnimodalworldmodels,
      title={Cosmos 3: Omnimodal World Models for Physical AI},
      author={NVIDIA: Aditi, Niket Agarwal, Arslan Ali, Jon Allen, Martin Antolini, Adeline Aubame, Alisson Azzolini, Junjie Bai, Maciej Bala, Yogesh Balaji, Josh Bapst, Aarti Basant, Mukesh Beladiya, Mohammad Qazim Bhat, Zaid Pervaiz Bhat, Dan Blick, Vanni Brighella, Han Cai, Tiffany Cai, Eric Cameracci, Jiaxin Cao, Yulong Cao, Mark Carlson, Carlos Casanova, Ting-Yun Chang, Yan Chang, Yu-Wei Chao, Prithvijit Chattopadhyay, Roshan Chaudhari, Chieh-Yun Chen, Junyu Chen, Ke Chen, Qizhi Chen, Wenkai Chen, Xiaotong Chen, Yu Chen, An-Chieh Cheng, Click Cheng, Xiu Chia, Jeana Choi, Chaeyeon Chung, Wenyan Cong, Yin Cui, Magdalena Dadela, Nalin Dadhich, Wenliang Dai, Joyjit Daw, Alperen Degirmenci, Rodrigo Vieira Del Monte, Robert Denomme, Sameer Dharur, Marco Di Lucca, Ke Ding, Wenhao Ding, Yifan Ding, Yuzhu Dong, Nicole Drumheller, Yilun Du, Aigul Dzhumamuratova, Aleksandr Efitorov, Hamid Eghbalzadeh, Naomi Eigbe, Imad El Hanafi, Hassan Eslami, Benedikt Falk, Jiaojiao Fan, Jim Fan, Amol Fasale, Sergiy Fefilatyev, Liang Feng, Francesco Ferroni, Sanja Fidler, Xiao Fu, Vikram Fugro, Prashant Gaikwad, TJ Galda, Katelyn Gao, Yihuai Gao, Wenhang Ge, Sreyan Ghosh, Arushi Goel, Vivek Goel, Akash Gokul, Rama Govindaraju, Jinwei Gu, Miguel Guerrero, Elfie Guo, Aryaman Gupta, Siddharth Gururani, Hugo Hadfield, Song Han, Ankur Handa, Zekun Hao, Mohammad Harrim, Ali Hassani, Nathan Hayes-Roth, Yufan He, Chris Helvig, Cyrus Hogg et al. (195 additional authors not shown)},
      year={2026},
      eprint={2606.02800},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.02800},
}
```
