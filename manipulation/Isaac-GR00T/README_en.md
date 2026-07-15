# Overview

GR00T N1.6 is a general robot foundation model released in December 2025. It addresses generalization bottlenecks of conventional VLA models in long-horizon embodied manipulation tasks and resolves temporal inconsistency issues of action generation under few-shot scenarios. By upgrading the VLM module and action prediction paradigm, it delivers capability improvements from short-horizon static tabletop manipulation to dynamic long-duration embodied tasks, and lowers the engineering deployment barriers for general humanoid robots in real-world environments.

# Supported Hardware

Atlas A3 Series

# Environment Preparation

1. Install CANN Environment

    Compilation and execution of this sample depend on the CANN development toolkit (cann-toolkit) and CANN binary operator package (cann-kernels), and the supported CANN version is `CANN 8.3.RC1`. Download the following packages from [the official software repository](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1.alpha002) `Ascend-cann-toolkit_${version}_linux-${arch}.run` and `Atlas-A3-cann-kernels_${version}_linux-${arch}.run`. Reference the official [CANN installation documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit) for deployment steps.
   
   * `${version}`: CANN version identifier, e.g., 8.3.RC1.
   * `${arch}`: CPU architecture, e.g.,aarch64, x86_64.

2. Clone Code Repositories
   GR00T relies on several submodules, so clone with submodule recursion enabled:
   
   ```
   git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
   cd Isaac-GR00T
   git checkout e29d8fc50b0e4745120ae3fb72447986fe638aa6
   cd ..
   
   git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git
   ```
   Copy all adaptation files from the cann recipe repo into the official Isaac-GR00T workspace:

   ```
   cp -rf cann-recipes-embodied-ai/manipulation/Isaac-GR00T/* Isaac-GR00T
   cd Isaac-GR00T
   ```

   
3. Environment Setup

   Two environment management workflows are provided: conda and uv.
   ## conda Environment
   ```
   conda create -n gr00t python=3.10 -y
   conda activate gr00t
   pip install -r requirements.txt
   chmod +x set_conda_env.sh
   ./set_conda_env.sh
   ```

   ## uv Environment
   
   > Note: Parsing `pyproject.toml` in `[tool.uv.extra-build-dependencies]` requires uv v0.8.4 or newer.
   
   
   Create environment and install GR00T dependencies:
   
   ```
   uv sync --python 3.10
   ```
   Install ffmpeg & decord
   ```
   chmod +x setup.sh
   ./setup.sh
   export LD_LIBRARY_PATH=$(pwd)/.venv/lib:$LD_LIBRARY_PATH
   ```
   Install decorator
   ```
   uv pip install decorator
   ```
   Activate uv virtual environment:

   ```
   source .venv/bin/activate
   ```



# Inference Execution

1. Dataset Preparation
   * Sample datasets are provided under the `demo_data` directory. Ensure full dataset download via git clone.
   * You may also prepare custom datasets on your own. For details, please refer to the [Data Preparation Guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md).
2. Quick Inference Script

   Run the standalone inference script to generate robot action trajectories after data preparation.

   conda Environment Command:

   ```
   python scripts/deployment/standalone_inference_script.py \
     --model-path nvidia/GR00T-N1.6-3B \
     --dataset-path demo_data/gr1.PickNPlace \
     --embodiment-tag GR1 \
     --traj-ids 2 \
     --video-backend decord \
     --seed 42 \
     --action-horizon 8
   ```
   uv Environment Command :
   ```
   uv run python scripts/deployment/standalone_inference_script.py \
     --model-path nvidia/GR00T-N1.6-3B \
     --dataset-path demo_data/gr1.PickNPlace \
     --embodiment-tag GR1 \
     --traj-ids 2 \
     --video-backend decord \
     --seed 42 \
     --action-horizon 8
   ```
  After launch, the model will run inference and output motion commands for robot components including left_arm, right_arm, left_hand, right_hand, waist.

# Citation
```
@inproceedings{gr00tn1_2025,
  archivePrefix = {arxiv},
  eprint     = {2503.14734},
  title      = {{GR00T} {N1}: An Open Foundation Model for Generalist Humanoid Robots},
  author     = {NVIDIA and Johan Bjorck and Fernando Castañeda, Nikita Cherniadev and Xingye Da and Runyu Ding and Linxi "Jim" Fan and Yu Fang and Dieter Fox and Fengyuan Hu and Spencer Huang and Joel Jang and Zhenyu Jiang and Jan Kautz and Kaushil Kundalia and Lawrence Lao and Zhiqi Li and Zongyu Lin and Kevin Lin and Guilin Liu and Edith Llontop and Loic Magne and Ajay Mandlekar and Avnish Narayan and Soroush Nasiriany and Scott Reed and You Liang Tan and Guanzhi Wang and Zu Wang and Jing Wang and Qi Wang and Jiannan Xiang and Yuqi Xie and Yinzhen Xu and Zhenjia Xu and Seonghyeon Ye and Zhiding Yu and Ao Zhang and Hao Zhang and Yizhou Zhao and Ruijie Zheng and Yuke Zhu},
  month      = {March},
  year       = {2025},
  booktitle  = {ArXiv Preprint},
}
```

# Appendix
## Final Repository Directory Tree
After patch replacement, the root directory structure of the Ascend-adapted GR00T N1.6 workspace is shown below:
```bash
.
├── adaptor_patches
│   ├── dit_patch.py
│   ├── gr00t_n1d6_patch.py
│   ├── gr00t_policy_patch.py
│   └── modeling_siglip2_patch.py
├── pyproject.toml
├── README.md
├── scripts
│   └── deployment
│       ├── model_adaptor.py
│       └── standalone_inference_script.py
├── requirements.txt
├── setup.sh
└── set_conda_env.sh
```
