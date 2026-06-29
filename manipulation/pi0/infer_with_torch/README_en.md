# π0 Robot VLA Large Model Ascend Usage Guide
<br>
## π0 Overview

**Paper Title:** π0: A Vision-Language-Action Flow Model for General Robot Control

### Introduction

π0 is a Vision-Language-Action (VLA) model designed for general robot control. It is built upon a pre-trained Vision-Language Model (VLM) and incorporates a flow matching mechanism to generate high-frequency continuous actions, enabling precise control of complex and dexterous robotic tasks. It integrates the OXE open-source dataset with proprietary datasets, totaling over 10,000 hours of robot manipulation data. It demonstrates outstanding performance on complex tasks such as laundry folding, desktop cleanup, and box packing, significantly surpassing existing baseline methods (OpenVLA, Octo, ACT, etc.) in both zero-shot and fine-tuning settings. It successfully completes long-horizon multi-stage tasks lasting 5–20 minutes, exhibiting strong robustness and generalization capabilities.
<br>
## Repository Cloning, Dataset and Model Download for π0

```bash
# Navigate to the local directory where you want to place the repository, then run:
git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git
chmod +x cann-recipes-embodied-ai/manipulation/pi0/infer_with_torch/download_code_and_data.sh
./cann-recipes-embodied-ai/manipulation/pi0/infer_with_torch/download_code_and_data.sh
```

After completing the above steps, the final directory tree of the lerobot root directory is shown in [Appendix: lerobot Root Directory Code Tree](#lerobot-root-directory-code-tree).
<br>
## Environment Configuration for π0 on Ascend A2

### Non-Ascend-Specific Environment Configuration

```bash
# Create the runtime environment
conda create -y -n lerobot python=3.10
conda activate lerobot
# Return to the lerobot root directory and install lerobot
cd lerobot
pip install -e .
```

### Ascend-Specific Environment Configuration

Install the CANN software packages. This sample requires the CANN Toolkit (cann-toolkit) and CANN Kernels (cann-kernels) for compilation and execution. The supported CANN software version is `CANN 8.3.RC1`.

Please download the `Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run` and `Ascend-cann-kernels-910b_8.3.RC1_linux-aarch64.run` packages from the [Software Download Page](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1), and follow the [CANN Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit) to install them sequentially.

```bash
# ${cann_install_path} is the actual installation directory of the CANN package.
# NOTE: Source set_env.sh every time a new terminal is opened.
# Option 1: Default path installation (root user example)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# Option 2: Custom path installation
source ${cann_install_path}/ascend-toolkit/set_env.sh
# Install the corresponding version of torch-npu in the above environment
pip install torch-npu==2.1.0.post12
```
<br>
### π0 Inference Steps on Ascend

Run the following commands to automatically load the Koch manipulator dataset, perform π0 model inference, and print the inference performance and robot actions:

```bash
# Navigate to the lerobot repository root directory
cd lerobot
conda activate lerobot
chmod +x run_pi0_inference.sh
./run_pi0_inference.sh koch_test pi0_model 10 100
```

Based on the above execution, the single inference time and results for π0 are as follows (for detailed optimization process, see [π0 Optimization Guide](../../../docs/manipulation/pi0/infer_with_torch/README.md)):

- **Inference Performance:** Single inference time reduced to 80 ms, achieving the expected inference time performance optimization target.
- **Inference Result:** A single inference produces 50 sets of manipulator joint angle sequences with a shape of [50, 6].
<br>
## π0 Accuracy Verification Steps on Ascend

### Verifying Ascend Inference Accuracy via ATE (Absolute Trajectory Error) of the Koch Manipulator End-Effector Pose

- To perform inference accuracy testing on the Ascend platform using a controlled variable approach, the Gaussian noise sampling in the `action_expert` of π0 inference is replaced with fixed noise file loading (i.e., using the same Gaussian noise sampling data).
- Based on the full trajectory six-joint angle sequence (dimension: 50×6) obtained from π0 model inference, forward kinematics computation is performed using the physical DH parameters of the Koch manipulator to obtain the actual pose (position x-y-z + orientation r-p-y) of the Koch manipulator's end-effector center. The ATE (Absolute Trajectory Error) method is then used to compute the L2 norm, yielding the end-effector pose error of the Koch manipulator on the Ascend platform. The error reference ranges are as follows:
  - Position ATE error reference range: [0, +0.03] m
  - Orientation ATE error reference range: [0, +0.2] rad
<br>
## Quick Start on GitCode Cloud Development Environment (One-Stop Platform)

This section is for users of the one-stop platform. The platform comes with a complete Ascend CANN environment pre-installed. Follow these steps to complete the π0 torch-based Ascend inference on a single card.

> Users of the one-stop platform should select a Python 3.12 instance on A2/A3 for creation.

### Code and Weights Preparation

```bash
# Navigate to the local directory where you want to place the repository, e.g., /mnt/workspace/gitCode/cann (adjust as needed):
cd /mnt/workspace/gitCode/cann
git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git
chmod +x cann-recipes-embodied-ai/manipulation/pi0/infer_with_torch/download_code_and_data.sh
./cann-recipes-embodied-ai/manipulation/pi0/infer_with_torch/download_code_and_data.sh
```

### Runtime Environment Configuration

```bash
# Create the runtime environment
conda create -y -n lerobot python=3.10
conda activate lerobot
# Return to the lerobot root directory and install lerobot
cd lerobot
pip install -e .
# Install the corresponding version of torch-npu in the lerobot environment
pip install torch-npu==2.1.0.post12
```

### Running the Inference Script

```bash
# Ensure you are in the lerobot repository root directory and the lerobot conda environment is activated
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
## Appendix

### lerobot Root Directory Code Tree

- After completing the above cloning and setup operations, the final code directory tree in the lerobot root directory adapted for Ascend is as follows:

```bash
├── koch_test                                 # Koch manipulator grasping task dataset (lerobot format)
├── lerobot                                   # π0 model training and inference framework
│   ├── common
│   │   ├── policies
│   │   │   ├── pi0
│   │   │   │   ├── modeling_pi0.py           # π0 model training and inference code
│   │   │   │   ├── paligemma_with_expert.py  # π0 model training and inference code
├── pi0_model                                 # Pre-trained π0 model for Koch manipulator grasping task
├── pyproject.toml                            # Third-party package version specifications
├── README_en.md                              # Environment configuration and operation guide for π0 inference on Ascend (English)
├── README.md                                 # Environment configuration and operation guide for π0 inference on Ascend (Chinese)
├── run_pi0_inference.sh                      # One-click launch script for π0 inference on Ascend
└── test_pi0_on_ascend.py                     # Main code for π0 inference on Ascend