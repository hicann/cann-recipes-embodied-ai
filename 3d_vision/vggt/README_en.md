# VGGT Model Inference Adaptation on Ascend Atlas A2/A3

This sample completes the inference adaptation of the VGGT model on NPU based on the [VGGT open-source model](https://github.com/facebookresearch/vggt), and provides accuracy evaluation scripts for three tasks: camera pose estimation, point cloud reconstruction, and depth estimation. For detailed information, please refer to the [Accuracy Evaluation Chapter](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/docs/3d_vision/vggt/vggt_accurancy_evaluation.md).

Additionally, this sample has optimized the VGGT model performance on NPU. Currently, with 25 images as input, the inference time has been reduced to 1.12 seconds. For detailed information, please refer to the [Performance Optimization Chapter](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/docs/3d_vision/vggt/vggt_optimization.md).

This sample supports single-card inference and multi-card sequence parallel inference on Ascend Atlas A2/A3 environment.

> Users using the one-stop platform can directly jump to the [「One-stop Platform Quick Start」](#one-stop-platform-quick-start) chapter.

***

## Running the Sample

### CANN Environment Preparation

1. This sample depends on the CANN development toolkit package (cann-toolkit) and CANN binary operator package (cann-kernels). Currently, the CANN software version used is `CANN.8.5.0`.
   Please download `Ascend-cann-toolkit_${version}_linux-${arch}.run` and `Ascend-cann-${chip_type}-ops_linux-${arch}.run` packages from [CANN Software Download](https://www.hiascend.com/developer/download/community/result?module=cann\&cann=8.5.0), and refer to [CANN Installation Documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/quickstart/instg_quick.html) for installation.

2. The torch and torch_npu versions required by this sample are 2.7.1.
   Please download torch and torch_npu installation packages from [Ascend Extension for PyTorch Plugin](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md). The torch and torch_npu versions required are 2.7.1 and 2.7.1.post2 respectively.
   ```shell
   conda create -n vggt python==3.11.13
   conda activate vggt
   pip3 install torch==2.7.1
   pip3 install torch_npu==2.7.1.post2
   ```

### Network Model Code Preparation

- This repository depends on the open-source repository code from [VGGT](https://github.com/facebookresearch/vggt/tree/main).
- Navigate to the official VGGT repository and download the VGGT model network code:
  ```shell
  git clone https://github.com/facebookresearch/vggt.git
  ```
- Download this repository code:
  ```shell
  git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git
  ```
- Download the VGGT model weights `model.pt` via HuggingFace (weights source: [VGGT model checkpoint](https://huggingface.co/facebook/VGGT-1B)), and place the weights file into the project `ckpt` directory. Follow these steps:
  ```shell
  pip install -U huggingface_hub
  export HF_ENDPOINT=https://hf-mirror.com
  hf download facebook/VGGT-1B model.pt --local-dir vggt
  mkdir -p cann-recipes-embodied-ai/3d_vision/vggt/ckpt
  mv vggt/model.pt cann-recipes-embodied-ai/3d_vision/vggt/ckpt/
  ```
- Copy the VGGT repository network model files to this project directory in **non-overwrite mode**:
  ```shell
  cp vggt/visual_util.py cann-recipes-embodied-ai/3d_vision/vggt/
  cp -r vggt/examples cann-recipes-embodied-ai/3d_vision/vggt/
  cp -rn vggt/vggt/dependency cann-recipes-embodied-ai/3d_vision/vggt/vggt/dependency
  cp -rn vggt/vggt/heads cann-recipes-embodied-ai/3d_vision/vggt/vggt/
  cp -rn vggt/vggt/layers cann-recipes-embodied-ai/3d_vision/vggt/vggt/
  cp -rn vggt/vggt/utils cann-recipes-embodied-ai/3d_vision/vggt/vggt/
  ```
- Install Python dependencies:
  ```shell
  cd cann-recipes-embodied-ai/3d_vision/vggt/
  pip3 install -r requirements.txt
  ```
- The model weights and model structure are listed in the file directory as follows:
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

### Quick Start

This sample uses YAML configuration files to manage parameters, supporting multi-configuration scenario switching. For detailed parameter descriptions and constraint conditions, please refer to [YAML Configuration File Description](config/README.md).

Before executing the script, please refer to the CANN installation tutorial in [Ascend Community](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/quickstart/instg_quick.html) to configure environment variables:

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

#### **Single-card/Multi-card Inference**

- Method 1: Python direct execution (single-card inference only)

```bash
# Use single.yaml configuration by default (single-card inference)
python demo_infer.py

# Specify configuration file (single-card inference)
python demo_infer.py --config config/single.yaml
```

- Method 2: Shell script execution (supports single-card and multi-card)

```bash
# Single-card inference (uses single.yaml by default)
bash infer_test.sh

# Multi-card inference (2-card parallel, internally calls torchrun)
bash infer_test.sh sp2.yaml

# Multi-card inference (4-card parallel)
bash infer_test.sh sp4.yaml

# Multi-card inference (8-card parallel)
bash infer_test.sh sp8.yaml
```

#### **int8 Quantized Model Inference**

You need to generate the int8 model first (in the current implementation, only the Linear layers with K=4096 in the VGGT model are quantized to 8bit).

**Step 1: Generate int8 Quantized Model**

Modify the `build` parameter in the YAML configuration file to true to build the quantized model:

```yaml
optimization:
  quantization:
    int8-w8a8:
      enable: false
      build: true
```

Then run (replace the yaml filename with actual configuration file):

```bash
python demo_infer.py --config config/xxx.yaml
```

The int8 model will be generated in the current path (filename: `VGGT_model_W8A8.pt`).

**Step 2: Use int8 Quantized Model for Inference**

Use the `config/single_w8a8.yaml` configuration file:

```bash
python demo_infer.py --config config/single_w8a8.yaml
```

## One-stop Platform Quick Start

This chapter is for users using the one-stop platform. The platform has pre-configured the complete CANN environment. Follow the steps below to complete VGGT 3D reconstruction inference on a single card.

> Users using the one-stop platform should select instances related to python3.11 on A2/A3 for creation.

### Modify Variables in File

Modify the `WORKSPACE_DIR` variable in `infer_platform_env_prepare.sh` to point to the path, such as `cann_recipes`

### Code and Weights Preparation

Run the following command to execute the script for code and weights preparation:

```bash
cd cann-recipes-embodied-ai/3d_vision/vggt
bash infer_platform_env_prepare.sh
```

### Run Inference Script

Run bf16 model single-card inference:

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