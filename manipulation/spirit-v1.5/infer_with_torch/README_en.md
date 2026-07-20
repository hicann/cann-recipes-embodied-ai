# Adaptation of Spirit-v1.5 Embodied Large Model for Ascend 310P

## Introduction to Spirit-v1.5

Spirit v1.5 is an embodied AI model independently developed by Spirit AI. It ranked first overall in the RoboChallenge evaluation held on January 12, 2026, maintaining a high success rate across multiple tasks. It delivers consistent performance especially in continuous multi-task execution, complex instruction decomposition, and cross-hardware configuration transfer scenarios.

This sample demonstrates how to adapt the Spirit v1.5 model and run inference on the Ascend 310P platform.

## Supported Product Models

Ascend 310P Series

## Code and Weight Preparation

1. Clone this repository: `git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git`


2. Clone the source code from the official open-source repository of Spirit AI:
`git clone https://github.com/Spirit-AI-Team/spirit-v1.5.git`


3. Copy the following files from this repository to the Spirit v1.5 repository (overwrite existing files with identical names if present):

    | Path in This Repository| Target Path in Spirit v1.5 Repository |
    | ---- | ---- |
    | `manipulation/spirit-v1.5/infer_with_torch/pyproject.toml` | `pyproject.toml` |
    | `manipulation/spirit-v1.5/infer_with_torch/requirements.txt` | `requirements.txt` |
    | `manipulation/spirit-v1.5/infer_with_torch/modeling_spirit_vla.py` | `model/modeling_spirit_vla.py` |
    | `manipulation/spirit-v1.5/infer_with_torch/attention_processor_patch.py` | `model/attention_processor_patch.py` |
    | `manipulation/spirit-v1.5/infer_with_torch/infer_mozrobot_ascend.py` | `scripts/infer_mozrobot_ascend.py` |


4. Download model weights provided by Spirit AI
    | Model | Type |
    |----------|-------------|
    | [Spirit-v1.5](https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5) | Base Model |
    | [Spirit-v1.5-move-objects-into-box](https://huggingface.co/Spirit-AI-robotics/Spirit-v1.5-for-RoboChallenge-move-objects-into-box) | Fine-tuned Model |

5. (Optional) Download weights of [Qwen/Qwen3VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)


## Runtime Environment Setup

### Install CANN Software Packages

This sample relies on the CANN development toolkit(cann-toolkit) and CANN binary operator package(cann-kernels). The supported CANN version is `CANN 8.3.RC1`.

Download from the [software download page](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1)`Ascend-cann-toolkit_8.3.RC1_linux-${arch}.run` and  `Ascend-cann-kernels_310p_8.3.RC1_linux-${arch}.run`，then complete installation by referring to the [CANN installation documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=netconda&OS=openEuler&Software=cannToolKit).

- `${arch}`denotes the CPU architecture; select either aarch64 or x86_64 matching your host machine.

### Configure Python Environment

To align with Spirit AI's official workflow, we use the `uv` tool for environment management.First check if uv is installed locally; if not, run `pip install uv` to install it.

Navigate to the Spirit v1.5 code directory and run `uv sync`,This command automatically resolves and installs all package dependencies.

A `.venv`folder will be generated at the repository root. Run `source .venv/bin/activate`to activate the virtual environment. You may execute `uv pip list` to verify all dependencies are fully installed.

## Inference

Follow the guidance in the README file inside the Spirit v1.5 repository to launch the inference script.
