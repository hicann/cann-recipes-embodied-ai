# OpenVLA on Ascend 310P: VLA Model Usage Guide

This directory describes how to convert the OpenVLA model to offline models and run inference on Ascend 310P. It also provides accuracy verification and simulation evaluation steps.

## OpenVLA Overview

The OpenVLA model was proposed in the paper "OpenVLA: An Open-Source Vision-Language-Action Model".

Paper link:  
https://arxiv.org/abs/2406.09246

Official OpenVLA repository:  
https://github.com/openvla/openvla

### Feature Introduction

OpenVLA is a typical Vision-Language-Action (VLA) general-purpose control model. Its core idea is to encode visual observations and language instructions into a unified sequence representation, and then generate action representations in an autoregressive manner, such as action tokens or discretized action sequences. These action representations are then decoded into executable continuous control values.

By learning a unified mapping from perception and semantics to actions on large-scale multi-task robot demonstration data, OpenVLA aims to improve generalization across tasks and scenarios, while reducing the cost of training a separate policy for each task.

## OpenVLA Code Repository, Simulation Dataset, and Model Download

The sample model used in this example is:  
https://huggingface.co/openvla/openvla-7b-finetuned-libero-object

This is the model officially released by OpenVLA after fine-tuning on the `libero_object` dataset.

### Model Inputs and Outputs

> Note: OpenVLA inputs consist of **text instructions (tokens)** and **image tensors (`pixel_values`)**.  
> When the *fused vision backbone* is enabled, the number of channels in `pixel_values` is **6 (3+3)**, which means that the same image frame is processed by two visual preprocessing pipelines and then concatenated along the channel dimension.

#### Inputs

| Input Name | Description | dtype | Shape (Example) | Notes |
| --- | --- | --- | --- | --- |
| `input_ids` | Token sequence of the instruction or prompt | `int64` | `[B, T]` | `T` is the text token length, including special tokens; `B` is the batch size, commonly 1 |
| `attention_mask` | Valid-position mask for text tokens | `bool` or `int64/int32`, depending on implementation | `[B, T]` | 1/True indicates a valid token, and 0/False indicates padding |
| `pixel_values` | Tensor obtained after preprocessing camera RGB images with the processor | `float16`, commonly | `[B, C, H, W]` | If `use_fused_vision_backbone=True`, `C=6 (3+3)`; otherwise, `C=3` |

#### Outputs

| Output Name | Description | dtype | Shape (Example) | Notes |
| --- | --- | --- | --- | --- |
| `actions` / `generated_ids` | Action tokens, or token IDs of a discretized action sequence | `int64`, commonly | `[B, A]` | `A` is the action dimension or number of action tokens, usually determined by `action_dim`; `bin_centers` and `action_norm_stats` are required for denormalization into continuous actions |

**Parameter descriptions:**

- `B`: batch size, usually 1 for offline verification.
- `T`: length of the text token sequence, which is determined by the prompt length and tokenizer rules, including special tokens.
- `H, W`: visual input resolution produced by the processor. A common value is 224x224, but the actual value depends on the processor configuration.
- `C`: number of image channels. When the fused backbone is enabled, `C=6=3+3`, which corresponds to the concatenated inputs of two vision towers. Otherwise, `C=3`.
- `A`: action sequence length or action dimension, usually equal to `action_dim`. It is related to the robot degrees of freedom and the action representation method.

## OpenVLA Runtime Configuration on Ascend 310P

### Ascend Platform Environment Configuration

Converting and running `.om` models requires CANN to be installed.

This sample depends on the CANN development toolkit package, `cann-toolkit`, and the CANN binary operator package, `cann-kernels`. The supported CANN versions are `CANN 8.0.0-8.2.RC1`. Download the corresponding software packages for your architecture from the software package download page, and install them by referring to the CANN installation documentation.

```bash
# ${cann_install_path} is the actual installation directory of the CANN package.
# Remember to source set_env.sh first whenever a new terminal is opened.

# Method 1: default installation path, using the root user as an example.
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Method 2: custom installation path.
source ${cann_install_path}/ascend-toolkit/set_env.sh
```

### Environment Configuration Unrelated to the Ascend Server

```bash
# Create the runtime environment.
conda create -y -n openvla python=3.10
conda activate openvla

# Clone and install the OpenVLA repository. Example:
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .
```

### Headless Simulation Rendering with MuJoCo

If the server or container lacks a display environment or an OpenGL rendering backend, MuJoCo may fail to render properly.

You can specify EGL headless rendering before running simulation or evaluation:

```bash
export MUJOCO_GL=egl
```

## OpenVLA Inference Steps on Ascend 310P

This section describes the deployment reference for offline inference with Ascend-friendly OM files. For more parameters, refer to the [ATC tool documentation](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/devaids/atctool/atlasatc_16_0003.html).

<img src="https://raw.gitcode.com/user-images/assets/10308584/d036e891-3c2f-4578-8f8b-3c2a3e8f5d0e/om_compile_workflow.png" style="zoom:50%;" />

The following is a recommended single-machine workflow:

1. Export ONNX on the 310P host machine using the host CPU.
2. Convert ONNX to OM with ATC on 310P.
3. Run simulation evaluation with the OM-backend sim-evaluator on 310P.

Additional ONNX Runtime dependencies need to be installed on the machine used for ONNX conversion:

```bash
pip install onnx

# For ONNX conversion based on the host CPU, run this on the 310P host:
pip install onnxruntime

# For ONNX conversion based on the host GPU:
pip install onnxruntime-gpu
```

#### 1. Export ONNX

Before export, apply the operator conversion fix to the `transformers` library in the environment. This ensures that OM model conversion can match Ascend-friendly operators.

```bash
cd /path/to/conda/envs/openvla/lib/python3.10/site-packages/transformers/models/llama
git apply --check -p1 /path/to/openvla/modeling_llama.patch
git apply -p1 /path/to/openvla/modeling_llama.patch
```

Run the following command on the host, either CPU or GPU:

```bash
# Example with a local directory: models/ contains files such as config.json.
# You can also use huggingface-cli to download the model to models/ first:
#   pip install -U huggingface_hub
#   huggingface-cli download openvla/openvla-7b-finetuned-libero-object --local-dir models

python3 convert_and_verify_onnx.py \
  --model-path models/openvla-7b-finetuned-libero-object \
  --vision-export-dir outputs/onnx/vision \
  --llama-prefill-export-dir outputs/onnx/llama_prefill \
  --llama-decoder-export-dir outputs/onnx/llama_decoder \
  --unnorm-key libero_object
```

Notes:

- By default, ONNXRuntime CPU is used to compare outputs against PyTorch and print the max/mean diff. To skip this step, add `--no-validate`.

Sample output:

```text
============================================================
Validating Full Inference Pipeline
============================================================

[1/2] Running PyTorch inference...
[2/2] Running ONNX inference...
Loading ONNX models with provider: CPUExecutionProvider...
ONNX models loaded successfully.

[3/3] Comparing results...
full_pipeline_action: max abs diff = 0.000000e+00
full_pipeline_action: mean abs diff = 0.000000e+00
full_pipeline_action: ✓ MATCH (rtol=0.001, atol=0.001, mean_diff_threshold=1e-2)

PyTorch action:
[ 1.43156521e-01  2.43907466e-02  9.26470588e-01 -3.15118654e-05
  7.75504180e-02 -3.35294148e-02  0.00000000e+00]

ONNX action:
[ 1.43156521e-01  2.43907466e-02  9.26470588e-01 -3.15118654e-05
  7.75504180e-02 -3.35294148e-02  0.00000000e+00]

✅ Full pipeline validation passed!

============================================================
```

#### 2. Convert ONNX to OM with ATC

Run the conversion script on 310P after CANN has been installed and sourced:

```bash
./convert_onnx_to_om.sh \
    --vision-onnx-dir outputs/onnx/vision \
    --llama-prefill-onnx-dir outputs/onnx/llama_prefill \
    --llama-decoder-onnx-dir outputs/onnx/llama_decoder \
    --vision-om-dir outputs/om/vision \
    --llama-prefill-om-dir outputs/om/llama_prefill \
    --llama-decoder-om-dir outputs/om/llama_decoder \
    --soc-version Ascend310P3
```

After conversion is complete, the `.om` models should be generated in their specified output directories. The terminal output should contain `ATC run success, welcome to the next use`.

#### 3. Run Simulation Evaluation with the OM-backend sim-evaluator

Run the following command on 310P. ACL/ACLLite Python dependencies must be available.

For reference, see the [ACLLite installation guide](https://gitee.com/ascend/ACLLite#%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B).

The simulation evaluation is modified based on the official LIBERO simulation evaluation provided by OpenVLA. You can apply the simulation adaptation patches included in this repository to obtain an OM-backend simulation evaluation environment.

The simulation-related patches are located in the `sim/` directory of the repository. They include three patches, `robot_utils.patch`, `openvla_utils.patch`, and `run_libero_eval.patch`, as well as a new file that needs to be added: `openvla_om_utils.py`.

```bash
# Make sure you are in the root directory of the openvla repository.
cd openvla
git apply --check /path/to/xxx.patch
git apply xxx.patch

# The new file must be placed under experiments/robot/.
cp /path/to/openvla_om_utils.py ./experiments/robot/
```

After the code environment is ready, run the following command for simulation evaluation:

```bash
python3 -m experiments.robot.libero.run_libero_eval \
    --model_family openvla \
    --pretrained_checkpoint models/openvla-7b-finetuned-libero-object/ \
    --task_suite_name libero_object \
    --center_crop True \
    --vision_backbone_om outputs/om/vision/vision_backbone.om \
    --projector_om outputs/om/vision/projector.om \
    --embedding_om outputs/om/vision/embedding.om \
    --prefill_om outputs/om/llama_prefill/vla_prefill.om \
    --decode_om outputs/om/llama_decoder/vla_decoder.om
```

Output:

- Evaluation result logs are written to `experiments/logs`, including information such as success rate.
- Simulation result videos are located under the `rollout/date` directory, where `date` is the date.

## OpenVLA Accuracy Verification on Ascend

This section introduces two methods for verifying the converted `.om` models running on NPU.

### 1. Compare Output Similarity Between CPU/GPU PyTorch and OM with Mock Inputs

Construct fixed inputs, such as an all-zero image and fixed instruction tokens, and compare the output accuracy between PyTorch CPU/GPU and OM on the 310P NPU:

```bash
# Run on 310P. ACL/ACLLite Python dependencies are required.
python3 verify_om_onnx.py \
   --model-path models/openvla-7b-finetuned-libero-object \
   --unnorm-key libero_object \
   --vision-backbone-om outputs/om/vision/vision_backbone.om \
   --projector-om outputs/om/vision/projector.om \
   --embedding-om outputs/om/vision/embedding.om \
   --prefill-om outputs/om/llama_prefill/vla_prefill.om \
   --decode-om outputs/om/llama_decoder/vla_decoder.om
```

### 2. Functional Test Based on the Simulation Environment (MuJoCo / LIBERO)

Use data from the `libero` simulation environment for inference on NPU, and run simulation rendering or the control loop on the host CPU:

```bash
python3 -m experiments.robot.libero.run_libero_eval \
    --model_family openvla \
    --pretrained_checkpoint models/openvla-7b-finetuned-libero-object/ \
    --task_suite_name libero_object \
    --center_crop True \
    --vision_backbone_om outputs/om/vision/vision_backbone.om \
    --projector_om outputs/om/vision/projector.om \
    --embedding_om outputs/om/vision/embedding.om \
    --prefill_om outputs/om/llama_prefill/vla_prefill.om \
    --decode_om outputs/om/llama_decoder/vla_decoder.om
```

Sample result:  
<img src="https://raw.gitcode.com/user-images/assets/7380116/7b2551e1-efab-4540-9bcd-77b3eda0b6e7/libero.gif " style="zoom:60%;" />

## Citation

```bibtex
@article{kim24openvla,
    title={OpenVLA: An Open-Source Vision-Language-Action Model},
    author={{Moo Jin} Kim and Karl Pertsch and Siddharth Karamcheti and Ted Xiao and Ashwin Balakrishna and Suraj Nair and Rafael Rafailov and Ethan Foster and Grace Lam and Pannag Sanketi and Quan Vuong and Thomas Kollar and Benjamin Burchfiel and Russ Tedrake and Dorsa Sadigh and Sergey Levine and Percy Liang and Chelsea Finn},
    journal = {arXiv preprint arXiv:2406.09246},
    year={2024}
}
```

## Appendix: Example Directory Tree of the OpenVLA Root Directory

After completing the steps above, check the overall code directory tree. An example project directory tree for adapting OpenVLA to Ascend is shown below:

```text
Format
|-- README.md                       # Chinese usage guide
|-- README_en.md                    # This file
|-- models                          # Models downloaded from Hugging Face or other sources
|-- openvla/
|   |-- convert_and_verify_onnx.py  # PyTorch -> ONNX conversion script
|   |-- verify_om_onnx.py           # Error comparison between PyTorch (CPU) and OM (NPU)
|   |-- vla_validation_utils.py     # Accuracy verification helper methods
|   |-- convert_onnx_to_om.sh       # ONNX -> OM conversion script
|   |-- lib
|   |   |-- modeling_llama.patch    # Adaptation patch for the transformers library
|   |
|   |-- sim
|       |-- robot_utils.patch       # Patch for the simulation file robot_utils.py
|       |-- openvla_utils.patch     # Patch for the simulation file openvla_utils.py
|       |-- run_libero_eval.patch   # Patch for the simulation file run_libero_eval.py
|       |-- openvla_om_utils.py     # New simulation file for OM-backend support
|
|-- outputs
    |-- onnx/                       # Output ONNX-format models
    |-- om/                         # Output OM-format models
```