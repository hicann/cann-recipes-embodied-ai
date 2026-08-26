# AgiBot Arm World Model

This sample packages an AgiBot robotic arm world model workflow based on `Wan2.1-Fun-V1.1-1.3B-Control`. It includes a training sample and a PyTorch-based inference sample for Ascend NPU environments.

## Summary

- Model: `Wan2.1-Fun-V1.1-1.3B-Control`
- Scenario: AgiBot robotic arm world model training and trajectory-conditioned inference
- Device: Ascend910
- Training form: multi-card training, default sample configuration uses 8 NPUs
- Inference form: single-card inference

## External Dependency

This sample depends on an external `DiffSynth-Studio` checkout from the official repository:

- Repository: `https://github.com/modelscope/DiffSynth-Studio.git`
- Model support reference: `docs/en/Model_Details/Wan.md`
- Suggested verification: run a smoke test with the exact sample scripts in this directory before submitting the PR

## Environment Requirements

- OS: Linux
- Python: 3.10 is recommended
- Device runtime: Ascend NPU environment with CANN properly configured
- Framework dependency: official `modelscope/DiffSynth-Studio`
- Python package dependency: see [requirements.txt](requirements.txt)

Platform-coupled packages such as `torch` and `torch_npu` should be installed according to the target Ascend and CANN environment, rather than pinned blindly in `requirements.txt`.Specifically,we use `CANN==8.5.1,torch==2.1.0 and torch-npu==2.1.0.post12`.

## Environment Setup

1. Prepare the Ascend runtime environment.
2. Make sure the Ascend environment scripts are sourced in the current shell.
3. Create and activate a Python environment.
4. Install the sample Python dependencies from `requirements.txt`.
5. Clone and install the official `DiffSynth-Studio`.

Example shell pattern:

```bash
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/toolbox/set_env.sh

conda create -n agibot-wm python=3.10 -y
conda activate agibot-wm

pip install -r requirements.txt

git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
cd ..
```

After that, install `torch` and `torch_npu` according to the target Ascend software stack if they are not already available in the environment.

You can verify the official framework installation with:

```bash
python -c "import diffsynth; print('diffsynth import ok')"
```

## Directory Layout

```text
agibot-arm-world-model/
  train/
  infer_with_torch/
  doc/
  lvdm/
```

- `train/`: training code and dataset loader
- `infer_with_torch/`: inference code
- `doc/`: optimization and adaptation notes
- `lvdm/`: third party modules

`DiffSynth-Studio` is an external dependency and is not included in this sample directory.

## Suggested Workspace Layout

It is recommended to place the official framework and this sample side by side in the same workspace:

```text
workspace/
  DiffSynth-Studio/
  agibot-arm-world-model/
```

Recommended setup flow:

```bash
cd workspace
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
cd ..
```

Then place this sample directory as:

```text
workspace/agibot-arm-world-model
```

After `DiffSynth-Studio` is installed, this sample can be run from its own directory and does not need to be copied into the `DiffSynth-Studio` repository.

## Included and Excluded Content

Included:

- sample source code
- launcher scripts
- config files
- dataset loading logic
- trajectory/action utility code
- README and optimization notes

Not included:

- datasets
- pretrained weights
- checkpoints
- generated outputs
- logs
- binary assets

## How To Use

1. Prepare an Ascend runtime environment and a conda environment with the required Python packages.
2. Clone and install the official `DiffSynth-Studio`.
3. Prepare model weights and the AgiBotWorld-format dataset.
4. Follow [doc/train.md](doc/train.md) for training.
5. Follow [doc/inference.md](doc/inference.md) for inference.
6. See [doc/optimization.md](doc/optimization.md) for the design and optimization rationale of this sample.
