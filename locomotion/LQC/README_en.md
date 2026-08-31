# Learning-based Quadruped Robot Controller

## Overview
The Learning-based Quadruped Robot Controller (LQC) is a reinforcement learning motion control algorithm for legged robots. It supports model import, one‑click training, and inference verification for multiple mainstream robot platforms such as Unitree G1 and GO2. This sample is based on the theoretical work presented at [IROS2025](#citation), and has been optimized and migrated on Ascend A2 servers. It aims to promote the adoption of Ascend servers in the legged‑robot domain and facilitate the intelligent upgrade of the quadruped robotics industry.

## Environment Setup

### Pull the Docker Image
Download the Docker image from the [ARM image repository](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-intelligence/lqc-image.tar) and upload it to the Ascend A2 server. Import the image with the following command:
```
docker load -i lqc-image.tar
```

After loading the image on the Ascend server, verify it with:

```
docker images
```

Start a container using the image on the Ascend server. Be sure to specify `${container_name}` and `${image_name}` accordingly:
```
docker run -itd  --privileged --net=host --ipc=host \
--device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 \
--device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 \
--device=/dev/davinci6 --device=/dev/davinci7 \
--device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
-v /etc/localtime:/etc/localtime -v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware -v /var/log/npu/:/usr/slog \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /data0:/data0 -v /data1:/data1 -v /data2:/data2 -v /home:/home \
--name=${container_name} ${image_name} /bin/bash
```

Enter the container:
```
docker exec -it ${container_name} /bin/bash
cd ~
```

### Clone the Repository
```
git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git /tmp/cann-recipes-embodied-ai
```

Copy and replace all files from the `cann-recipes-embodied-ai/locomotion/LQC` directory in the repository to the corresponding path inside the container:
```
cp -rf /tmp/cann-recipes-embodied-ai/locomotion/LQC/* ~/cann-recipes-embodied-ai/locomotion/LQC/
```

Switch to the project directory:
```
cd cann-recipes-embodied-ai/locomotion/LQC
```

Clone the remaining submodules:
```
cd extern
git config --global http.sslVerify false
git clone https://github.com/bab2min/EigenRand
git clone https://github.com/ArashPartow/exprtk
cd EigenRand && git checkout f3190cd7 && cd ..
cd exprtk && git checkout master && cd ../..
```

### Environment Setup
LQC uses Conda for environment management and requires MuJoCo as the physics engine, FastNoise2 for noise generation, as well as dependencies such as CMake and GLFW3. This sample provides an environment script for one‑click downloading and installation of all required components.

```
chmod +x set_env.sh
./set_env.sh
```

After configuring the environment, activate the Conda environment:
```
source ~/.bashrc
conda activate lltk
```

Install the remaining Python libraries:
```
pip install -r requirements.txt
```

> If you see warnings about missing `getopt`, `inspect`, or `multiprocessing`, these are false positive dependency alerts triggered by a declaration flaw in the third‑party package `op‑compile‑tool 0.1.0`. They do not affect any code execution.

Enable performance optimisation for multi‑card parallel training:
```
chmod +x auto_patch_config.sh
./auto_patch_config.sh
```

Compile the project:
```
python build.py --backend mujoco
```

After compilation, verify that the training environment is ready; the following command will list the robot types supported by the current training environment:
```
python -c "from lltk import registry; print(registry.list_envs())"
```

### Model Assets Preparation
Before starting training, you need to prepare 3D model files for MuJoCo to load. This sample provides physics model files for various robots including Unitree G1 and GO2 with different degrees of freedom. [Download the assets](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-intelligence/LQC/resources.tar) and extract them into the `./resources` folder.

## Single‑Card Training

Use the following command to start training. The default mode is `Headless`, and training logs will be printed in the terminal:
```
python scripts/train.py -r <ROBOT> -n <RUN_NAME>
```

where `<ROBOT>` supports three preset configurations for Unitree G1:
- `g1`: 12 DoF legs, suitable for flat terrain.
- `g1_15dof`: 15 DoF (12 DoF legs + 3 DoF waist), suitable for flat terrain.
- `g1_15dof.rough`: 15 DoF with elevation map enabled, suitable for rough terrain and stairs.

When training is launched, a folder `logs/<TASK_NAME>/<RUN_NAME>` will be created to save the model weights.

## Multi‑Card Training
### Multi‑Card Parallel Training Command
```bash
# Example for single‑node 8‑card training
export WANDB_MODE=disabled
export MASTER_PORT=29999
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 scripts/train.py -r g1_15dof.rough
```
### Detailed Explanation of Environment Variables for Multi‑Card Parallel Training
| Environment Variable | Description | Purpose |
| ---- | ---- | ---- |
| `WANDB_MODE=disabled` | Disable WandB online logging | Optional. If no external network access is available, disable online logging to avoid startup errors. |
| `MASTER_PORT=29999` | Master process port for distributed training | PyTorch DDP communication port; ensure no port conflicts within the node. |
| `ASCEND_RT_VISIBLE_DEVICES` | Visible NPU cards | Specify which NPU cards to use for this training run. The example enables cards 0–7 (8 cards). |
| `nproc_per_node` | Number of processes per node | Number of training processes to launch on a single node. This should match the number of visible NPU cards (e.g., 8 for 8 cards). |


### Multi‑Card Training Time Performance Comparison
The following table shows the time performance of LQC reinforcement learning training for the Unitree G1 humanoid robot on Ascend A2 (32 GB VRAM) with multiple cards:

| num_cards | num_envs (total MuJoCo environments across all cards) | env step_time (s) | observe_time (s) | agent infer_time (s) | update_time (s) | Avg training time per iteration (s) | Speedup over single‑card |
|---|-------|------|------|------|-------|-------|-----|
| 1 | 12288 | 5.14 | 1.08 | 0.67 | 11.02 | 17.93 | 1.0 |
| 2 | 12288 | 4.70 | 1.21 | 0.49 | 5.26 | 11.67 | 1.5 |
| 4 | 12288 | 4.45 | 1.45 | 0.41 | 3.49 | 9.82 | 1.8 |
| 8 | 12288 | 4.26 | 1.55 | 0.37 | 2.80 | 9.00 | 2.0 |

In the table, num_envs is defined in cann-recipes-embodied-ai/locomotion/LQC/configs/<ROBOT>/env.yml. On the Ascend A2 (32 GB VRAM), the recommended maximum num_envs per card is 12288; exceeding this may cause out‑of‑memory errors.


## Inference
This sample supports both online and offline rendering during inference. To make the best use of Ascend server performance, offline rendering is recommended for model inference validation.


### Offline Rendering
For offline rendering, we recommend installing the PyOpenGL-accelerate library:
```
pip install PyOpenGL-accelerate
```
Run the following command to enable offline rendering. It will create a subfolder under /results and record simulation data of the specified robot, generating an MP4 video file:
```
python scripts/play.py <RUN_DIR/WEIGHT_PATH> --command random --offline-render --record-video
```

For more options, see:
```
python scripts/play.py -h
```

We also provide pre‑trained weights for Unitree G1 for quick validation. [Download the weight](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-intelligence/LQC-G1-15DOF-Rough/G1_15DOF_rough.tar) and extract them into the /logs/g1_15dof.rough folder.


### Online Rendering
In online rendering mode, X11 forwarding is required to display the rendering window on your local screen. We recommend using  [MobaXterm](https://mobaxterm.mobatek.net/)to connect to the server.


If you are using SSH to connect, ensure that x11forwarding yes is set in /etc/ssh/sshd_config on the server, and that MobaXterm has X11 Forwarding enabled. Refer to the [MobaXterm documentation](https://mobaxterm.mobatek.net/documentation.html) for configuration steps.


A one‑click rendering environment setup script is provided. Run it with:
```
chmod +x set_render.sh
./set_render.sh
```

You will be prompted to enter the IP address of your local machine:
```
Please enter your Windows host IP address: <ip>
```

After the script finishes, you should have a working MuJoCo viewer. Set the Linux GUI environment variables with your local IP and run the inference command to see the robot simulation in the MuJoCo window, including robot motion, perception, and terrain environment:
```
export DISPLAY=<ip>:0.0
export LIBGL_ALWAYS_INDIRECT=0
python scripts/play.py <RUN_DIR/WEIGHT_PATH> --command random
```

<p align="middle">
  <img src="doc/images/MuJoCo_G1_demo.png" width="50%" align="top" />
</p>

## Project Structure

```bash
The overall project structure is as follows:
├── algorithms             # RL algorithm library
├── configs                # Robot training configurations (g1, g1_15dof, g1_15dof.rough, go2)
├── doc
├── extern                 # Third‑party libraries; oomj is a MuJoCo wrapper
├── lltk                   # RL environment toolkit, connecting lower‑level C++ and upper‑level Python training algorithms
├── resources              # Physical model files for robots
└── scripts                # Training and inference entry points
└── README.md
└── set_env.sh             # One‑click training environment setup script
└── set_render.sh          # One‑click online rendering environment setup script
```
## Citation
```
Chengrui Zhu, Zhen Zhang, Siqi Li, Qingpeng Li, and Yong Liu. Learning Symmetric Legged Locomotion via State Distribution Symmetrization. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025.
```







