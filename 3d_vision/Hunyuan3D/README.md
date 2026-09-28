# 在昇腾Atlas A2平台上适配Hunyuan3D 2.0 模型的推理

Hunyuan3D模型是腾讯混元系列在2025年推出的一款3D资产创作模型，用于生成带有高分辨率纹理贴图的高保真度3D模型，本项目旨在提供Hunyuan3D的NPU适配版本，方便用户能够在昇腾生态上直接使用Hunyuan3D。

此外，本样例基于Hunyuan3D模型在NPU进行了性能优化，目前texgen在2万平面mesh网格输入下，推理时间降至26秒。详细内容可至[性能优化章节](../../docs/3d_vision/Hunyuan3D/Hunyuan3D_optimization.md)进行查看。

本样例支持在昇腾Atlas A2环境的单机单卡推理。

## 执行样例


### 环境准备
1. 本样例采用CANN 8.2.RC1。请从[CANN软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-kernels-${chip_type}_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0007.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)进行安装。

    ```shell
    conda create -n hunyuan3d python==3.10.18
    conda activate hunyuan3d
    ```
- 本样例的torch以及torch_npu版本为2.6，请从[Ascend Extension for PyTorch插件](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0004.html)下载torch与torch_npu安装包，同时指定decorator版本
    ```shell
    pip install torch==2.6.0
    pip install torchvision==0.21.0
    pip install torch-npu==2.6.0.post3
    pip install decorator==5.2.1
    pip install ninja
    ```
### 网络模型代码准备
- 本仓库依赖[Hunyuan3D](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)的开源仓库代码。
- 进入Hunyuan3D的官方仓库，下载Hunyuan3D模型网络结构代码
    ```shell
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
    ```
- 下载本仓库代码
    ```shell
    git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git
    ```

- 将Hunyuan3D仓库的网络模型文件以非覆盖模式复制到本项目目录下。其中下方命令里的```/path/to/Hunyuan3D-2/```改为Hunyuan3D-2文件路径。
    ```shell
    cd cann-recipes-embodied-ai/3d_vision/Hunyuan3D
    cp -rn /path/to/Hunyuan3D-2/* ./ 
    ```

- 安装Python依赖
    ```shell
    pip install -r requirements.txt
    ```
- 编译第三方代码
    ```shell
    cd hy3dgen/texgen/differentiable_renderer
    python3 setup.py install
    ```

- 模型结构修改部分如下所示
    ```  
    Hunyuan3D
        +--- hy3dgen
            +--- texgen
                +--- custom_rasterizer ==> custom_rasterizer in current depository #替换当前文件
                +--- differentiable_renderer
                    +--- mesh_render.py ==> differentiable_renderer/mesh_render.py #替换文件
                    +--- rasterizer.py #添加新文件
        +--- minimal_demo_npu.py #添加新文件

    ```


### 权重模型
NPU环境支持Hunyuan3D模型列表，用户可根据需要下载蒸馏版本：


**Hunyuan3D-2mv 系列**

| Model                     | Description                    | Date       | Size | Huggingface                                                                                  |
|---------------------------|--------------------------------|------------|------|----------------------------------------------------------------------------------------------| 
| Hunyuan3D-DiT-v2-mv       | Multiview Image to Shape Model | 2025-03-18 | 1.1B | [Download](https://huggingface.co/tencent/Hunyuan3D-2mv/tree/main/hunyuan3d-dit-v2-mv)       |

**Hunyuan3D-2 系列**

| Model                      | Description                 | Date       | Size | Huggingface                                                                               |
|----------------------------|-----------------------------|------------|------|-------------------------------------------------------------------------------------------| 
| Hunyuan3D-DiT-v2-0         | Image to Shape Model        | 2025-01-21 | 1.1B | [Download](https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-dit-v2-0)         |
| Hunyuan3D-Paint-v2-0       | Texture Generation Model    | 2025-01-21 | 1.3B | [Download](https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-paint-v2-0)       |
| Hunyuan3D-Delight-v2-0     | Image Delight Model         | 2025-01-21 | 1.3B | [Download](https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-delight-v2-0)     | 

- 权重安装地址为```base_dir+model_path```，默认```base_dir```为```~/.cache/hy3dgen/```，如需修改```base_dir```，需将```./hy3dgen/shapegen/utils.py```中第97行和```./hy3dgen/texgen/pipelines.py```中第61行进行修改
    ```python
    class Hunyuan3DPaintPipeline:
        @classmethod
        def from_pretrained(cls, model_path, subfolder='hunyuan3d-paint-v2-0-turbo'):
            original_model_path = model_path
            if not os.path.exists(model_path):
                # try local path
                base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen/')
                model_path = os.path.expanduser(os.path.join(base_dir, model_path))
    ```

- 最后模型权重路径为
    ```  
    base_dir+model_path
        +--- hunyuan3d-dit-v2-0 #单视角图像DIT模型
        +--- hunyuan3d-dit-v2-mv #多视角图像DIT模型
        +--- hunyuan3d-delight-v2-0 
        +--- hunyuan3d-paint-v2-0
    ```

## 执行推理
### 执行推理
由于本项目位于 `3d_vision/Hunyuan3D` 目录下，并依靠仓库根目录下 `module/` 提供的公共优化模块，因此在开始推理前，需要设置环境变量，执行如下指令
```bash
source set_env.sh
```
即可将当前样例目录及仓库根目录加入环境变量中，之后执行
```bash
python minimal_demo_npu.py
```
``` minimal_demo_npu.py```采用默认设置执行单图像推理，输入图片默认读取当前目录下的 `assets/demo.png`，生成结果默认输出到当前目录下的 `outputs/`。运行不同配置可参考以下脚本执行：

```bash
python minimal_demo_npu.py --model_path tencent/Hunyuan3D-2 --multiview --face_reduce --full_graph --multi_thread --save_render
```
```model_path``` 选择模型路径，```multiview``` 设置是否采用多视角推理，```face_reduce``` 设置是否减少三角面片，```full_graph```设置是否采用图模式，```multi_thread```设置是否采用多线程并行执行光栅化，```use_render_npu```设置是否采用npu方式执行光栅化（不能与```multi_thread```共同使用），```save_render```设置是否复用光栅化结果。

### 推理步骤介绍

Hunyuan3D源码参考```diffusers```接口API，用于物体生成，包括**Hunyuan3D-DiT**和**Hunyuan3D-Paint**两个部分。

运行**Hunyuan3D-DiT**方式如下：
```python
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from torch_npu.contrib import transfer_to_npu

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(image='assets/demo.png')[0]
```

运行**Hunyuan3D-Paint**方式如下：
```python
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from torch_npu.contrib import transfer_to_npu

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(image='assets/demo.png')[0]

pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(mesh, image='assets/demo.png')
```
**Hunyuan3D-DiT**执行过程中，我们在NPU环境下提供多视角模型推理，具体方式如下：
```python
import torch
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from torch_npu.contrib import transfer_to_npu

images = {
    "front": "assets/example_mv_images/1/front.png",
    "left": "assets/example_mv_images/1/left.png",
    "back": "assets/example_mv_images/1/back.png"
}

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mv',
    subfolder='hunyuan3d-dit-v2-mv',
    variant='fp16'
)
mesh = pipeline(
    image=images,
    num_inference_steps=50,
    octree_resolution=380,
    num_chunks=20000,
    generator=torch.manual_seed(12345),
    output_type='trimesh'
)[0]

pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(mesh, image=list(images.values()))
```

## Citation
```
@misc{lai2025hunyuan3d25highfidelity3d,
      title={Hunyuan3D 2.5: Towards High-Fidelity 3D Assets Generation with Ultimate Details}, 
      author={Tencent Hunyuan3D Team},
      year={2025},
      eprint={2506.16504},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.16504}, 
}

@misc{hunyuan3d22025tencent,
    title={Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation},
    author={Tencent Hunyuan3D Team},
    year={2025},
    eprint={2501.12202},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{yang2024hunyuan3d,
    title={Hunyuan3D 1.0: A Unified Framework for Text-to-3D and Image-to-3D Generation},
    author={Tencent Hunyuan3D Team},
    year={2024},
    eprint={2411.02293},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
