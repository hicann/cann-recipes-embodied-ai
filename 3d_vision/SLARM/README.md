# 在昇腾Atlas A3平台上适配SLARM模型的推理

SLARM（Streaming and Language-Aligned Reconstruction Model for Dynamic Scenes）是发表于CVPR 2026的前馈式动态场景重建模型，能够从稀疏多视角序列中统一实现动态场景重建、语义理解与实时流式推理。模型联合学习**3D高斯**与**场景流**，支持实时渲染与语义分割。本项目旨在提供SLARM的NPU适配版本，方便用户能够在昇腾生态上直接使用SLARM。

SLARM在动态估计、渲染质量、场景解析等多项任务上达到**SOTA**水平，相比已有方法运动精度提升**21%**，重建PSNR提升**1.6 dB**，分割mIoU提升**20%**。

本样例支持在昇腾Atlas A3环境（910C）的推理。

## 执行样例


### 环境准备
1. 本样例采用CANN 8.2.RC1。请从[CANN软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-kernels-${chip_type}_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0007.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)进行安装。

    ```shell
    conda create -n SLARM python=3.10 -y
    conda activate SLARM
    ```
- 本样例的torch以及torch_npu版本为2.1.0，请从[Ascend Extension for PyTorch插件](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0004.html)下载torch与torch_npu安装包。请根据对应的torch和cann版本选择正确的torch_npu版本，可参考[Ascend/pytorch](https://gitcode.com/Ascend/pytorch)。可通过`cat /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info`查看CANN版本（例如：version=8.2.RC1）。
    ```shell
    # 安装PyTorch
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

    # 安装torch-npu
    pip install torch-npu==2.1.0.post13

    # 安装torch-scatter
    pip install setuptools==69.5.1
    pip install torch-scatter==2.0.9 --no-build-isolation
    ```

### 网络模型代码准备
- 下载本仓库代码
    ```shell
    git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git
    ```
- 进入本项目目录
    ```shell
    cd cann-recipes-embodied-ai/3d_vision/SLARM
    ```
- 安装Python依赖
    ```shell
    pip install -r requirements_npu.txt
    ```
- 安装渲染算子meta_gauss_render（适配torch 2.1.0与昇腾910C(A3)的特定版本）。请参考我们专门为昇腾硬件适配的3DGS代码，并从源码编译安装：[gaussian_splatting README](../gaussian_splatting/README.md)。
- 安装用于语义对齐的CLIP
    ```shell
    pip install git+https://github.com/openai/CLIP.git
    ```

### 数据集和模型权重

**Waymo数据集**
- 准备Waymo Open Dataset【全量数据集，也可使用我们提供的demo数据快速实现】，请参考[Waymo数据说明](docs/WAYMO.md)。

- 我们提供了SLARM的权重和demo数据集，用户可以以下链接获取：[权重和demo数据集](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/SLARM/SLARM.zip)。

## 执行推理
本项目已将推理所需的NPU环境变量、性能优化配置及模型参数封装在 `inference.sh` 、`evaluation.sh` 脚本中。使用我们提供的样例数据集与脚本默认配置，直接执行如下命令即可完成推理：

```bash
bash inference.sh # 出可视化结果
bash evaluation.sh # 出评测指标
```

> **提示：**
> - 使用脚本前，需要配置以下路径参数（目前为占位符）：
>   - `DATA_ROOT`：数据集根目录路径
>   - `CKPT_PTH`：SLARM checkpoint 权重路径
>   - `--lseg_model_scratch_path`：LSeg 模型 scratch 权重路径（`inference.sh` 中）
>   - `--lseg_model_pretrained_path`：LSeg 模型预训练权重路径（`inference.sh` 中）

使用我们提供的样例数据集，跑出来的评测结果如下：

```text
Average PSNR: 27.3069
Average SSIM: 0.8419
Average Depth RMSE (0.01-100m): 1.9343
Average Depth RMSE (100-200m): -1.0000
Average Depth RMSE (0.01-200m): 1.9343
Average Occupied PSNR: 27.4185
Average Occupied SSIM: 0.8366
Average Dynamic PSNR: 24.8740
Average Dynamic SSIM: 0.7683
Average Dynamic Depth RMSE (0.01-100m): 2.9485
Average Dynamic Depth RMSE (100-200m): -1.0000
Average Dynamic Depth RMSE (0.01-200m): 2.9485
Evaluated on 468 samples.
Valid depth samples (0.01-100m): 468
Valid depth samples (100-200m): 0
Valid dynamic depth samples (0.01-100m): 450
Valid dynamic depth samples (100-200m): 0


flow:
Average Flow EPE: 0.1406
Average Flow Acc Strict: 80.5886
Average Flow Acc Relax: 85.1571
Average Flow Angle: 0.3104
Average Flow RMSE: 0.2307
Evaluated on 182.0 samples.


segment:
Average Semantic mIOU: 0.6058
Average Semantic Accuracy: 0.9025
Evaluated on 21.0 samples.
```

## 与其他模型的性能对比

#### 表1：与 SOTA 方法在 WOD 数据集上的对比

我们对比了光写实性和几何指标。PSNR、SSIM 和深度 RMSE（D-RMSE）结果如下。SLARM-F 表示使用全注意力的离线模式，SLARM-W 表示使用窗口注意力的在线模式。

| 方法 | Dynamic-only |  |  | Full image† |  |  |
|------|--------------|--------------|--------------|--------------|--------------|--------------|
|  | PSNR↑ | SSIM↑ | D-RMSE↓ | PSNR↑ | SSIM↑ | D-RMSE↓ |
| LGM | 17.36 | 0.216 | 11.09 | 18.53 | 0.447 | 9.07 |
| LGM* | 19.58 | 0.443 | 9.43 | 23.59 | 0.691 | 8.02 |
| GS-LRM* | 20.02 | 0.520 | 9.95 | 25.18 | 0.753 | 7.94 |
| MapAnything | - | - | - | - | - | 13.53 |
| STORM* | 22.03 | 0.623 | 7.50 | 25.86 | 0.804 | 5.47 |
| **Ours** |  |  |  |  |  |  |
| SLARM-W | **23.20** | **0.676** | **6.38** | **27.30** | **0.825** | **4.75** |
| SLARM-F | 23.51 | 0.691 | 6.16 | 27.49 | 0.828 | 4.57 |

*注：*: 由我们复现。†: 非天空区域。

#### 表2：语义分割性能定量对比

SLARM 在所有方法中实现了最佳的 mIoU 和准确率。

| 方法 | mIoU↑ | Acc↑ |
|------|-------|------|
| EfficientViT-Seg | 0.4352 | 0.7637 |
| Mask2Former-R50 | 0.4429 | 0.7082 |
| SegMAN | 0.4567 | 0.7186 |
| SegFormer | 0.4660 | 0.7572 |
| OffSeg-B | 0.4612 | 0.7417 |
| OffSeg-L | 0.4868 | 0.7635 |
| LSeg | 0.4876 | 0.7976 |
| Mask2Former-Swin | 0.5505 | 0.8192 |
| **SLARM** | **0.6663** | **0.8923** |


## Citation
```bibtex
@InProceedings{Qiu_2026_CVPR,
    author    = {Qiu, Zhicheng and Meng, Jiarui and Luo, Tong-an and Huang, Yican and Feng, Xuan and Li, Xuanfu and Xu, Zhan},
    title     = {SLARM: Streaming and Language-Aligned Reconstruction Model for Dynamic Scenes},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {29023-29034}
}
```
