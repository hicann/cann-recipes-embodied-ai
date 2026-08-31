# Adapting the SLARM Model for Inference on the Ascend Atlas A3

SLARM (Streaming and Language-Aligned Reconstruction Model for Dynamic Scenes) is a feed-forward dynamic scene reconstruction model published at CVPR 2026. It unifies dynamic scene reconstruction, semantic understanding, and real-time streaming inference from sparse multi-view sequences. The model jointly learns **3D Gaussians** and **scene flow**, supporting real-time rendering and semantic segmentation. This project provides an NPU-adapted version of SLARM, enabling users to run SLARM directly within the Ascend ecosystem.

SLARM achieves **SOTA** performance across multiple tasks including motion estimation, rendering quality, and scene parsing. Compared with existing methods, it improves motion accuracy by **21%**, reconstruction PSNR by **1.6 dB**, and segmentation mIoU by **20%**.

This sample supports inference on the Ascend Atlas A3 environment (910C).

## Running the Sample


### Environment Setup
1. This sample uses CANN 8.2.RC1. Please download the `Ascend-cann-toolkit_${version}_linux-${arch}.run` and `Ascend-cann-kernels-${chip_type}_${version}_linux-${arch}.run` packages from the [CANN download page](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1), and follow the [CANN installation guide](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0007.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit) to install them.

    ```shell
    conda create -n SLARM python=3.10 -y
    conda activate SLARM
    ```
- This sample uses torch and torch_npu version 2.1.0. Please download the torch and torch_npu installation packages from the [Ascend Extension for PyTorch plugin](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0004.html). Select the correct torch_npu version according to your torch and CANN versions; refer to [Ascend/pytorch](https://gitcode.com/Ascend/pytorch). You can check the CANN version with `cat /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info` (e.g., version=8.2.RC1).
    ```shell
    # Install PyTorch
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

    # Install torch-npu
    pip install torch-npu==2.1.0.post13

    # Install torch-scatter
    pip install setuptools==69.5.1
    pip install torch-scatter==2.0.9 --no-build-isolation
    ```

### Preparing the Model Code
- Clone this repository
    ```shell
    git clone https://gitcode.com/cann/cann-recipes-embodied-ai.git
    ```
- Enter the project directory
    ```shell
    cd cann-recipes-embodied-ai/3d_vision/SLARM
    ```
- Install Python dependencies
    ```shell
    pip install -r requirements_npu.txt
    ```
- Install the rendering operator meta_gauss_render (a specific version adapted for torch 2.1.0 and Ascend 910C (A3)). Please refer to our 3DGS code specifically adapted for Ascend hardware, and build and install it from source: [gaussian_splatting README](../gaussian_splatting/README.md).
- Install CLIP for semantic alignment
    ```shell
    pip install git+https://github.com/openai/CLIP.git
    ```

### Datasets and Model Weights

**Waymo Dataset**
- Prepare the Waymo Open Dataset [the full dataset, or you can use our provided demo data for a quick start]. Please refer to the [Waymo data guide](docs/WAYMO.md).

- We provide the SLARM weights and demo dataset, which users can obtain from the following link: [weights and demo dataset](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-recipes-embodied-ai/SLARM/SLARM.zip).

## Running Inference
This project encapsulates the NPU environment variables, performance optimization settings, and model parameters required for inference in the `inference.sh` and `evaluation.sh` scripts. Using our provided sample dataset and the default script configuration, you can complete inference by simply running the following commands:

```bash
bash inference.sh # produce visualization results
bash evaluation.sh # produce evaluation metrics
```

> **Note:**
> - Before using the scripts, you need to configure the following path parameters (currently placeholders):
>   - `DATA_ROOT`: root directory path of the dataset
>   - `CKPT_PTH`: path to the SLARM checkpoint weights
>   - `--lseg_model_scratch_path`: path to the LSeg model scratch weights (in `inference.sh`)
>   - `--lseg_model_pretrained_path`: path to the LSeg model pretrained weights (in `inference.sh`)

Using our provided sample dataset, the evaluation results are as follows:

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

## Performance Comparison with Other Models

#### Table 1: Comparison with SOTA methods on the WOD dataset

We compare photorealistic and geometric metrics. The PSNR, SSIM, and depth RMSE (D-RMSE) results are shown below. SLARM-F denotes the offline mode using full attention, and SLARM-W denotes the online mode using window attention.

| Method | Dynamic-only |  |  | Full image† |  |  |
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

*Notes:* \*: reproduced by us. †: non-sky regions.

#### Table 2: Quantitative comparison of semantic segmentation performance

SLARM achieves the best mIoU and accuracy among all methods.

| Method | mIoU↑ | Acc↑ |
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
