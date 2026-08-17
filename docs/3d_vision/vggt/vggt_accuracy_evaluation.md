# VGGT精度测量
由于原论文并未提供全部的精度评测脚本，本样例中的相关评测脚本基于论文介绍、其他基于VGGT的论文开源代码以及相关开发人员的理解进行实现。

---
## 数据集准备
- 论文在相机位姿估计、点云三维重建和深度估计这三个任务上分别使用了Co3D数据集、ETH3D数据集和DTU数据集，按照下列数据集链接下载对应数据集
  - Co3D 数据集下载链接：[Co3D](https://ai.meta.com/datasets/co3d-downloads/)
    - 考虑到Co3D数据集规模庞大，可按需下载：直接选取[SEEN_CATEGORIES](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/3d_vision/vggt/eval/pose_evaluation/dataset_prepare/categories.py)列表中列出的类别，或直接编辑该列表，以精准控制数据规模。
  - ETH3D 数据集下载链接：[ETH3D](https://www.eth3d.net/datasets)
    1. 下载High-res multi-view下Training data中的[multi_view_training_dslr_jpg](https://www.eth3d.net/data/multi_view_training_dslr_jpg.7z)。
    2. 下载High-res multi-view下Training data中**各类别**的真值深度值数据，如courtyard类别下载[courtyard_dslr_depth](https://www.eth3d.net/data/courtyard_dslr_depth.7z)。
  - DTU 数据集下载链接：[1. 原始DTU数据集](https://roboimagedata.compute.dtu.dk/?page_id=36) 、 [2. 预处理过的DTU数据](https://aistudio.baidu.com/datasetdetail/129222)、[3. Depths_raw数据](https://aistudio.baidu.com/datasetdetail/207802)
    - 从链接1中下载[Points](http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip)和[SampleSet](http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip)，解压SampleSet中数据，提取ObsMask。
    - 从链接2中下载`dtu_testing.zip`。
    - 从链接3中下载`Depths.zip`，解压后重命名为Depths_raw。
- 下载好的数据集分别放至datasets目录下，datasets文件夹在文件目录中的结果如下
  ```
  VGGT
    +--- datasets
          +--- co3d
                +--- apple_000.zip
                +--- apple_001.zip
                ...
          +--- multi_view_training_dslr_jpg
                +--- courtyard
                      +--- dslr_calibration_jpg
                      +--- images
                      +--- ground_truth_depth
                +--- delivery_area
                ...
          +--- dtu_testing
                +--- Depths_raw
                +--- Points
                +--- ObsMask
                +--- scan1
                ...
    +--- eval
    +--- demo_gradio.py
    +--- demo_infer.py
    +--- ckpt
    +--- vggt
  ```

---
## 测量程序运行
### 相机位姿估计
1. **数据预处理**：解压各类别数据放至`$VGGT_DIR/datasets/co3d/co3d_data/`目录下，并为各类别数据生成annotation文件。
   ```shell
   export VGGT_DIR=$(pwd)
   cd eval/pose_evaluation/dataset_prepare
   python preprocess_co3d.py --category all --co3d_v2_dir $VGGT_DIR/datasets/co3d/co3d_data/ --output_dir $VGGT_DIR/datasets/co3d/co3d_anno/ 
   ```
   当`datasets/co3d`目录结构显示如下，则说明数据预处理成功。
    ```
    VGGT
      +--- datasets
            +--- co3d
                  +--- co3d_data
                        +--- apple
                        +--- backpack
                        ....
                  +--- co3d_anno
                        +--- apple_train.jgz
                        +--- apple_test.jgz
                        ... 
                  +--- apple_000.zip
                  +--- apple_001.zip
                  ...
    ```
2. **精度评测脚本运行**：运行eval_co3d.py进行相机位姿估计任务精度测量。
   ```shell
   export VGGT_DIR=$(pwd)
   # use bf16 model
   cd eval/pose_evaluation
   python eval_co3d.py --co3d_dir $VGGT_DIR/datasets/co3d/co3d_data/ --co3d_anno_dir $VGGT_DIR/datasets/co3d/co3d_anno/  --ckpt $VGGT_DIR/ckpt/model.pt
   
   # use bf16_sp model
   cd eval/pose_evaluation
   torchrun --nproc_per_node=1 eval_co3d.py \
    --ckpt $VGGT_DIR/ckpt/model.pt \
    --co3d_dir $VGGT_DIR/datasets/co3d/co3d_data/ \
    --co3d_anno_dir $VGGT_DIR/datasets/co3d/co3d_anno/ \
    --use_sp \
    --ulysses_degree 1 \
    --ring_degree 1
    
   # use int8 model
   cd eval/pose_evaluation
   python eval_co3d.py --co3d_dir $VGGT_DIR/datasets/co3d/co3d_data/ --co3d_anno_dir $VGGT_DIR/datasets/co3d/co3d_anno/  --ckpt ../../VGGT_model_W8A8.pt --enableW8A8
   ```
### 点云三维重建
1. **点云重建评测脚本运行**：运行eval_eth3d.py 进行点云三维重建任务精度测量
  ```shell
  export VGGT_DIR=$(pwd)
  cd eval/point_map_estimation
  # use bf16 model
  python eval_eth3d.py --ckpt $VGGT_DIR/ckpt/model.pt --dataset_dir $VGGT_DIR/datasets/multi_view_training_dslr_jpg
  
 # use bf16_sp model
   cd eval/point_map_estimation
   torchrun --nproc_per_node=1 eval_eth3d.py \
    --ckpt $VGGT_DIR/ckpt/model.pt \
	--dataset_dir $VGGT_DIR/datasets/multi_view_training_dslr_jpg
    --use_sp \
    --ulysses_degree 1 \
    --ring_degree 1
    
  # use int8 model
  python eval_eth3d.py --ckpt ../../VGGT_model_W8A8.pt --dataset_dir $VGGT_DIR/datasets/multi_view_training_dslr_jpg --enableW8A8
  ```

### 深度估计
1. **深度图生成**：运行VGGT模型，根据各输入图片生成对应深度图。
   ```shell
    export VGGT_DIR=$(pwd)
    cd eval/depth_estimation
    # 深度图输出结果将保存在$VGGT_DIR/outputs文件夹下(bf16模型)
    python eval_dtu.py --testlist dataset_utils/lists/test.txt --testpath $VGGT_DIR/datasets/dtu_testing/ --ckpt $VGGT_DIR/ckpt/model.pt #使用全量数据集
    # python eval_dtu.py --testlist dataset_utils/lists/sub_test.txt --testpath $VGGT_DIR/datasets/dtu_testing/ --ckpt $VGGT_DIR/ckpt/model.pt #使用部分数据集
    # 深度图输出结果将保存在$VGGT_DIR/outputs文件夹下(in8模型)
    python eval_dtu.py --testlist dataset_utils/lists/test.txt --testpath $VGGT_DIR/datasets/dtu_testing/ --ckpt ../../VGGT_model_W8A8.pt --enableW8A8 #使用全量数据集
    # python eval_dtu.py --testlist dataset_utils/lists/sub_test.txt --testpath $VGGT_DIR/datasets/dtu_testing/ --ckpt ../../VGGT_model_W8A8.pt --enableW8A8 #使用部分数据集
    # use sp model
   cd eval/point_map_estimation
   torchrun --nproc_per_node=1 eval_dtu.py \
    --testlist dataset_utils/lists/test.txt 
    --testpath $VGGT_DIR/datasets/dtu_testing/ 
    --ckpt $VGGT_DIR/ckpt/model.pt
    --use_sp \
    --ulysses_degree 1 \
    --ring_degree 1
   ```

2. **深度图数值分布对齐**：由于VGGT模型基于归一化后的图像进行训练的，因此得到的深度图取值范围与真实深度图取值范围不接近，因此需要将生成的深度图与真值进行对齐。
   ```shell
   python align_depth.py --testlist dataset_utils/lists/test.txt --output_path ./outputs --depth_conf_thres 3 #使用全量数据集
   # python align_depth.py --testlist dataset_utils/lists/sub_test.txt --output_path ./outputs --depth_conf_thres 3 #使用部分数据集
   ```
3. **点云生成**：通过融合场景下多视角的预测深度图，生成该场景的点云。
   ```shell
   python _open3d_fusion.py --data_list dataset_utils/lists/test.txt --depth_path ./outputs #使用全量数据集
   # python _open3d_fusion.py --data_list dataset_utils/lists/sub_test.txt --depth_path ./outputs #使用部分数据集

   ```
4. **计算精度**：通过计算由深度图融合得到的点云与真实点云之间的距离，衡量深度估计任务的精度。
   ```shell
   python measure.py --gt_dir $VGGT_DIR/datasets/dtu_testing --pred_dir outputs/dtu/open3d_fusion_plys/ --data_list dataset_utils/lists/test.txt #使用全量数据集
   # python measure.py --gt_dir $VGGT_DIR/datasets/dtu_testing --pred_dir outputs/dtu/open3d_fusion_plys/ --data_list dataset_utils/lists/sub_test.txt #使用部分数据集
   ```
---
## 精度测量结果
下文分别给出原始VGGT模型在GPU上和优化后的VGGT模型在NPU上的任务精度结果。
### 相机位姿估计
- 论文中采用**AUC@30**指标来衡量相机位姿的估计结果。
  - **AUC@30**是相机位姿误差 $\leq$ 30°的累积准确率，AUC@30越大，说明在0~30°误差范围内，模型正确估计相机位姿的累积准确率越高。
- 实验结果表明，VGGT模型在NPU上进行相机位姿估计的精度结果与论文结果基本一致。
  ||AUC@30|
  |:---:|:---:|
  |论文实验数据|88.2|
  |NPU运行结果|91.1|
   |GPU运行结果|90.5|
  <!-- |NPU运行结果|91.3|85.3 -->
### 点云三维重建
- 论文中采用**Accuracy (Acc)、Completeness（Comp）、Average Overall**三项指标衡量点云三维重建的精度结果。
  - **Acc**指的是预测点云中所有点到真值点云上最近邻的平均距离。
  - **Comp**指的是真值点云中所有点到预测点云上最近邻的平均距离。
  - **Overall**则是通过 (Acc+Comp)/2进行计算。
  - 这三项指标的数值越低，说明重建精度越高。
- 实验结果表明，本评测程序在GPU和NPU设备上输出的重建精度基本一致。
  ||Acc|Comp|Overall|
  |:---:|:---:|:---:|:---:|
  |论文实验数据|0.873|0.482|0.677|
  |NPU实验数据|0.491|0.422|0.456|
  |GPU实验数据|0.498|0.429|0.464|
  <!-- |NPU实验数据|0.485|0.443|0.464| -->
### 深度估计
- 论文对该任务的精度评测是将场景下多视角的深度图进行融合，得到重建点云。通过比较重建点云与真实点云之间的距离衡量深度估计的精度结果。因此这里用到的依旧是**Accuracy (Acc)、Completeness（Comp）、Average Overall**三项指标。这三项指标的数值越低，说明重建精度越高。
- 实验结果表明，本评测程序在GPU和NPU设备上输出的重建精度基本一致。
  
  ||Acc|Comp|Overall|
  |:---:|:---:|:---:|:---:|
  |论文实验数据|0.389|0.374|0.382|
  |NPU实验数据|1.4669|0.5475|1.0072|
  |GPU实验数据|1.4675|0.5813|1.0244|
  <!-- |NPU实验数据|1.4702|0.5823|1.0262| -->
- **已知问题**：由于论文官方仓库中为开源相关评测代码，复现过程主要参考官方仓库相关[issue](https://github.com/facebookresearch/vggt/issues/208)，目前在该任务上的精度复现结果与论文结果存在一定差异，而与官方仓库相关讨论中其他人复现的结果接近（[结果讨论1](https://github.com/facebookresearch/vggt/issues/208#issuecomment-3172136901)，[结果讨论2](https://github.com/facebookresearch/vggt/issues/208#issuecomment-3128037216)）。其原因在于复现过程和论文实验设置上存在细微区别，后续会继续根据作者公布的细节或代码进一步调整该任务的评测程序。