import argparse
from src.dataset.constants import DATASET_DICT
import torch

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass


def get_args_parser():
    parser = argparse.ArgumentParser("PerceptualModel training", add_help=False)
    _add_model_core_args(parser)
    _add_model_head_and_render_args(parser)
    _add_render_loss_args(parser)
    _add_special_and_reg_loss_args(parser)
    _add_optimization_args(parser)
    _add_dataset_args(parser)
    _add_infra_args(parser)

    return parser


def _add_model_core_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="slarm", type=str)
    parser.add_argument("--num_context_timesteps", default=4, type=int)
    parser.add_argument("--num_target_timesteps", default=4, type=int)
    parser.add_argument("--gs_dim", default=3, type=int, help="Number of gs dimensions")
    parser.add_argument("--use_time_token", action="store_true")
    parser.add_argument("--use_sky_token", action="store_true")
    parser.add_argument("--use_affine_token", action="store_true")
    parser.add_argument("--pred_gs_conf", action="store_true")
    parser.add_argument("--enable_lifespan", action="store_true")
    parser.add_argument("--voxelize", action="store_true")
    parser.add_argument("--voxel_size", type=float, default=0.2)
    parser.add_argument("--sigmoid_rgb", action="store_true")
    parser.add_argument("--decoder_type", type=str, choices=["dummy", "conv"], default="dummy",
                        help="XXX or LatentXXX")
    parser.add_argument("--num_motion_tokens", default=16, type=int, help="Number of motion tokens")
    parser.add_argument("--use_pred_camera_pose", action="store_true")
    parser.add_argument("--use_pred_depth", action="store_true")

    parser.add_argument("--add_patch_plucker_embed", action="store_true")
    parser.add_argument("--add_camera_embed", action="store_true")
    parser.add_argument("--concat_plucker_embed", action="store_true")
    parser.add_argument("--use_last_token", action="store_true")

    parser.add_argument("--vggt_pretrained_weight_filepath", type=str, default="",
                        help="filepath of vggt pre-trained model ckpts")

    parser.add_argument("--embed_dim", type=int, default=1024, help="token embedding dimension")
    parser.add_argument("--depth", type=int, default=24, help="model attention layers number")
    parser.add_argument("--patch_size", type=int, default=14, help="token patchify size")
    parser.add_argument("--patch_embed", type=str, default="dinov2_vitl14_reg",
                        help="image token patchify model")


def _add_model_head_and_render_args(parser: argparse.ArgumentParser) -> None:
    # model
    parser.add_argument("--enable_depth_head", action="store_true")
    parser.add_argument("--enable_camera_head", action="store_true")
    parser.add_argument("--enable_point_head", action="store_true")
    parser.add_argument("--shortcut_rgb", action="store_true")

    parser.add_argument("--use_ms3_motion", action="store_true")
    parser.add_argument("--add_angular_velocity", action="store_true")
    parser.add_argument("--max_scale", type=float, default=0.5)
    parser.add_argument("--gs_marbles", action="store_true")
    parser.add_argument("--use_2dgs", action="store_true")
    parser.add_argument("--pesudo_3dgs", action="store_true")
    parser.add_argument("--save_gaussian", action="store_true")
    parser.add_argument("--gaussian_save_path", type=str, default="output_gs")
    parser.add_argument("--save_rendered_pc", action="store_true")
    parser.add_argument("--rendered_pc_save_path", type=str, default="output_rendered_pc")
    parser.add_argument("--render_context_view", action="store_true")
    parser.add_argument("--render_context_frame_contribution", action="store_true")
    parser.add_argument("--without_feat", action="store_true", help="pred gs without semantic feature")

    parser.add_argument("--use_render_novel_view", action="store_true")

    parser.add_argument("--lseg_model_pretrained_path", type=str,
                        default="ckpts/lseg/lseg_model_pretrained.pth",
                        help="filepath of lseg pre-trained vit model ckpts")
    parser.add_argument("--lseg_model_scratch_path", type=str,
                        default="ckpts/lseg/lseg_model_scratch.pth",
                        help="filepath of lseg pre-trained scratch model ckpts")


def _add_render_loss_args(parser: argparse.ArgumentParser) -> None:
    """添加损失函数及权重相关参数。"""
    parser.add_argument("--rgb_loss_coeff", type=float, default=1.0)
    parser.add_argument("--enable_context_rgb_loss", action="store_true")
    parser.add_argument("--context_rgb_loss_coeff", type=float, default=0.1)

    parser.add_argument("--enable_depth_loss", action="store_true")

    parser.add_argument("--enable_pseudo_depth_loss", action="store_true")
    parser.add_argument("--pseudo_depth_coeff", type=float, default=0.1)

    parser.add_argument("--enable_context_depth_loss", action="store_true")
    parser.add_argument("--context_depth_loss_coeff", type=float, default=0.1)
    parser.add_argument("--context_depth_loss_with_conf", action="store_true")

    parser.add_argument("--enable_context_point_loss", action="store_true")
    parser.add_argument("--context_point_loss_coeff", type=float, default=0.1)
    parser.add_argument("--context_point_loss_with_conf", action="store_true")

    parser.add_argument("--enable_feat_loss", action="store_true")
    parser.add_argument("--enable_context_feat_loss", action="store_true")
    parser.add_argument("--feat_loss_coeff", type=float, default=1.0)
    parser.add_argument(
        "--feat_loss_type",
        type=str,
        choices=["mse", "cos_dist", "cls_prob"],
        default="mse",
        help="MSE or Cosine Distance",
    )


def _add_special_and_reg_loss_args(parser: argparse.ArgumentParser) -> None:
    # Option 1: push the sky depth to a fixed value
    parser.add_argument("--enable_sky_depth_loss", action="store_true")
    parser.add_argument("--sky_depth", type=float, default=300.0)
    # Option 2: make sky gaussians transparent and use a sky token to represent sky
    parser.add_argument("--enable_sky_opacity_loss", action="store_true")
    parser.add_argument("--sky_opacity_loss_coeff", type=float, default=0.1)
    parser.add_argument("--enable_context_sky_opacity_loss", action="store_true")
    parser.add_argument("--context_sky_opacity_loss_coeff", type=float, default=0.01)

    parser.add_argument("--enable_camera_loss", action="store_true")
    parser.add_argument("--camera_loss_coeff", type=float, default=5.0)

    parser.add_argument("--context_prediction_loss_warmup_steps", type=int, default=0)

    # long lifespan regularization loss
    parser.add_argument("--enable_lifespan_reg_loss", action="store_true")
    parser.add_argument("--lifespan_reg_coeff", type=float, default=0.005)

    # flow regularization loss
    parser.add_argument("--enable_flow_reg_loss", action="store_true")
    parser.add_argument("--flow_reg_coeff", type=float, default=0.005)

    # flow loss
    parser.add_argument("--enable_flow_loss", action="store_true")
    parser.add_argument("--flow_coeff", type=float, default=0.1)
    parser.add_argument("--flow_loss_start_iter", default=10000, type=int)

    # perceptual loss
    parser.add_argument("--enable_perceptual_loss", action="store_true")
    parser.add_argument("--perceptual_weight", default=0.05, type=float, help="LPIPS weight")
    parser.add_argument("--perceptual_loss_start_iter", default=5000, type=int)
    parser.add_argument("--lpips_model", type=str, default="VGG16", choices=["VGG16", "VGG19"])


def _add_optimization_args(parser: argparse.ArgumentParser) -> None:
    """添加优化器、学习率及训练流控制相关参数。"""
    parser.add_argument("--lr", type=float, default=4e-4, help="learning rate (absolute lr)")
    parser.add_argument("--blr", type=float, default=8e-4, help="base learning rate")
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--lr_sched", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--warmup_iters", type=int, default=5000, help="iters to warmup LR")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=3.0, help="Gradient clip")
    parser.add_argument("--disable_grad_checkpointing", action="store_true")

    parser.add_argument("--start_iteration", default=0, type=int, help="start iteration")
    parser.add_argument("--num_iterations", default=200_000, type=int, help="num of iterations")
    parser.add_argument("--resume_from", default=None, help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--load_from", type=str, default=None)


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    """添加数据集加载及预处理相关参数。"""
    parser.add_argument("--data_root", default="./data/SLARM_data", type=str, help="dataset path")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size per GPU")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--input_size", default=(160, 240), type=int, nargs=2)
    parser.add_argument("--num_max_cameras", type=int, default=3)
    parser.add_argument("--timespan", type=float, default=2.0)
    parser.add_argument("--load_ground", action="store_true")
    parser.add_argument("--load_semantic_label", action="store_true")
    parser.add_argument("--load_depth", action="store_true")
    parser.add_argument("--load_flow", action="store_true")
    parser.add_argument("--dataset", default=["waymo"], type=str, nargs='+', choices=DATASET_DICT.keys())
    parser.add_argument("--load_pseudo_depth", action="store_true")
    parser.add_argument("--subset_ratio", default=1.0, type=float)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--skip_sky_mask", action="store_true", help="skip sky mask loading")
    parser.add_argument("--online_feat", action="store_true", help="online feature extraction")
    parser.add_argument("--img_norm_for_online_feat", action="store_true",
                        help="image normalization for online feature extraction")


def _add_infra_args(parser: argparse.ArgumentParser) -> None:
    """添加日志、监控、硬件以及评估相关的基础设施参数。"""
    parser.add_argument("--output_dir", default="./work_dirs")
    parser.add_argument("--num_vis_samples", type=int, default=1)
    parser.add_argument("--log_every_n_iters", type=int, default=50)
    parser.add_argument("--vis_every_n_iters", type=int, default=5000)
    parser.add_argument("--ckpt_every_n_iters", type=int, default=5000)
    parser.add_argument("--eval_every_n_iters", type=int, default=50000000000)
    parser.add_argument("--total_elapsed_time", type=float, default=0.0, help="total time elapsed")
    parser.add_argument("--keep_n_ckpts", default=1, type=int)
    parser.add_argument("--profiling_name", default="profiling")

    # ============= Miscellaneous ============= #
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--visualization_only", action="store_true")

    # ============= WandB ============= #
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--project", default="debug", type=str)
    parser.add_argument("--entity", default="YOUR_ENTITY", type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--overwrite_wandb", action="store_true")

    # ============= TensorBoard ============= #
    parser.add_argument("--enable_tensorboard", action="store_true")

    # ============= Evaluation ============= #
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--similarity_probs_threshold", type=float, default=0.2)

    # ============= TensorBoard ============= #
    parser.add_argument("--mode", type=str, default="full",
                        help="Attention mode: 'full' = full attention, 'causal' = causal attention, "
                             "'window_N' = sliding window attention with window size N (e.g., 'window_3').")
