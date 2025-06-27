import logging
import os
import argparse
import logging
from contextlib import nullcontext

import torch
import numpy as np
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel
from diffusers.models.controlnet import ControlNetModel
from diffusers.utils import check_min_version

from pipeline import SDaIGControlNetPipeline
from data import build_dataset_from_cfg
from utils.img_utils import (
    concat_6_views,
    disparity2depth,
    concat_and_visualize_6_depths,
    set_inf_to_max,
)
from utils.nvs_utils import render_novel_view
from third_party.Lotus.utils.seed_all import seed_all

check_min_version('0.28.0.dev0')

def parse_args():
    '''Set the Args'''
    parser = argparse.ArgumentParser(
        description="Simultaneous Depth and Image Generation..."
    )
    # model settings
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="pretrained model path from hugging face or local dir",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Config file for the model.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="sample",
        help="The used prediction_type. ",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=999,
    )
    parser.add_argument(
        "--max_timesteps",
        type=int,
        default=1000,
        help="Maximum timesteps for multi-step diffusion training.",
    )
    parser.add_argument(
        "--multi_step",
        action="store_true",
        help="Enable multi-step diffusion.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps to use during validation when multi-step training is enabled.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    # inference settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`. Default: `bilinear`",
    )

    args = parser.parse_args()

    return args

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Run inference...")

    args = parse_args()

    # -------------------- Preparation --------------------
    # Random seed
    if args.seed is not None:
        seed_all(args.seed)

    # Output directories
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output dir = {args.output_dir}")

    output_dir = args.output_dir
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # half_precision
    if args.half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32

    # processing_res
    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"Device = {device}")

    # -------------------- Data --------------------
    dataset_cfg = OmegaConf.load(args.dataset_config)
    dataset = build_dataset_from_cfg(dataset_cfg.data.inference)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        pin_memory_device=device,
    )
    # -------------------- Model --------------------

    # Check if loading from a checkpoint directory with separate unet/controlnet folders
    checkpoint_unet_path = os.path.join(args.pretrained_model_name_or_path, "unet")
    checkpoint_controlnet_path = os.path.join(
        args.pretrained_model_name_or_path, "controlnet"
    )

    if os.path.exists(checkpoint_unet_path) and os.path.exists(
        checkpoint_controlnet_path
    ):
        # Load base pipeline from original model (jingheya/lotus-depth-g-v2-1-disparity)
        base_model_path = "jingheya/lotus-depth-g-v2-1-disparity"
        logging.info(f"Loading base pipeline from {base_model_path}")
        pipeline = SDaIGControlNetPipeline.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
        )

        # Load fine-tuned UNet
        logging.info(f"Loading fine-tuned UNet from {checkpoint_unet_path}")

        fine_tuned_unet = UNet2DConditionModel.from_pretrained(
            checkpoint_unet_path,
            torch_dtype=dtype,
        )
        pipeline.unet = fine_tuned_unet

        # Load fine-tuned ControlNet
        logging.info(f"Loading fine-tuned ControlNet from {checkpoint_controlnet_path}")
        fine_tuned_controlnet = ControlNetModel.from_pretrained(
            checkpoint_controlnet_path,
            torch_dtype=dtype,
        )
        pipeline.controlnet = fine_tuned_controlnet

        logging.info(
            f"Successfully loaded fine-tuned models from {args.pretrained_model_name_or_path}"
        )
    else:
        # Loading from pretrained model (original behavior)
        logging.info(
            f"Loading pretrained pipeline from {args.pretrained_model_name_or_path}"
        )
        pipeline = SDaIGControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=dtype,
        )

        if pipeline.controlnet is None:
            logging.info("Initializing ControlNet weights from unet")
            pipeline.controlnet = ControlNetModel.from_unet(
                pipeline.unet, conditioning_channels=4
            )
        logging.info(
            f"Successfully loading pipeline from {args.pretrained_model_name_or_path}."
        )
    logging.info(f"processing_res = {processing_res or pipeline.default_processing_resolution}")

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for frame, data in enumerate(tqdm(dataloader)):
            data = {k: v.to(device) for k, v in data.items()}
            pixel_values_rendered = render_novel_view(
                (data[("pixel_values", 0)].squeeze() + 1) / 2,
                set_inf_to_max(
                    disparity2depth((data[("disparity_maps", 0)].squeeze() + 1) / 2)
                ),
                data["ego_masks"].squeeze(),
                data["intrinsics"].squeeze(),
                data["extrinsics", 0].squeeze(),
                data["intrinsics"].squeeze(),
                data["extrinsics", 1].squeeze(),
                (dataset_cfg.height, dataset_cfg.width),
                render_depth=False,
            )

            pixel_values_rendered = pixel_values_rendered.permute([0,3,1,2])  # [6, H, W, 3] -> [6, 3, H, W]
            concat_6_views(pixel_values_rendered).save(
                os.path.join(args.output_dir, f"{frame+1:04d}_rgb_cond.png")
            )
            if torch.backends.mps.is_available():
                autocast_ctx = nullcontext()
            else:
                autocast_ctx = torch.autocast(pipeline.device.type)
            with autocast_ctx:
                # Run inference with appropriate parameters based on training mode
                if args.multi_step:
                    # Multi-step inference with proper denoising steps
                    preds = pipeline(
                        disparity_cond=data["box_disparity_maps", 1][0],
                        rgb_cond=pixel_values_rendered * 2 - 1,
                        prompt=["" for _ in range(data["pixel_values", 1].shape[1])],
                        num_inference_steps=args.num_inference_steps,
                        generator=generator,
                        output_type="np",
                    ).images
                else:
                    # Single-step inference
                    preds = pipeline(
                        disparity_cond=data["box_disparity_maps", 1][0],
                        rgb_cond=pixel_values_rendered,
                        prompt=["" for _ in range(data["pixel_values", 1].shape[1])],
                        num_inference_steps=1,
                        generator=generator,
                        output_type="np",
                        timesteps=[args.timestep],
                    ).images

                rgb_out, disparity_out = np.split(
                    preds, 2, axis=0
                )  # [6, H, W, 3] #TODO inverse
                disparity_out = disparity_out.mean(axis=-1)  # [B, H, W]
                rgb_out = rgb_out.transpose(0, 3, 1, 2)  # [B, 3, H, W]

                disparity_out[disparity_out < 0.005] = (
                    0.005  # Thresholding to remove noise
                )
                depth_out = disparity2depth(disparity_out)
                concat_and_visualize_6_depths(
                    depth_out,
                    save_path=os.path.join(output_dir, f"{frame+1:04d}_depth_out.png"),
                )
                disparity_gt = (data["disparity_maps", 1][0].cpu().numpy() + 1) / 2
                depth_gt = disparity2depth(disparity_gt)
                depth_gt = set_inf_to_max(depth_gt)
                concat_and_visualize_6_depths(
                    depth_gt,
                    save_path=os.path.join(output_dir, f"{frame+1:04d}_depth_gt.png"),
                )
                concat_6_views(
                    (data["pixel_values", 1][0] + 1) / 2,  # Normalize to [0, 1]
                ).save(os.path.join(output_dir, f"{frame+1:04d}_rgb_gt.png"))
                concat_6_views(
                    rgb_out,
                ).save(os.path.join(output_dir, f"{frame+1:04d}_rgb_out.png"))
            breakpoint()
            torch.cuda.empty_cache()

    print('==> Inference is done. \n==> Results saved to:', args.output_dir)


if __name__ == '__main__':
    main()
