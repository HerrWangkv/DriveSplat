import logging
import os
import argparse
from contextlib import nullcontext

import torch
import numpy as np
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models.controlnet import ControlNetModel
from diffusers.utils import check_min_version

from pipeline import SDaIGControlNetPipeline
from datasets import build_dataset_from_cfg
from utils.img_utils import concat_6_views, disparity2depth, concat_and_visualize_6_depths
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
        default="datasets/NuScenes/sdaig.yaml",
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
        "--disparity",
        action="store_true",
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
    dataset = build_dataset_from_cfg(dataset_cfg.data.trainval)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        pin_memory_device=device,
    )
    # -------------------- Model --------------------

    pipeline = SDaIGControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
    )

    if pipeline.controlnet is None:
        logging.info("Initializing ControlNet weights from unet")
        pipeline.controlnet = ControlNetModel.from_unet(
            pipeline.unet,
            conditioning_channels=1
        )
    logging.info(f"Successfully loading pipeline from {args.pretrained_model_name_or_path}.")
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
            if torch.backends.mps.is_available():
                autocast_ctx = nullcontext()
            else:
                autocast_ctx = torch.autocast(pipeline.device.type)
            with autocast_ctx:
                # Run
                preds = pipeline(
                    rgb_in=data["pixel_values", 0][0],
                    disparity_in=data["disparity_maps", 0][0],
                    prompt=["" for _ in range(data["pixel_values", 0].shape[1])],
                    num_inference_steps=1,
                    generator=generator,
                    # guidance_scale=0,
                    output_type="np",
                    timesteps=[args.timestep],
                    processing_res=processing_res,
                    match_input_res=match_input_res,
                    resample_method=resample_method,
                ).images

                disparity_out, rgb_out = np.split(preds, 2, axis=0)  # [6, H, W, 3]
                disparity_out = disparity_out.mean(axis=-1)  # [B, H, W]
                rgb_out = rgb_out.transpose(0, 3, 1, 2)  # [B, 3, H, W]

                disparity_out[disparity_out < 0.005] = (
                    0.005  # Thresholding to remove noise
                )
                depth = disparity2depth(disparity_out)
                concat_and_visualize_6_depths(
                    depth, 
                    save_path=os.path.join(output_dir, f"{frame:04d}_depth.png")
                )
                concat_6_views(
                    (data["pixel_values", 0][0] + 1) / 2,  # Normalize to [0, 1]
                ).save(os.path.join(output_dir, f"{frame:04d}_camera.png"))
                concat_6_views(
                    rgb_out,
                ).save(os.path.join(output_dir, f"{frame:04d}_rgb.png"))
            novel_images, novel_depth = render_novel_view(
                rgb_out,
                depth,
                data["ego_masks"][0].squeeze(),
                data["intrinsics"][0],
                data["extrinsics", 0][0],
                data["intrinsics"][0],
                data["extrinsics", 1][0],
                (dataset_cfg.height, dataset_cfg.width),
            )
            novel_depth[novel_depth > 200] = 200  # Thresholding to remove noise
            concat_6_views(
                novel_images.squeeze().permute([0, 3, 1, 2]),
            ).save(os.path.join(output_dir, f"{frame:04d}_novel_view.png"))
            concat_and_visualize_6_depths(
                novel_depth,
                save_path=os.path.join(output_dir, f"{frame:04d}_novel_view_depth.png"),
            )
            breakpoint()
            torch.cuda.empty_cache()

    print('==> Inference is done. \n==> Results saved to:', args.output_dir)


if __name__ == '__main__':
    main()
