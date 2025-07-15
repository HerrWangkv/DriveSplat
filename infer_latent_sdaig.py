import logging
import os
import argparse
from contextlib import nullcontext

import torch
import numpy as np
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel
from diffusers.models.controlnet import ControlNetModel
from diffusers.utils import check_min_version

from pipeline import SDaIGControlNetPipeline, LatentDisparityDecoder
from data import build_dataset_from_cfg
from utils.img_utils import (
    concat_6_views,
    disparity2depth,
    concat_and_visualize_6_depths,
    set_inf_to_max,
)
from utils.nvs_utils import render_novel_views_using_point_cloud
from utils.video_utils import create_videos_from_images
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

    # Video creation arguments
    parser.add_argument(
        "--create_videos",
        action="store_true",
        help="Create videos from generated images after inference",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=1,
        help="Frames per second for generated videos",
    )
    parser.add_argument(
        "--video_quality",
        choices=["high", "medium", "low"],
        default="high",
        help="Video quality for generated videos",
    )
    parser.add_argument(
        "--create_comparison_videos",
        action="store_true",
        help="Create side-by-side comparison videos",
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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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
    dataset = build_dataset_from_cfg(dataset_cfg.data.val)
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
    latent_disparity_decoder_path = os.path.join(
        args.pretrained_model_name_or_path, "latent_disparity_decoder"
    )

    if os.path.exists(checkpoint_unet_path) and os.path.exists(
        checkpoint_controlnet_path and os.path.exists(latent_disparity_decoder_path)
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

        # Load disparity decoder
        logging.info(f"Loading disparity decoder from {latent_disparity_decoder_path}")
        latent_disparity_decoder = LatentDisparityDecoder.from_pretrained(
            latent_disparity_decoder_path,
            torch_dtype=dtype,
        )
        pipeline.latent_disparity_decoder = latent_disparity_decoder

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
        if pipeline.latent_disparity_decoder is None:
            logging.info("Initializing disparity decoder weights randomly")
            pipeline.latent_disparity_decoder = LatentDisparityDecoder(
                latent_channels=4, out_channels=1
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
        for index, data in enumerate(tqdm(dataloader)):
            data = {k: v.to(device) for k, v in data.items()}
            seq_idx, frame_idx = dataset.nusc.seq_indices[index]
            if frame_idx == 0:
                scene_idx = dataset.nusc.seqs[seq_idx]
                output_dir = os.path.join(args.output_dir, f"{scene_idx:02d}")
                os.makedirs(output_dir, exist_ok=True)
                concat_6_views(
                    (data["pixel_values", 0][0] + 1) / 2,  # Normalize to [0, 1]
                ).save(os.path.join(output_dir, f"{frame_idx:04d}_rgb_gt.png"))
                concat_and_visualize_6_depths(
                    set_inf_to_max(
                        disparity2depth((data["disparity_maps", 0][0] + 1) / 2)
                    ),
                    save_path=os.path.join(output_dir, f"{frame_idx:04d}_depth_gt.png"),
                    vmax=200,
                )
                concat_and_visualize_6_depths(
                    set_inf_to_max(
                        disparity2depth((data["latent_box_disparity_maps", 0][0] + 1) / 2)
                    ),
                    save_path=os.path.join(
                        output_dir, f"{frame_idx:04d}_depth_cond.png"
                    ),
                    vmax=200,
                )
                current_features = pipeline.vae.encode(
                    data[("pixel_values", 0)].squeeze().to(dtype)
                ).latent_dist.sample()
                current_features *= pipeline.vae.config.scaling_factor
                current_features = current_features.float()
                if torch.backends.mps.is_available():
                    autocast_ctx = nullcontext()
                else:
                    autocast_ctx = torch.autocast(pipeline.device.type)
                with autocast_ctx:
                    if args.multi_step:
                        preds = pipeline(
                            disparity_cond=data["latent_box_disparity_maps", 0][0],
                            rgb_cond=current_features,
                            prompt=[
                                "" for _ in range(data["pixel_values", 0].shape[1])
                            ],
                            num_inference_steps=args.num_inference_steps,
                            generator=generator,
                            output_type="latent",
                        ).images
                    else:
                        preds = pipeline(
                            disparity_cond=data["latent_box_disparity_maps", 0][0],
                            rgb_cond=current_features,
                            prompt=[
                                "" for _ in range(data["pixel_values", 0].shape[1])
                            ],
                            num_inference_steps=1,
                            generator=generator,
                            output_type="latent",
                            timesteps=[args.timestep],
                        ).images
                rgb_latents = preds[: len(preds) // 2]  # [B, 4, H//8, W//8]
                disparity_latents = preds[len(preds) // 2 :]  # [B, 4, H//8, W//8]
                rgb_latents = rgb_latents.to(dtype) 
                rgb_out = pipeline.vae.decode(
                    rgb_latents / pipeline.vae.config.scaling_factor,
                    return_dict=False,
                    generator=generator,
                )[0]
                rgb_out = (rgb_out / 2 + 0.5).clamp(0, 1)  # Normalize to [0, 1]
                concat_6_views(
                    rgb_out,
                ).save(os.path.join(output_dir, f"{frame_idx:04d}_rgb_out.png"))

                disparity_out = pipeline.vae.decode(
                    disparity_latents / pipeline.vae.config.scaling_factor,
                    return_dict=False,
                    generator=generator,
                )[0]
                disparity_out = disparity_out.mean(dim=1)
                disparity_out = (disparity_out / 2 + 0.5).clamp(0, 1)  # Normalize to [0, 1]
                concat_and_visualize_6_depths(
                    set_inf_to_max(disparity2depth(disparity_out)),
                    save_path=os.path.join(output_dir, f"{frame_idx:04d}_depth_out.png"),
                    vmax=200,
                )
                latent_disparity_out = pipeline.decode_latent_disparity(
                    disparity_latents, pipeline.device
                )
                latent_disparity_out = (latent_disparity_out / 2 + 0.5).clamp(
                    0, 1
                )  # Normalize to [0, 1]
                concat_and_visualize_6_depths(
                    set_inf_to_max(disparity2depth(latent_disparity_out)),
                    save_path=os.path.join(output_dir, f"{frame_idx:04d}_latent_depth_out.png"),
                    vmax=200,
                )
                rgb_latents = current_features # Use original features for rendering
            elif frame_idx > 6:
                # Create videos if requested
                if frame_idx == 7 and args.create_videos:
                    create_videos_from_images(
                        output_dir,
                        fps=args.video_fps,
                        quality=args.video_quality,
                        create_comparison=args.create_comparison_videos,
                    )
                continue
            novel_features = render_novel_views_using_point_cloud(
                current_features=rgb_latents.float(),  # Ensure float32 for PyTorch3D compatibility
                current_depths=set_inf_to_max(disparity2depth(latent_disparity_out.squeeze())),
                current_ego_mask=data["latent_ego_masks"].squeeze(),
                current_intrinsics=data["latent_intrinsics"].squeeze(),
                current_extrinsics=data["extrinsics", 0].squeeze(),
                novel_intrinsics=data["latent_intrinsics"].squeeze(),
                novel_extrinsics=data["extrinsics", 1].squeeze(),
                point_radius=0.05,
                points_per_pixel=8,
                current_objs_to_world=data["objs_to_world", 0][0],
                current_box_sizes=data["box_sizes", 0][0],
                current_obj_ids=data["obj_ids", 0][0],
                transforms_cur_to_next=data["transforms", 0, 1][0],
                expanding_factor=1.5,
                image_size=(dataset_cfg.height//8, dataset_cfg.width//8),
                background_color=(0, 0, 0, 0),
                return_novel_depths=False,
            )["novel_features"]
            novel_features = novel_features.permute(
                [0, 3, 1, 2]
            )  # [6, H, W, C] -> [6, C, H, W]
            novel_features = novel_features.to(dtype)
            novel_images = pipeline.vae.decode(
                novel_features / pipeline.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]
            concat_6_views(((novel_images + 1) / 2).clamp(0, 1)).save(
                os.path.join(output_dir, f"{frame_idx+1:04d}_rgb_cond.png")
            )

            concat_and_visualize_6_depths(
                set_inf_to_max(
                    disparity2depth((data["latent_box_disparity_maps", 1][0] + 1) / 2)
                ),
                save_path=os.path.join(output_dir, f"{frame_idx+1:04d}_depth_cond.png"),
                vmax=200,
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
                        disparity_cond=data["latent_box_disparity_maps", 1][0],
                        rgb_cond=novel_features.float(),
                        prompt=["" for _ in range(data["pixel_values", 1].shape[1])],
                        num_inference_steps=args.num_inference_steps,
                        generator=generator,
                        output_type="latent",
                    ).images
                else:
                    # Single-step inference
                    preds = pipeline(
                        disparity_cond=data["latent_box_disparity_maps", 1][0],
                        rgb_cond=novel_features.float(),
                        prompt=["" for _ in range(data["pixel_values", 1].shape[1])],
                        num_inference_steps=1,
                        generator=generator,
                        output_type="latent",
                        timesteps=[args.timestep],
                    ).images
 
                rgb_latents = preds[: len(preds) // 2]  # [B, 4, H//8, W//8]
                disparity_latents = preds[len(preds) // 2 :]  # [B, 4, H//8, W//8]
                rgb_out = pipeline.vae.decode(
                    rgb_latents / pipeline.vae.config.scaling_factor,
                    return_dict=False,
                    generator=generator,
                )[0]
                rgb_out = (rgb_out / 2 + 0.5).clamp(0, 1)  # Normalize to [0, 1]
                concat_6_views(
                    rgb_out,
                ).save(os.path.join(output_dir, f"{frame_idx+1:04d}_rgb_out.png"))

                disparity_out = pipeline.vae.decode(
                    disparity_latents / pipeline.vae.config.scaling_factor,
                    return_dict=False,
                    generator=generator,
                )[0]
                disparity_out = disparity_out.mean(dim=1)
                disparity_out = (disparity_out / 2 + 0.5).clamp(0, 1)  # Normalize to [0, 1]
                concat_and_visualize_6_depths(
                    set_inf_to_max(disparity2depth(disparity_out)),
                    save_path=os.path.join(output_dir, f"{frame_idx+1:04d}_depth_out.png"),
                    vmax=200,
                )
                latent_disparity_out = pipeline.decode_latent_disparity(
                    disparity_latents, pipeline.device
                )
                latent_disparity_out = (latent_disparity_out / 2 + 0.5).clamp(
                    0, 1
                )  # Normalize to [0, 1]
                concat_and_visualize_6_depths(
                    set_inf_to_max(disparity2depth(latent_disparity_out)),
                    save_path=os.path.join(output_dir, f"{frame_idx+1:04d}_latent_depth_out.png"),
                    vmax=200,
                )
                disparity_gt = (data["disparity_maps", 1][0].cpu().numpy() + 1) / 2
                concat_and_visualize_6_depths(
                    set_inf_to_max(disparity2depth(disparity_gt)),
                    save_path=os.path.join(
                        output_dir, f"{frame_idx+1:04d}_depth_gt.png"
                    ),
                    vmax=200,
                )
                concat_6_views(
                    (data["pixel_values", 1][0] + 1) / 2,  # Normalize to [0, 1]
                ).save(os.path.join(output_dir, f"{frame_idx+1:04d}_rgb_gt.png"))
            torch.cuda.empty_cache()
    print('==> Inference is done. \n==> Results saved to:', args.output_dir)


if __name__ == '__main__':
    main()
