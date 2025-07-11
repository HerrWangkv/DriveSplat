#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from PIL import Image
from glob import glob
from easydict import EasyDict

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.nn as nn
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
from omegaconf import OmegaConf

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.models.controlnet import ControlNetModel

from pipeline import SDaIGControlNetPipeline, DisparityVAE
from data import build_dataset_from_cfg
from utils.img_utils import (
    concat_6_views,
    disparity2depth,
    multiply_each_disparity_by_different_factors,
    add_gaussian_noise,
    concat_and_visualize_6_depths,
    set_inf_to_max,
)
from utils.nvs_utils import render_novel_views_using_point_cloud

import tensorboard

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def run_example_validation(pipeline, batch, args, step, generator, weight_dtype):

    dataset_cfg = OmegaConf.load(args.dataset_config_path)
    current_features = pipeline.vae.encode(
        batch[("pixel_values", 0)].squeeze().to(weight_dtype)
    ).latent_dist.sample()
    current_features *= pipeline.vae.config.scaling_factor
    current_disparity_maps = (batch[("disparity_maps", 0)].squeeze() + 1) / 2
    novel_features = render_novel_views_using_point_cloud(
        current_features=current_features.float(),
        current_depths=set_inf_to_max(disparity2depth(current_disparity_maps)),
        current_ego_mask=batch["ego_masks"].squeeze(),
        current_intrinsics=batch["latent_intrinsics"].squeeze(),
        current_extrinsics=batch["extrinsics", 0].squeeze(),
        novel_intrinsics=batch["latent_intrinsics"].squeeze(),
        novel_extrinsics=batch["extrinsics", 1].squeeze(),
        point_radius=0.05,
        points_per_pixel=8,
        current_objs_to_world=batch["objs_to_world", 0][0],
        current_box_sizes=batch["box_sizes", 0][0],
        current_obj_ids=batch["obj_ids", 0][0],
        transforms_cur_to_next=batch["transforms", 0, 1][0],
        expanding_factor=1.5,
        image_size=(
            dataset_cfg.height // 8,
            dataset_cfg.width // 8,
        ),
        background_color=(0, 0, 0, 0),
        return_novel_depths=False,
    )["novel_features"]
    novel_features = novel_features.permute(
        [0, 3, 1, 2]
    )  # [6, H, W, C] -> [6, C, H, W]
    novel_features = novel_features.to(weight_dtype)
    novel_images = pipeline.vae.decode(
        novel_features / pipeline.vae.config.scaling_factor, return_dict=False
    )[
        0
    ]  # [6, 3, H, W]
    concat_6_views(
        ((novel_images + 1) / 2).clamp(0, 1),
    ).save(os.path.join(args.output_dir, f"{step}_rgb_cond.png"))
    concat_6_views(
        (batch["pixel_values", 1][0] + 1) / 2,  # Normalize to [0, 1]
    ).save(os.path.join(args.output_dir, f"{step}_rgb_gt.png"))

    disparity_gt = (batch["disparity_maps", 1][0].cpu().numpy() + 1) / 2
    depth_gt = disparity2depth(disparity_gt)
    depth_gt = set_inf_to_max(depth_gt)
    concat_and_visualize_6_depths(
        depth_gt,
        save_path=os.path.join(args.output_dir, f"{step}_depth_gt.png"),
    )
    box_disparity_gt = (batch["box_disparity_maps", 1][0].cpu().numpy() + 1) / 2
    box_depth_gt = disparity2depth(box_disparity_gt)
    box_depth_gt = set_inf_to_max(box_depth_gt)
    concat_and_visualize_6_depths(
        box_depth_gt,
        save_path=os.path.join(args.output_dir, f"{step}_depth_cond.png"),
    )
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(pipeline.device.type)
    with autocast_ctx:
        # Run inference with appropriate parameters based on training mode
        if args.multi_step_training:
            # Multi-step inference with proper denoising steps
            preds = pipeline(
                disparity_cond=batch["box_disparity_maps", 1][0],
                rgb_cond=novel_features,
                prompt=["" for _ in range(batch["pixel_values", 1].shape[1])],
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                output_type="latent",
            ).images
        else:
            # Single-step inference
            preds = pipeline(
                disparity_cond=batch["box_disparity_maps", 1][0],
                rgb_cond=novel_features,
                prompt=["" for _ in range(batch["pixel_values", 1].shape[1])],
                num_inference_steps=1,
                generator=generator,
                output_type="latent",
                timesteps=[args.timestep],
            ).images
        rgb_latents = preds[: len(preds) // 2]  # [B, 4, H//8, W//8]
        disparity_latents = preds[len(preds) // 2 :]  # [B, 4, H//8, W//8]

        # Debug: Print shapes and some statistics
        print(f"VALIDATION DEBUG:")
        # print(f"  preds.shape = {preds.shape}")
        # print(f"  rgb_latents.shape = {rgb_latents.shape}")
        # print(f"  disparity_latents.shape = {disparity_latents.shape}")
        print(
            f"  rgb_latents stats: min={rgb_latents.min():.4f}, max={rgb_latents.max():.4f}, mean={rgb_latents.mean():.4f}"
        )
        print(
            f"  disparity_latents stats: min={disparity_latents.min():.4f}, max={disparity_latents.max():.4f}, mean={disparity_latents.mean():.4f}"
        )

        rgb_out = pipeline.vae.decode(
            rgb_latents / pipeline.vae.config.scaling_factor,
            return_dict=False,
            generator=generator,
        )[0]
        rgb_out = (rgb_out / 2 + 0.5).clamp(0, 1)  # Normalize to [0, 1]

        disparity_out = pipeline.decode_disparity(
            disparity_latents, pipeline.device
        )  # [B, 1, H//8, W//8]
        print(
            f"  disparity_out (before norm) stats: min={disparity_out.min():.4f}, max={disparity_out.max():.4f}, mean={disparity_out.mean():.4f}, std={disparity_out.std():.4f}"
        )

        # Check if output is degenerate (all similar values)
        unique_vals = torch.unique(
            disparity_out.flatten()[:1000]
        )  # Sample first 1000 values
        print(
            f"  Number of unique values in disparity_out (sample): {len(unique_vals)}"
        )
        if len(unique_vals) < 10:
            print(
                f"  WARNING: disparity_out appears degenerate, unique values: {unique_vals[:10]}"
            )

        disparity_out = (disparity_out / 2 + 0.5).clamp(0, 1)  # Normalize to [0, 1]
        print(
            f"  disparity_out (after norm) stats: min={disparity_out.min():.4f}, max={disparity_out.max():.4f}, mean={disparity_out.mean():.4f}, std={disparity_out.std():.4f}"
        )

        # Compare with ground truth disparity range
        gt_disparity_normalized = (batch["disparity_maps", 1][0] + 1) / 2
        print(
            f"  GT disparity stats: min={gt_disparity_normalized.min():.4f}, max={gt_disparity_normalized.max():.4f}, mean={gt_disparity_normalized.mean():.4f}, std={gt_disparity_normalized.std():.4f}"
        )

        # Debug: Compare with ground truth disparity
        disparity_gt_tensor = batch["disparity_maps", 1][0]

        # Debug: Test encoder-decoder round trip on ground truth
        gt_encoded = pipeline.encode_disparity(disparity_gt_tensor, pipeline.device)
        gt_decoded = pipeline.decode_disparity(gt_encoded, pipeline.device)
        gt_decoded_norm = gt_decoded
        print(f"  GT encode-decode test:")
        print(
            f"    GT original: min={disparity_gt_tensor.min():.4f}, max={disparity_gt_tensor.max():.4f}, mean={disparity_gt_tensor.mean():.4f}"
        )
        print(
            f"    GT encoded: min={gt_encoded.min():.4f}, max={gt_encoded.max():.4f}, mean={gt_encoded.mean():.4f}"
        )
        print(
            f"    GT decoded: min={gt_decoded_norm.min():.4f}, max={gt_decoded_norm.max():.4f}, mean={gt_decoded_norm.mean():.4f}"
        )
        print(
            f"    Round-trip MSE: {F.mse_loss(gt_decoded_norm, disparity_gt_tensor):.6f}"
        )

        depth_out = set_inf_to_max(disparity2depth(disparity_out))
        concat_and_visualize_6_depths(
            depth_out,
            save_path=os.path.join(args.output_dir, f"{step}_depth_out.png"),
            vmax=np.max(depth_gt) if depth_gt is not None else None,
        )
        concat_6_views(
            rgb_out,
        ).save(os.path.join(args.output_dir, f"{step}_rgb_out.png"))


def log_validation(
    batch,
    vae,
    text_encoder,
    tokenizer,
    unet,
    controlnet,
    disparity_vae,
    args,
    accelerator,
    weight_dtype,
    step,
):
    logger.info("Running validation")

    # Create scheduler
    scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    scheduler.register_to_config(prediction_type=args.prediction_type)

    # Create pipeline directly with our trained components
    # Since the base model doesn't have ControlNet, we skip loading from pretrained
    pipeline = SDaIGControlNetPipeline(
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        controlnet=accelerator.unwrap_model(controlnet),
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        disparity_vae=accelerator.unwrap_model(disparity_vae),
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            pipeline.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    run_example_validation(pipeline, batch, args, step, generator, weight_dtype)
    del pipeline
    torch.cuda.empty_cache()
    # Force garbage collection to free memory
    import gc

    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--disparity_encoder_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained disparity encoder model. If not specified, disparity encoder weights are initialized randomly.",
    )
    parser.add_argument(
        "--disparity_decoder_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained disparity decoder model. If not specified, disparity decoder weights are initialized randomly.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_config_path",
        type=str,
        default=None,
        help=(
            "path to the dataset config file"
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sdaig-controlnet",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=999,
        help="Fixed timestep for single-step training. Set to None for multi-step training.",
    )
    parser.add_argument(
        "--max_timesteps",
        type=int,
        default=1000,
        help="Maximum timesteps for multi-step diffusion training.",
    )
    parser.add_argument(
        "--multi_step_training",
        action="store_true",
        help="Enable multi-step diffusion training with random timestep sampling.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of inference steps to use during validation when multi-step training is enabled.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="sample",
        help="The prediction_type that shall be used for training. ",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_sdaig",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if not os.path.exists(args.dataset_config_path):
        raise ValueError(f"Dataset config file {args.dataset_config_path} does not exist. Please provide a valid path.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision,
        class_embed_type="projection", projection_class_embeddings_input_dim=4,
        low_cpu_mem_usage=False, device_map=None,
    )

    if args.controlnet_model_name_or_path:
        logger.info(f"Loading controlnet from {args.controlnet_model_name_or_path}")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    elif args.pretrained_model_name_or_path and os.path.exists(
        os.path.join(args.pretrained_model_name_or_path, "controlnet")
    ):
        logger.info("Loading controlnet weights from pretrained model path")
        controlnet = ControlNetModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="controlnet"
        )
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(
            unet, conditioning_channels=8, conditioning_embedding_out_channels=(256,)
        )

    # Initialize disparity encoder and decoder
    # Try loading from specific paths first, then check controlnet path, finally initialize new
    if args.disparity_encoder_model_name_or_path:
        logger.info(
            f"Loading disparity VAE from {args.disparity_encoder_model_name_or_path}"
        )
        disparity_vae = DisparityVAE.from_pretrained(
            args.disparity_encoder_model_name_or_path
        )
    elif args.pretrained_model_name_or_path and os.path.exists(
        os.path.join(args.pretrained_model_name_or_path, "disparity_vae")
    ):
        logger.info("Loading existing disparity VAE weights from pretrained model path")
        disparity_vae = DisparityVAE.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="disparity_vae"
        )
    else:
        logger.info("Initializing new disparity VAE weights")
        disparity_vae = DisparityVAE(in_channels=1, latent_channels=4)

    # Freeze vae and text_encoder and set unet, controlnet, and disparity_vae to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()
    controlnet.train()
    disparity_vae.train()
    disparity_vae.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                # Save UNet, ControlNet, and disparity encoder/decoder
                # The models list contains [unet, controlnet, disparity_vae] in the order from accelerator.prepare()
                if len(models) >= 3:
                    unet_model = models[0]
                    controlnet_model = models[1]
                    disparity_vae_model = models[2]

                    unet_model.save_pretrained(os.path.join(output_dir, "unet"))
                    controlnet_model.save_pretrained(os.path.join(output_dir, "controlnet"))

                    # Save disparity VAE using its save_pretrained method
                    disparity_vae_model.save_pretrained(
                        os.path.join(output_dir, "disparity_vae")
                    )

                    # Pop weights for all models
                    weights.pop()  # UNet
                    weights.pop()  # ControlNet
                    weights.pop()  # DisparityVAE
                else:
                    # Fallback for backward compatibility
                    models[0].save_pretrained(os.path.join(output_dir, "unet"))
                    weights.pop()

                # Note: optimizer, lr_scheduler, and other training states are automatically
                # saved by accelerator.save_state() - no need to handle them manually here

        def load_model_hook(models, input_dir):
            # Load UNet, ControlNet, and disparity VAE
            if len(models) >= 3:
                # Pop and load DisparityVAE (third model)
                disparity_vae_model = models.pop()
                if os.path.exists(os.path.join(input_dir, "disparity_vae")):
                    load_disparity_vae = DisparityVAE.from_pretrained(
                        input_dir, subfolder="disparity_vae"
                    )
                    disparity_vae_model.load_state_dict(load_disparity_vae.state_dict())
                    del load_disparity_vae

                # Pop and load ControlNet (second model)
                controlnet_model = models.pop()
                if os.path.exists(os.path.join(input_dir, "controlnet")):
                    load_controlnet = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                    controlnet_model.register_to_config(**load_controlnet.config)
                    controlnet_model.load_state_dict(load_controlnet.state_dict())
                    del load_controlnet

                # Pop and load UNet (first model)
                unet_model = models.pop()
                if os.path.exists(os.path.join(input_dir, "unet")):
                    load_unet = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet", in_channels=8)
                    unet_model.register_to_config(**load_unet.config)
                    unet_model.load_state_dict(load_unet.state_dict())
                    del load_unet
            else:
                # Fallback for backward compatibility
                unet_model = models.pop()
                if os.path.exists(os.path.join(input_dir, "unet")):
                    load_unet = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet", in_channels=8)
                    unet_model.register_to_config(**load_unet.config)
                    unet_model.load_state_dict(load_unet.state_dict())
                    del load_unet

            # Note: optimizer, lr_scheduler, and other training states are automatically
            # loaded by accelerator.load_state() - no need to handle them manually here

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()
        # Only enable gradient checkpointing if the models support it
        if (
            hasattr(disparity_vae, "enable_gradient_checkpointing")
            and hasattr(disparity_vae, "_supports_gradient_checkpointing")
            and disparity_vae._supports_gradient_checkpointing
        ):
            disparity_vae.enable_gradient_checkpointing()
        else:
            logger.warning(
                "DisparityVAE does not support gradient checkpointing, skipping."
            )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # Combine parameters from UNet, ControlNet, and disparity VAE for training
    trainable_params = (
        list(unet.parameters())
        + list(controlnet.parameters())
        + list(disparity_vae.parameters())
    )

    optimizer = optimizer_cls(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets and dataloaders.
    # Use main_process_first to ensure dataset is loaded consistently across all processes
    with accelerator.main_process_first():
        dataset_cfg = OmegaConf.load(args.dataset_config_path)
        train_dataset = build_dataset_from_cfg(dataset_cfg.data.train)

        if args.max_train_samples is not None:
            raise NotImplementedError(
                "max_train_samples is not implemented for this training script. Please set it to None."
            )

    if accelerator.num_processes > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True,
            seed=args.seed if args.seed is not None else 42,
        )
        shuffle = False  # Don't use DataLoader shuffle when using DistributedSampler
    else:
        sampler = None
        shuffle = True

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=shuffle,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # Lr_scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    assert args.max_train_steps is not None or args.num_train_epochs is not None, "max_train_steps or num_train_epochs should be provided"
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    (
        unet,
        controlnet,
        disparity_vae,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        unet,
        controlnet,
        disparity_vae,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encoder and vae to gpu and cast to weight_dtype
    # Note: UNet, ControlNet, and disparity VAE are handled by accelerator.prepare() and should not be moved manually
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples Hypersim = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if args.multi_step_training:
        logger.info(
            f"  Training mode: Multi-step diffusion (max timesteps: {args.max_timesteps})"
        )
    else:
        logger.info(f"  Training mode: Single-step (fixed timestep: {args.timestep})")
    logger.info(f"Output Workspace: {args.output_dir}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        # Set epoch for distributed sampler to ensure different shuffle each epoch
        if accelerator.num_processes > 1 and hasattr(
            train_dataloader.sampler, "set_epoch"
        ):
            train_dataloader.sampler.set_epoch(epoch)

        iter_nuscenes = iter(train_dataloader)

        train_loss = 0.0
        log_disparity_loss = 0.0
        log_reconstruction_loss = 0.0
        log_rgb_loss = 0.0

        for _ in range(len(train_dataloader)):
            batch = next(iter_nuscenes)

            with accelerator.accumulate(unet, controlnet):
                # Access data using tuple keys - these will be in [batch_size, 6, ...] format
                if args.train_batch_size != 1:
                    raise NotImplemented
                with torch.no_grad():
                    current_features = vae.encode(batch[("pixel_values", 0)].squeeze().to(weight_dtype)).latent_dist.sample()
                    current_features *= vae.config.scaling_factor
                    current_disparity_maps = (
                        batch[("disparity_maps", 0)].squeeze() + 1
                    ) / 2

                    novel_features = render_novel_views_using_point_cloud(
                        current_features=current_features.float(),
                        current_depths=set_inf_to_max(
                            disparity2depth(current_disparity_maps)
                        ),
                        current_ego_mask=batch["ego_masks"].squeeze(),
                        current_intrinsics=batch["latent_intrinsics"].squeeze(),
                        current_extrinsics=batch["extrinsics", 0].squeeze(),
                        novel_intrinsics=batch["latent_intrinsics"].squeeze(),
                        novel_extrinsics=batch["extrinsics", 1].squeeze(),
                        point_radius=0.05,
                        points_per_pixel=8,
                        current_objs_to_world=batch["objs_to_world", 0][0],
                        current_box_sizes=batch["box_sizes", 0][0],
                        current_obj_ids=batch["obj_ids", 0][0],
                        transforms_cur_to_next=batch["transforms", 0, 1][0],
                        expanding_factor=1.5,
                        image_size=(
                            dataset_cfg.height // 8,
                            dataset_cfg.width // 8,
                        ),
                        background_color=(0, 0, 0, 0),
                        return_novel_depths=False,
                    )["novel_features"]
                    novel_features = novel_features.permute(
                        [0, 3, 1, 2]
                    )  # [6, H, W, C] -> [6, C, H, W]

                    # novel_features = novel_features.to(
                    #     weight_dtype
                    # )  # Convert back to weight_dtype
                    # novel_images = vae.decode(
                    #     novel_features / vae.config.scaling_factor,
                    #     return_dict=False,
                    # )[0]
                    # novel_images = (novel_images / 2 + 0.5).clamp(0, 1)
                    # concat_6_views(novel_images).save("novel_pred.png")
                    # concat_6_views(
                    #     batch[("pixel_values", 1)].squeeze() / 2 + 0.5
                    # ).save("novel_gt.png")
                pixel_values = batch[("pixel_values", 1)]  # Shape: [batch_size, 6, 3, H, W]
                latent_disparity_maps = batch[
                    ("disparity_maps", 1)
                ]  # Shape: [batch_size, 6, 1, H//8, W//8]
                ego_masks = batch["ego_masks"]  # [batch_size, 6, 1, H//8, W//8]
                latent_box_disparity_maps = batch[("box_disparity_maps", 1)]  # Shape: [batch_size, 6, 1, H//8, W//8]

                # Reshape from [batch_size, 6, ...] to [6*batch_size, ...] using view
                pixel_values = pixel_values.view(
                    6 * args.train_batch_size, *pixel_values.shape[2:]
                )  # [6*batch_size, 3, H, W]
                latent_disparity_maps = latent_disparity_maps.view(
                    6 * args.train_batch_size, *latent_disparity_maps.shape[2:]
                )  # [6*batch_size, 1, H//8, W//8]
                ego_masks = ego_masks.view(
                    6 * args.train_batch_size, *ego_masks.shape[2:]
                )  # [6*batch_size, 1, H, W]
                latent_box_disparity_maps = latent_box_disparity_maps.view(
                    6 * args.train_batch_size, *latent_box_disparity_maps.shape[2:]
                )  # [6*batch_size, 1, H//8, W//8]

                # Convert images to latent space
                rgb_latents = vae.encode(
                    torch.cat((pixel_values, pixel_values), dim=0).to(weight_dtype)
                    ).latent_dist.sample()
                rgb_latents = rgb_latents * vae.config.scaling_factor

                # Encode disparity maps to latent space using disparity VAE
                # Use the VAE encoder to get the latent representation
                # Access the underlying module when using distributed training
                disparity_vae_module = (
                    disparity_vae.module
                    if hasattr(disparity_vae, "module")
                    else disparity_vae
                )
                # Get the actual dtype of the VAE parameters to match input
                vae_dtype = next(disparity_vae_module.parameters()).dtype
                disparity_input = torch.cat(
                    (latent_disparity_maps, latent_disparity_maps), dim=0
                ).to(vae_dtype)
                mu, logvar = disparity_vae_module.encode(disparity_input)
                disparity_latents = (
                    disparity_vae_module.reparameterize(mu, logvar)
                    if disparity_vae_module.training
                    else mu
                )
                disparity_latents_detached = disparity_latents.detach()

                torch.cuda.empty_cache()

                bsz = disparity_latents.shape[0]
                bsz_per_task = int(bsz/2)

                # Get the valid mask for the latent space
                valid_mask = ~ego_masks

                # Debug: Check valid mask dimensions and coverage
                # print(f"DEBUG: ego_masks shape: {ego_masks.shape}")
                # print(f"DEBUG: valid_mask shape: {valid_mask.shape}")
                # print(f"DEBUG: disparity_latents shape: {disparity_latents.shape}")
                # print(f"DEBUG: valid_mask coverage: {valid_mask.float().mean():.4f}")

                # Ensure valid_mask is properly sized for latent space
                if valid_mask.shape[-2:] != disparity_latents.shape[-2:]:
                    # Resize valid_mask to match latent dimensions
                    valid_mask = F.interpolate(
                        valid_mask.float(),
                        size=disparity_latents.shape[-2:],
                        mode="nearest",
                    ).bool()
                    print(f"DEBUG: Resized valid_mask to shape: {valid_mask.shape}")

                # Sample noise that we'll add to the latents
                rgb_noise = torch.randn_like(rgb_latents)
                disparity_noise = torch.randn_like(disparity_latents)

                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    rgb_noise += args.noise_offset * torch.randn(
                        (rgb_latents.shape[0], rgb_latents.shape[1], 1, 1), device=rgb_latents.device
                    )
                    disparity_noise += args.noise_offset * torch.randn(
                        (disparity_latents.shape[0], disparity_latents.shape[1], 1, 1),
                        device=disparity_latents.device,
                    )
                if args.input_perturbation:
                    new_rgb_noise = rgb_noise + args.input_perturbation * torch.randn_like(rgb_noise)
                    new_disparity_noise = disparity_noise + args.input_perturbation * torch.randn_like(disparity_noise)

                # Set timesteps for training
                if args.multi_step_training:
                    # Random timestep sampling for multi-step training
                    timesteps = torch.randint(
                        0, args.max_timesteps, (bsz,), device=disparity_latents.device
                    )
                    timesteps = timesteps.long()
                else:
                    # Fixed timestep for single-step training
                    timesteps = torch.tensor(
                        [args.timestep], device=disparity_latents.device
                    ).repeat(bsz)
                    timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_rgb_latents = noise_scheduler.add_noise(rgb_latents, new_rgb_noise, timesteps)
                    noisy_disparity_latents = noise_scheduler.add_noise(
                        disparity_latents_detached, new_disparity_noise, timesteps
                    )
                else:
                    noisy_rgb_latents = noise_scheduler.add_noise(rgb_latents, rgb_noise, timesteps)
                    noisy_disparity_latents = noise_scheduler.add_noise(
                        disparity_latents_detached, disparity_noise, timesteps
                    )

                # Concatenate rgb and depth using VAE encoder
                # Ensure input dtype matches VAE parameters
                vae_dtype = next(disparity_vae_module.parameters()).dtype
                latent_box_mu, latent_box_logvar = disparity_vae_module.encode(
                    latent_box_disparity_maps.to(vae_dtype)
                )
                latent_box_encoded = (
                    disparity_vae_module.reparameterize(
                        latent_box_mu, latent_box_logvar
                    )
                    if disparity_vae_module.training
                    else latent_box_mu
                )
                controlnet_cond = torch.cat(
                    [novel_features, latent_box_encoded],
                    dim=1,
                )  # [6*batch_size, 5, H, W]
                controlnet_cond = torch.cat([controlnet_cond, controlnet_cond], dim=0)
                controlnet_cond = controlnet_cond.to(weight_dtype)
                noisy_latents = torch.cat(
                    [noisy_rgb_latents, noisy_disparity_latents], dim=1
                )

                # Get the empty text embedding for conditioning
                prompt = ""
                text_inputs = tokenizer(
                    prompt,
                    padding="do_not_pad",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(
                    noisy_disparity_latents.device
                )
                encoder_hidden_states = text_encoder(text_input_ids, return_dict=False)[0]
                encoder_hidden_states = encoder_hidden_states.repeat(bsz, 1, 1)

                # Get the task embedding
                task_emb_rgb = (
                    torch.tensor([0, 1]).float().unsqueeze(0).to(accelerator.device)
                )
                task_emb_rgb = torch.cat(
                    [torch.sin(task_emb_rgb), torch.cos(task_emb_rgb)], dim=-1
                ).repeat(bsz_per_task, 1)
                task_emb_disparity = torch.tensor([1, 0]).float().unsqueeze(0).to(accelerator.device)
                task_emb_disparity = torch.cat([torch.sin(task_emb_disparity), torch.cos(task_emb_disparity)], dim=-1).repeat(bsz_per_task, 1)
                task_emb = torch.cat((task_emb_rgb, task_emb_disparity), dim=0)

                # Predict
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_cond,
                    return_dict=False,
                    class_labels=task_emb,
                )
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype)
                        for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(
                        dtype=weight_dtype
                    ),
                    return_dict=False,
                    class_labels=task_emb,
                )[0]
                # Compute loss
                rgb_loss = F.mse_loss(
                    model_pred[:bsz_per_task].float(),
                    rgb_latents[:bsz_per_task].float(),
                    reduction="mean",
                )

                # Decode predicted disparity latents and compute loss in disparity space
                predicted_disparity_latents = model_pred[bsz_per_task:]
                target_disparity_latents = disparity_latents_detached[bsz_per_task:]

                # Check for degenerate predictions
                pred_unique = torch.unique(predicted_disparity_latents.flatten()[:1000])
                target_unique = torch.unique(target_disparity_latents.flatten()[:1000])

                # Debug: Check what we're actually comparing in the loss
                pred_valid = predicted_disparity_latents[
                    valid_mask.expand(-1, 4, -1, -1)
                ].float()
                target_valid = target_disparity_latents[
                    valid_mask.expand(-1, 4, -1, -1)
                ].float()

                # print(
                #     f"DEBUG: Valid region stats - pred: min={pred_valid.min():.4f}, max={pred_valid.max():.4f}, mean={pred_valid.mean():.4f}, std={pred_valid.std():.4f}"
                # )
                # print(
                #     f"DEBUG: Valid region stats - target: min={target_valid.min():.4f}, max={target_valid.max():.4f}, mean={target_valid.mean():.4f}, std={target_valid.std():.4f}"
                # )
                # print(f"DEBUG: Valid region size: {len(pred_valid)} elements")

                disparity_loss = F.mse_loss(pred_valid, target_valid, reduction="mean")

                # Add VAE reconstruction loss to train encoder-decoder mapping
                # This ensures the encoder-decoder learns to preserve disparity information
                # and regularizes the latent space with KL divergence

                # Compute VAE loss using the disparity VAE model with warmup
                # VAE warmup: gradually increase KL weight to prevent collapse
                warmup_steps = 1000
                kl_weight = (
                    min(0.001, 0.001 * (global_step / warmup_steps))
                    if global_step < warmup_steps
                    else 0.001
                )

                vae_loss, vae_reconstruction_loss, kl_loss, target_reconstructed = (
                    disparity_vae_module.compute_vae_loss(
                        latent_disparity_maps,
                        beta=kl_weight,  # Use dynamic KL weight
                    )
                )

                total_disparity_loss = (
                    disparity_loss + 0.01 * vae_loss
                )  # Further reduce VAE weight

                # The variance regularization is now handled by the KL divergence in the VAE
                # No need for manual variance penalty anymore

                # Debug: Print training statistics occasionally
                if global_step % 100 == 0:
                    print(f"TRAINING DEBUG (step {global_step}):")
                    print(
                        f"  predicted_disparity_latents stats: min={predicted_disparity_latents.min():.4f}, max={predicted_disparity_latents.max():.4f}, mean={predicted_disparity_latents.mean():.4f}"
                    )
                    print(f"  valid_mask sum: {valid_mask.sum()}")
                    print(f"  kl_weight: {kl_weight:.6f}")
                    print(f"  disparity_loss: {disparity_loss:.6f}")
                    print(f"  vae_reconstruction_loss: {vae_reconstruction_loss:.6f}")
                    print(f"  kl_loss: {kl_loss:.6f}")
                    print(f"  vae_loss: {vae_loss:.6f}")
                    print(f"  total_disparity_loss: {total_disparity_loss:.6f}")
                    print(f"  rgb_loss: {rgb_loss:.6f}")

                    # Test encoder-decoder round trip on target disparity (should be very similar to reconstruction loss)
                    round_trip_mse = F.mse_loss(
                        target_reconstructed, latent_disparity_maps
                    )
                    print(f"  Encoder-decoder round-trip MSE: {round_trip_mse:.6f}")

                    # Additional debugging for disparity understanding
                    print(
                        f"  Original latent_disparity_maps stats: min={latent_disparity_maps.min():.4f}, max={latent_disparity_maps.max():.4f}, mean={latent_disparity_maps.mean():.4f}"
                    )
                    print(
                        f"  Encoded disparity_latents stats: min={disparity_latents[bsz_per_task:].min():.4f}, max={disparity_latents[bsz_per_task:].max():.4f}, mean={disparity_latents[bsz_per_task:].mean():.4f}"
                    )
                    print(
                        f"  Reconstructed target_reconstructed stats: min={target_reconstructed.min():.4f}, max={target_reconstructed.max():.4f}, mean={target_reconstructed.mean():.4f}"
                    )

                    # Check if the predicted latents are actually meaningful
                    predicted_reconstructed = disparity_vae_module.decode(
                        predicted_disparity_latents
                    )
                    pred_recon_mse = F.mse_loss(
                        predicted_reconstructed, latent_disparity_maps
                    )
                    print(f"  Predicted->Reconstructed MSE: {pred_recon_mse:.6f}")

                    # Check latent space variance
                    latent_variance = torch.var(
                        predicted_disparity_latents, dim=[0, 2, 3]
                    )
                    print(f"  Predicted latent variance per channel: {latent_variance}")
                    target_variance = torch.var(target_disparity_latents, dim=[0, 2, 3])
                    print(f"  Target latent variance per channel: {target_variance}")

                    # Check for degenerate solutions
                    print(
                        f"  Predicted latent unique values (sample): {len(pred_unique)}"
                    )
                    print(
                        f"  Target latent unique values (sample): {len(target_unique)}"
                    )
                    if len(pred_unique) < 50:
                        print(f"  WARNING: Predicted latents appear degenerate!")

                    # Check valid mask coverage
                    valid_ratio = valid_mask.float().mean()
                    print(
                        f"  Valid mask coverage: {valid_ratio:.4f} ({valid_mask.sum()} / {valid_mask.numel()} pixels)"
                    )

                    # Check disparity range in original space
                    pred_disparity_decoded = disparity_vae_module.decode(
                        predicted_disparity_latents
                    )
                    target_disparity_decoded = disparity_vae_module.decode(
                        target_disparity_latents
                    )
                    print(
                        f"  Predicted disparity (decoded) stats: min={pred_disparity_decoded.min():.4f}, max={pred_disparity_decoded.max():.4f}, std={pred_disparity_decoded.std():.4f}"
                    )
                    print(
                        f"  Target disparity (decoded) stats: min={target_disparity_decoded.min():.4f}, max={target_disparity_decoded.max():.4f}, std={target_disparity_decoded.std():.4f}"
                    )

                loss = total_disparity_loss + rgb_loss

                # Gather loss
                avg_disparity_loss = accelerator.gather(disparity_loss.repeat(args.train_batch_size*6)).mean()
                log_disparity_loss += avg_disparity_loss.item() / args.gradient_accumulation_steps
                avg_vae_reconstruction_loss = accelerator.gather(
                    vae_reconstruction_loss.repeat(args.train_batch_size * 6)
                ).mean()
                log_reconstruction_loss += (
                    avg_vae_reconstruction_loss.item()
                    / args.gradient_accumulation_steps
                )
                avg_rgb_loss = accelerator.gather(rgb_loss.repeat(args.train_batch_size*6)).mean()
                log_rgb_loss += avg_rgb_loss.item() / args.gradient_accumulation_steps
                train_loss = log_disparity_loss + log_reconstruction_loss + log_rgb_loss

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # Clip gradients for UNet, ControlNet, and disparity VAE
                    accelerator.clip_grad_norm_(
                        list(unet.parameters())
                        + list(controlnet.parameters())
                        + list(disparity_vae.parameters()),
                        args.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Clear cache to prevent memory buildup
                if global_step % 50 == 0:  # Clear cache every 50 steps
                    torch.cuda.empty_cache()

            logs = {
                "SL": loss.detach().item(),
                "SL_D": disparity_loss.detach().item(),
                "SL_VAE": vae_loss.detach().item(),
                "SL_KL": kl_loss.detach().item(),
                "SL_R": rgb_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "disparity_loss": log_disparity_loss,
                        "reconstruction_loss": log_reconstruction_loss,
                        "rgb_loss": log_rgb_loss,
                    },
                    step=global_step,
                )
                train_loss = 0.0
                log_disparity_loss = 0.0
                log_reconstruction_loss = 0.0
                log_rgb_loss = 0.0

                checkpointing_steps = args.checkpointing_steps
                validation_steps = args.validation_steps

                if accelerator.is_main_process:
                    if global_step % checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        torch.cuda.empty_cache()

                    if global_step % validation_steps == 0:
                        log_validation(
                            batch,
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
                            disparity_vae,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            if global_step >= args.max_train_steps:
                break
        torch.cuda.empty_cache()

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        controlnet = unwrap_model(controlnet)
        disparity_vae = unwrap_model(disparity_vae)

        pipeline = SDaIGControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            controlnet=controlnet,
            disparity_vae=disparity_vae,
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.save_pretrained(os.path.join(args.output_dir, "final"))

    accelerator.end_training()


if __name__ == "__main__":
    main()
