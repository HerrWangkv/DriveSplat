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
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.models.controlnet import ControlNetModel

# LoRA specific imports
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers

from pipeline import SDaIGControlNetPipeline
from data import build_dataset_from_cfg
from utils.img_utils import (
    concat_6_views,
    disparity2depth,
    depth2disparity,
    concat_and_visualize_6_depths,
    set_inf_to_max,
)
from utils.nvs_utils import render_novel_view

import tensorboard

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def run_example_validation(pipeline, batch, args, step, generator):

    dataset_cfg = OmegaConf.load(args.dataset_config_path)
    pixel_values_rendered = render_novel_view(
        current_images=(batch[("pixel_values", 0)].squeeze() + 1) / 2,
        current_depths=set_inf_to_max(
            disparity2depth((batch[("disparity_maps", 0)].squeeze() + 1) / 2)
        ),
        current_ego_mask=batch["ego_masks"].squeeze(),
        current_intrinsics=batch["intrinsics"].squeeze(),
        current_extrinsics=batch["extrinsics", 0].squeeze(),
        novel_intrinsics=batch["intrinsics"].squeeze(),
        novel_extrinsics=batch["extrinsics", 1].squeeze(),
        current_objs_to_world=batch["objs_to_world", 0][0],
        current_box_sizes=batch["box_sizes", 0][0],
        transforms_cur_to_next=batch["transforms", 0, 1][0],
        expanding_factor=1.5,
        image_size=(dataset_cfg.height, dataset_cfg.width),
        render_depth=False,
    )
    pixel_values_rendered = pixel_values_rendered.permute([0,3,1,2])  # [6, H, W, 3] -> [6, 3, H, W]
    concat_6_views(pixel_values_rendered).save(
        os.path.join(args.output_dir, f"{step}_rgb_cond.png")
    )
    disparity_gt = (batch["disparity_maps", 1][0].cpu().numpy() + 1) / 2
    depth_gt = disparity2depth(disparity_gt)
    depth_gt = set_inf_to_max(depth_gt)
    concat_and_visualize_6_depths(
        depth_gt,
        save_path=os.path.join(args.output_dir, f"{step}_depth_gt.png"),
    )
    concat_6_views(
        (batch["pixel_values", 1][0] + 1) / 2,  # Normalize to [0, 1]
    ).save(os.path.join(args.output_dir, f"{step}_rgb_gt.png"))
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
                rgb_cond=pixel_values_rendered * 2 - 1,
                prompt=["" for _ in range(batch["pixel_values", 1].shape[1])],
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                output_type="np",
            ).images
        else:
            # Single-step inference
            preds = pipeline(
                disparity_cond=batch["box_disparity_maps", 1][0],
                rgb_cond=pixel_values_rendered,
                prompt=["" for _ in range(batch["pixel_values", 1].shape[1])],
                num_inference_steps=1,
                generator=generator,
                output_type="np",
                timesteps=[args.timestep],
            ).images
        rgb_out, disparity_out = np.split(preds, 2, axis=0)  # [6, H, W, 3]
        disparity_out = disparity_out.mean(axis=-1)  # [B, H, W]

        depth_out = disparity2depth(disparity_out)
        concat_and_visualize_6_depths(
            depth_out,
            save_path=os.path.join(args.output_dir, f"{step}_depth_out.png"),
            vmax=np.max(depth_gt) if depth_gt is not None else None,
        )
        rgb_out = rgb_out.transpose(0, 3, 1, 2)  # [B, 3, H, W]
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

    run_example_validation(pipeline, batch, args, step, generator)
    del pipeline
    torch.cuda.empty_cache()
    # Force garbage collection to free memory
    import gc

    gc.collect()


def save_lora_weights(unet, output_dir, step, args):
    """Save LoRA weights for both UNet and ControlNet."""

    # Save UNet LoRA weights
    unet_lora_state_dict = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(unet)
    )
    from diffusers import StableDiffusionPipeline

    StableDiffusionPipeline.save_lora_weights(
        save_directory=os.path.join(output_dir, f"unet_lora_step_{step}"),
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
    )
    logger.info(f"Saved LoRA weights at step {step}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning script for SDaIGControlNetPipeline."
    )

    # LoRA specific arguments
    parser.add_argument(
        "--lora_rank", type=int, default=64, help="The rank of the LoRA adaptation."
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="The alpha parameter for LoRA scaling.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="Dropout probability for LoRA layers.",
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        choices=["none", "all", "lora_only"],
        help="Bias type for LoRA adaptation.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["to_k", "to_q", "to_v", "to_out.0"],
        help="Target modules for LoRA adaptation.",
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
        default="sdaig-controlnet-lora",
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
        help="Number of inference steps to use during autoregressive training and validation.",
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
        raise ValueError(
            f"Dataset config file {args.dataset_config_path} does not exist. Please provide a valid path."
        )

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
    if args.multi_step_training:
        # Load DDIM scheduler for both noising and denoising processes
        ddim_scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        ddim_scheduler.register_to_config(prediction_type=args.prediction_type)
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
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet, conditioning_channels=4)

    # # Replace the first layer to accept 8 in_channels.
    # _weight = unet.conv_in.weight.clone()
    # _bias = unet.conv_in.bias.clone()
    # _weight = _weight.repeat(1, 2, 1, 1)
    # _weight *= 0.5
    # # unet.config.in_channels *= 2
    # config_dict = EasyDict(unet.config)
    # config_dict.in_channels *= 2
    # unet._internal_dict = config_dict

    # # new conv_in channel
    # _n_convin_out_channel = unet.conv_in.out_channels
    # _new_conv_in =nn.Conv2d(
    #     8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    # )
    # _new_conv_in.weight = nn.Parameter(_weight)
    # _new_conv_in.bias = nn.Parameter(_bias)
    # unet.conv_in = _new_conv_in

    # Freeze vae and text_encoder and set unet and controlnet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.train()

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # LoRA Configuration
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
    )

    trainable_params = list(controlnet.parameters())

    logger.info("Applying LoRA to UNet")
    unet.add_adapter(lora_config)
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)
    trainable_params.extend([p for p in unet.parameters() if p.requires_grad])

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
                if len(models) >= 2:
                    unet_model = models[0]
                    controlnet_model = models[1]
                    unet_lora_state_dict = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(unet_model)
                    )

                    SDaIGControlNetPipeline.save_lora_weights(
                        save_directory=os.path.join(output_dir, "unet_lora"),
                        unet_lora_layers=unet_lora_state_dict,
                        safe_serialization=True,
                    )
                    controlnet_model.save_pretrained(
                        os.path.join(output_dir, "controlnet")
                    )
                    # Pop weights for both models
                    weights.pop()  # UNet
                    weights.pop()  # ControlNet
                else:
                    unet_lora_state_dict = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(models[0])
                    )

                    SDaIGControlNetPipeline.save_lora_weights(
                        save_directory=os.path.join(output_dir, "unet_lora"),
                        unet_lora_layers=unet_lora_state_dict,
                        safe_serialization=True,
                    )
                    weights.pop()  # UNet

        def load_model_hook(models, input_dir):
            if len(models) >= 2:
                controlnet_model = models.pop()
                if os.path.exists(os.path.join(input_dir, "controlnet")):
                    load_controlnet = ControlNetModel.from_pretrained(
                        input_dir, subfolder="controlnet"
                    )
                    controlnet_model.register_to_config(**load_controlnet.config)
                    controlnet_model.load_state_dict(load_controlnet.state_dict())
                    del load_controlnet

                unet_model = models.pop()
                if os.path.exists(os.path.join(input_dir, "unet_lora")):
                    # Load UNet LoRA weights
                    lora_path = os.path.join(input_dir, "unet_lora")
                    unet_model.load_adapter(lora_path)
            else:
                unet_model = models.pop()
                if os.path.exists(os.path.join(input_dir, "unet_lora")):
                    # Load UNet LoRA weights
                    lora_path = os.path.join(input_dir, "unet_lora")
                    unet_model.load_adapter(lora_path)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()

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

    # Only train LoRA parameters
    trainable_params = [p for p in trainable_params if p.requires_grad]

    if not trainable_params:
        raise ValueError(
            "No trainable parameters found. Make sure at least one LoRA option is enabled."
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
        train_dataset = build_dataset_from_cfg(dataset_cfg.data.finetune)

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
    unet, controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # Move text_encoder and vae to gpu and cast to weight_dtype
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
        # Filter out non-serializable values from args for tracker config
        tracker_config = {}
        for key, value in vars(args).items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                tracker_config[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings for tracking
                tracker_config[key] = ",".join(str(v) for v in value) if value else ""
            else:
                # Skip non-serializable objects
                continue
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
            f"  Training mode: Autoregressive multi-step diffusion(DDIM inference steps: {args.num_inference_steps}, max timesteps: {args.max_timesteps})"
        )
    else:
        logger.info(
            f"  Training mode: Autoregressive single-step diffusion(fixed timestep: {args.timestep})"
        )

    logger.info(f"  LoRA Configuration:")
    logger.info(f"    Rank: {args.lora_rank}")
    logger.info(f"    Alpha: {args.lora_alpha}")
    logger.info(f"    Dropout: {args.lora_dropout}")
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
        log_rgb_loss = 0.0
        log_rgb_losses = [0.0, 0.0, 0.0]

        for _ in range(len(train_dataloader)):
            batch = next(iter_nuscenes)
            with accelerator.accumulate(unet, controlnet):
                # Access data using tuple keys - these will be in [batch_size, 6, ...] format
                if args.train_batch_size != 1:
                    raise NotImplemented
                rgb_cond = batch[("pixel_values", 0)].view(
                    6 * args.train_batch_size, *batch[("pixel_values", 0)].shape[2:]
                )  # [6*batch_size, 3, H, W]
                ego_masks = batch["ego_masks"].view(
                    6 * args.train_batch_size, *batch["ego_masks"].shape[2:]
                )  # [6*batch_size, 1, H, W]
                bsz_per_task = rgb_cond.shape[0]
                bsz = bsz_per_task * 2
                # Get the empty text embedding for conditioning
                prompt = ""
                text_inputs = tokenizer(
                    prompt,
                    padding="do_not_pad",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(rgb_cond.device)
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

                rgb_losses = []
                loss = 0.0
                for frame in range(3):
                    disparity_cond = batch[("box_disparity_maps", frame)].view(
                        6 * args.train_batch_size,
                        *batch[("box_disparity_maps", 0)].shape[2:],
                    )  # [6*batch_size, 1, H, W]
                    controlnet_cond = torch.cat(
                        [rgb_cond, disparity_cond], dim=1
                    )  # [6*batch_size, 4, H, W]
                    controlnet_cond = torch.cat(
                        [controlnet_cond, controlnet_cond], dim=0
                    )
                    controlnet_cond = controlnet_cond.to(weight_dtype)

                    if args.multi_step_training:
                        # Set timesteps for training - Always use autoregressive training
                        # Set up DDIM scheduler for inference
                        ddim_scheduler.set_timesteps(
                            args.num_inference_steps, device=controlnet_cond.device
                        )
                        inference_timesteps = ddim_scheduler.timesteps

                        # Directly sample from inference timesteps for autoregressive training
                        target_step = torch.randint(
                            0,
                            len(inference_timesteps),
                            (1,),
                            device=controlnet_cond.device,
                        ).item()

                    # Encode target images (ground truth) for loss computation
                    target_rgb = batch[("pixel_values", frame)].view(
                        6 * args.train_batch_size,
                        *batch[("pixel_values", frame)].shape[2:],
                    )  # [6*batch_size, 3, H, W]

                    # Encode with VAE to get ground truth latents
                    rgb_latents = vae.encode(
                        target_rgb.to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    rgb_latents = rgb_latents * vae.config.scaling_factor

                    # Clear intermediate VAE tensors
                    del target_rgb
                    torch.cuda.empty_cache()

                    # Start from pure noise for inference (not ground truth latents!)
                    noisy_rgb_latents = torch.randn_like(rgb_latents)
                    noisy_disparity_latents = torch.randn_like(rgb_latents)

                    # Add noise offset if specified
                    if args.noise_offset:
                        noisy_rgb_latents += args.noise_offset * torch.randn(
                            (
                                noisy_rgb_latents.shape[0],
                                noisy_rgb_latents.shape[1],
                                1,
                                1,
                            ),
                            device=noisy_rgb_latents.device,
                        )
                        noisy_disparity_latents += args.noise_offset * torch.randn(
                            (
                                noisy_disparity_latents.shape[0],
                                noisy_disparity_latents.shape[1],
                                1,
                                1,
                            ),
                            device=noisy_disparity_latents.device,
                        )

                    # Concatenate for processing
                    noisy_latents = torch.cat(
                        [noisy_rgb_latents, noisy_disparity_latents], dim=1
                    )

                    if args.multi_step_training:
                        # DDIM inference WITHOUT gradients to the target timestep
                        with torch.no_grad():
                            for i, t in enumerate(inference_timesteps[:target_step]):
                                # Prepare model inputs
                                latent_model_input = ddim_scheduler.scale_model_input(
                                    noisy_latents, t
                                )

                                # ControlNet forward
                                down_block_res_samples, mid_block_res_sample = (
                                    controlnet(
                                        torch.cat(
                                            [latent_model_input, latent_model_input],
                                            dim=0,
                                        ),
                                        timestep=t,
                                        encoder_hidden_states=encoder_hidden_states,
                                        controlnet_cond=controlnet_cond,
                                        return_dict=False,
                                        class_labels=task_emb,
                                    )
                                )

                                # UNet forward (predicting x0, not noise)
                                x0_pred = unet(
                                    torch.cat(
                                        [latent_model_input, latent_model_input], dim=0
                                    ),
                                    t,
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
                                x0_pred = torch.cat(
                                    [x0_pred[:bsz_per_task], x0_pred[bsz_per_task:]],
                                    dim=1,
                                )
                                # DDIM step
                                noisy_latents = ddim_scheduler.step(
                                    x0_pred, t, noisy_latents, return_dict=False
                                )[0]

                        # Now use the partially denoised latents for final prediction WITH gradients
                        target_timestep = inference_timesteps[target_step]
                        timesteps = target_timestep.repeat(bsz)
                    else:
                        # Fixed timestep for single-step training
                        timesteps = torch.tensor(
                            [args.timestep], device=rgb_cond.device
                        ).repeat(bsz)
                    torch.cuda.empty_cache()
                    # Predict
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        torch.cat([noisy_latents, noisy_latents], dim=0),
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=controlnet_cond,
                        return_dict=False,
                        class_labels=task_emb,
                    )
                    model_pred = unet(
                        torch.cat([noisy_latents, noisy_latents], dim=0),
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
                    breakpoint()
                    # Compute loss - since we're predicting x0, compare directly with ground truth latents
                    rgb_loss = F.mse_loss(
                        model_pred[:bsz_per_task].float(),
                        rgb_latents[:bsz_per_task].float(),
                        reduction="mean",
                    )
                    loss += rgb_loss
                    rgb_losses.append(rgb_loss.item())

                    # Gather loss
                    avg_rgb_loss = accelerator.gather(
                        rgb_loss.repeat(args.train_batch_size * 6)
                    ).mean()
                    log_rgb_loss += (
                        avg_rgb_loss.item() / args.gradient_accumulation_steps
                    )
                    log_rgb_losses[frame] += log_rgb_loss
                    train_loss += log_rgb_loss
                    if frame != 2:
                        # Ensure proper type casting for VAE decode operations
                        rgb_out = vae.decode(
                            (model_pred[:bsz_per_task] / vae.config.scaling_factor).to(
                                dtype=weight_dtype
                            )
                        ).sample()
                        rgb_out = rgb_out / 2 + 0.5
                        rgb_out = rgb_out.clamp(0, 1)
                        concat_6_views(rgb_out).save("rgb_out.png")

                        disparity_out = vae.decode(
                            (model_pred[bsz_per_task:] / vae.config.scaling_factor).to(
                                dtype=weight_dtype
                            )
                        ).sample()
                        disparity_out = disparity_out / 2 + 0.5
                        disparity_out = disparity_out.mean(dim=1)
                        disparity_out = disparity_out.clamp(0, 1)
                        concat_and_visualize_6_depths(
                            set_inf_to_max(disparity2depth(disparity_out)),
                            save_path="disparity_out.png",
                            vmax=50,
                        )

                        # Clear intermediate tensors to free memory
                        del rgb_input, disparity_input

                        rgb_cond = render_novel_view(
                            current_images=rgb_out,
                            current_depths=set_inf_to_max(
                                disparity2depth(disparity_out)
                            ),
                            current_ego_masks=ego_masks.squeeze(),
                            current_intrinsics=batch["intrinsics"].squeeze(),
                            current_extrinsics=batch["extrinsics", frame].squeeze(),
                            novel_intrinsics=batch["intrinsics"].squeeze(),
                            novel_extrinsics=batch["extrinsics", frame + 1].squeeze(),
                            current_objs_to_world=batch[
                                "objs_to_world", frame
                            ].squeeze(),
                            current_box_sizes=batch["box_sizes", frame].squeeze(),
                            transforms_cur_to_next=batch[
                                "transforms", frame, frame + 1
                            ].squeeze(),
                            expanding_factor=1.0,
                            image_size=(dataset_cfg.height, dataset_cfg.width),
                            render_depth=False,
                        )
                    torch.cuda.empty_cache()

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # Clip gradients for both UNet and ControlNet
                    accelerator.clip_grad_norm_(list(unet.parameters()) + list(controlnet.parameters()), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Clear cache to prevent memory buildup
                if global_step % 50 == 0:  # Clear cache every 50 steps
                    torch.cuda.empty_cache()

            logs = {
                "SL": loss.detach().item(),
                "SL_R0": rgb_losses[0],
                "SL_R1": rgb_losses[1],
                "SL_R2": rgb_losses[2],
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
                        "rgb_loss0": log_rgb_losses[0],
                        "rgb_loss1": log_rgb_losses[1],
                        "rgb_loss2": log_rgb_losses[2],
                    },
                    step=global_step,
                )
                train_loss = 0.0
                log_rgb_loss = 0.0
                log_rgb_losses = [0.0, 0.0, 0.0]

                checkpointing_steps = args.checkpointing_steps
                validation_steps = args.validation_steps

                if accelerator.is_main_process:
                    if global_step % checkpointing_steps == 0:
                        # Save LoRA weights separately
                        save_lora_weights(
                            unet, controlnet, args.output_dir, global_step, args
                        )
                        torch.cuda.empty_cache()

                    if global_step % validation_steps == 0:
                        log_validation(
                            batch,
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
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

        pipeline = SDaIGControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            controlnet=controlnet,
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
