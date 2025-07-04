import cv2
import numpy as np
import torch
from PIL import Image
from typing import Tuple
import matplotlib.pyplot as plt


def img_concat_h(im1, *args, color="black"):
    if len(args) == 1:
        im2 = args[0]
    else:
        im2 = img_concat_h(*args)
    height = max(im1.height, im2.height)
    mode = im1.mode
    dst = Image.new(mode, (im1.width + im2.width, height), color=color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def img_concat_v(im1, *args, color="black"):
    if len(args) == 1:
        im2 = args[0]
    else:
        im2 = img_concat_v(*args)
    width = max(im1.width, im2.width)
    mode = im1.mode
    dst = Image.new(mode, (width, im1.height + im2.height), color=color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def img_concat_h(im1, *args, color="black"):
    if len(args) == 1:
        im2 = args[0]
    else:
        im2 = img_concat_h(*args)
    height = max(im1.height, im2.height)
    mode = im1.mode
    dst = Image.new(mode, (im1.width + im2.width, height), color=color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def img_concat_v(im1, *args, color="black"):
    if len(args) == 1:
        im2 = args[0]
    else:
        im2 = img_concat_v(*args)
    width = max(im1.width, im2.width)
    mode = im1.mode
    dst = Image.new(mode, (width, im1.height + im2.height), color=color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def concat_6_views(imgs: Tuple[Image.Image, ...], oneline=False):
    if isinstance(imgs, torch.Tensor):
        if imgs.shape[1] == 3:
            imgs = [
                Image.fromarray(
                    (img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                ).convert("RGB")
                for img in imgs
            ]
        else:
            assert imgs.shape[1] == 1
            imgs = [
                Image.fromarray(
                    (img.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
                ).convert("L")
                for img in imgs
            ]
    elif isinstance(imgs, np.ndarray):
        if imgs.shape[1] == 3:
            imgs = [
                Image.fromarray(
                    (img.transpose(1, 2, 0) * 255).astype(np.uint8)
                ).convert("RGB")
                for img in imgs
            ]
        else:
            assert imgs.shape[1] == 1
            imgs = [
                Image.fromarray((img.squeeze() * 255).astype(np.uint8)).convert("L")
                for img in imgs
            ]
    if oneline:
        image = img_concat_h(*imgs)
    else:
        image = img_concat_v(img_concat_h(*imgs[:3]), img_concat_h(*imgs[3:]))
    return image


def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity


def disparity2depth(disparity, return_mask=False):
    if isinstance(disparity, torch.Tensor):
        depth = torch.ones_like(disparity) * float("inf")
    elif isinstance(disparity, np.ndarray):
        depth = np.ones_like(disparity) * float("inf")
    non_negtive_mask = disparity > 0
    depth[non_negtive_mask] = 1.0 / disparity[non_negtive_mask]
    if return_mask:
        return depth, non_negtive_mask
    else:
        return depth


def set_inf_to_max(depth_maps):
    # Set inf values in depth_maps to the maximal finite value
    if torch.is_tensor(depth_maps):
        finite_mask = torch.isfinite(depth_maps)
        if finite_mask.any():
            max_val = torch.max(depth_maps[finite_mask])
            depth_maps = torch.where(torch.isinf(depth_maps), max_val, depth_maps)
    else:
        finite_mask = np.isfinite(depth_maps)
        if finite_mask.any():
            max_val = np.max(depth_maps[finite_mask])
            depth_maps = np.where(np.isinf(depth_maps), max_val, depth_maps)
    return depth_maps


def concat_and_visualize_6_depths(
    depths: Tuple[torch.Tensor, ...], save_path="concat_6_depths.png", vmax=None
):
    if isinstance(depths, torch.Tensor):
        depths = [depth.squeeze().detach().cpu().numpy() for depth in depths]
    elif isinstance(depths, np.ndarray):
        depths = [depth.squeeze() for depth in depths]

    fig, axes = plt.subplots(2, 3, figsize=(12, 4))
    vmin = 0  # min([np.min(d) for d in depths])
    vmax = max([np.max(d) for d in depths]) if vmax is None else vmax
    depths = [np.clip(depth, vmin, vmax) for depth in depths]

    ims = []
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(depths[i], cmap='jet', vmin=vmin, vmax=vmax)
        ax.axis('off')
        ims.append(im)

    fig.subplots_adjust(right=0.85, left=0.02, top=0.98, bottom=0.02, wspace=0.02, hspace=0.05)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    fig.colorbar(ims[0], cax=cbar_ax)
    plt.savefig(save_path)
    plt.close(fig)
