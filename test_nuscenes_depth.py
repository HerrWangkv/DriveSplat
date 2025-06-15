"""
Test script for NuScenesBoxDepth to ensure it works with the new box_depth_projection utility.
"""
import os
import sys
import torch
import numpy as np
import torchvision.utils as vutils

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import necessary modules
from datasets.Nuscenes.NuScenes import NuScenesBase, NuScenesBoxDepth, NuScenesCameraImages

def main():
    print("Testing NuScenesBoxDepth with the new box_depth_projection utility...")

    nusc = NuScenesBase(
        version='v1.0-mini',
        dataroot='/storage_local/kwang/nuscenes/raw',  # Update this to your NuScenes data path
        cache_dir='/storage_local/kwang/nuscenes/cache',    # Update this to your cache directory
        split='mini_train',
        view_order=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        object_classes=[],
        map_classes=[],
        lane_classes=[],
        map_bound=[-50, 50, 0.5],
        N=0,  # No interpolation for simple testing
        verbose=False
    )
    
    print("NuScenesBase initialized successfully")
    
    # Create the NuScenesBoxDepth dataset
    cam_img = NuScenesCameraImages(nusc, img_size=(900,1600))
    box_depth = NuScenesBoxDepth(nusc, img_size=(900, 1600))
    print(f"NuScenesBoxDepth initialized with {len(box_depth)} frames")
        
    first_cam = cam_img[0]["pixel_values"]
    first_depth = box_depth[0]
    
    # Save a depth visualization
    import matplotlib.pyplot as plt
    for cam in range(len(first_depth)):
        depth_map = first_depth[cam]
        cam_img = first_cam[cam]
        plt.imshow(cam_img.cpu().numpy().transpose(1, 2, 0))
        plt.title(f"Camera {cam} Image")
        plt.axis('off')
        plt.savefig(f"cam_{cam}_image.png", bbox_inches='tight')
        plt.close()
        print(f"Depth map shape for {cam}: {depth_map.shape}")

        plt.imshow(depth_map.cpu().numpy(), cmap='viridis')
        plt.title("Depth Map Visualization")
        plt.colorbar(label='Depth')
        plt.axis('off')
        plt.savefig(f"{cam}.png", bbox_inches='tight')
        plt.close()
            
    
    print("\nDone!")

if __name__ == "__main__":
    main()
