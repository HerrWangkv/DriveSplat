import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    PointsRenderer,
)
from pytorch3d.transforms import Transform3d


def depth_to_pointcloud(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    features: Optional[torch.Tensor] = None,
    ego_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert depth map to 3D point cloud in world coordinates.

    Args:
        depth: Depth map tensor of shape (H, W) or (B, H, W)
        intrinsics: Camera intrinsic matrix of shape (3, 3) or (B, 3, 3)
        extrinsics: Camera extrinsic matrix of shape (4, 4) or (B, 4, 4)
        features: Optional feature tensor of shape (H, W, C) or (B, H, W, C)
        ego_mask: Optional ego mask tensor of shape (H, W) or (B, H, W)
                  True/1 for ego pixels to be filtered out, False/0 for valid pixels

    Returns:
        points: 3D points in world coordinates (N, 3)
        featues: features for points (N, C)
    """
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
        batch_size = 1
        squeeze_output = True
    else:
        batch_size = depth.shape[0]
        squeeze_output = False

    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0).expand(batch_size, -1, -1)
    if extrinsics.dim() == 2:
        extrinsics = extrinsics.unsqueeze(0).expand(batch_size, -1, -1)

    # Handle ego mask dimensions
    if ego_mask is not None:
        if ego_mask.dim() == 2:
            ego_mask = ego_mask.unsqueeze(0).expand(batch_size, -1, -1)
        elif ego_mask.dim() == 3 and ego_mask.shape[0] != batch_size:
            # If ego_mask has different batch size, broadcast it
            ego_mask = ego_mask.expand(batch_size, -1, -1)

    device = depth.device
    H, W = depth.shape[-2:]

    # Create pixel coordinates
    y, x = torch.meshgrid(torch.arange(H, device=device), 
                         torch.arange(W, device=device), indexing='ij')
    x = x.float()
    y = y.float()

    # Convert to homogeneous coordinates
    ones = torch.ones_like(x)
    pixel_coords = torch.stack([x, y, ones], dim=-1)  # (H, W, 3)
    pixel_coords = pixel_coords.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, H, W, 3)

    # Get valid depth points (non-zero depth values and not in ego mask)
    valid_mask = depth > 0
    if ego_mask is not None:
        # Filter out ego mask pixels (ego_mask=True means ego pixels to be filtered out)
        valid_mask = valid_mask & (~ego_mask.bool())

    points_list = []
    features_list = []

    for b in range(batch_size):
        valid_pixels = pixel_coords[b][valid_mask[b]]  # (N_valid, 3)
        valid_depths = depth[b][valid_mask[b]]  # (N_valid,)

        if len(valid_pixels) == 0:
            # Handle case with no valid depth points
            points_list.append(torch.zeros((0, 3), device=device))
            if features is not None:
                features_list.append(torch.zeros((0, 3), device=device))
            continue

        # Convert pixel coordinates to camera coordinates
        K_inv = torch.inverse(intrinsics[b])
        cam_coords = torch.matmul(valid_pixels, K_inv.T)  # (N_valid, 3)
        cam_coords = cam_coords * valid_depths.unsqueeze(-1)  # Scale by depth

        # Convert to homogeneous coordinates
        cam_coords_homo = torch.cat([cam_coords, torch.ones(cam_coords.shape[0], 1, device=device)], dim=-1)

        # Transform to world coordinates
        world_coords = torch.matmul(cam_coords_homo, torch.inverse(extrinsics[b]).T)
        points_list.append(world_coords[:, :3])

        # Extract features if provided
        if features is not None:
            # Handle batch dimension for features
            if features.dim() == 3 and batch_size == 1:
                # Single image case: (C, H, W) -> (1, C, H, W) -> (1, H, W, C)
                features_batch = features.unsqueeze(0).permute(0, 2, 3, 1)
            elif features.dim() == 4:
                features_batch = features
            else:
                # Already correct format or other case
                features_batch = features
                if features_batch.dim() == 3:  # (H, W, C) -> (1, H, W, C)
                    features_batch = features_batch.unsqueeze(0)

            # Now features_batch should be in (B, H, W, C) format
            valid_features = features_batch[b][valid_mask[b]]  # (N_valid, 3)
            features_list.append(valid_features)

    # Concatenate results
    if squeeze_output and batch_size == 1:
        points = points_list[0]
        features = features_list[0] if features is not None else None
    else:
        # Concatenate all points from different cameras
        if points_list:
            points = torch.cat(points_list, dim=0)  # (N_total, 3)
            if features is not None:
                features = torch.cat(features_list, dim=0)  # (N_total, 3)
            else:
                features = None
        else:
            points = torch.zeros((0, 3), device=device)
            features = (
                torch.zeros((0, 3), device=device) if features is not None else None
            )

    return points, features


def assign_points_to_objects(
    points: torch.Tensor,
    objects_to_world: torch.Tensor,
    box_sizes: torch.Tensor,
    object_ids: torch.Tensor,
    expanding_factor: float = 1.0,
) -> torch.Tensor:
    """
    Assign points to objects based on their 3D positions, assuming objects are vertical.
    Uses 2D bounding box check in X-Y plane with Z bounds for autonomous driving scenarios.

    Args:
        points: 3D points tensor of shape (N, 3)
        objects_to_world: Object transformation matrices of shape (M, 4, 4)
        box_sizes: Sizes of the boxes for each object of shape (M, 3) in [length, width, height]
        object_ids: Object IDs for current objects, shape (M,) or None
        expanding_factor: Factor to expand the bounding boxes (default: 1.0, no expansion)

    Returns:
        assignments: Tensor of shape (N,) with object indices. -1 indicates background points.
    """
    device = points.device
    N = points.shape[0]
    M = objects_to_world.shape[0]

    assignments = torch.full((N,), -1, dtype=torch.long, device=device)

    if M == 0:
        return assignments

    for obj_idx in range(M):
        # Get object center in world coordinates (translation part of transformation matrix)
        obj_center_world = objects_to_world[obj_idx][:3, 3]  # Shape: (3,)
        obj_id = object_ids[obj_idx]

        # For vertical objects, we primarily care about X-Y plane membership
        # Calculate X-Y bounds in world coordinates with expanding factor
        box_half_length = (
            box_sizes[obj_idx][0] / 2.0
        ) * expanding_factor  # length/2 * factor
        box_half_width = (
            box_sizes[obj_idx][1] / 2.0
        ) * expanding_factor  # width/2 * factor
        box_height = box_sizes[obj_idx][2] * expanding_factor  # full height * factor

        # First, do a rough distance filter to avoid rotating all points
        # Use a conservative bounding circle in X-Y plane
        max_radius = (
            torch.sqrt(box_half_length**2 + box_half_width**2) * 1.1
        )  # 10% margin

        # Quick distance check in X-Y plane
        xy_distances = torch.norm(
            points[:, :2] - obj_center_world[:2].unsqueeze(0), dim=1
        )
        rough_candidates = xy_distances <= max_radius

        # Quick Z bounds check
        z_center = obj_center_world[2]
        z_min = z_center - box_height / 2.0
        z_max = z_center + box_height / 2.0
        z_candidates = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)

        # Combine rough filters
        candidate_mask = rough_candidates & z_candidates

        if not candidate_mask.any():
            continue  # No points are even close to this object

        # Only process candidate points for precise checking
        candidate_points = points[candidate_mask]

        # Extract object rotation matrix for X-Y plane transformation
        obj_rotation_xy = objects_to_world[obj_idx][:2, :2]  # 2x2 rotation in X-Y plane

        # Transform candidate points to object-centered coordinate system (X-Y plane only)
        points_centered = candidate_points[:, :2] - obj_center_world[:2].unsqueeze(
            0
        )  # (N_candidates, 2)

        # Rotate to object-aligned coordinates in X-Y plane (only for candidates)
        obj_rotation_xy_inv = torch.inverse(obj_rotation_xy)
        points_obj_xy = torch.matmul(
            points_centered, obj_rotation_xy_inv.T
        )  # (N_candidates, 2)

        # Check X-Y plane membership (primary condition for vertical objects)
        xy_inside_mask = (
            torch.abs(points_obj_xy[:, 0]) <= box_half_length
        ) & (  # length check
            torch.abs(points_obj_xy[:, 1]) <= box_half_width
        )  # width check

        # Final Z bounds check for candidates (redundant but precise)
        z_inside_mask = (candidate_points[:, 2] >= z_min) & (
            candidate_points[:, 2] <= z_max
        )

        # Combine X-Y and Z conditions for candidates
        inside_mask_candidates = xy_inside_mask & z_inside_mask

        # Map back to original point indices and assign
        if inside_mask_candidates.any():
            candidate_indices = torch.where(candidate_mask)[0]
            final_indices = candidate_indices[inside_mask_candidates]
            assignments[final_indices] = obj_id

    return assignments


def move_objects_in_pointcloud(
    points: torch.Tensor,
    objects_to_world: torch.Tensor,
    box_sizes: torch.Tensor,
    object_ids: torch.Tensor,
    transforms_cur_to_next: torch.Tensor,
    expanding_factor: float = 1.0,
) -> torch.Tensor:
    """
    Move objects in point cloud based on their transformations.

    Args:
        points: 3D points tensor of shape (N, 3)
        objects_to_world: Object transformation matrices of shape (M, 4, 4)
        box_sizes: Sizes of the boxes for each object of shape (M, 3)
        object_ids: Object IDs for current objects, shape (M,) or None
        transforms_cur_to_next: Combined transformations (rotation + translation + scaling)
                               from current to next frame of shape (M, 4, 4)
        expanding_factor: Factor to expand the bounding boxes for point assignment (default: 1.0)
    Returns:
        moved_points: Points after applying transformations, shape (N, 3)
        point_object_assignments: Tensor of shape (N,) with object indices.
    """
    device = points.device
    moved_points = points.clone()

    if objects_to_world.shape[0] == 0:
        # No objects to transform
        return moved_points, torch.full(
            (points.shape[0],), -1, dtype=torch.long, device=device
        )

    # Convert points to homogeneous coordinates
    homogeneous_points = torch.cat(
        [points, torch.ones(points.shape[0], 1, device=device)], dim=-1
    )

    # Automatically assign points to objects
    point_object_assignments = assign_points_to_objects(
        points=points,
        objects_to_world=objects_to_world,
        box_sizes=box_sizes,
        object_ids=object_ids,
        expanding_factor=expanding_factor,
    )

    # Transform points based on their object assignments
    for obj_idx in range(objects_to_world.shape[0]):
        if transforms_cur_to_next[obj_idx].abs().sum() == 0:
            continue  # No transformation for this object
        # Find points belonging to this object
        point_mask = point_object_assignments == object_ids[obj_idx]
        if not point_mask.any():
            continue

        # Get points for this object
        obj_points = homogeneous_points[point_mask]

        transformed_points = torch.matmul(obj_points, transforms_cur_to_next[obj_idx].T)

        # Update the moved points for this object
        moved_points[point_mask] = transformed_points[:, :3]

    return moved_points, point_object_assignments


def paste_ego_area(
    current: torch.Tensor,
    current_ego_mask: torch.Tensor,
    novel: torch.Tensor,
):
    if current.dim() == 3:
        novel = torch.where(current_ego_mask.bool(), current, novel)
    elif current.dim() == 4:
        novel = torch.where(current_ego_mask.bool().unsqueeze(-1), current, novel)
    else:
        raise ValueError("Unsupported tensor dimensions for pasting ego area")

    return novel


def render_novel_views_using_point_cloud(
    current_features: Union[torch.Tensor, np.ndarray],
    current_depths: Union[torch.Tensor, np.ndarray],
    current_ego_mask: torch.Tensor,
    current_intrinsics: torch.Tensor,
    current_extrinsics: torch.Tensor,
    novel_intrinsics: torch.Tensor,
    novel_extrinsics: torch.Tensor,
    current_objs_to_world: Optional[torch.Tensor] = None,
    current_box_sizes: Optional[torch.Tensor] = None,
    current_obj_ids: Optional[torch.Tensor] = None,
    transforms_cur_to_next: Optional[torch.Tensor] = None,
    expanding_factor: float = 1.0,
    image_size: Tuple[int, int] = (512, 512),
    point_radius: float = 0.01,
    points_per_pixel: int = 8,
    background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    return_novel_depths: bool = True,
    return_current_points: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Render novel view using PyTorch3D point cloud rendering with optional object movement.

    Args:
        current_features: Current view RGB images or latents, shape (B, C, H, W), (C, H, W), (B, H, W, C), or (H, W, C)
                       Can be numpy array or torch tensor
        current_depths: Current view depth maps, shape (B, H, W) or (H, W)
                       Can be numpy array or torch tensor
        current_ego_mask: Mask for ego objects, shape (B, H, W) or (H, W)
        current_intrinsics: Current camera intrinsics, shape (B, 3, 3) or (3, 3)
        current_extrinsics: Current camera extrinsics, shape (B, 4, 4) or (4, 4)
        novel_intrinsics: Novel view intrinsics, shape (B, 3, 3) or (3, 3)
        novel_extrinsics: Novel view extrinsics, shape (B, 4, 4) or (4, 4)
        current_objs_to_world: Optional object-to-world matrices, shape (M, 4, 4)
        current_box_sizes: Optional object box sizes, shape (M, 3)
        current_obj_ids: Optional object IDs for current objects, shape (M,) or None
        transforms_cur_to_next: Optional transformations from current to next frame, shape (M, 4, 4)
        expanding_factor: Factor to expand bounding boxes for point assignment (default: 1.0)
        image_size: Output image size (height, width)
        point_radius: Radius of rendered points
        points_per_pixel: Number of points to consider per pixel
        background_color: Background feature
        return_novel_depths: Whether to return depth maps for novel views
        return_current_points: If True, returns the current point cloud as well

    Returns:
        novel_features: Novel view RGB images or latents, shape (B, H, W, C)
        novel_depths: Novel view depth maps (if return_novel_depths is True), shape (B, H, W)
        cur_points: Current point cloud (if return_current_points is True), shape (N, 3)
        cur_features: Current point features (if return_current_points is True), shape (N, 3)
        cur_assignments: Current point assignments (if return_current_points is True), shape (N,)
    """
    ret = {}
    # Convert numpy arrays to torch tensors if needed
    if isinstance(current_features, np.ndarray):
        current_features = torch.from_numpy(current_features).float()
    if isinstance(current_depths, np.ndarray):  
        current_depths = torch.from_numpy(current_depths).float()

    # Move to same device as intrinsics
    device = current_intrinsics.device
    current_features = current_features.to(device)
    current_depths = current_depths.to(device)
    current_ego_mask = current_ego_mask.to(device)

    # Ensure batch dimension and convert to (B, H, W, C) format for depth_to_pointcloud
    if current_features.dim() == 3:
        # Single image case (C, H, W) -> (1, H, W, C)
        current_features = current_features.permute(1, 2, 0).unsqueeze(0)
        batch_size = 1
        squeeze_output = True
    elif current_features.dim() == 4:
        current_features = current_features.permute(
            0, 2, 3, 1
        )  # (B, C, H, W) -> (B, H, W, C)
        batch_size = current_features.shape[0]
        squeeze_output = False
    else:
        raise ValueError("current_features must be of shape (B, C, H, W) or (C, H, W)")

    if current_depths.dim() == 2:
        current_depths = current_depths.unsqueeze(0)
    if current_ego_mask.dim() == 2:
        current_ego_mask = current_ego_mask.unsqueeze(0)
    if current_intrinsics.dim() == 2:
        current_intrinsics = current_intrinsics.unsqueeze(0)
    if current_extrinsics.dim() == 2:
        current_extrinsics = current_extrinsics.unsqueeze(0)
    if novel_intrinsics.dim() == 2:
        novel_intrinsics = novel_intrinsics.unsqueeze(0)
    if novel_extrinsics.dim() == 2:
        novel_extrinsics = novel_extrinsics.unsqueeze(0)

    # Convert depth maps to point clouds (concatenated from all cameras)
    points, features = depth_to_pointcloud(
        current_depths,
        current_intrinsics,
        current_extrinsics,
        current_features,
        current_ego_mask,
    )
    if return_current_points:
        ret["cur_points"] = points.clone()  # Save current points if needed
        ret["cur_features"] = features.clone()

    # Apply object movement if transformation data is provided
    if (
        current_objs_to_world is not None
        and current_box_sizes is not None
        and transforms_cur_to_next is not None
    ):

        # Move objects in the point cloud based on transformations
        points, assignments = move_objects_in_pointcloud(
            points=points,
            objects_to_world=current_objs_to_world,
            box_sizes=current_box_sizes,
            object_ids=current_obj_ids,
            transforms_cur_to_next=transforms_cur_to_next,
            expanding_factor=expanding_factor,
        )
        if return_current_points:
            ret["cur_assignments"] = assignments.clone()

    # Create single PyTorch3D point cloud from all cameras
    # Filter out zero points
    if points.dim() == 2:
        valid_points_mask = torch.any(points != 0, dim=-1)
    else:
        # This shouldn't happen with concatenation, but handle just in case
        raise ValueError("Unexpected point cloud dimensions after concatenation")

    # Set up novel view cameras using batch processing
    R_batch = torch.zeros(batch_size, 3, 3, dtype=torch.float32, device=device)
    T_batch = torch.zeros(batch_size, 3, dtype=torch.float32, device=device)
    K_batch = torch.zeros(batch_size, 4, 4, dtype=torch.float32, device=device)

    # Create R_flip tensor for PyTorch3D camera convention
    R_flip = torch.tensor([
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device=device)

    for b in range(batch_size):
        # Convert extrinsics to PyTorch3D camera format
        # The extrinsics matrix is already world-to-camera transformation
        world_to_cam = novel_extrinsics[b].clone()

        # Apply PyTorch3D camera convention
        world_to_cam[:3, :3] = torch.matmul(R_flip, world_to_cam[:3, :3])
        world_to_cam[:3, 3] = torch.matmul(R_flip, world_to_cam[:3, 3])

        R = world_to_cam[:3, :3].T  # PyTorch3D expects transposed rotation
        T = world_to_cam[:3, 3]

        # Prepare camera matrices for batch
        R_batch[b] = R.contiguous()
        T_batch[b] = T.contiguous()
        K_batch[b, :3, :3] = novel_intrinsics[b]
        K_batch[b, 3, 2] = 1.0

    # Create batched camera
    cameras = PerspectiveCameras(
        K=K_batch,
        R=R_batch,
        T=T_batch,
        in_ndc=False,
        image_size=((image_size[0], image_size[1]),),
        device=device,
    )

    # Set up rasterization settings
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=point_radius,
        points_per_pixel=points_per_pixel,
    )

    # Create renderer with batched cameras
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    compositor = AlphaCompositor(background_color=background_color)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)

    # Create replicated point clouds for batch rendering
    replicated_points = [points[valid_points_mask] for _ in range(batch_size)]
    if features is not None:
        replicated_features = [features[valid_points_mask] for _ in range(batch_size)]
        point_cloud_batch = Pointclouds(points=replicated_points, features=replicated_features)
    else:
        point_cloud_batch = Pointclouds(points=replicated_points)

    # Render all novel views at once
    rendered = renderer(point_cloud_batch)

    # Extract RGB and depth
    novel_features = rendered  # (B, H, W, 3)
    ret["novel_features"] = paste_ego_area(
        current_features, current_ego_mask, novel_features
    )
    if return_novel_depths:
        # Extract depth from rasterizer's z-buffer
        # The rasterizer provides depth information through its fragments
        # Re-rasterize to get fragments with depth information
        fragments = rasterizer(point_cloud_batch)
        # Extract depth from z-buffer - fragments.zbuf shape: (B, H, W, K) where K is points_per_pixel
        zbuf = fragments.zbuf[..., 0]  # Take closest point depth: (B, H, W)

        # Create depth mask - valid where zbuf > 0 (valid depth)
        valid_depth_mask = zbuf > 0

        # Initialize depth maps
        novel_depths = torch.ones(batch_size, *image_size, device=device) * float(
            "inf"
        )  # Set to inf initially

        # Set depth values where valid
        novel_depths[valid_depth_mask] = zbuf[valid_depth_mask]

        if squeeze_output:
            novel_features = novel_features.squeeze(0)
            novel_depths = novel_depths.squeeze(0)
        ret["novel_depths"] = novel_depths
    return ret


def create_camera_matrices(fx: float, fy: float, cx: float, cy: float,
                          rotation: torch.Tensor, translation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create intrinsic and extrinsic camera matrices.
    
    Args:
        fx, fy: Focal lengths
        cx, cy: Principal point coordinates
        rotation: Rotation matrix (3, 3) or rotation vector (3,)
        translation: Translation vector (3,)
    
    Returns:
        intrinsics: Intrinsic matrix (3, 3)
        extrinsics: Extrinsic matrix (4, 4)
    """
    device = rotation.device
    
    # Create intrinsic matrix
    intrinsics = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    # Handle rotation
    if rotation.shape == (3,):
        # Convert rotation vector to matrix (simplified - in practice use proper rodrigues formula)
        angle = torch.linalg.norm(rotation)
        if angle > 1e-8:
            axis = rotation / angle
            # Rodrigues' rotation formula (simplified version)
            K = torch.tensor([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ], device=device)
            R = torch.eye(3, device=device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * torch.matmul(K, K)
        else:
            R = torch.eye(3, device=device)
    else:
        R = rotation
    
    # Create extrinsic matrix
    extrinsics = torch.eye(4, device=device)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = translation
    
    return intrinsics, extrinsics


def interpolate_camera_path(start_extrinsics: torch.Tensor, 
                           end_extrinsics: torch.Tensor, 
                           num_frames: int) -> torch.Tensor:
    """
    Interpolate camera path between two camera poses.
    
    Args:
        start_extrinsics: Starting camera extrinsic matrix (4, 4)
        end_extrinsics: Ending camera extrinsic matrix (4, 4)
        num_frames: Number of interpolated frames
    
    Returns:
        interpolated_extrinsics: Interpolated camera poses (num_frames, 4, 4)
    """
    device = start_extrinsics.device
    
    # Extract rotation and translation
    R_start = start_extrinsics[:3, :3]
    t_start = start_extrinsics[:3, 3]
    R_end = end_extrinsics[:3, :3]
    t_end = end_extrinsics[:3, 3]
    
    # Linear interpolation for translation
    alphas = torch.linspace(0, 1, num_frames, device=device)
    interpolated_translations = torch.stack([
        (1 - alpha) * t_start + alpha * t_end for alpha in alphas
    ])
    
    # SLERP for rotation (simplified - for proper SLERP, convert to quaternions)
    # This is a simplified version - for production use proper SLERP
    interpolated_rotations = torch.stack([
        (1 - alpha) * R_start + alpha * R_end for alpha in alphas
    ])
    
    # Orthogonalize rotation matrices (Gram-Schmidt process)
    for i in range(num_frames):
        U, _, V = torch.svd(interpolated_rotations[i])
        interpolated_rotations[i] = torch.matmul(U, V.T)
    
    # Combine into extrinsic matrices
    interpolated_extrinsics = torch.eye(4, device=device).unsqueeze(0).repeat(num_frames, 1, 1)
    interpolated_extrinsics[:, :3, :3] = interpolated_rotations
    interpolated_extrinsics[:, :3, 3] = interpolated_translations
    
    return interpolated_extrinsics


def depth_consistency_check(depth1: torch.Tensor, depth2: torch.Tensor, 
                           intrinsics1: torch.Tensor, extrinsics1: torch.Tensor,
                           intrinsics2: torch.Tensor, extrinsics2: torch.Tensor,
                           threshold: float = 0.1) -> torch.Tensor:
    """
    Check depth consistency between two views.
    
    Args:
        depth1, depth2: Depth maps for view 1 and 2
        intrinsics1, intrinsics2: Intrinsic matrices for both views
        extrinsics1, extrinsics2: Extrinsic matrices for both views
        threshold: Consistency threshold
    
    Returns:
        consistency_mask: Boolean mask indicating consistent depths
    """
    device = depth1.device
    H, W = depth1.shape
    
    # Project depth1 to view 2
    points1, _ = depth_to_pointcloud(depth1, intrinsics1, extrinsics1)
    
    if len(points1) == 0:
        return torch.zeros_like(depth1, dtype=torch.bool)
    
    # Transform to camera 2 coordinates
    points1_homo = torch.cat([points1, torch.ones(points1.shape[0], 1, device=device)], dim=-1)
    cam2_coords = torch.matmul(points1_homo, extrinsics2.T)[:, :3]
    
    # Project to image 2
    cam2_coords_2d = cam2_coords[:, :2] / cam2_coords[:, 2:3]
    pixel_coords2 = torch.matmul(
        torch.cat([cam2_coords_2d, torch.ones(cam2_coords_2d.shape[0], 1, device=device)], dim=-1),
        intrinsics2.T
    )[:, :2]
    
    # Check consistency
    consistency_mask = torch.zeros_like(depth1, dtype=torch.bool)
    
    valid_mask = (
        (pixel_coords2[:, 0] >= 0) & (pixel_coords2[:, 0] < W) &
        (pixel_coords2[:, 1] >= 0) & (pixel_coords2[:, 1] < H) &
        (cam2_coords[:, 2] > 0)
    )
    
    if valid_mask.any():
        valid_pixels = pixel_coords2[valid_mask].long()
        projected_depths = cam2_coords[valid_mask, 2]
        actual_depths = depth2[valid_pixels[:, 1], valid_pixels[:, 0]]
        
        depth_diff = torch.abs(projected_depths - actual_depths)
        consistent = depth_diff < threshold
        
        # Map back to original coordinates (this is simplified)
        # In practice, you'd need to properly track the correspondence
        
    return consistency_mask


def visualize_point_cloud(points: torch.Tensor, colors: Optional[torch.Tensor] = None,
                         save_path: Optional[str] = None) -> None:
    """
    Simple point cloud visualization utility.
    
    Args:
        points: 3D points (N, 3)
        colors: RGB colors (N, 3), optional
        save_path: Path to save visualization, optional
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        points_np = points.detach().cpu().numpy()

        if colors is not None:
            colors_np = colors.detach().cpu().numpy()
            ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], 
                      c=colors_np, s=1)
        else:
            ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        ax.set_title('Point Cloud Visualization')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    except ImportError:
        print("Matplotlib not available for visualization")
