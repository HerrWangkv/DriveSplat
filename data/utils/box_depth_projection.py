import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
)
import numpy as np
from functools import lru_cache


class BoxDepthProjector:
    """
    Utility class for projecting 3D bounding boxes to 2D depth maps using PyTorch3D.
    This can be reused across different datasets that need to generate synthetic depth maps
    from 3D bounding box representations.
    """
    def __init__(self, img_size, device=None):
        """
        Initialize the box depth projector.
        
        Args:
            img_size: Tuple of (height, width) for output depth maps
            device: PyTorch device to use (defaults to CUDA if available, otherwise CPU)
        """
        self.img_height, self.img_width = img_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define PyTorch3D rasterization settings
        self.raster_settings = RasterizationSettings(
            image_size=(self.img_height, self.img_width),
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
            bin_size=0,  # Set bin_size to 0 to bypass binning for faster performance
        )

        # Pre-compute reusable data structures
        self._ones_8x1 = torch.ones((8, 1), device=self.device)

        # Create R_flip tensor for PyTorch3D camera convention
        self.R_flip = torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Precompute unit box mesh
        _ = self._create_unit_bbox_mesh()  # Warm up cache

    # Cache the mesh creation for better performance
    @lru_cache(maxsize=128)
    def _create_unit_bbox_mesh(self):
        """
        Create a unit 3D mesh for a bounding box (1,1,1).
        This will be scaled later for actual box sizes.

        Returns:
            vertices: Tensor of vertices [8, 3]
            faces: Tensor of face indices [12, 3]
        """
        # Define vertices (8 corners of the unit box)
        vertices = torch.tensor(
            [
                [-0.5, -0.5, -0.5],  # 0: front-bottom-left
                [+0.5, -0.5, -0.5],  # 1: front-bottom-right
                [+0.5, +0.5, -0.5],  # 2: back-bottom-right
                [-0.5, +0.5, -0.5],  # 3: back-bottom-left
                [-0.5, -0.5, +0.5],  # 4: front-top-left
                [+0.5, -0.5, +0.5],  # 5: front-top-right
                [+0.5, +0.5, +0.5],  # 6: back-top-right
                [-0.5, +0.5, +0.5],  # 7: back-top-left
            ],
            device=self.device,
        )

        # Define faces (12 triangular faces - each rectangular face is divided into 2 triangles)
        faces = torch.tensor(
            [
                [0, 1, 2],
                [0, 2, 3],  # bottom face
                [4, 6, 5],
                [4, 7, 6],  # top face
                [0, 4, 5],
                [0, 5, 1],  # front face
                [2, 6, 3],
                [3, 6, 7],  # back face
                [0, 3, 4],
                [3, 7, 4],  # left face
                [1, 5, 2],
                [2, 5, 6],  # right face
            ],
            device=self.device,
            dtype=torch.int64,
        )

        return vertices, faces

    def create_bbox_mesh(self, box_size):
        """
        Create 3D meshes for bounding boxes in a batch-efficient manner.

        Args:
            box_size: 3D dimensions of the box(es)
                     - Single box: [length, width, height] or [3]
                     - Batch: [N, 3] where N is the number of boxes

        Returns:
            If single box:
                vertices: Tensor of vertices [8, 3]
                faces: Tensor of face indices [12, 3]
            If batch:
                vertices: Tensor of vertices [N, 8, 3]
                faces: Tensor of face indices [N, 12, 3]
        """
        # Convert box_size to tensor if it's not already
        if not isinstance(box_size, torch.Tensor):
            box_size = torch.tensor(box_size, dtype=torch.float32, device=self.device)
        else:
            # Ensure float32 data type and move to correct device
            box_size = box_size.to(dtype=torch.float32, device=self.device)

        # Get unit box vertices and faces (cached)
        unit_vertices, unit_faces = self._create_unit_bbox_mesh()

        # Handle both single box and batch cases
        if box_size.dim() == 1:
            # Single box case: [3] -> [1, 3] for broadcasting
            if box_size.shape[0] != 3:
                raise ValueError(
                    f"Single box_size must have 3 elements, got {box_size.shape[0]}"
                )

            # Scale the vertices by box_size
            vertices = unit_vertices * box_size.reshape(1, 3)
            faces = unit_faces

        elif box_size.dim() == 2:
            # Batch case: [N, 3]
            batch_size = box_size.shape[0]
            if box_size.shape[1] != 3:
                raise ValueError(
                    f"Batch box_size must have shape [N, 3], got {box_size.shape}"
                )

            # Expand unit vertices for batch processing: [8, 3] -> [N, 8, 3]
            batch_unit_vertices = unit_vertices.unsqueeze(0).expand(batch_size, -1, -1)

            # Scale vertices for each box: [N, 8, 3] * [N, 1, 3] -> [N, 8, 3]
            vertices = batch_unit_vertices * box_size.unsqueeze(1)

            # Expand faces for batch: [12, 3] -> [N, 12, 3]
            faces = unit_faces.unsqueeze(0).expand(batch_size, -1, -1)

        else:
            raise ValueError(f"box_size must be 1D or 2D tensor, got {box_size.dim()}D")

        return vertices, faces

    def render_depth_from_boxes(self, object_list, camera_params):
        """
        Render depth maps for multiple cameras from 3D bounding boxes.
        This fully batched version processes all cameras and objects together.

        Args:
            object_list: List of objects with their transformations and dimensions
                Each object should be a dict with:
                - obj_to_world: 4x4 transformation matrix from object to world coordinates
                - box_size: 3D dimensions of the box [length, width, height]

            camera_params: Dict mapping camera names to their parameters
                Each camera should have:
                - world_to_cam: 4x4 transformation matrix from world to camera coordinates
                - intrinsics: 3x3 camera intrinsic matrix

        Returns:
            depth_maps: Dict mapping camera names to depth maps (torch tensor)
        """
        # Quick exit for empty object list
        if len(object_list) == 0:
            return {
                cam_name: torch.zeros(
                    (self.img_height, self.img_width),
                    dtype=torch.float32,
                    device=self.device,
                )
                for cam_name in camera_params
            }

        # Process all cameras at once using batched rendering
        cam_names = list(camera_params.keys())
        num_cameras = len(cam_names)

        # Initialize outputs
        outputs = {
            cam_name: torch.ones(
                (self.img_height, self.img_width),
                dtype=torch.float32,
                device=self.device,
            )
            * float("inf")
            for cam_name in cam_names
        }

        # Prepare batched camera parameters
        K_batch = torch.zeros(
            num_cameras, 4, 4, dtype=torch.float32, device=self.device
        )
        R_batch = torch.zeros(
            num_cameras, 3, 3, dtype=torch.float32, device=self.device
        )
        T_batch = torch.zeros(num_cameras, 3, dtype=torch.float32, device=self.device)

        for i, cam_name in enumerate(cam_names):
            params = camera_params[cam_name]

            # Convert to tensors and apply PyTorch3D camera convention
            if isinstance(params["world_to_cam"], np.ndarray):
                world_to_cam = torch.tensor(
                    params["world_to_cam"], dtype=torch.float32, device=self.device
                )
            else:
                world_to_cam = params["world_to_cam"].to(
                    dtype=torch.float32, device=self.device
                )

            if isinstance(params["intrinsics"], np.ndarray):
                intrinsics = torch.tensor(
                    params["intrinsics"], dtype=torch.float32, device=self.device
                )
            else:
                intrinsics = params["intrinsics"].to(
                    dtype=torch.float32, device=self.device
                )

            # Apply PyTorch3D camera convention
            world_to_cam[:3, :3] = torch.matmul(self.R_flip, world_to_cam[:3, :3])
            world_to_cam[:3, 3] = torch.matmul(self.R_flip, world_to_cam[:3, 3])

            # Prepare camera matrices for batch
            K_batch[i, :3, :3] = intrinsics
            K_batch[i, 3, 2] = 1.0
            R_batch[i] = world_to_cam[:3, :3].T.contiguous()
            T_batch[i] = world_to_cam[:3, 3].contiguous()

        # Create batched camera
        cameras = PerspectiveCameras(
            K=K_batch,
            R=R_batch,
            T=T_batch,
            in_ndc=False,
            image_size=((self.img_height, self.img_width),),
            device=self.device,
        )

        # Create batched rasterizer
        rasterizer = MeshRasterizer(
            cameras=cameras, raster_settings=self.raster_settings
        )
        for obj in object_list:
            vertices, faces = self.create_bbox_mesh(obj["box_size"])
            obj_to_world = torch.tensor(
                obj["obj_to_world"], dtype=torch.float32, device=self.device
            )
            verts_world = vertices @ obj_to_world[:3, :3].T + obj_to_world[:3, 3]
            replicated_verts = [verts_world for _ in range(num_cameras)]
            replicated_faces = [faces for _ in range(num_cameras)]
            mesh = Meshes(verts=replicated_verts, faces=replicated_faces)
            fragments = rasterizer(mesh)
            zbuf = fragments.zbuf[..., 0]
            valid_masks = zbuf > 0
            # Update outputs for each camera
            for cam_idx, cam_name in enumerate(cam_names):
                # Get the depth map for this camera
                depth_map = zbuf[cam_idx]
                valid_mask = valid_masks[cam_idx]

                # Update the output depth map only where valid
                outputs[cam_name][valid_mask] = torch.minimum(
                    outputs[cam_name][valid_mask], depth_map[valid_mask]
                )
        # Replace inf with zeros
        final_outputs = {cam_name: depth_map for cam_name, depth_map in outputs.items()}

        return final_outputs
