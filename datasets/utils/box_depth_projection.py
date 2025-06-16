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
        Create a 3D mesh for a bounding box.
        
        Args:
            box_size: 3D dimensions of the box [length, width, height]
            
        Returns:
            vertices: Tensor of vertices [8, 3]
            faces: Tensor of face indices [12, 3]
        """
        # Convert box_size to tensor if it's not already
        if not isinstance(box_size, torch.Tensor):
            box_size = torch.tensor(box_size, dtype=torch.float32, device=self.device)
        else:
            # Ensure float32 data type
            box_size = box_size.to(dtype=torch.float32)

        # Get unit box vertices and faces (cached)
        unit_vertices, faces = self._create_unit_bbox_mesh()

        # Scale the vertices by box_size
        vertices = unit_vertices * box_size.reshape(1, 3)

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

        # Group objects by box size for efficient processing
        box_size_to_objects = {}
        for obj_idx, obj in enumerate(object_list):
            box_size = obj["box_size"]
            if isinstance(box_size, np.ndarray):
                box_size_key = tuple(box_size.flatten())
            elif isinstance(box_size, torch.Tensor):
                box_size_key = tuple(box_size.cpu().numpy().flatten())
            else:
                box_size_key = tuple(box_size)

            if box_size_key not in box_size_to_objects:
                box_size_to_objects[box_size_key] = []
            box_size_to_objects[box_size_key].append(obj_idx)

        # Process each group of objects with same box size
        for box_size_key, obj_indices in box_size_to_objects.items():
            if len(obj_indices) == 0:
                continue

            num_objects = len(obj_indices)
            vertices, faces = self.create_bbox_mesh(box_size_key)

            # Batch all object transformations
            obj_to_world_batch = torch.zeros(
                num_objects, 4, 4, dtype=torch.float32, device=self.device
            )

            for i, obj_idx in enumerate(obj_indices):
                obj = object_list[obj_idx]
                obj_to_world = obj["obj_to_world"]

                if isinstance(obj_to_world, np.ndarray):
                    obj_to_world_batch[i] = torch.tensor(
                        obj_to_world, dtype=torch.float32, device=self.device
                    )
                else:
                    obj_to_world_batch[i] = obj_to_world.to(
                        dtype=torch.float32, device=self.device
                    )

            # Transform vertices to world space (batched)
            vertices_batch = vertices.unsqueeze(0).expand(num_objects, -1, -1)
            ones_batch = torch.ones(
                num_objects, vertices.shape[0], 1, device=self.device
            )
            verts_obj_homo_batch = torch.cat([vertices_batch, ones_batch], dim=2)

            verts_world_batch = torch.bmm(
                verts_obj_homo_batch, obj_to_world_batch.transpose(-2, -1)
            )[:, :, :3]

            # Create mesh lists for PyTorch3D batched rendering
            verts_world_list = [verts_world_batch[i] for i in range(num_objects)]
            faces_list = [faces for _ in range(num_objects)]

            # Render all objects across all cameras at once
            if verts_world_list:
                # Create replicated vertex and face lists
                replicated_verts = []
                replicated_faces = []

                for cam_idx in range(num_cameras):
                    for obj_idx in range(num_objects):
                        replicated_verts.append(verts_world_list[obj_idx])
                        replicated_faces.append(faces_list[obj_idx])

                mesh = Meshes(verts=replicated_verts, faces=replicated_faces)
                fragments = rasterizer(mesh)

                zbuf = fragments.zbuf[..., 0]

                # Reshape to [num_cameras, num_objects, H, W]
                zbuf = zbuf.view(
                    num_cameras, num_objects, self.img_height, self.img_width
                )

                # Vectorized processing: combine depths by taking minimum across objects for all cameras
                # Take minimum depth across all objects: [num_cameras, H, W]
                min_depths = torch.min(zbuf, dim=1)[0]

                # Create valid masks for all cameras: [num_cameras, H, W]
                valid_masks = min_depths > 0

                # Update outputs for all cameras at once
                for cam_idx, cam_name in enumerate(cam_names):
                    valid_mask = valid_masks[cam_idx]
                    min_depth = min_depths[cam_idx]
                    outputs[cam_name][valid_mask] = torch.minimum(
                        outputs[cam_name][valid_mask], min_depth[valid_mask]
                    )

        # Replace inf with zeros
        final_outputs = {cam_name: depth_map for cam_name, depth_map in outputs.items()}

        return final_outputs
