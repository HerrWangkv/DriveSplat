import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
)
import numpy as np


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
        self.img_height, self.img_width  = img_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define PyTorch3D rasterization settings
        self.raster_settings = RasterizationSettings(
            image_size=(self.img_height, self.img_width),
            blur_radius=0.0, 
            faces_per_pixel=1,
            perspective_correct=True,
        )

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

        # Define 8 corners of the bounding box
        half_size = box_size / 2.0

        # Define vertices (8 corners of the box)
        vertices = torch.tensor([
            [-half_size[0], -half_size[1], -half_size[2]],  # 0: front-bottom-left
            [+half_size[0], -half_size[1], -half_size[2]],  # 1: front-bottom-right
            [+half_size[0], +half_size[1], -half_size[2]],  # 2: back-bottom-right
            [-half_size[0], +half_size[1], -half_size[2]],  # 3: back-bottom-left
            [-half_size[0], -half_size[1], +half_size[2]],  # 4: front-top-left
            [+half_size[0], -half_size[1], +half_size[2]],  # 5: front-top-right
            [+half_size[0], +half_size[1], +half_size[2]],  # 6: back-top-right
            [-half_size[0], +half_size[1], +half_size[2]],  # 7: back-top-left
        ], device=self.device)

        # Define faces (12 triangular faces - each rectangular face is divided into 2 triangles)
        faces = torch.tensor([
            [0, 1, 2], [0, 2, 3],  # bottom face
            [4, 6, 5], [4, 7, 6],  # top face
            [0, 4, 5], [0, 5, 1],  # front face
            [2, 6, 3], [3, 6, 7],  # back face
            [0, 3, 4], [3, 7, 4],  # left face
            [1, 5, 2], [2, 5, 6],  # right face
        ], device=self.device, dtype=torch.int64)

        return vertices, faces

    def setup_camera(self, camera_intrinsics, world_to_cam):
        """
        Setup a PyTorch3D camera from intrinsic and extrinsic parameters.
        
        Args:
            camera_intrinsics: 3x3 intrinsic matrix (numpy array or torch tensor)
            world_to_cam: 4x4 extrinsic matrix (world to camera) (numpy array or torch tensor)
            
        Returns:
            camera: PyTorch3D camera object
        """
        # Convert to torch tensors if they are numpy arrays
        if isinstance(camera_intrinsics, np.ndarray):
            camera_intrinsics = torch.tensor(camera_intrinsics, dtype=torch.float32, device=self.device)
        else:
            camera_intrinsics = camera_intrinsics.to(dtype=torch.float32)

        if isinstance(world_to_cam, np.ndarray):
            world_to_cam = torch.tensor(world_to_cam, dtype=torch.float32, device=self.device)
        else:
            world_to_cam = world_to_cam.to(dtype=torch.float32)

        K = torch.zeros((4,4), device=self.device, dtype=torch.float32)
        K[:3, :3] = camera_intrinsics
        K[3, 2] = 1.0  # Add homogeneous coordinate

        # Extract rotation and translation from extrinsics#
        # All transforms in Pytorch3D assume row-vectors, see https://github.com/facebookresearch/pytorch3d/issues/1183
        R = world_to_cam[:3, :3].T
        T = world_to_cam[:3, 3]

        # Create camera
        cameras = PerspectiveCameras(
            K=K.unsqueeze(0),
            R=R.unsqueeze(0),
            T=T.unsqueeze(0),
            in_ndc=False,
            image_size=((self.img_height, self.img_width),),
            device=self.device
        )

        return cameras

    def create_rasterizer(self, camera_intrinsics, world_to_cam):
        """
        Create a mesh rasterizer with the given camera parameters.
        
        Args:
            camera_intrinsics: 3x3 intrinsic matrix
            world_to_cam: 4x4 extrinsic matrix (world to camera)
            
        Returns:
            rasterizer: PyTorch3D MeshRasterizer object
        """
        camera = self.setup_camera(camera_intrinsics, world_to_cam)
        rasterizer = MeshRasterizer(
            cameras=camera, 
            raster_settings=self.raster_settings
        )
        return rasterizer

    def render_depth_from_boxes(self, object_list, camera_params):
        """
        Render depth maps for multiple cameras from 3D bounding boxes.
        
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
        # Initialize output dict with depth maps for each camera
        outputs = {}
        rasterizers = {}

        # Prepare empty depth maps for all cameras
        for cam_name in camera_params:
            outputs[cam_name] = torch.ones((self.img_height, self.img_width), 
                                          dtype=torch.float32, device=self.device) * float('inf')
            # According to https://pytorch3d.org/docs/cameras, +x points left and +y points up in camera view coordinate system.
            R_flip = np.array(
                [
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            camera_params[cam_name]["world_to_cam"][:3, :3] = (
                R_flip @ camera_params[cam_name]["world_to_cam"][:3, :3]
            )  # Flip to match PyTorch3D camera convention
            camera_params[cam_name]["world_to_cam"][:3, 3] = (
                R_flip @ camera_params[cam_name]["world_to_cam"][:3, 3]
            )  # Apply same transformation to translation

            # Create rasterizer for this camera
            rasterizer = self.create_rasterizer(camera_params[cam_name]['intrinsics'], camera_params[cam_name]['world_to_cam'])
            rasterizers[cam_name] = rasterizer

        # Process each object
        for obj in object_list:
            obj_to_world = obj['obj_to_world']
            box_size = obj['box_size']

            # Convert to tensor if needed
            if isinstance(obj_to_world, np.ndarray):
                obj_to_world = torch.tensor(obj_to_world, dtype=torch.float32, device=self.device)
            else:
                # Ensure float32 data type
                obj_to_world = obj_to_world.to(dtype=torch.float32)
            # Create mesh vertices and faces
            vertices, faces = self.create_bbox_mesh(box_size)

            # For each camera, render the depth of this object
            for cam_name, params in camera_params.items():
                world_to_cam = params["world_to_cam"]
                if isinstance(world_to_cam, np.ndarray):
                    world_to_cam = torch.tensor(world_to_cam, dtype=torch.float32, device=self.device)
                else:
                    # Ensure float32 data type
                    world_to_cam = world_to_cam.to(dtype=torch.float32)
                # Transform vertices from object to world space
                verts_obj = vertices  # Vertices in object space
                verts_obj_homo = torch.cat([verts_obj, torch.ones((8, 1), device=self.device)], dim=1)
                verts_world = torch.matmul(verts_obj_homo, obj_to_world.T)[:, :3]

                # Check if all vertices are behind the camera
                verts_world_homo = torch.cat([verts_world, torch.ones((8, 1), device=self.device)], dim=1)
                verts_cam = torch.matmul(verts_world_homo, world_to_cam.T)[:, :3]
                if torch.all(verts_cam[:, 2] <= 0):
                    continue  # Skip if all vertices are behind the camera

                # Create mesh batch with the transformed vertices
                mesh = Meshes(
                    verts=[verts_world],  # Batch of 1 mesh
                    faces=[faces]
                )

                # Rasterize mesh to get fragments (z-buffer and face_idx)
                fragments = rasterizers[cam_name](mesh)

                # Extract zbuf (depth buffer)
                zbuf = fragments.zbuf[0, :, :, 0]  # [H, W]

                # Update output depth map with the zbuf data (only where valid)
                valid_mask = zbuf > 0
                outputs[cam_name][valid_mask] = torch.minimum(outputs[cam_name][valid_mask], zbuf[valid_mask])

        # Convert infinite values to zero
        final_outputs = {}
        for cam_name in outputs:
            depth_map = outputs[cam_name].clone()
            depth_map[torch.isinf(depth_map)] = 0
            final_outputs[cam_name] = depth_map

        return final_outputs

    def render_single_camera_depth(self, object_list, camera_intrinsics, world_to_cam):
        """
        Render depth map for a single camera from 3D bounding boxes.
        
        Args:
            object_list: List of objects with their transformations and dimensions
                Each object should be a dict with:
                - obj_to_world: 4x4 transformation matrix from object to world coordinates
                - box_size: 3D dimensions of the box [length, width, height]
            camera_intrinsics: 3x3 camera intrinsic matrix
            world_to_cam: 4x4 extrinsic matrix (world to camera)
        
        Returns:
            depth_map: Depth map as torch tensor
        """
        camera_params = {'camera': {'intrinsics': camera_intrinsics, 'world_to_cam': world_to_cam}}
        result = self.render_depth_from_boxes(object_list, camera_params)
        return result['camera']
