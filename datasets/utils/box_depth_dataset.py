import torch
from torch.utils.data import Dataset
from datasets.utils.box_depth_projection import BoxDepthProjector

class BoxDepthDataset(Dataset):
    """
    A base dataset class for generating depth maps from 3D bounding boxes.
    This class can be extended for specific datasets by implementing the
    required methods.
    """
    def __init__(self, img_size, device=None):
        """
        Initialize the box depth dataset.
        
        Args:
            img_size: Tuple of (width, height) for output depth maps
            device: PyTorch device to use
        """
        super().__init__()
        self.img_width, self.img_height = img_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.projector = BoxDepthProjector(img_size, device=self.device)
    
    def __len__(self):
        """
        Returns the number of frames in the dataset.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement __len__")
    
    def get_frame_objects(self, index):
        """
        Get all 3D objects for a given frame.
        
        Args:
            index: Frame index
            
        Returns:
            A list of objects where each object is a dict with keys:
                - obj_to_world: 4x4 transformation matrix from object to world coordinates
                - box_size: 3D dimensions of the box [length, width, height]
        
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_frame_objects")
    
    def get_camera_params(self, index):
        """
        Get camera parameters for all cameras in a given frame.
        
        Args:
            index: Frame index
            
        Returns:
            Dict mapping camera names to their parameters:
                - world_to_cam: 4x4 transformation matrix from world to camera coordinates
                - intrinsics: 3x3 camera intrinsic matrix
        
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_camera_params")
    
    def __getitem__(self, index):
        """
        Generate depth maps for all cameras in a given frame.
        
        Args:
            index: Frame index
            
        Returns:
            Dict mapping camera names to depth maps
        """
        # Get all objects in the frame
        objects = self.get_frame_objects(index)
        
        # Get camera parameters for all cameras
        camera_params = self.get_camera_params(index)
        
        # Render depth maps for all cameras
        depth_maps = self.projector.render_depth_from_boxes(objects, camera_params)
        
        return depth_maps
