import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.common.utils import Quaternion

import warnings
from typing import List
import matplotlib.pyplot as plt
import time
import json
from tqdm import tqdm

import sys
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from data.utils.box_depth_projection import BoxDepthProjector
from utils.img_utils import (
    concat_6_views,
    depth2disparity,
    disparity2depth,
    set_inf_to_max,
    concat_and_visualize_6_depths,
)
from utils.nvs_utils import render_novel_view

warnings.filterwarnings("ignore")


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        print(f"Done in {time.time() - start_time:.2f} seconds")
        print("=" * 6)
        return ret

    return wrapper


#################################################
################## Data loader ##################
#################################################


class NuScenesBase(NuScenes):

    def __init__(
        self,
        version,
        dataroot,
        cache_dir,
        split,
        view_order,
        # object_classes,
        # map_classes,
        # lane_classes,
        # map_bound,
        N,
        verbose=True,
        **kwargs,
    ):
        """
        Args:
            version (str): version of the dataset, e.g. 'v1.0-trainval'
            dataroot (str): directory of the dataset
            verbose (bool): whether to print information of the dataset
            seqs (list): list of scene indices to load
            N (int): number of interpolated frames between keyframes
            keys (list): list of keys to load
        """
        super().__init__(version=version, dataroot=dataroot, verbose=verbose, **kwargs)
        self.cache_dir = os.path.join(cache_dir, version)
        self.split = split
        # self.object_classes = object_classes
        # self.map_classes = map_classes
        # self.lane_classes = lane_classes
        # self.map_bound = map_bound
        # self.map_size = map_bound[1] - map_bound[0]
        # self.canvas_size = int(self.map_size / map_bound[2])
        if isinstance(self.split, str):
            self.seqs = []
            self.accumulate_seqs()
        elif isinstance(self.split, int):
            self.seqs = [self.split]
            print(f"Scene index: {self.split}")
            print("=" * 6)
        else:
            self.seqs = list(self.split)
            print(f"Number of scenes: {len(self.seqs)}")
            print("=" * 6)
        self.N = N  # Number of interpolated frames between keyframes

        self.lidar = "LIDAR_TOP"
        self.lidar_data_tokens = []  # Stacked list of LiDAR data tokens
        self.lidar_calib_tokens = []  # Stacked list of LiDAR calibration tokens
        self.seq_indices = []  # [(seq_idx, timestamp_idx)]
        self.accumulate_lidar_tokens()

        self.cameras = {k: i for i, k in enumerate(view_order)}
        self.camera_data_tokens = [[] for _ in range(len(self.cameras))]
        self.camera_calib_tokens = [[] for _ in range(len(self.cameras))]
        self.accumulate_img_and_calib_tokens()

        self.objects_mapping = {
            "movable_object.barrier": "barrier",
            "vehicle.bicycle": "bicycle",
            "vehicle.bus.bendy": "bus",
            "vehicle.bus.rigid": "bus",
            "vehicle.car": "car",
            "vehicle.construction": "construction_vehicle",
            "vehicle.motorcycle": "motorcycle",
            "human.pedestrian.adult": "pedestrian",
            "human.pedestrian.child": "pedestrian",
            "human.pedestrian.construction_worker": "pedestrian",
            "human.pedestrian.police_officer": "pedestrian",
            "movable_object.trafficcone": "traffic_cone",
            "vehicle.trailer": "trailer",
            "vehicle.truck": "truck",
        }
        self.instance_infos = [None for _ in self.seqs]
        self.frame_instances = [None for _ in self.seqs]
        self.accumulate_objects()
        # self.maps = {}
        # self.accumulate_maps()

    def accumulate_seqs(self):
        if self.version == "v1.0-mini" and not self.split.startswith("mini_"):
            self.split = "mini_" + self.split
        assert self.split in [
            "train",
            "val",
            "trainval",
            "test",
            "mini_train",
            "mini_val",
            "mini_trainval",
        ], f"Invalid split: {self.split}"
        if self.split == "trainval":
            scene_names = (
                create_splits_scenes()["train"] + create_splits_scenes()["val"]
            )
        elif self.split == "mini_trainval":
            scene_names = (
                create_splits_scenes()["mini_train"]
                + create_splits_scenes()["mini_val"]
            )
        else:
            scene_names = create_splits_scenes()[self.split]
        for i in range(len(self.scene)):
            if self.scene[i]["name"] in scene_names:
                self.seqs.append(i)
        print(f"Current split: {self.split}, number of scenes: {len(self.seqs)}")
        print("=" * 6)

    def get_keyframe_timestamps(self, scene_data):
        """Get keyframe timestamps from a scene data"""
        first_sample_token = scene_data["first_sample_token"]
        last_sample_token = scene_data["last_sample_token"]
        curr_sample_record = self.get("sample", first_sample_token)

        keyframe_timestamps = []

        while True:
            # Add the timestamp of the current keyframe
            keyframe_timestamps.append(
                self.get("sample_data", curr_sample_record["data"]["LIDAR_TOP"])["timestamp"]
            )

            if curr_sample_record["token"] == last_sample_token:
                break

            # Move to the next keyframe
            curr_sample_record = self.get("sample", curr_sample_record["next"])

        return keyframe_timestamps

    def get_interpolated_timestamps(self, keyframe_timestamps: List[int]):
        """Interpolate timestamps between keyframes."""
        interpolated_timestamps = []

        for i in range(len(keyframe_timestamps) - 1):
            start_time = keyframe_timestamps[i]
            end_time = keyframe_timestamps[i + 1]

            # Calculate the time step for interpolation
            time_step = (end_time - start_time) / (self.N + 1)

            # Add the start timestamp
            interpolated_timestamps.append(start_time)

            # Add N interpolated timestamps
            for j in range(1, self.N + 1):
                interpolated_time = start_time + j * time_step
                interpolated_timestamps.append(int(interpolated_time))

        # Add the last keyframe timestamp
        interpolated_timestamps.append(keyframe_timestamps[-1])

        return interpolated_timestamps

    def get_timestamps(self, scene_idx):
        scene_data = self.scene[scene_idx]
        frame_num = (self.N + 1) * (scene_data["nbr_samples"] - 1) + 1
        keyframe_timestamps = self.get_keyframe_timestamps(scene_data)
        assert len(keyframe_timestamps) == scene_data["nbr_samples"]
        interpolated_timestamps = self.get_interpolated_timestamps(keyframe_timestamps)
        assert len(interpolated_timestamps) == frame_num
        return interpolated_timestamps

    def find_closest_lidar_tokens(self, scene_data, timestamps: List[int]):
        """Find the closest LiDAR tokens for given timestamps."""
        first_sample_token = scene_data["first_sample_token"]
        first_sample_record = self.get("sample", first_sample_token)
        lidar_token = first_sample_record["data"]["LIDAR_TOP"]
        lidar_data = self.get("sample_data", lidar_token)

        # Collect all LiDAR timestamps and tokens
        lidar_timestamps = []
        lidar_tokens = []
        current_lidar = lidar_data
        while True:
            lidar_timestamps.append(current_lidar["timestamp"])
            lidar_tokens.append(current_lidar["token"])
            if current_lidar["next"] == "":
                break
            current_lidar = self.get("sample_data", current_lidar["next"])

        lidar_timestamps = np.array(lidar_timestamps, dtype=np.int64)

        # Find closest LiDAR tokens for each timestamp
        closest_tokens = []
        for timestamp in timestamps:
            idx = np.argmin(np.abs(lidar_timestamps - timestamp))
            closest_tokens.append(lidar_tokens[idx])

        # DEBUG USAGE: find is there any duplicated tokens
        # if len(closest_tokens) != len(set(closest_tokens)):
        #     duplicates = [token for token, count in Counter(closest_tokens).items() if count > 1]
        #     print(f"\nWARNING: {len(duplicates)} duplicated tokens in lidar")

        return closest_tokens

    def find_closest_img_tokens(self, scene_data, timestamps: List[int], cam_name):
        """Find the closest image tokens for given timestamps for a specific camera."""
        first_sample_token = scene_data["first_sample_token"]
        first_sample_record = self.get("sample", first_sample_token)
        img_token = first_sample_record["data"][cam_name]
        img_data = self.get("sample_data", img_token)

        # Collect all image timestamps and tokens for the specified camera
        img_timestamps = []
        img_tokens = []
        current_img = img_data
        while True:
            img_timestamps.append(current_img["timestamp"])
            img_tokens.append(current_img["token"])
            if current_img["next"] == "":
                break
            current_img = self.get("sample_data", current_img["next"])

        img_timestamps = np.array(img_timestamps, dtype=np.int64)

        # Find closest image tokens for each timestamp
        closest_tokens = []
        for timestamp in timestamps:
            idx = np.argmin(np.abs(img_timestamps - timestamp))
            closest_tokens.append(img_tokens[idx])

        # DEBUG USAGE: find is there any duplicated tokens
        # if len(closest_tokens) != len(set(closest_tokens)):
        #     duplicates = [token for token, count in Counter(closest_tokens).items() if count > 1]
        #     print(f"\nWARNING: {len(duplicates)} duplicated tokens in {cam_name}")

        return closest_tokens

    @timer
    def accumulate_lidar_tokens(self):
        print("Accumulating LiDAR data tokens...")
        for i, scene_idx in enumerate(tqdm(self.seqs)):
            scene_data = self.scene[scene_idx]
            timestamps = self.get_timestamps(scene_idx)
            closest_lidar_tokens = self.find_closest_lidar_tokens(
                scene_data, timestamps
            )
            self.lidar_data_tokens += closest_lidar_tokens
            self.lidar_calib_tokens += [
                self.get("sample_data", token)["calibrated_sensor_token"]
                for token in closest_lidar_tokens
            ]
            self.seq_indices += [(i, t) for t in range(len(timestamps))]
        assert len(self.lidar_data_tokens) == len(self.seq_indices)

    @timer
    def accumulate_img_and_calib_tokens(self):
        print("Accumulating image and calibration tokens...")
        for i, scene_idx in enumerate(tqdm(self.seqs)):
            scene_data = self.scene[scene_idx]
            for cam_name in self.cameras.keys():
                timestamps = self.get_timestamps(scene_idx)
                img_tokens = self.find_closest_img_tokens(
                    scene_data, timestamps, cam_name
                )
                self.camera_data_tokens[self.cameras[cam_name]] += img_tokens
                calib_tokens = [
                    self.get("sample_data", token)["calibrated_sensor_token"]
                    for token in img_tokens
                ]
                self.camera_calib_tokens[self.cameras[cam_name]] += calib_tokens
        assert all(
            [len(tokens) == len(self.seq_indices) for tokens in self.camera_data_tokens]
        )
        assert all(
            [
                len(tokens) == len(self.seq_indices)
                for tokens in self.camera_calib_tokens
            ]
        )

    def fetch_keyframe_objects(self, scene_data):
        """Parse and save the objects annotation data."""
        first_sample_token, last_sample_token = (
            scene_data["first_sample_token"],
            scene_data["last_sample_token"],
        )
        curr_sample_record = self.get("sample", first_sample_token)
        key_frame_idx = 0

        instances_info = {}
        while True:
            anns = [
                self.get("sample_annotation", token)
                for token in curr_sample_record["anns"]
            ]

            for ann in anns:
                if ann["category_name"] not in self.objects_mapping:
                    continue

                instance_token = ann["instance_token"]
                if instance_token not in instances_info:
                    instances_info[instance_token] = {
                        "id": instance_token,
                        "class_name": self.objects_mapping[ann["category_name"]],
                        "frame_annotations": {
                            "frame_idx": [],
                            "obj_to_world": [],
                            "box_size": [],
                        },
                    }

                # Object to world transformation
                o2w = np.eye(4)
                o2w[:3, :3] = Quaternion(ann["rotation"]).rotation_matrix
                o2w[:3, 3] = np.array(ann["translation"])

                # Key frames are spaced (N + 1) frames apart in the new sequence
                obj_frame_idx = key_frame_idx * (self.N + 1)
                instances_info[instance_token]["frame_annotations"]["frame_idx"].append(
                    obj_frame_idx
                )
                instances_info[instance_token]["frame_annotations"][
                    "obj_to_world"
                ].append(o2w.tolist())
                # convert wlh to lwh
                lwh = [ann["size"][1], ann["size"][0], ann["size"][2]]
                instances_info[instance_token]["frame_annotations"]["box_size"].append(
                    lwh
                )

            if (
                curr_sample_record["next"] == ""
                or curr_sample_record["token"] == last_sample_token
            ):
                break
            key_frame_idx += 1
            curr_sample_record = self.get("sample", curr_sample_record["next"])

        # Correct ID mapping
        id_map = {}
        for i, (k, v) in enumerate(instances_info.items()):
            id_map[v["id"]] = i

        # Update keys in instances_info
        new_instances_info = {}
        for k, v in instances_info.items():
            new_instances_info[id_map[v["id"]]] = v

        return new_instances_info

    def interpolate_boxes(self, instances_info, max_frame_idx):
        """Interpolate object positions and sizes between keyframes."""
        new_instances_info = {}
        new_frame_instances = {}

        for obj_id, obj_info in instances_info.items():
            frame_annotations = obj_info["frame_annotations"]
            keyframe_indices = frame_annotations["frame_idx"]
            obj_to_world_list = frame_annotations["obj_to_world"]
            box_size_list = frame_annotations["box_size"]

            new_frame_idx = []
            new_obj_to_world = []
            new_box_size = []

            for i in range(len(keyframe_indices) - 1):
                start_frame = keyframe_indices[i]
                start_transform = np.array(obj_to_world_list[i])
                end_transform = np.array(obj_to_world_list[i + 1])
                start_quat = Quaternion(matrix=start_transform[:3, :3])
                end_quat = Quaternion(matrix=end_transform[:3, :3])
                start_size = np.array(box_size_list[i])
                end_size = np.array(box_size_list[i + 1])

                for j in range(self.N + 1):
                    t = j / (self.N + 1)
                    current_frame = start_frame + j

                    # Interpolate translation
                    translation = (1 - t) * start_transform[:3, 3] + t * end_transform[
                        :3, 3
                    ]

                    # Interpolate rotation using Quaternions
                    current_quat = Quaternion.slerp(start_quat, end_quat, t)

                    # Construct interpolated transformation matrix
                    current_transform = np.eye(4)
                    current_transform[:3, :3] = current_quat.rotation_matrix
                    current_transform[:3, 3] = translation

                    # Interpolate box size
                    current_size = (1 - t) * start_size + t * end_size

                    new_frame_idx.append(current_frame)
                    new_obj_to_world.append(current_transform.tolist())
                    new_box_size.append(current_size.tolist())

            # Add the last keyframe
            new_frame_idx.append(keyframe_indices[-1])
            new_obj_to_world.append(obj_to_world_list[-1])
            new_box_size.append(box_size_list[-1])

            # Update instance info
            new_instances_info[obj_id] = {
                "id": obj_info["id"],
                "class_name": obj_info["class_name"],
                "frame_annotations": {
                    new_frame_idx[f]: (new_obj_to_world[f], new_box_size[f])
                    for f in range(len(new_frame_idx))
                },
            }

            # Update frame instances
            for frame in new_frame_idx:
                if frame not in new_frame_instances:
                    new_frame_instances[frame] = []
                new_frame_instances[frame].append(obj_id)

        for k in range(max_frame_idx):
            if k not in new_frame_instances:
                new_frame_instances[k] = []
        return new_instances_info, new_frame_instances

    @timer
    def accumulate_objects(self):
        objects_cache_dir = os.path.join(self.cache_dir, "objects")
        print("Accumulating objects...")
        for i, scene_idx in enumerate(tqdm(self.seqs)):
            os.makedirs(os.path.join(objects_cache_dir, f"{scene_idx}"), exist_ok=True)
            instances_info_cache_path = os.path.join(
                objects_cache_dir, f"{scene_idx}/instances_info.json"
            )
            frame_instances_cache_path = os.path.join(
                objects_cache_dir, f"{scene_idx}/frame_instances.json"
            )
            if os.path.exists(instances_info_cache_path) and os.path.exists(
                frame_instances_cache_path
            ):
                with open(instances_info_cache_path, "r") as f:
                    instances_info = json.load(f)
                with open(frame_instances_cache_path, "r") as f:
                    frame_instances = json.load(f)
                self.instance_infos[i] = instances_info
                self.frame_instances[i] = frame_instances
            else:
                scene_data = self.scene[scene_idx]
                instances_info = self.fetch_keyframe_objects(scene_data)
                max_frame_idx = (self.N + 1) * (scene_data["nbr_samples"] - 1) + 1
                instances_info, frame_instances = self.interpolate_boxes(
                    instances_info, max_frame_idx
                )
                with open(instances_info_cache_path, "w") as f:
                    json.dump(instances_info, f)
                with open(frame_instances_cache_path, "w") as f:
                    json.dump(frame_instances, f)
                self.instance_infos[i] = instances_info
                self.frame_instances[i] = frame_instances

    def get_location(self, index):
        seq_idx, _ = self.seq_indices[index]
        scene_idx = self.seqs[seq_idx]
        scene_data = self.scene[scene_idx]
        log = self.get("log", scene_data["log_token"])
        return log["location"]

    def get_map_info(self, index):
        seq_idx, frame_idx = self.seq_indices[index]
        scene_idx = self.seqs[seq_idx]
        map_info = np.load(
            os.path.join(self.cache_dir, "maps", f"{scene_idx}/map_{frame_idx}.npy"),
            allow_pickle=True,
        ).item()
        return (
            map_info["drivable_mask"],
            map_info["gt_map_pts"],
            map_info["gt_vecs_label"],
        )

    def is_keyframe(self, index):
        return self.seq_indices[index][1] % (self.N + 1) == 0

    def get_frame_instances(self, index):
        seq_idx, frame_idx = self.seq_indices[index]
        if frame_idx in self.frame_instances[seq_idx]:
            return self.frame_instances[seq_idx][frame_idx]
        else:
            return self.frame_instances[seq_idx][str(frame_idx)]

    def get_frame_annotation(self, index, instance_id):
        seq_idx, frame_idx = self.seq_indices[index]
        if instance_id in self.instance_infos[seq_idx]:
            instance_info = self.instance_infos[seq_idx][instance_id]
        else:
            instance_info = self.instance_infos[seq_idx][str(instance_id)]
        class_name = instance_info["class_name"]
        if frame_idx in instance_info["frame_annotations"]:
            obj_to_world, box_size = instance_info["frame_annotations"][frame_idx]
        else:
            obj_to_world, box_size = instance_info["frame_annotations"][str(frame_idx)]
        return class_name, np.array(obj_to_world), box_size

    def get_frame_objects(self, index):
        frame_instances = self.get_frame_instances(index)
        objects = []

        for instance_id in frame_instances:
            _, obj_to_world, box_size = self.get_frame_annotation(index, instance_id)
            objects.append({"obj_to_world": obj_to_world, "box_size": box_size})
        return objects

    def get_ego_pose(self, index, sensor):
        if sensor == "LIDAR_TOP":
            sensor_data_token = self.lidar_data_tokens[index]
        else:
            cam_idx = self.cameras[sensor]
            sensor_data_token = self.camera_data_tokens[cam_idx][index]
        sensor_data = self.get("sample_data", sensor_data_token)
        ego_pose_token = sensor_data["ego_pose_token"]
        ego_pose = self.get("ego_pose", ego_pose_token)
        return ego_pose

    def get_ego_to_world(self, index, sensor):
        ego_pose = self.get_ego_pose(index, sensor)
        ego_to_world = np.eye(4)
        ego_to_world[:3, :3] = Quaternion(ego_pose["rotation"]).rotation_matrix
        ego_to_world[:3, 3] = np.array(ego_pose["translation"])
        return ego_to_world

    def get_world_to_ego(self, index, sensor):
        return np.linalg.inv(self.get_ego_to_world(index, sensor))

    def get_reference_ego_to_world(self, index):
        return self.get_ego_to_world(index, "LIDAR_TOP")

    def get_world_to_reference_ego(self, index):
        return np.linalg.inv(self.get_reference_ego_to_world(index))

    def get_cam_to_ego(self, index, cam):
        if isinstance(cam, int):
            cam = list(self.cameras.keys())[cam]
        cam_idx = self.cameras[cam]
        calib_token = self.camera_calib_tokens[cam_idx][index]
        calib_data = self.get("calibrated_sensor", calib_token)
        cam_to_ego = np.eye(4)
        cam_to_ego[:3, :3] = Quaternion(calib_data["rotation"]).rotation_matrix
        cam_to_ego[:3, 3] = np.array(calib_data["translation"])
        return cam_to_ego

    def get_ego_to_cam(self, index, cam):
        return np.linalg.inv(self.get_cam_to_ego(index, cam))

    def get_cam_to_reference_ego(self, index, cam):
        return self.get_world_to_reference_ego(index) @ self.get_cam_to_world(
            index, cam
        )

    def get_reference_ego_to_cam(self, index, cam):
        return np.linalg.inv(self.get_cam_to_reference_ego(index, cam))

    def get_lidar_to_ego(self, index):
        calib_token = self.lidar_calib_tokens[index]
        calib_data = self.get("calibrated_sensor", calib_token)
        lidar_to_ego = np.eye(4)
        lidar_to_ego[:3, :3] = Quaternion(calib_data["rotation"]).rotation_matrix
        lidar_to_ego[:3, 3] = np.array(calib_data["translation"])
        return lidar_to_ego

    def get_ego_to_lidar(self, index):
        return np.linalg.inv(self.get_lidar_to_ego(index))

    def get_lidar_to_reference_ego(self, index):
        return self.get_world_to_reference_ego(index) @ self.get_lidar_to_world(index)

    def get_reference_ego_to_lidar(self, index):
        return np.linalg.inv(self.get_lidar_to_reference_ego(index))

    def get_cam_to_lidar(self, index, cam):
        return self.get_world_to_lidar(index) @ self.get_cam_to_world(index, cam)

    def get_lidar_to_cam(self, index, cam):
        return np.linalg.inv(self.get_cam_to_lidar(index, cam))

    def get_cam_to_world(self, index, cam):
        if isinstance(cam, int):
            cam = list(self.cameras.keys())[cam]
        cam_to_ego = self.get_cam_to_ego(index, cam)
        ego_to_world = self.get_ego_to_world(index, cam)
        cam_to_world = ego_to_world @ cam_to_ego
        return cam_to_world

    def get_world_to_cam(self, index, cam):
        return np.linalg.inv(self.get_cam_to_world(index, cam))

    def get_cam_front_to_world(self, index):
        return self.get_cam_to_world(index, "CAM_FRONT")

    def get_world_to_cam_front(self, index):
        return np.linalg.inv(self.get_cam_front_to_world(index))

    def get_lidar_to_world(self, index):
        lidar_to_ego = self.get_lidar_to_ego(index)
        ego_to_world = self.get_ego_to_world(index, "LIDAR_TOP")
        lidar_to_world = ego_to_world @ lidar_to_ego
        return lidar_to_world

    def get_world_to_lidar(self, index):
        return np.linalg.inv(self.get_lidar_to_world(index))

    def is_first_frame(self, index):
        return (index == 0) or (
            self.seq_indices[index][0] != self.seq_indices[index - 1][0]
        )

    def is_last_frame(self, index):
        return (index + 1 == len(self.seq_indices)) or (
            self.seq_indices[index][0] != self.seq_indices[index + 1][0]
        )

    def get_caption(self, index):
        seq_idx, frame_idx = self.seq_indices[index]
        scene_idx = self.seqs[seq_idx]
        scene_data = self.scene[scene_idx]
        location = self.get_location(index)
        description = scene_data["description"]
        return f"A driving scene image at {location}. {description}"

    def get_camera_intrinsics(self, index, cam, img_size=(900, 1600)):
        assert cam in self.cameras, f"Camera {cam} not found in cameras list."
        cam_idx = self.cameras[cam]
        calib_token = self.camera_calib_tokens[cam_idx][index]
        calib_data = self.get("calibrated_sensor", calib_token)
        camera_intrinsics = np.array(calib_data["camera_intrinsic"])
        camera_intrinsics[0] *= img_size[1] / 1600
        camera_intrinsics[1] *= img_size[0] / 900
        return camera_intrinsics

    def get_camera_params(self, index, img_size=(900, 1600)):
        camera_params = {}

        for cam in self.cameras:
            world_to_cam = self.get_world_to_cam(index, cam)
            camera_intrinsics = self.get_camera_intrinsics(
                index, cam, img_size=img_size
            )

            camera_params[cam] = {
                "world_to_cam": world_to_cam,
                "intrinsics": camera_intrinsics,
            }
        return camera_params


class NuScenesLidarDisparity(Dataset):
    def __init__(self, nusc, img_size):
        """
        Lidar disparity map

        Args:
            nusc: NuScenesBase instance
            height (int): Height of the disparity map
            width (int): Width of the disparity map
        """
        super().__init__()
        self.nusc = nusc
        self.img_size = img_size

    def __len__(self):
        return len(self.nusc.lidar_data_tokens)

    def __getitem__(self, index):
        lidar_token = self.nusc.lidar_data_tokens[index]
        lidar_data = self.nusc.get("sample_data", lidar_token)
        lidar_points = np.fromfile(
            os.path.join(self.nusc.dataroot, lidar_data["filename"]), dtype=np.float32
        ).reshape(-1, 5)[
            :, :3
        ]  # Extract x, y, z coordinates

        disparity_maps = []
        for cam in self.nusc.cameras:
            cam_idx = self.nusc.cameras[cam]
            calib_token = self.nusc.camera_calib_tokens[cam_idx][index]
            calib_data = self.nusc.get("calibrated_sensor", calib_token)
            camera_intrinsics = np.array(calib_data["camera_intrinsic"])
            camera_intrinsics[0] *= self.img_size[1] / 1600
            camera_intrinsics[1] *= self.img_size[0] / 900
            lidar_to_cam = self.nusc.get_lidar_to_cam(index, cam)

            # Transform lidar points to camera frame
            lidar_points_h = np.hstack(
                (lidar_points, np.ones((lidar_points.shape[0], 1)))
            )
            points_in_cam = lidar_points_h @ lidar_to_cam.T
            depth = points_in_cam[:, 2]  # Extract depth values

            # Project points onto the image plane
            front_mask = points_in_cam[:, 2] > 0  # Keep points in front
            points_in_cam = points_in_cam[front_mask]
            depth = depth[front_mask]
            points_in_img = points_in_cam[:, :3] @ camera_intrinsics.T
            points_in_img = points_in_img[:, :2] / points_in_img[:, 2:3]  # Normalize
            points_in_img = np.round(points_in_img)

            # Filter points within image bounds
            img_width, img_height = self.img_size[1], self.img_size[0]
            valid_mask = (
                (points_in_img[:, 0] >= 0)
                & (points_in_img[:, 0] < img_width)
                & (points_in_img[:, 1] >= 0)
                & (points_in_img[:, 1] < img_height)
            )
            # Create an empty image with the same size as the target image
            img_width, img_height = self.img_size[1], self.img_size[0]
            disparity_image = np.zeros((img_height, img_width))

            # Round the coordinates to integers for indexing
            points_in_img = points_in_img[valid_mask]
            depth = depth[valid_mask]
            x_coords = points_in_img[:, 0].astype(int)
            y_coords = points_in_img[:, 1].astype(int)

            # Assign depth values to the corresponding pixel locations
            disparity_image[y_coords, x_coords] = 1 / depth

            # Add the depth image to the projected points
            disparity_maps.append(torch.tensor(disparity_image).unsqueeze(0).float())
        disparity_maps = torch.stack(disparity_maps, axis=0)
        return {"lidar_disparity_maps": disparity_maps}


class NuScenesBoxDisparity(Dataset):
    def __init__(self, nusc, img_size):
        """
        Accurate disparity map generation using PyTorch3D rendering capabilities.

        Args:
            nusc: NuScenesBase instance
            height (int): Height of the disparity map
            width (int): Width of the disparity map
        """
        super().__init__()
        self.nusc = nusc
        self.img_size = img_size
        self.depth_projector = BoxDepthProjector(
            img_size=img_size,
        )

    def __len__(self):
        return len(self.nusc.lidar_data_tokens)

    def __getitem__(self, index):
        objects = self.nusc.get_frame_objects(index)
        camera_params = self.nusc.get_camera_params(index, img_size=self.img_size)
        depth_maps = self.depth_projector.render_depth_from_boxes(
            objects, camera_params
        )
        disparity_maps = []
        for cam in self.nusc.cameras:
            depth_map = depth_maps[cam]
            disparity_map = depth2disparity(depth_map)
            disparity_maps.append(disparity_map)
        disparity_maps = torch.stack(disparity_maps, axis=0)[:, None]
        return {"box_disparity_maps": disparity_maps}


class NuScenesDisparity(Dataset):
    def __init__(self, nusc, img_size):
        """
        Accurate disparity map using OmniRe

        Args:
            nusc: NuScenesBase instance
            height (int): Height of the disparity map
            width (int): Width of the disparity map
        """
        super().__init__()
        self.nusc = nusc
        self.img_size = img_size

    def __len__(self):
        return len(self.nusc.lidar_data_tokens)

    def __getitem__(self, index):
        seq_idx, frame_idx = self.nusc.seq_indices[index]
        scene_idx = self.nusc.seqs[seq_idx]
        disparity_dir = os.path.join(
            self.nusc.cache_dir, "disparity", f"{scene_idx:03d}"
        )

        disparity_maps = []

        for cam in self.nusc.cameras:
            filename = os.path.join(disparity_dir, f"{frame_idx:03d}_{cam}.png")

            if os.path.exists(filename):
                # Load 16-bit PNG disparity image
                img = Image.open(filename)
                disparity_uint16 = np.array(img)

                # Extract metadata for reconstruction
                if hasattr(img, "text") and img.text:
                    cam_min = float(img.text.get("min_val", 0.0))
                    cam_max = float(img.text.get("max_val", 1.0))

                    # Reconstruct original disparity values
                    disparity_array = (
                        disparity_uint16.astype(np.float32) / 65535.0
                    ) * (cam_max - cam_min) + cam_min
                else:
                    raise ValueError(
                        f"Disparity image {filename} does not contain metadata."
                    )

                # Resize if needed
                if disparity_array.shape != self.img_size:
                    disparity_pil = Image.fromarray(disparity_array.astype(np.float32))
                    disparity_pil = disparity_pil.resize(
                        (self.img_size[1], self.img_size[0])
                    )
                    disparity_array = np.array(disparity_pil)

                disparity_maps.append(disparity_array)
            else:
                raise FileNotFoundError(f"Disparity image {filename} not found.")

        disparity_maps = torch.tensor(np.stack(disparity_maps, axis=0))[
            :, None
        ]  # Add channel dimension

        return {"disparity_maps": disparity_maps}


class NuScenesCameraImages(Dataset):
    def __init__(self, nusc, img_size):
        super().__init__()
        self.nusc = nusc
        self.img_size = img_size

    def __len__(self):
        return len(self.nusc.camera_data_tokens[1])

    def __getitem__(self, index):
        pixel_values = []
        for cam in self.nusc.cameras:
            cam_idx = self.nusc.cameras[cam]
            filename = self.nusc.get(
                "sample_data", self.nusc.camera_data_tokens[cam_idx][index]
            )["filename"]
            img = Image.open(os.path.join(self.nusc.dataroot, filename)).convert("RGB")
            img = img.resize((self.img_size[1], self.img_size[0]))
            img = np.array(img).transpose(2, 0, 1)
            img = img / 255.0
            pixel_values.append(img)
        pixel_values = torch.tensor(pixel_values)
        return {"pixel_values": pixel_values}


class NuScenesEgoMasks(Dataset):
    def __init__(self, nusc, img_size):
        super().__init__()
        self.nusc = nusc
        self.img_size = img_size

    def __len__(self):
        return len(self.nusc.seq_indices)

    def __getitem__(self, index):
        seq_idx, _ = self.nusc.seq_indices[index]
        scene_idx = self.nusc.seqs[seq_idx]
        ego_masks = np.zeros(
            (len(self.nusc.cameras), 1, self.img_size[0], self.img_size[1])
        ).astype(bool)
        for cam_idx, cam in enumerate(self.nusc.cameras):
            if cam == "CAM_BACK":
                trunk_mask = Image.open(
                    os.path.join(self.nusc.cache_dir, "masks", f"{scene_idx}.png")
                ).convert("L")
                trunk_mask = trunk_mask.resize((self.img_size[1], self.img_size[0]))
                ego_masks[cam_idx, 0] = np.array(trunk_mask).astype(bool)
        return {"ego_masks": torch.tensor(ego_masks)}


class SDaIGNuScenesDataset(Dataset):
    def __init__(
        self,
        version,
        dataroot,
        cache_dir,
        split,
        view_order,
        N,
        height,
        width,
        **kwargs,
    ):
        self.nusc = NuScenesBase(
            version=version,
            dataroot=dataroot,
            cache_dir=cache_dir,
            split=split,
            view_order=view_order,
            N=N,
            **kwargs,
        )
        self.img_size = (height, width)
        self.pixel_values = NuScenesCameraImages(self.nusc, self.img_size)
        self.box_disparity_maps = NuScenesBoxDisparity(self.nusc, self.img_size)
        self.disparity_maps = NuScenesDisparity(self.nusc, self.img_size)
        self.ego_masks = NuScenesEgoMasks(self.nusc, img_size=self.img_size)

    def __len__(self):
        return len(self.nusc.lidar_data_tokens)

    def is_first_frame(self, index):
        return self.nusc.is_first_frame(index)

    def is_last_frame(self, index):
        return self.nusc.is_last_frame(index)


class SDaIGNuScenesTrainDataset(SDaIGNuScenesDataset):
    def __getitem__(self, index):
        interval = 1 if not self.is_last_frame(index) else -1
        ret = {}
        ret["pixel_values", 0] = (
            self.pixel_values[index]["pixel_values"] * 2 - 1
        )  # Normalize to [-1, 1]
        ret["pixel_values", 1] = (
            self.pixel_values[index + interval]["pixel_values"] * 2 - 1
        )  # Normalize to [-1, 1]
        ret["box_disparity_maps", 1] = (
            self.box_disparity_maps[index + interval]["box_disparity_maps"] * 2 - 1
        )  # Normalize to [-1, 1]
        ret["disparity_maps", 0] = (
            self.disparity_maps[index]["disparity_maps"] * 2 - 1
        )  # Normalize to [-1, 1]
        ret["disparity_maps", 1] = (
            self.disparity_maps[index + interval]["disparity_maps"] * 2 - 1
        )  # Normalize to [-1, 1]
        ret["ego_masks"] = self.ego_masks[index]["ego_masks"]
        intrinsics = []
        extrinsics0 = []
        extrinsics1 = []
        for cam in self.nusc.cameras:
            intrinsics.append(
                torch.tensor(
                    self.nusc.get_camera_intrinsics(index, cam, img_size=self.img_size)
                )
            )
            extrinsics0.append(torch.tensor(self.nusc.get_world_to_cam(index, cam)))
            extrinsics1.append(
                torch.tensor(self.nusc.get_world_to_cam(index + interval, cam))
            )
        ret["intrinsics"] = torch.stack(intrinsics, dim=0)
        ret["extrinsics", 0] = torch.stack(extrinsics0, dim=0)
        ret["extrinsics", 1] = torch.stack(extrinsics1, dim=0)
        return {k: v.cpu().to(torch.float32) for k, v in ret.items()}


class SDaIGNuScenesTestDataset(SDaIGNuScenesDataset):
    def __getitem__(self, index):
        interval = 1 if not self.is_last_frame(index) else -1
        ret = {}
        ret["pixel_values", 0] = (
            self.pixel_values[index]["pixel_values"] * 2 - 1
        )  # Normalize to [-1, 1]
        ret["pixel_values", 1] = (
            self.pixel_values[index + interval]["pixel_values"] * 2 - 1
        )  # Normalize to [-1, 1]
        ret["box_disparity_maps", 0] = (
            self.box_disparity_maps[index]["box_disparity_maps"] * 2 - 1
        )  # Normalize to [-1, 1]
        ret["disparity_maps", 0] = (
            self.disparity_maps[index]["disparity_maps"] * 2 - 1
        )  # Normalize to [-1, 1]
        ret["disparity_maps", 1] = (
            self.disparity_maps[index + interval]["disparity_maps"] * 2 - 1
        )  # Normalize to [-1, 1]
        ret["ego_masks"] = self.ego_masks[index]["ego_masks"]
        intrinsics = []
        extrinsics0 = []
        extrinsics1 = []
        for cam in self.nusc.cameras:
            intrinsics.append(
                torch.tensor(
                    self.nusc.get_camera_intrinsics(index, cam, img_size=self.img_size)
                )
            )
            extrinsics0.append(torch.tensor(self.nusc.get_world_to_cam(index, cam)))
            extrinsics1.append(
                torch.tensor(self.nusc.get_world_to_cam(index + interval, cam))
            )
        ret["intrinsics"] = torch.stack(intrinsics, dim=0)
        ret["extrinsics", 0] = torch.stack(extrinsics0, dim=0)
        ret["extrinsics", 1] = torch.stack(extrinsics1, dim=0)
        return {k: v.cpu().to(torch.float32) for k, v in ret.items()}

    def vis(self, index):
        os.makedirs("vis", exist_ok=True)
        item = self[index]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        item = {k: v.to(device) for k, v in item.items()}
        cond_disparity_maps = (
            item["box_disparity_maps", 0] + 1
        ) / 2  # Normalize to [0, 1]
        cond_depth_maps = disparity2depth(cond_disparity_maps)
        concat_and_visualize_6_depths(
            set_inf_to_max(cond_depth_maps), save_path="vis/cond_depth_maps.png"
        )
        pixel_values = (item["pixel_values", 0] + 1) / 2  # Normalize to [0, 1]
        ego_masks = item["ego_masks"]
        concat_6_views(pixel_values * (1 - ego_masks)).save(f"vis/pixel_values.png")
        disparity_maps = (item["disparity_maps", 0] + 1) / 2  # Normalize to [0, 1]
        depth_maps = disparity2depth(disparity_maps)

        concat_and_visualize_6_depths(set_inf_to_max(depth_maps), f"vis/depth_maps.png")
        with torch.no_grad():
            novel_images, novel_depth = render_novel_view(
                pixel_values,
                set_inf_to_max(depth_maps).squeeze(),
                ego_masks.squeeze(),
                item["intrinsics"],
                item["extrinsics", 0],
                item["intrinsics"],
                item["extrinsics", 1],
                self.img_size,
            )
        concat_6_views(
            novel_images.squeeze().permute([0, 3, 1, 2]),
        ).save("vis/predicted_novel_view.png")
        concat_and_visualize_6_depths(
            set_inf_to_max(novel_depth), save_path="vis/predicted_novel_view_depth.png"
        )

        next_disparity_maps = (item["disparity_maps", 1] + 1) / 2  # Normalize to [0, 1]
        next_depth_maps = disparity2depth(next_disparity_maps)
        concat_and_visualize_6_depths(
            set_inf_to_max(next_depth_maps),
            save_path="vis/real_novel_view_depth.png",
        )
        next_pixel_values = (item["pixel_values", 1] + 1) / 2  # Normalize to [0, 1]
        concat_6_views(next_pixel_values).save("vis/real_novel_view.png")
