import os
from os.path import join
from typing import Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from src.data_loader.joints import Joints
from src.utils import read_json
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class IH_DB(Dataset):
    """Class to load samples from the Interhand dataset.
    https://mks0601.github.io/InterHand2.6M/
    Inherits from the Dataset class in  torch.utils.data.
    Note: The keypoints are mapped to format used at AIT.
    Refer to joint_mapping.json in src/data_loader/utils.
    """

    IS_LEFT = True

    def __init__(
        self,
        root_dir: str,
        split: str,
        annotor: str = "all",
        seed: int = 5,
        train_ratio: float = 0.9,
    ):
        """Initializes the Interhand dataset class, relevant paths, meta_info jsons,
        dataframes and the Joints class for remappinng interhand formatted joints to
        that of AIT.
        Args:
            root_dir (str): Path to the directory with image samples.
            split (str): set to 'train', 'val' or 'test'.
            annotor (str, optional): [description]. Defaults to 'all'. Other options are
              'human_annot' and 'machine_annot' .
        """
        self.root_dir = root_dir
        # To convert from freihand to AIT format.
        self.joints = Joints()
        self.seed = seed
        self.train_ratio = train_ratio
        self.annotor = annotor  # "human_annot" and "machine_annot" possible.
        self.annotation_sampling_folder = "InterHand2.6M.annotations.5.fps"
        self.image_sampling_folder = "InterHand2.6M_5fps_batch0/images"
        self._split = split
        self.split = "train" if split in ["train", "val"] else split
        (
            self.image_info,
            self.annotations_info,
            self.camera_info,
            self.joints_dict,
        ) = self.get_meta_info()
        self.indices = self.create_train_val_split()

    def get_meta_info(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        data = read_json(
            os.path.join(
                self.root_dir,
                self.annotation_sampling_folder,
                self.annotor,
                f"InterHand2.6M_{self.split}_data.json",
            )
        )
        camera_info = pd.DataFrame(
            read_json(
                os.path.join(
                    self.root_dir,
                    self.annotation_sampling_folder,
                    self.annotor,
                    f"InterHand2.6M_{self.split}_camera.json",
                )
            )
        ).T
        joints_dict = read_json(
            os.path.join(
                self.root_dir,
                self.annotation_sampling_folder,
                self.annotor,
                f"InterHand2.6M_{self.split}_joint_3d.json",
            )
        )
        annotations_info = pd.DataFrame(data["annotations"])
        # selecting only single hand images
        annotations_info = annotations_info[
            annotations_info["hand_type"] != "interacting"
        ]
        annotations_info = annotations_info.set_index(np.arange(len(annotations_info)))
        image_info = pd.DataFrame(data["images"]).set_index("id")
        return image_info, annotations_info, camera_info, joints_dict

    def get_camera_params(
        self, camera, capture_id
    ) -> Tuple[np.array, np.array, np.array]:
        camera_param = self.camera_info.loc[str(capture_id)]
        t, r, (fx, fy), (px, py) = (
            camera_param.campos[camera],
            camera_param.camrot[camera],
            camera_param.focal[camera],
            camera_param.princpt[camera],
        )
        intrinsic_param = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1.0]])
        # intrinsic_param = np.array([[fx, 0, px], [0, fy, py]])
        return intrinsic_param, np.array(r), np.array(t)

    def get_joints(
        self, capture_id: Union[int, str], frame_idx: Union[int, str]
    ) -> Tuple[np.array, np.array, bool]:
        joint_item = self.joints_dict[str(capture_id)][str(frame_idx)]
        if joint_item["hand_type"] == "left":
            return (
                np.array(joint_item["world_coord"][-21:]),
                np.array(joint_item["joint_valid"][-21:]),
                IH_DB.IS_LEFT,
            )
        elif joint_item["hand_type"] == "right":
            return (
                np.array(joint_item["world_coord"][:21]),
                np.array(joint_item["joint_valid"][:21]),
                not IH_DB.IS_LEFT,
            )
        else:
            raise NotImplementedError

    def create_train_val_split(self) -> np.array:
        """Creates split for train and val data in mpii
        Raises:
            NotImplementedError: In case the split doesn't match test, train or val.
        Returns:
            np.array: array of indices
        """
        num_images = len(self.annotations_info)
        train_indices, val_indices = train_test_split(
            np.arange(num_images), train_size=self.train_ratio, random_state=self.seed
        )
        if self._split == "train":
            return np.sort(train_indices)
        elif self._split == "val":
            return np.sort(val_indices)
        elif self._split == "test":
            return np.arange(len(self.annotations_info))
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        """Returns a sample corresponding to the index.

        Args:
            idx (int): index

        Returns:
            dict: item with following elements.
                "image" in opencv bgr format.
                "K": camera params
                "joints3D": 3D coordinates of joints in AIT format.
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx_ = self.indices[idx]
        image_id = self.annotations_info.loc[idx_]["image_id"]
        image_item = self.image_info.loc[image_id]
        image = cv2.imread(
            os.path.join(
                self.root_dir,
                self.image_sampling_folder,
                self.split,
                image_item.file_name,
            )
        )
        joints, joints_valid, is_left = self.get_joints(
            image_item.capture, image_item.frame_idx
        )
        joints, joints_valid = (
            self.joints.interhand_to_ait(joints),
            self.joints.interhand_to_ait(joints_valid),
        )
        intrinsic_camera_matrix, camera_rot, camera_t = self.get_camera_params(
            image_item.camera, image_item.capture
        )
        if is_left:
            image = cv2.flip(image, 1)
            _, W = image.shape[:2]
            intrinsic_camera_matrix = (
                np.float32([[-1, 0, W - 1], [0, 1, 0], [0, 0, 1]])
                @ intrinsic_camera_matrix
            )

        joints_camera_frame = (joints - camera_t) @ camera_rot.T
        # To avoid division by zero.
        joints_camera_frame[:, -1] += 1e-5
        sample = {
            "image": image,
            "K": torch.tensor(intrinsic_camera_matrix).float(),
            "joints3D": torch.tensor(joints_camera_frame).float() / 1000.0,
            "joints_valid": torch.tensor(joints_valid),
        }
        return sample
