from src.data_loader.data_set import Data_Set
import unittest
from easydict import EasyDict as edict
import torch
from src.utils import read_json
from src.constants import TRAINING_CONFIG_PATH
from src.models.utils import cal_l1_loss, cal_3d_loss
from torch.utils.data import DataLoader


class TestStringMethods(unittest.TestCase):
    def test_cal_l1_loss(self):
        print("Running test on supervised loss calulation")
        pred_joints = torch.ones((12, 21, 3), dtype=torch.float16)
        true_joints = torch.ones((12, 21, 3), dtype=torch.float16) * 2
        scale = torch.ones((12), dtype=torch.float16) * 10.0
        joints_valid = torch.ones((12, 21, 1), dtype=torch.float16)
        joints_valid[10, 5:20] = 0.0
        loss_2d, loss_z, loss_z_unscaled = cal_l1_loss(
            pred_joints, true_joints, scale, joints_valid
        )
        self.assertTrue(((loss_z - 1.0) < 1e-6).tolist())
        self.assertTrue(((loss_2d - 1.0) < 1e-6).tolist())
        self.assertTrue(((loss_z_unscaled - 10.0) < 1e-6).tolist())

    def test_cal_loss3d(self):
        print("Running test on 3d loss calculation")
        train_param = edict(read_json(TRAINING_CONFIG_PATH))
        data = DataLoader(
            Data_Set(
                train_param,
                None,
                split="val",
                experiment_type="supervised",
                source="freihand",
            ),
            batch_size=12,
        )
        loss3d = 100
        for i in iter(data):
            sample = i
            loss3d = cal_3d_loss(
                sample["joints"],
                sample["joints3D"] + 5,
                sample["scale"],
                sample["K"],
                sample["joints_valid"],
            )
            print(loss3d)
            break

        self.assertTrue((loss3d - 5 < 1e-6).tolist())


if __name__ == "__main__":
    unittest.main()
