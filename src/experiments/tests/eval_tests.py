from src.data_loader.utils import get_data
from src.experiments.evaluation_utils import (
    calc_procrustes_transform,
    calc_procrustes_transform2,
)
from math import cos, sin
import unittest

import torch


def get_rot_mat(angle):
    return torch.tensor(
        [[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]]
    )


class TestExperimentMethods(unittest.TestCase):
    def test_calc_procrustes_transform(self):
        print("Perfroming test on procrsutes transform")
        joints1 = torch.rand((128, 21, 3))
        translate, scale, angle = torch.tensor([5.0, 6.0, 0.0]), 5.0, 90.0
        rot_mat = get_rot_mat(angle).view(1, 3, 3).repeat(128, 1, 1)
        joints2 = joints1.clone()
        joints2 = torch.bmm(joints2, rot_mat.transpose(2, 1)) * scale
        joints2[..., 0] += translate[0]
        joints2[..., 1] += translate[1]
        joints2[..., 2] += translate[2]

        (
            joints_transformed,
            rot_cal,
            scale_cal,
            translation_cal,
        ) = calc_procrustes_transform(joints1, joints2)
        self.assertTrue(((joints_transformed - joints1).abs().max() < 1e-5).tolist())
        # calculated rotation matrix is in batch transposed format.
        self.assertTrue(
            (
                (rot_cal - get_rot_mat(-angle).T.view(1, 3, 3)).abs().max() < 1e-6
            ).tolist()
        )
        self.assertTrue(((scale - 1 / scale_cal).abs().max() < 1e-3).tolist())
        self.assertTrue(
            (
                (
                    translation_cal
                    + (get_rot_mat(-angle) @ translate.view(3, 1) / scale).view(1, 1, 3)
                )
                .abs()
                .max()
                < 1e-5
            ).tolist()
        )

    def test_calc_procrustes_transform2(self):
        print("Perfroming test on procrsutes transform2 with validaity flags")
        joints1 = torch.rand((1, 21, 3))
        translate, scale, angle = torch.tensor([5.0, 6.0, 0.0]), 5.0, 90.0
        rot_mat = get_rot_mat(angle).view(1, 3, 3)
        joints2 = joints1.clone()
        joints2 = torch.bmm(joints2, rot_mat.transpose(2, 1)) * scale
        joints2[..., 0] += translate[0]
        joints2[..., 1] += translate[1]
        joints2[..., 2] += translate[2]
        validity_flags = torch.ones((1, 21), dtype=torch.bool)
        validity_flags[0, 20] = not validity_flags[0, 20]
        (
            joints_transformed2,
            rot_cal_,
            scale_ca_l,
            translation_cal_,
        ) = calc_procrustes_transform(joints1, joints2)
        # This change should not effect procrustes transform 2
        faulty_joints = joints1.clone()
        faulty_joints[:, 20, 2] += 100000
        (
            joints_transformed,
            rot_cal,
            scale_cal,
            translation_cal,
        ) = calc_procrustes_transform2(faulty_joints, joints2, validity_flags)

        self.assertTrue(((joints_transformed - joints1).abs().max() < 1e-5).tolist())
        self.assertTrue(
            ((joints_transformed2 - joints_transformed).abs().max() < 1e-5).tolist()
        )


if __name__ == "__main__":
    unittest.main()
