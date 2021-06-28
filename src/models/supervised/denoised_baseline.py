from typing import Dict

import torch
from easydict import EasyDict as edict
from src.data_loader.utils import PARENT_JOINT, get_root_depth
from src.models.supervised.baseline_model import BaselineModel
from src.models.utils import cal_3d_loss, cal_l1_loss, get_denoiser
from src.utils import get_console_logger
from torch import Tensor
from torch.nn.modules.loss import L1Loss


class DenoisedBaselineModel(BaselineModel):
    """Class wrapper for the baseline supervised model.
    Uses Resnet as the base model.
    Appends more layers in the end to fit the HPE Task.
    """

    def __init__(self, config: edict):
        super().__init__(config)
        self.console_logger = get_console_logger("denoised_baseline_model")
        self.denoiser = get_denoiser()

    def training_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:

        x, y, scale, k, joints_valid = (
            batch["image"],
            batch["joints"],
            batch["scale"],
            batch["K"],
            batch["joints_valid"],
        )
        prediction = self(x)
        loss_2d, loss_z, loss_z_unscaled = cal_l1_loss(
            prediction, y, scale, joints_valid
        )
        loss = loss_2d + self.config.alpha * loss_z

        z_root_denoised = self.get_denoised_z_root_calc(prediction.detach(), k)

        z_root_gt = batch["joints3D"][:, PARENT_JOINT, -1] / scale
        loss_z_denoise = (
            L1Loss(reduction="none")(z_root_gt, z_root_denoised.view(-1))
            * joints_valid[:, PARENT_JOINT, -1]
            / (joints_valid[:, PARENT_JOINT, -1]).sum()
        ).sum()
        loss += loss_z_denoise
        loss3d = None
        with torch.no_grad():
            loss3d = cal_3d_loss(
                prediction,
                batch["joints3D"],
                batch["scale"],
                batch["K"],
                joints_valid,
                z_root_denoised,
            )
        self.train_metrics = {
            "loss": loss.detach(),
            "loss_z": loss_z.detach(),
            "loss_2d": loss_2d.detach(),
            "loss_z_unscaled": loss_z_unscaled.detach(),
            "loss_z_denoise": loss_z_denoise.detach(),
            "loss_3d": loss3d,
        }
        self.plot_params = {
            "prediction": prediction.detach(),
            "ground_truth": y,
            "input": x,
        }
        return {
            "loss": loss,
            "loss_z": loss_z.detach(),
            "loss_2d": loss_2d.detach(),
            "loss_z_unscaled": loss_z_unscaled.detach(),
            "loss_z_denoise": loss_z_denoise.detach(),
            "loss_3d": loss3d,
        }

    def get_denoised_z_root_calc(self, joints25D: Tensor, k: Tensor) -> Tensor:

        z_root_calc, k_inv = get_root_depth(joints25D, k, is_batch=True)
        joints2d = torch.cat(
            (joints25D[..., :-1], torch.ones_like(joints25D[..., -1:])), dim=-1
        )
        xy = torch.bmm(joints2d, torch.transpose(k_inv, 1, 2))
        z_root_calc = z_root_calc.view((-1, 1))
        batch_size = joints25D.size()[0]
        denoising_input = torch.cat(
            (
                z_root_calc,
                xy[..., :-1].reshape((batch_size, -1)),
                joints25D[..., -1].reshape(batch_size, -1),
            ),
            dim=1,
        )

        return self.denoiser(denoising_input.detach()) + z_root_calc.detach()

    def validation_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        x, y, scale, k, joints_valid = (
            batch["image"],
            batch["joints"],
            batch["scale"],
            batch["K"],
            batch["joints_valid"],
        )
        prediction = self(x)
        loss_2d, loss_z, loss_z_unscaled = cal_l1_loss(
            prediction, y, scale, joints_valid
        )
        loss = loss_2d + self.config.alpha * loss_z

        z_root_denoised = self.get_denoised_z_root_calc(y, k)

        z_root_gt = batch["joints3D"][:, PARENT_JOINT, -1] / scale
        loss_z_denoise = (
            L1Loss(reduction="none")(z_root_gt, z_root_denoised.view(-1))
            * joints_valid[:, PARENT_JOINT, -1]
            / (joints_valid[:, PARENT_JOINT, -1]).sum()
        ).sum()
        loss += loss_z_denoise
        loss += loss_z_denoise
        loss3d = None
        with torch.no_grad():
            loss3d = cal_3d_loss(
                prediction,
                batch["joints3D"],
                batch["scale"],
                batch["K"],
                joints_valid,
                z_root_denoised,
            )
        metrics = {
            "loss": loss,
            "loss_z": loss_z,
            "loss_2d": loss_2d,
            "loss_z_unscaled": loss_z_unscaled,
            "loss_z_denoise": loss_z_denoise,
            "loss_3d": loss3d,
        }
        self.plot_params = {"prediction": prediction, "ground_truth": y, "input": x}

        return metrics
