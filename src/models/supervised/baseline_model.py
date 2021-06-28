from typing import Dict

import torch
from easydict import EasyDict as edict
from src.models.base_model import BaseModel
from src.models.utils import cal_3d_loss, cal_l1_loss
from src.utils import get_console_logger
from torch import Tensor, nn


class BaselineModel(BaseModel):
    """Class wrapper for the fully supervised model used as baseline.
    It uses Resnet as the base model and appends more layers in the end to fit the
    HPE Task.
    """

    def __init__(self, config: edict):
        super().__init__(config)
        self.console_logger = get_console_logger("baseline_model")
        if not self.config["resnet_trainable"]:
            self.console_logger.warning("Freeizing the underlying  Resnet!")
            for param in self.encoder.parameters():
                param.requires_grad = False
        # self.final_layers = nn.Sequential(
        #     nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 21 * 3)
        # )
        self.final_layers = nn.Sequential(nn.Linear(512, 21 * 3))

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.final_layers(x)
        x = x.view(-1, 21, 3)
        return x

    def training_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        x, y, scale, joints_valid = (
            batch["image"],
            batch["joints"],
            batch["scale"],
            batch["joints_valid"],
        )
        prediction = self(x)
        loss_2d, loss_z, loss_z_unscaled = cal_l1_loss(
            prediction, y, scale, joints_valid
        )
        loss = loss_2d + self.config.alpha * loss_z
        loss3d = None
        with torch.no_grad():
            loss3d = cal_3d_loss(
                prediction, batch["joints3D"], batch["scale"], batch["K"], joints_valid
            )
        self.train_metrics = {
            "loss": loss.detach(),
            "loss_z": loss_z.detach(),
            "loss_2d": loss_2d.detach(),
            "loss_z_unscaled": loss_z_unscaled.detach(),
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
            "loss_3d": loss3d,
        }

    def validation_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        x, y, scale, joints_valid = (
            batch["image"],
            batch["joints"],
            batch["scale"],
            batch["joints_valid"],
        )
        prediction = self(x)
        loss_2d, loss_z, loss_z_unscaled = cal_l1_loss(
            prediction, y, scale, joints_valid
        )
        loss = loss_2d + self.config.alpha * loss_z
        loss3d = None
        with torch.no_grad():
            loss3d = cal_3d_loss(
                prediction, batch["joints3D"], batch["scale"], batch["K"], joints_valid
            )
        metrics = {
            "loss": loss,
            "loss_z": loss_z,
            "loss_2d": loss_2d,
            "loss_z_unscaled": loss_z_unscaled,
            "loss_3d": loss3d,
        }
        self.plot_params = {"prediction": prediction, "ground_truth": y, "input": x}

        return metrics
