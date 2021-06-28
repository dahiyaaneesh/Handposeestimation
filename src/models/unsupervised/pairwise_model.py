from typing import Dict, Tuple

import torch
from src.models.base_model import BaseModel
from torch import nn
from torch.nn import L1Loss
from torch.tensor import Tensor


class PairwiseModel(BaseModel):
    """
    Pairwise self-supervised model. The transformation parameters are regressed in this
        model.
    """

    def __init__(self, config):
        super().__init__(config)
        self.regress_rotate = False
        self.regress_jitter = False
        self.regress_color_jitter = False
        self.regress_scale = False
        self.log_keys = ["loss"]

        # transformation head.
        if "rotate" in self.config.augmentation:
            self.rotation_head = self.get_rotation_head()
        if "crop" in self.config.augmentation:
            self.jitter_head = self.get_jitter_head()
        if "color_jitter" in self.config.augmentation:
            self.color_jitter_head = self.get_color_jitter_head()
        if "random_crop" in self.config.augmentation:
            self.scale_head = self.get_scale_head()

    def get_base_transformation_head(self, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(
                self.config.transformation_head_input_dim * 2,
                self.config.transformation_head_hidden_dim,
                bias=True,
            ),
            nn.BatchNorm1d(self.config.transformation_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.config.transformation_head_hidden_dim, output_dim, bias=False
            ),
        )

    def get_rotation_head(self) -> nn.Sequential:
        self.regress_rotate = True
        self.log_keys += ["loss_rotation", "sigma_rotation"]
        self.log_sigma_rotate = nn.Parameter(torch.zeros(1, 1))
        rotation_head = self.get_base_transformation_head(output_dim=1)
        return rotation_head

    def get_jitter_head(self) -> nn.Sequential:
        self.regress_jitter = True
        self.log_keys += ["loss_jitter", "sigma_jitter"]
        self.log_sigma_jitter = nn.Parameter(torch.zeros(1, 1))
        return self.get_base_transformation_head(output_dim=2)

    def get_color_jitter_head(self) -> nn.Sequential:
        self.regress_color_jitter = True
        self.log_keys += ["loss_color_jitter", "sigma_color_jitter"]
        self.log_sigma_color_jitter = nn.Parameter(torch.zeros(1, 1))
        return self.get_base_transformation_head(output_dim=4)

    def get_scale_head(self) -> nn.Sequential:
        self.regress_scale = True
        self.log_keys += ["loss_scale", "sigma_scale"]
        self.log_sigma_scale = nn.Parameter(torch.zeros(1, 1))
        return self.get_base_transformation_head(output_dim=1)

    def regress_rotation(
        self,
        rotation_gt: Tensor,
        encoding: Tensor,
        loss: Tensor,
        log: dict,
        pred_gt: Tensor,
    ) -> Tensor:
        rotation_pred = self.rotation_head(encoding)
        loss_rotation = L1Loss()(rotation_gt, rotation_pred)
        loss += loss_rotation / torch.exp(self.log_sigma_rotate) + self.log_sigma_rotate
        log.update(
            {
                "loss_rotation": loss_rotation.detach(),
                "sigma_rotation": torch.exp(self.log_sigma_rotate).detach(),
            }
        )
        pred_gt.update({"rotation": [rotation_gt, rotation_pred]})
        return loss

    def regress_scaling(
        self,
        scale_gt: Tensor,
        encoding: Tensor,
        loss: Tensor,
        log: dict,
        pred_gt: Tensor,
    ) -> Tensor:
        scale_pred = self.scale_head(encoding)
        loss_scale = L1Loss()(scale_gt, scale_pred)
        loss += loss_scale / torch.exp(self.log_sigma_scale) + self.log_sigma_scale
        log.update(
            {
                "loss_scale": loss_scale.detach(),
                "sigma_scale": torch.exp(self.log_sigma_scale).detach(),
            }
        )
        pred_gt.update({"scale": [scale_gt, scale_pred]})
        return loss

    def regress_jittering(
        self,
        jitter_gt: Tensor,
        encoding: Tensor,
        loss: Tensor,
        log: Dict[str, Tensor],
        pred_gt: Tensor,
    ) -> Tensor:
        jitter_pred = self.jitter_head(encoding)
        loss_jitter = L1Loss()(jitter_gt, jitter_pred)
        loss += loss_jitter / torch.exp(self.log_sigma_jitter) + self.log_sigma_jitter
        log.update(
            {
                "loss_jitter": loss_jitter.detach(),
                "sigma_jitter": torch.exp(self.log_sigma_jitter).detach(),
            }
        )
        pred_gt.update({"jitter": [jitter_gt, jitter_pred]})
        return loss

    def regress_color_jittering(
        self,
        color_jitter_gt: Tensor,
        encoding: Tensor,
        loss: Tensor,
        log: Dict[str, Tensor],
        pred_gt: Tensor,
    ) -> Tensor:
        color_jitter_pred = self.color_jitter_head(encoding)
        loss_color_jitter = L1Loss()(color_jitter_gt, color_jitter_pred)
        loss += (
            loss_color_jitter / torch.exp(self.log_sigma_color_jitter)
            + self.log_sigma_color_jitter
        )
        log.update(
            {
                "loss_color_jitter": loss_color_jitter.detach(),
                "sigma_color_jitter": torch.exp(self.log_sigma_color_jitter).detach(),
            }
        )
        pred_gt.update({"color_jitter": [color_jitter_gt, color_jitter_pred]})
        return loss

    def transformation_regression_step(
        self, batch: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
        batch_images = torch.cat(
            (batch["transformed_image1"], batch["transformed_image2"]), dim=0
        )
        batch_size = len(batch_images) // 2
        encoding = self.get_encodings(batch_images)
        # concatentating embeddings of augmented sample pairs
        encoding = torch.cat((encoding[:batch_size], encoding[batch_size:]), 1)
        loss = 0
        log = {}
        pred_gt = {}
        # Rotation regression
        if self.regress_rotate:
            rotate_gt = batch["rotation"]
            loss = self.regress_rotation(rotate_gt, encoding, loss, log, pred_gt)
        # Translation  jitter regression
        if self.regress_jitter:
            jitter_gt = batch["jitter"]
            loss = self.regress_jittering(jitter_gt, encoding, loss, log, pred_gt)
        # Color jitter regression
        if self.regress_color_jitter:
            color_jitter_gt = batch["color_jitter"]
            loss = self.regress_color_jittering(
                color_jitter_gt, encoding, loss, log, pred_gt
            )
        if self.regress_scale:
            scale_gt = batch["scale"]
            loss = self.regress_scaling(scale_gt, encoding, loss, log, pred_gt)
        return (loss, log, pred_gt)

    def get_encodings(self, batch_images: Tensor) -> Tensor:
        return self.encoder(batch_images)

    def forward(self, x: Tensor) -> Tensor:
        embedding = self.get_encodings(x)
        return embedding

    def training_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:
        loss, losses, gt_pred = self.transformation_regression_step(batch)
        self.train_metrics = {**{"loss": loss.detach()}, **losses}
        self.plot_params = {
            **{
                "image1": batch["transformed_image1"],
                "image2": batch["transformed_image2"],
            },
            **{"gt_pred": gt_pred},
        }
        return {**{"loss": loss}, **losses}

    def validation_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:
        loss, losses, gt_pred = self.transformation_regression_step(batch)
        self.plot_params = {
            **{
                "image1": batch["transformed_image1"],
                "image2": batch["transformed_image2"],
            },
            **{"gt_pred": gt_pred},
        }
        return {**{"loss": loss}, **losses}
