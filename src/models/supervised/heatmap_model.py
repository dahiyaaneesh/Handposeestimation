import torch
from src.constants import HRNET_CONFIG
from src.models.external.HRnet.pose_hrnet import get_pose_net
from src.models.external.spatial_2d_soft_argmax import spatial_soft_argmax2d
from src.models.supervised.baseline_model import BaselineModel
from src.utils import read_yaml
from torch import Tensor, nn


class HeatmapPoseModel(BaselineModel):
    def __init__(self, config):
        super().__init__(config)
        self.epsilon = 1e-6
        hrnet_config = read_yaml(HRNET_CONFIG)
        self.encoder = get_pose_net(hrnet_config.MODEL36, True)
        # self.final_layers = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 42, kernel_size=(3, 3), stride=1),
        # )
        self.final_layers = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1),
            nn.Conv2d(64, 42, kernel_size=(3, 3), stride=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        image_h, image_w = x.size()[-2:]
        x = self.encoder(x)
        x = self.final_layers(x)
        h_star_2d, h_star_z = x[:, :21], x[:, 21:]
        out = self.heatmap_to_joints(h_star_2d, h_star_z, image_h, image_w)
        return out

    def heatmap_to_joints(
        self, h_star_2d: Tensor, h_star_z: Tensor, image_h: int, image_w: int
    ) -> Tensor:
        heat_h, heat_w = h_star_2d.size()[-2:]
        scale = (
            torch.tensor([image_h * 1.0 / heat_h, image_w * 1.0 / heat_w])
            .view(1, 1, 2)
            .to(h_star_2d.device)
        )
        joints2d, h_2d = spatial_soft_argmax2d(
            h_star_2d, normalized_coordinates=False, output_probability=True
        )
        joints2d = joints2d * scale
        z_r = torch.sum(h_2d * h_star_z, dim=[2, 3]).view(-1, 21, 1)

        return torch.cat([joints2d, z_r], dim=2)

    # def normalize_heatmap(self, heatmap: Tensor) -> Tensor:
    #     batch, channels, height, width = heatmap.size()
    #     heatmap = heatmap.view(batch, channels, -1)
    #     heatmap = torch.exp(heatmap - torch.max(heatmap, dim=-1, keepdim=True)[0])
    #     heatmap = heatmap / (heatmap.sum(dim=-1, keepdim=True) + self.epsilon)

    #     return heatmap.view(batch, channels, height, width)
