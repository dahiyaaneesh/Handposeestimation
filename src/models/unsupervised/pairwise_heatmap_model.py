from src.constants import HRNET_CONFIG
from src.models.external.HRnet.pose_hrnet import get_pose_net
from src.models.unsupervised.pairwise_model import PairwiseModel
from src.utils import read_yaml
from torch import nn, Tensor


class PairwiseHeatmapModel(PairwiseModel):
    def __init__(self, config):
        super().__init__(config)
        self.epsilon = 1e-6
        hrnet_config = read_yaml(HRNET_CONFIG)
        self.encoder = get_pose_net(hrnet_config.MODEL36, True)
        self.layer_flattener = nn.Sequential(
            nn.Conv2d(42, 32, kernel_size=(8, 8), stride=8), nn.Flatten()
        )

    def get_encodings(self, batch_images: Tensor) -> Tensor:
        encoding_heatmap = self.encoder(batch_images)
        return self.layer_flattener(encoding_heatmap)
