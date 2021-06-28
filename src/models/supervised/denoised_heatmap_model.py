from typing import Dict

from src.models.supervised.denoised_baseline import DenoisedBaselineModel
from src.models.supervised.heatmap_model import HeatmapPoseModel
from src.models.utils import get_denoiser
from torch import Tensor


class DenoisedHeatmapmodel(HeatmapPoseModel, DenoisedBaselineModel):
    def __init__(self, config):
        HeatmapPoseModel.__init__(self, config)
        self.denoiser = get_denoiser()

    def training_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        return DenoisedBaselineModel.training_step(self, batch, batch_idx)

    def validation_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        return DenoisedBaselineModel.validation_step(self, batch, batch_idx)

    def forward(self, x: Tensor) -> Tensor:
        return HeatmapPoseModel.forward(self, x)
