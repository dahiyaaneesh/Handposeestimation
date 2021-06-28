from easydict import EasyDict as edict
from src.models.supervised.denoised_heatmap_model import DenoisedHeatmapmodel
from src.models.utils import get_encoder_state_dict


class DenoisedHeatmapHead(DenoisedHeatmapmodel):
    def __init__(self, config: edict):
        super().__init__(config)
        encoder_state_dict = get_encoder_state_dict(
            saved_model_path=config.saved_model_name, checkpoint=config.checkpoint
        )
        self.encoder.load_state_dict(encoder_state_dict)
        if not self.config.encoder_trainable:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
