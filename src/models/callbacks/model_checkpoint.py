from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
# from src.models.external.wrapper_model import WrapperModel
import torch


class UpdatedModelCheckpoint(ModelCheckpoint):
    def _save_model(self, filepath: str, trainer, pl_module):
        """Incase the encoder was based on wrapperModel instance,
        then only encpoder weights are saved, else standard lightening checkpoint saving occurs.
        """
        ModelCheckpoint._save_model(self, filepath, trainer, pl_module)
        # if isinstance(pl_module.encoder, WrapperModel):
        #     torch.save(pl_module.encoder.state_dict(), filepath)
        # else:
            # ModelCheckpoint._save_model(self, filepath, trainer, pl_module)
