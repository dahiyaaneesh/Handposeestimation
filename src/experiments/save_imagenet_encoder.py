import os

import torch
import torchvision
from src.constants import SAVED_MODELS_BASE_PATH
from src.models.utils import get_resnet


def main():
    resnet_encoder = get_resnet(resnet_size="50", pretrained=False)
    # resnet_encoder.fc = torch.nn.Sequential()
    path_to_checkpoint = os.path.join(SAVED_MODELS_BASE_PATH, "ResNet50", "checkpoints")
    if not os.path.exists(path_to_checkpoint):
        os.makedirs(path_to_checkpoint)
    state_dict = {f"encoder.{k}": v for k, v in resnet_encoder.state_dict().items()}
    torch.save(
        {"state_dict": state_dict}, os.path.join(path_to_checkpoint, "epoch=0.ckpt")
    )


if __name__ == "__main__":
    main()
