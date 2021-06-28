import argparse
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import torch
from easydict import EasyDict as edict
from src.constants import FREIHAND_DATA, TRAINING_CONFIG_PATH
from src.data_loader.freihand_loader import F_DB
from src.data_loader.joints import Joints
from src.data_loader.sample_augmenter import SampleAugmenter
from src.evaluation.utils import load_model, model_refined_inference, normalize_joints
from src.utils import read_json, save_json
from torchvision import transforms
from tqdm import tqdm

JOINTS = Joints()


def main():
    """
    Main eval loop: Iterates over all evaluation samples and saves the corresponding
    predictions as json and zip file. This is the format expected at
    https://competitions.codalab.org/competitions/21238#learn_the_details-overview
    """
    parser = argparse.ArgumentParser(description="Evaluation on Freihand eval set.")
    parser.add_argument(
        "-key", type=str, help="Add comet key of experiment to restore."
    )
    parser.add_argument(
        "-resnet_size",
        type=str,
        help="Resnet sizes",
        choices=["18", "34", "50", "101", "152"],
        default=50,
    )
    parser.add_argument(
        "--heatmap", action="store_true", help="Choose Resnet", default=False
    )
    parser.add_argument(
        "--palm_trained",
        action="store_true",
        help="Use when palm is regressed during training.",
        default=False,
    )
    parser.add_argument(
        "-split",
        type=str,
        help="For debugging select val split",
        default="test",
        choices=["test", "val"],
    )
    parser.add_argument(
        "-checkpoint", type=str, help="selectign checkpoint", default=""
    )
    args = parser.parse_args()
    model = load_model(args.key, args.resnet_size, args.heatmap, args.checkpoint)
    if args.split == "val":
        print(
            "DEBUG MODE ACTIVATED.\n Evaluation pipeline is executed on validation set"
        )
    train_param = edict(read_json(TRAINING_CONFIG_PATH))
    train_param.augmentation_flags.resize = True
    train_param.augmentation_flags.crop = True
    # train_param.augmentation_params.crop_margin = 1.5
    train_param.augmentation_params.crop_box_jitter = [0.0, 0.0]
    augmenter = SampleAugmenter(
        train_param.augmentation_flags, train_param.augmentation_params
    )
    # Normalization for BGR mode.
    # transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             (0.485, 0.456, 0.406)[::-1], (0.229, 0.224, 0.225)[::-1]
    #         ),
    #     ]
    # )
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    data = F_DB(FREIHAND_DATA, split=args.split)
    xyz_pred = []
    debug_mean = []
    with torch.no_grad():
        for i in tqdm(range(len(data))):
            joints3d_normalized = normalize_joints(
                model_refined_inference(
                    model, data[i], augmenter, transform, args.palm_trained
                )
            )
            if args.split == "val":
                # DEBUG CODE:
                joints3d = joints3d_normalized * data.scale[data.indices[i] % 32560]
                debug_mean.append(torch.mean(torch.abs(joints3d - data[i]["joints3D"])))
            else:
                joints3d = joints3d_normalized * data.scale[data.indices[i]]

            xyz_pred.append(JOINTS.ait_to_freihand(joints3d).tolist())

    if args.split == "val":
        # DEBUG CODE:
        print(
            f"MAE 3d\nMean : {np.mean(debug_mean)}\nMax: { np.max(debug_mean)}"
            "\nMedian: { np.median(debug_mean)}"
        )
        exit()

    verts = np.zeros((len(xyz_pred), 778, 3)).tolist()
    save_json([xyz_pred, verts], f"{args.key}_pred.json")
    subprocess.call(["zip", "-j", f"{args.key}_pred.zip", f"{args.key}_pred.json"])
    subprocess.call(["rm", f"{args.key}_pred.json"])


if __name__ == "__main__":
    main()
