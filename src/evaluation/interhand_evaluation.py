import argparse

from src.data_loader.data_set import Data_Set
from src.experiments.evaluation_utils import (
    calculate_epe_statistics,
    get_predictions_and_ground_truth,
    cal_auc_joints,
    get_pck_curves,
    get_procrustes_statistics,
)
import pandas as pd
import numpy as np
import torch
from easydict import EasyDict as edict
from src.constants import TRAINING_CONFIG_PATH
from src.data_loader.joints import Joints
from src.evaluation.utils import load_model
from src.utils import read_json
from torchvision import transforms
from tqdm import tqdm

JOINTS = Joints()


def main():
    """
    Main eval loop: Iterates over all evaluation samples and saves the corresponding
    predictions as json and zip file. This is the format expected at
    https://competitions.codalab.org/competitions/21238#learn_the_details-overview
    """
    parser = argparse.ArgumentParser(description="Evaluation on Inerhand eval set.")
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
    train_param.augmentation_params.crop_box_jitter = [0.0, 0.0]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    data = Data_Set(
        transform=transform,
        split=args.split,
        experiment_type="supervised",
        source="interhand",
        config=train_param,
    )
    with torch.no_grad():
        prediction_dict = get_predictions_and_ground_truth(
            model, data, num_workers=8, batch_size=128
        )
        epe_2D = calculate_epe_statistics(
            prediction_dict["predictions"],
            prediction_dict["ground_truth"],
            dim=2,
            validitiy_flags=prediction_dict["validitiy_flags"],
        )
        epe_3D = calculate_epe_statistics(
            prediction_dict["predictions_3d"]
            - prediction_dict["predictions_3d"][:, :1],
            prediction_dict["ground_truth_3d"]
            - prediction_dict["ground_truth_3d"][:, :1],
            dim=3,
            validitiy_flags=prediction_dict["validitiy_flags"],
        )
        procrustes_result = get_procrustes_statistics(
            prediction_dict, use_visibitiy=True
        )
        if hasattr(model, "denoiser"):
            epe_3D_gt_vs_denoised = calculate_epe_statistics(
                prediction_dict["ground_truth_3d"]
                - prediction_dict["ground_truth_3d"][:, :1],
                prediction_dict["predictions_3d_denoised"]
                - prediction_dict["predictions_3d_denoised"][:, :1],
                dim=3,
                validitiy_flags=prediction_dict["validitiy_flags"],
            )
            auc_denoised = np.mean(
                cal_auc_joints(epe_3D_gt_vs_denoised["eucledian_dist"])
            )
            denoised_results = {
                "Mean_EPE_3D_denoised": epe_3D_gt_vs_denoised["mean"].cpu(),
                "Median_EPE_3D_denoised": epe_3D_gt_vs_denoised["median"].cpu(),
                "auc_denoised": auc_denoised,
            }
        else:
            denoised_results = {}
        eucledian_dist = epe_3D["eucledian_dist"]
        y, x = get_pck_curves(eucledian_dist, per_joint=True)
        auc = cal_auc_joints(eucledian_dist, per_joint=True)
        results = {
            **{
                "Mean_EPE_2D": epe_2D["mean"].cpu().numpy(),
                "Median_EPE_2D": epe_2D["median"].cpu().numpy(),
                "Mean_EPE_3D": epe_3D["mean"].cpu().numpy(),
                "Median_EPE_3D": epe_3D["median"].cpu().numpy(),
                # "Mean_EPE_3D_R": epe_3D_recreated["mean"].cpu().numpy(),
                # "Median_EPE_3D_R": epe_3D_recreated["median"].cpu().numpy(),
                # "Mean_EPE_3D_R_v_3D": epe_3D__gt_vs_3D_recreated["mean"].cpu().numpy(),
                # "Median_EPE_3D_R_V_3D": epe_3D__gt_vs_3D_recreated["median"].cpu().numpy(),
                "AUC": np.mean(auc),
            },
            **denoised_results,
            **procrustes_result,
        }
        results_df = pd.DataFrame.from_dict(
            {k: ["{:.3f}".format(float(v))] for k, v in results.items()}
        ).T
        results_df = results_df.rename(columns={0: "value"})
        results_df.index.name = "Metric"
        results_df = results_df.reset_index()
        print(results_df)
        results_df.to_csv(f"IH_{args.key}_root_align.csv")


if __name__ == "__main__":
    main()
