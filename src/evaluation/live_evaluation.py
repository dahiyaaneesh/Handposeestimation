import os
import argparse

os.environ["MASTER_THESIS_PATH"] = "/home/aneesh/Documents/work/Handposeestimation"
os.environ[ "SAVED_MODELS_BASE_PATH"] = "/home/aneesh/Documents/work/Handposeestimation/models"

from src.models.supervised.denoised_baseline import DenoisedBaselineModel
from src.models.supervised.denoised_heatmap_model import DenoisedHeatmapmodel
from src.experiments.utils import restore_model
from easydict import EasyDict as edict
from src.utils import read_json
from src.constants import SUPERVISED_CONFIG_PATH
import cv2
from src.visualization.visualize import draw_hand_over_image
from src.evaluation.utils import make_raw_inference, capture_frame


def main():
    parser = argparse.ArgumentParser(description="Evaluation on Live capture")
    parser.add_argument(
        "-key",
        type=str,
        help="Add key of the experiment",
        default="89947329cacb423da8224635f5958d54",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Select model as heatmap type",
        default=False,
    )
    args = parser.parse_args()

    # Load model
    model_config = edict(read_json(SUPERVISED_CONFIG_PATH))
    model_config["resnet_size"] = "18"
    if args.heatmap:
        model = DenoisedHeatmapmodel(model_config)
    else:
        model = DenoisedBaselineModel(model_config)
    restore_model(model, experiment_key=args.key, checkpoint="")
    model.eval()

    # Start capturing frames.
    vc = cv2.VideoCapture(0)
    frame = capture_frame(vc)
    if frame is not None:
        while True:
            frame = capture_frame(vc)
            prediction = make_raw_inference(model, frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = draw_hand_over_image(frame, prediction)
            cv2.imshow(f"HandPose Model {args.key}", frame)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

    vc.release()
    cv2.destroyAllWindows()


main()
