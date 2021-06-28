from src.data_loader.data_set import Data_Set
from src.utils import read_json
from easydict import EasyDict as edict
from src.constants import TRAINING_CONFIG_PATH
from src.data_loader.utils import error_in_conversion, get_data
from tqdm import tqdm


def main():
    train_param = edict(read_json(TRAINING_CONFIG_PATH))
    data = get_data(
        Data_Set,
        train_param,
        sources=["youtube"],
        experiment_type="supervised",
        split="train",
    )
    for id in tqdm(range(len(data))):
        sample = data[id]
        # joints25D = sample["joints"]
        # scale = sample["scale"]
        # K = sample["K"]
        true_joints_3D = sample["joints3D"]
        cal_joints_3D = sample["joints3D_recreated"]
        error = error_in_conversion(true_joints_3D, cal_joints_3D)
        if error > 1e-3:
            print(f"High error found {error} of the true ")
            break


if __name__ == "__main__":
    main()
