import os

# This project was a part of my master thesis, hence the naming.
#  Set this envrionment variable to the root of the cloned directory. 
# eg: for me it was    " /home/aneesh/Documents/work/Handposeestimation"
MASTER_THESIS_DIR = os.environ.get("MASTER_THESIS_PATH")


# Data paths
# Download and put the data files at this path.
DATA_PATH = os.environ.get("DATA_PATH")
FREIHAND_DATA = os.path.join(DATA_PATH, "freihand_dataset")
INTERHAND_DATA = os.path.join(DATA_PATH, "InterHand")
YOUTUBE_DATA = os.path.join(DATA_PATH, "youtube_3d_hands", "data")
MPII_DATA = os.path.join(DATA_PATH, "mpii_dataset", "hand_labels")

# config paths
CONFIG_PATH = os.path.join(MASTER_THESIS_DIR, "src", "experiments", "config")
TRAINING_CONFIG_PATH = os.path.join(CONFIG_PATH, "training_config.json")
SUPERVISED_CONFIG_PATH = os.path.join(CONFIG_PATH, "supervised_config.json")
SIMCLR_CONFIG = os.path.join(CONFIG_PATH, "simclr_config.json")
SIMCLR_HEATMAP_CONFIG = os.path.join(CONFIG_PATH, "simclr_heatmap_config.json")
SSL_CONFIG = os.path.join(CONFIG_PATH, "semi_supervised_config.json")
PAIRWISE_CONFIG = os.path.join(CONFIG_PATH, "pairwise_config.json")
PAIRWISE_HEATMAP_CONFIG = os.path.join(CONFIG_PATH, "pairwise_heatmap_config.json")
HYBRID1_CONFIG = os.path.join(CONFIG_PATH, "hybrid1_config.json")
HYBRID1_HEATMAP_CONFIG = os.path.join(CONFIG_PATH, "hybrid1_heatmap_config.json")
HYBRID2_CONFIG = os.path.join(CONFIG_PATH, "hybrid2_config.json")
HYBRID2_HEATMAP_CONFIG = os.path.join(CONFIG_PATH, "hybrid2_heatmap_config.json")
NIPS_A1_CONFIG = os.path.join(CONFIG_PATH, "nips_a1_config.json")
NIPS_A2_CONFIG = os.path.join(CONFIG_PATH, "nips_a2_config.json")
DOWNSTREAM_CONFIG = os.path.join(CONFIG_PATH, "downstream_config.json")
HYBRID1_AUGMENTATION_CONFIG = os.path.join(
    CONFIG_PATH, "hybrid1_augmentation_config.json"
)
HRNET_CONFIG = os.path.join(
    MASTER_THESIS_DIR, "src", "models", "external", "HRnet", "hrnet_config.yaml"
)
HEATMAP_CONFIG_PATH = os.path.join(CONFIG_PATH, "heatmap_config.json")

ANGLES = [i for i in range(10, 360, 10)]
# Set SAVED_MODELS_BASE_PATH  path to your models directory.
# SAVED_MODELS_BASE_PATH to path_to_base_repo/models   or wherever the models are downloaded.
SAVED_MODELS_BASE_PATH = os.environ.get("SAVED_MODELS_BASE_PATH")
SAVED_META_INFO_PATH = os.environ.get("SAVED_META_INFO_PATH")
STD_LOGGING_FORMAT = "%(name)s -%(levelname)s - %(message)s"
COMET_KWARGS = {
    "api_key": os.environ.get("COMET_API_KEY"),
    "project_name": "master-thesis",
    "workspace": "dahiyaaneesh",
    "save_dir": SAVED_META_INFO_PATH,
}

# MANO mesh to joint matrix
# This is used to convert the surface model of youtube hands into keypoints.
MANO_MAT = os.path.join(
    MASTER_THESIS_DIR, "src", "data_loader", "mano_mesh_to_joints_mat.pth"
)
