Exploring self-supervised learning techniques for hand pose estimation
==============================
Abstract

3D hand pose estimation from monocular RGB is a challenging problem due to significantly varying environmental conditions such as lighting or variation in subject appearances. One way to improve performance across board is to introduce more data. However, acquiring 3D annotated data for hands is a laborious task, as it involves heavy multi-camera set up leading to lab-like training data which does not generalize well. Alternatively, one could make use of unsupervised pre-training in order to significantly increase the training data size one can train on. More recently, contrastive learning has shown promising results on tasks such as image classification. Yet, no study has been made on how it affects structured regression problems such as hand pose estimation. We hypothesize that the contrastive objective does extend well to such downstream task due to its inherent invariance and instead propose a relation objective, promoting equivariance. Our goal is to perform extensive experiments to validate our hypothesis.

-------

**This paper was submitted to [Pre-register workshop](https://preregister.science/) at NeurIPS 2020.** Paper Id 50

# Note:
Paper is in result phase: Final paper will be updated here after the result review phase is over ;) 

Setup
------------
```
make env      # to make a virtual environment 
make requirements  # to install all the requirements and setting up of githooks
source env_name/bin/activate  #to activate the environment
```
# Models
## Contrastive model
![Contrastive model](pmlr_contrastive.png)

## Relative model
![Relative model](pmlr_relative.png)
# Training
The models were trained on [ETH's GPU cluster](https://scicomp.ethz.ch/wiki/Leonhard) 
For reference, use ``pmlr_launcher.sh`` to retrain the models on other architectures

### Augmentations visualization and Model Evaluation.
```
voila notebooks/01-Data_handler.ipynb --theme=dark
voila notebooks/02-Model-Evaluation.ipynb --theme=dark
```

Project Organization
------------
```
.
|
├── models
├── notebooks
│   ├── 01-Data_handler.ipynb
│   ├── 02-Model-Evaluation.ipynb
├── reports
│   ├── figures
├── src
│   ├── data_loader # classes and files to read data.
│   │   ├── data_set.py
│   │   ├── freihand_loader.py
│   │   ├── interhand_loader.py
│   │   ├── joint_mapping.json
│   │   ├── joints.py
│   │   ├── mano_mesh_to_joints_mat.pth
│   │   ├── mpii_loader.py
│   │   ├── sample_augmenter.py
│   │   ├── tests.py
│   │   ├── utils.py
│   │   └── youtube_loader.py
│   ├── evaluation # classes and scripts for model evaluation.
│   │   ├── freihand_evaluation.py
│   │   ├── interhand_evaluation.py
│   │   └── utils.py
│   ├── experiments # experiment scripts
│   │   ├── config # model and training configs
│   │   │   ├── heatmap_config.json
│   │   │   ├── hybrid1_augmentation_config.json
│   │   │   ├── hybrid1_config.json
│   │   │   ├── hybrid1_heatmap_config.json
│   │   │   ├── hybrid2_config.json
│   │   │   ├── hybrid2_heatmap_config.json
│   │   │   ├── pairwise_config.json
│   │   │   ├── pairwise_heatmap_config.json
│   │   │   ├── semi_supervised_config.json
│   │   │   ├── simclr_config.json
│   │   │   ├── simclr_heatmap_config.json
│   │   │   ├── supervised_config.json
│   │   │   └── training_config.json
│   │   ├── tests # unit tests for evaluation
│   │   │   └── eval_tests.py
│   │   ├── baseline_experiment.py # script to train supervised model
│   │   ├── evaluation_utils.py
│   │   ├── pairwise_experiment.py # script to train relative model encoder
│   │   ├── save_imagenet_encoder.py
│   │   ├── semi_supervised_experiment.py # script to train pre-trained models
│   │   ├── simclr_experiment.py # script to train simclr model encoder
│   │   └── utils.py
│   ├── models # Model definitions.
│   │   ├── callbacks
│   │   │   ├── model_checkpoint.py
│   │   │   └── upload_comet_logs.py
│   │   ├── external
│   │   │   ├── HRnet
│   │   │   │   ├── hrnet_config.yaml
│   │   │   │   ├── pose_hrnet.py
│   │   │   │   └── README.md
│   │   │   ├── __init__.py
│   │   │   └── spatial_2d_soft_argmax.py
│   │   ├── semisupervised
│   │   │   ├── __init__.py
│   │   │   ├── denoised_heatmap_head_model.py
│   │   │   ├── denoised_supervised_head_model.py
│   │   │   ├── heatmap_head_model.py
│   │   │   └── supervised_head_model.py
│   │   ├── supervised
│   │   │   ├── __init__.py
│   │   │   ├── baseline_model.py
│   │   │   ├── denoised_baseline.py
│   │   │   ├── denoised_heatmap_model.py
│   │   │   └── heatmap_model.py
│   │   ├── tests
│   │   │   └── loss_test.py
│   │   ├── unsupervised
│   │   │   ├── __init__.py
│   │   │   ├── pairwise_heatmap_model.py
│   │   │   ├── pairwise_model.py
│   │   │   ├── simclr_heatmap_model.py
│   │   │   └── simclr_model.py
│   │   ├── __init__.py
│   │   ├── .gitkeep
│   │   ├── base_model.py
│   │   ├── model_restoring_utils.py
│   │   └── utils.py
│   ├── visualization # uitls for visulaizations
│   │   ├── __init__.py
│   │   ├── .gitkeep
│   │   ├── joint_color.json
│   │   └── visualize.py
│   ├── __init__.py
│   ├── constants.py
│   ├── types.py
│   └── utils.py
├── LICENSE
├── Makefile
├── pmlr_launcher.sh
├── README.md
├── requirements.txt
├── setup.py
├── test_environment.py
└── tox.ini
```
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
