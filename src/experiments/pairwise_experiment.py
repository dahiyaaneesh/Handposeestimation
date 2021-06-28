import os
from pprint import pformat

from easydict import EasyDict as edict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CometLogger
from src.constants import (
    COMET_KWARGS,
    MASTER_THESIS_DIR,
    PAIRWISE_CONFIG,
    PAIRWISE_HEATMAP_CONFIG,
    TRAINING_CONFIG_PATH,
)
from src.data_loader.data_set import Data_Set
from src.data_loader.utils import get_data, get_train_val_split
from src.experiments.utils import (
    get_callbacks,
    get_general_args,
    get_model,
    prepare_name,
    update_train_params,
    update_model_params,
    save_experiment_key,
)
from src.utils import get_console_logger, read_json


def main():
    experiment_type = "pairwise"
    console_logger = get_console_logger(__name__)
    args = get_general_args("Pairwise model training script.")
    train_param = edict(read_json(TRAINING_CONFIG_PATH))
    train_param = update_train_params(args, train_param)
    model_param_path = PAIRWISE_HEATMAP_CONFIG if args.heatmap else PAIRWISE_CONFIG
    model_param = edict(read_json(model_param_path))
    seed_everything(train_param.seed)

    # data preperation
    data = get_data(
        Data_Set, train_param, sources=args.sources, experiment_type=experiment_type
    )
    train_data_loader, val_data_loader = get_train_val_split(
        data, batch_size=train_param.batch_size, num_workers=train_param.num_workers
    )
    # logger
    experiment_name = prepare_name(
        f"{experiment_type}_", train_param, hybrid_naming=False
    )
    comet_logger = CometLogger(**COMET_KWARGS, experiment_name=experiment_name)
    # model

    model_param = update_model_params(model_param, args, len(data), train_param)
    model_param.augmentation = [
        k for k, v in train_param.augmentation_flags.items() if v
    ]
    console_logger.info(f"Model parameters {pformat(model_param)}")
    model = get_model(
        experiment_type="pairwise",
        heatmap_flag=args.heatmap,
        denoiser_flag=args.denoiser,
    )(config=model_param)

    # callbacks
    callbacks = get_callbacks(
        logging_interval=args.log_interval,
        experiment_type=experiment_type,
        save_top_k=args.save_top_k,
        period=args.save_period,
    )
    # trainer
    trainer = Trainer(
        accumulate_grad_batches=train_param.accumulate_grad_batches,
        gpus="0",
        logger=comet_logger,
        max_epochs=train_param.epochs,
        precision=train_param.precision,
        amp_backend="native",
        **callbacks,
    )
    trainer.logger.experiment.set_code(
        overwrite=True,
        filename=os.path.join(
            MASTER_THESIS_DIR, "src", "experiments", "pairwise_experiment.py"
        ),
    )
    if args.meta_file is not None:
        save_experiment_key(
            experiment_name=experiment_name,
            experiment_key=trainer.logger.experiment.get_key(),
            filename=args.meta_file,
        )
    trainer.logger.experiment.log_parameters(train_param)
    trainer.logger.experiment.log_parameters(model_param)
    trainer.logger.experiment.add_tags(["pretraining", "pairwise"] + args.tag)
    # training
    trainer.fit(model, train_data_loader, val_data_loader)


if __name__ == "__main__":
    main()
