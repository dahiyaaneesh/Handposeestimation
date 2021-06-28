import os
from pprint import pformat

from easydict import EasyDict as edict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CometLogger
from src.constants import (
    COMET_KWARGS,
    HEATMAP_CONFIG_PATH,
    MASTER_THESIS_DIR,
    SUPERVISED_CONFIG_PATH,
    TRAINING_CONFIG_PATH,
)
from src.data_loader.data_set import Data_Set
from src.data_loader.utils import get_data, get_train_val_split
from src.experiments.utils import (
    downstream_evaluation,
    get_callbacks,
    get_general_args,
    get_model,
    prepare_name,
    restore_model,
    update_model_params,
    update_train_params,
)
from src.utils import get_console_logger, read_json
from torchvision import transforms


def main():
    # get configs and args
    console_logger = get_console_logger(__name__)
    args = get_general_args("Baseline model training script.")
    train_param = edict(read_json(TRAINING_CONFIG_PATH))
    model_param_path = HEATMAP_CONFIG_PATH if args.heatmap else SUPERVISED_CONFIG_PATH
    model_param = edict(read_json(model_param_path))
    train_param = update_train_params(args, train_param)
    console_logger.info(f"Train parameters {pformat(train_param)}")

    # seed
    seed_everything(train_param.seed)

    # data preperation
    data = get_data(
        Data_Set, train_param, sources=args.sources, experiment_type="supervised"
    )
    train_data_loader, val_data_loader = get_train_val_split(
        data, num_workers=train_param.num_workers, batch_size=train_param.batch_size
    )

    # logger
    comet_logger = CometLogger(
        experiment_name=prepare_name("sup", train_param), **COMET_KWARGS
    )

    # Model
    model_param = update_model_params(model_param, args, len(data), train_param)
    console_logger.info(f"Model parameters {pformat(model_param)}")
    model = get_model(
        experiment_type="supervised",
        heatmap_flag=args.heatmap,
        denoiser_flag=args.denoiser,
    )(config=model_param)

    # callbacks
    callbacks = get_callbacks(
        logging_interval=args.log_interval,
        experiment_type="supervised",
        save_top_k=args.save_top_k,
        period=args.save_period,
    )

    # Trainer setup
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
            MASTER_THESIS_DIR, "src", "experiments", "baseline_experiment.py"
        ),
    )
    tags = ["SUPERVISED", "downstream", "baseline"]
    tags += ["heatmap"] if args.heatmap else []
    trainer.logger.experiment.add_tags(tags + args.tag)
    trainer.logger.experiment.log_parameters(train_param)
    trainer.logger.experiment.log_parameters(model_param)

    # fit model
    trainer.fit(model, train_data_loader, val_data_loader)

    # restore the best model
    model = restore_model(model, trainer.logger.experiment.get_key())

    # evaluation
    downstream_evaluation(
        model, data, train_param.num_workers, train_param.batch_size, trainer.logger
    )


if __name__ == "__main__":
    main()
