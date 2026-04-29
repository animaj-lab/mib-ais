import logging
import os
import tempfile
from enum import StrEnum, auto
from pathlib import Path

import mlflow
import torch
import typer
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from pytorch_lightning.tuner.tuning import Tuner

from motion_inbetweening.config.train import GlobalTrainingConfig, LoggingConfig
from motion_inbetweening.infra.configs.train import get_best_model_config, get_debug_config, get_default_config
from motion_inbetweening.infra.configs.train_diffusion import (
    get_diffusion_best_model_config,
    get_diffusion_debug_config,
)
from motion_inbetweening.infra.dev_settings import get_dataset_directory
from motion_inbetweening.infra.hf_dataset import ensure_hf_dataset
from motion_inbetweening.infra.loading.safetensors import save_safetensors
from motion_inbetweening.infra.pytorch_lightning.data_module import MotionInbetweeningDatamodule
from motion_inbetweening.lightning_callbacks.create_callbacks import create_callbacks
from motion_inbetweening.lightning_modules.init_training_module import init_training_module
from motion_inbetweening.lightning_modules.seq2seq_module import Seq2SeqModule
from shared.domain.entities.ip import CharacterName
from shared.domain.services.ip import validate_character_name
from shared.rig.trainable_controllers import load_trainable_controllers_from_conf
from shared.utils import get_device, get_save_dir, save_fig_lr_finder, save_yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()
torch.set_float32_matmul_precision("medium")


def create_loggers(config: LoggingConfig, save_dir: Path):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") if config.upload_to_mlflow else None
    mlflow_logger = MLFlowLogger(
        experiment_name=config.project_name,
        tracking_uri=tracking_uri,
        run_name=config.experiment_name,
        save_dir=save_dir.as_posix(),
        tags=config.tags,
    )
    mlflow_logger.log_hyperparams(config.hyperparameters)
    tb_logger = TensorBoardLogger(save_dir=save_dir.as_posix(), default_hp_metric=False)
    tb_logger.log_hyperparams(config.hyperparameters, metrics={"test/loss": 0, "val/loss": 0})
    return [mlflow_logger, tb_logger]


def log_config_files(logger: MLFlowLogger, config: GlobalTrainingConfig) -> None:
    client = MlflowClient(logger._tracking_uri)
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dict = config.model_dump(mode="python")
        config_path = Path(temp_dir) / "config.yaml"
        save_yaml(config_path, config_dict)
        client.log_artifact(logger.run_id, config_path)
    for config_file_path in config.controllers.model_dump().values():
        if config_file_path is None:
            continue
        client.log_artifact(logger.run_id, config_file_path)


def get_custom_config(character_name: CharacterName) -> GlobalTrainingConfig:
    model_name = "lstm"
    config = get_default_config(model_name, character_name)
    config.logging.project_name = f"MIB_{character_name}_custom"
    return config


def run_single_training(config: GlobalTrainingConfig) -> float:
    seed_everything(config.seed)
    trainable_controllers = load_trainable_controllers_from_conf(config.controllers)
    device = get_device()
    save_dir = get_save_dir(config)
    loggers = create_loggers(config.logging, save_dir)
    callbacks = create_callbacks(
        config.callbacks,
        save_dir,
        trainable_controllers,
        config.data_module.predict_dataset,
        config.data_module.controller_keyframes_preprocessing_config,
    )
    data_module = MotionInbetweeningDatamodule(config.data_module, trainable_controllers)
    data_module.setup("fit")
    pose_normalizer = data_module.pose_normalizer
    train_module = init_training_module(
        config=config.train_module,
        controllers_config=config.controllers,
        trainable_controllers=trainable_controllers,
        pose_normalizer=pose_normalizer,
        controller_keyframes_preprocessing_config=config.data_module.controller_keyframes_preprocessing_config,
    )
    trainer = Trainer(
        accelerator=device,
        devices=1,
        max_epochs=config.training.max_epochs,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        val_check_interval=config.training.val_check_interval,
        logger=loggers,
        callbacks=callbacks,
        precision=config.training.precision,
        gradient_clip_val=config.training.gradient_clip_val,
    )
    if config.train_module.optimization.optimizer.tune_lr:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(train_module, datamodule=data_module)
        save_fig_lr_finder(lr_finder, save_dir)
    trainer.validate(train_module, datamodule=data_module)
    log_config_files(loggers[0], config)
    trainer.fit(train_module, datamodule=data_module)

    ckpt_path = save_dir / "checkpoints" / "model.ckpt"
    save_safetensors(ckpt_path, save_dir / "checkpoints" / "safetensors")

    # For pytorch 2.6.0 update: reloading manually the best checkpoint with weight_only=False (set in the BaseModule)
    best_train_module = Seq2SeqModule.load_from_checkpoint(save_dir / "checkpoints" / "model.ckpt")
    evaluate_output = trainer.test(model=best_train_module, datamodule=data_module)
    shifted_distance = evaluate_output[0]["test/shifted_distance"]

    if config.data_module.predict_dataset is not None:
        trainer.predict(datamodule=data_module, ckpt_path="best")
    if config.logging.save_model_to_mlflow:
        with mlflow.start_run(run_id=loggers[0].run_id):
            mlflow.pytorch.log_model(train_module, "model")
    return shifted_distance


class TrainingType(StrEnum):
    best = auto()
    custom = auto()
    debug = auto()
    from_file = auto()


def main(
    character_name_str: str,
    training_type: TrainingType = TrainingType.debug,
    model: str = "lstm",
    config_file_path: Path | None = None,
) -> None:
    character_name = validate_character_name(character_name_str)
    dataset_dir = get_dataset_directory(character_name)
    ensure_hf_dataset(dataset_dir)
    if model == "lstm" or model == "citl" or model == "delta_interpolator":
        match training_type:
            case TrainingType.best:
                config = get_best_model_config(model_name=model, character_name=character_name)
            case TrainingType.debug:
                config = get_debug_config(model_name=model, character_name=character_name)
            case TrainingType.custom:
                config = get_custom_config(character_name)
            case TrainingType.from_file:
                if config_file_path is None:
                    raise ValueError("config_file_path must be provided for from_file training type")
                config = GlobalTrainingConfig.from_yaml(config_file_path)
            case _:
                raise ValueError(f"Unknown training type: {training_type}")
    if model == "diffusion":
        match training_type:
            case TrainingType.best:
                config = get_diffusion_best_model_config(character_name)
            case TrainingType.debug:
                config = get_diffusion_debug_config(character_name)
            case TrainingType.from_file:
                if config_file_path is None:
                    raise ValueError("config_file_path must be provided for from_file training type")
                config = GlobalTrainingConfig.from_yaml(config_file_path)
            case _:
                raise ValueError(f"Unknown training type: {training_type}")
    run_single_training(config)


if __name__ == "__main__":
    typer.run(main)
