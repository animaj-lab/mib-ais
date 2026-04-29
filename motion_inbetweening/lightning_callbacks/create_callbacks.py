from pathlib import Path

from pytorch_lightning.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint

from motion_inbetweening.config.controller_keyframes_preprocessing import ControllerKeyframesPreprocessingConfig
from motion_inbetweening.config.data import DatasetConfig
from motion_inbetweening.config.train import CallbacksConfig
from motion_inbetweening.lightning_callbacks.rig_csv_writer import RigCSVWriter
from shared.rig.trainable_controllers import TrainableController


def create_callbacks(
    config: CallbacksConfig,
    experiment_directory: Path,
    trainable_controllers: list[TrainableController],
    predict_dataset_config: DatasetConfig | None,
    controller_keyframes_preprocessing_config: ControllerKeyframesPreprocessingConfig | None,
) -> list[Callback]:
    """Create and return a list of callbacks based on the config.

    Args:
        config (CallbacksConfig): Callbacks configuration
        experiment_directory (Path): Experiment directory
        predict_dataset_config (DatasetConfig | None): Configuration of the dataset used for prediction. If None, no
            scene writing will be done.

    Returns:
        list[Callback]: List of callbacks
    """
    checkpoints_dir = experiment_directory / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="model",
        monitor="val/loss",
        mode="min",
        every_n_epochs=config.model_checkpoint_params.every_n_epochs,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]
    if predict_dataset_config is not None:
        csv_writer = RigCSVWriter(
            predict_dataset_config.dataset_directory,
            experiment_directory,
            trainable_controllers,
            predict_dataset_config.relative_controller_transformations,
            controller_keyframes_preprocessing_config,
            write_interval="epoch",
        )
        callbacks.append(csv_writer)
    if config.use_early_stopping:
        early_stop_callback = EarlyStopping(
            monitor=config.early_stopping_params.monitor,
            min_delta=config.early_stopping_params.min_delta,
            patience=config.early_stopping_params.patience,
            mode=config.early_stopping_params.mode,
        )
        callbacks.append(early_stop_callback)
    return callbacks
