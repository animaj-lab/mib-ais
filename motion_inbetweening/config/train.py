from pathlib import Path
from typing import Literal, Self

import pydantic
from pydantic import Field

from motion_inbetweening.app_services.online_preprocessing.mask_generator import MaskGeneratorConfig
from motion_inbetweening.config.base import BaseModel, MutableBaseModel
from motion_inbetweening.config.data import TrainDataModuleConfig
from motion_inbetweening.config.gaussian_diffusion import SpacedDiffusionConfig
from motion_inbetweening.config.inference_diffusion import InferenceDiffusionConfig
from motion_inbetweening.config.losses import LossesConfig
from motion_inbetweening.config.mdm import MotionDiffusionModelConfig
from motion_inbetweening.config.model import ModelConfig
from motion_inbetweening.config.optimization import OptimizationConfig
from motion_inbetweening.config.post_processing import PostProcessingConfig
from motion_inbetweening.domain.mask_applier import MaskApplierConfig
from shared.rig.config import ControllerConfig


class TrainConfig(BaseModel):
    max_epochs: int
    precision: str
    accumulate_grad_batches: int
    val_check_interval: float
    gradient_clip_val: float


class LoggingConfig(MutableBaseModel):
    """Logging configuration

    Attributes:
        project_name (str): Name of the project
        experiment_name (str): Name of the experiment
        save_dir_experiments (Path): Path to the directory where the experiments are saved
        save_model_to_mlflow (bool): Whether to save the best model weights to MLFlow
        tags (dict[str, str]): Tags to be added to the experiment
        hyperparameters (dict[str, str]): Hyperparameters to be added to the experiment. Fill these in if you want to be
            able to compare experiments in MLFlow and/or TensorBoard.
    """

    project_name: str
    experiment_name: str
    save_dir_experiments: Path
    save_model_to_mlflow: bool
    tags: dict[str, str]
    hyperparameters: dict[str, str]
    upload_to_mlflow: bool


class ModelCheckpointParams(BaseModel):
    every_n_epochs: int


class EarlyStoppingParams(BaseModel):
    monitor: str
    mode: str
    patience: int
    min_delta: float


class SceneWriterParams(BaseModel):
    video_resolution: int
    fps: int


class CallbacksConfig(BaseModel):
    model_checkpoint_params: ModelCheckpointParams
    use_early_stopping: bool
    model_checkpoint_params: ModelCheckpointParams
    early_stopping_params: EarlyStoppingParams
    scene_writer_params: SceneWriterParams
    write_controller_keyframes_curves: bool


class MaskingConfig(BaseModel):
    mask_generator: MaskGeneratorConfig = Field(discriminator="strategy_name")
    mask_applier: MaskApplierConfig
    test_mask_generator: MaskGeneratorConfig = Field(discriminator="strategy_name")
    predict_mask_generator: MaskGeneratorConfig = Field(discriminator="strategy_name")


class Seq2SeqTrainingModuleConfig(BaseModel):
    character_name: str
    module_name: Literal["seq2seq", "diffusion"]
    model: ModelConfig
    masking: MaskingConfig
    optimization: OptimizationConfig
    losses: LossesConfig
    do_controller_keyframes_prediction: bool
    post_processing: PostProcessingConfig

    @pydantic.model_validator(mode="after")
    def validate_controller_keyframes_prediction(self) -> Self:
        if self.do_controller_keyframes_prediction and self.losses.weights.controller_keyframes_prediction == 0:
            raise ValueError("If controller keyframes prediction is enabled, the weight for the loss has to be > 0")
        elif not self.do_controller_keyframes_prediction and self.losses.weights.controller_keyframes_prediction != 0:
            raise ValueError(
                "Weight for the controller keyframes prediction loss is > 0, "
                "but controller keyframes prediction is disabled"
            )
        return self


class SchedulerSamplerConfig(BaseModel):
    schedule_sampler_type: str


class DiffusionTrainingModuleConfig(BaseModel):
    character_name: str
    model: MotionDiffusionModelConfig
    diffusion: SpacedDiffusionConfig
    masking: MaskingConfig
    optimization: OptimizationConfig
    scheduler: SchedulerSamplerConfig
    inference: InferenceDiffusionConfig
    losses: LossesConfig
    do_controller_keyframes_prediction: bool
    post_processing: PostProcessingConfig


class GlobalTrainingConfig(BaseModel):
    data_module: TrainDataModuleConfig
    training: TrainConfig
    logging: LoggingConfig
    callbacks: CallbacksConfig
    controllers: ControllerConfig
    train_module: Seq2SeqTrainingModuleConfig | DiffusionTrainingModuleConfig
    seed: int

    @pydantic.model_validator(mode="after")
    def validate_controller_keyframes_prediction(self) -> Self:
        if (
            self.train_module.do_controller_keyframes_prediction
            and self.data_module.controller_keyframes_preprocessing_config is None
        ):
            raise ValueError(
                "If controller keyframes prediction is enabled, the controller keyframes "
                "preprocessing config has to be set."
            )
        if (
            not self.train_module.do_controller_keyframes_prediction
            and self.data_module.controller_keyframes_preprocessing_config
        ):
            raise ValueError(
                "Controller keyframes prediction is disabled, but controller keyframes preprocessing config is set."
            )
        return self
