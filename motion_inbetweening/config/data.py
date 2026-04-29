from pathlib import Path

import pydantic

from motion_inbetweening.app_services.online_preprocessing.scene_splitting import PredictSceneSplitting, SceneSplitting
from motion_inbetweening.config.base import BaseModel
from motion_inbetweening.config.block_keyframes_schedule_modifier import BlockScheduleAugmentation
from motion_inbetweening.config.controller_keyframes_preprocessing import ControllerKeyframesPreprocessingConfig
from shared.domain.entities.rig_controllers_values import AttributeName, ControllerName


class RelativeControllerTransformationConfig(BaseModel):
    attribute: AttributeName
    train_range_filter_threshold: int | None = None
    test_range_filter_threshold: int | None = None

    def __str__(self) -> str:
        return f"{self.attribute}-train={self.train_range_filter_threshold}-test={self.test_range_filter_threshold}"


class DatasetConfig(BaseModel):
    """Config for the pytorch dataset class.

    Attributes:
        dataset_directory (Path): Path to the directory containing the dataset.
        dataset_csv_filename (str): Name of the csv file containing the dataset.
        subset_size (int): If different from -1, only use a subset of the dataset of the given size.
        relative_controller_transformations (dict[str, list[str]]): Dictionary of controller names and their
            transformations to apply.
    """

    dataset_directory: Path
    subset_size: int = -1
    relative_controller_transformations: dict[ControllerName, list[RelativeControllerTransformationConfig]] = (
        pydantic.Field(default_factory=dict)
    )
    block_schedule_augmentation: BlockScheduleAugmentation | None


class TrainDataModuleConfig(BaseModel):
    """
    Configuration for the training data module.

    Attributes:
        batch_size (int): The number of samples per batch.
        validation_ratio (float): The ratio of the dataset to be used for validation.
        train_dataset (DatasetConfig): Configuration for the dataset.
        test_dataset (DatasetConfig | None): Configuration for the test dataset, if any.
        predict_dataset (DatasetConfig | None): Configuration for the prediction dataset, if any.
        num_workers (int): The number of worker processes to use for data loading.
        do_normalization (bool): Whether to perform normalization on the data.
        train_scene_splitting (SceneSplitting): Configuration for splitting the training scenes, based on strategy_name.
        test_scene_splitting (SceneSplitting): Configuration for splitting the test scenes, based on strategy_name.
        predict_scene_splitting (PredictSceneSplitting): Configuration for splitting the prediction scenes,
            based on strategy_name.
    """

    batch_size: int
    validation_ratio: float
    train_dataset: DatasetConfig
    test_dataset: DatasetConfig | None
    predict_dataset: DatasetConfig | None
    num_workers: int
    do_normalization: bool
    train_scene_splitting: SceneSplitting = pydantic.Field(discriminator="strategy_name")
    test_scene_splitting: SceneSplitting = pydantic.Field(discriminator="strategy_name")
    predict_scene_splitting: PredictSceneSplitting = pydantic.Field(discriminator="strategy_name")
    controller_keyframes_preprocessing_config: ControllerKeyframesPreprocessingConfig | None


class PredictDataModuleConfig(BaseModel):
    """
    Configuration class for the prediction data module.

    Attributes:
        batch_size (int): The size of the batches for prediction.
        predict_dataset (DatasetConfig): Configuration for the prediction dataset.
        num_workers (int): The number of worker threads to use for data loading.
        predict_scene_splitting (PredictSceneSplitting): Configuration for scene splitting strategy during prediction.
    """

    batch_size: int
    predict_dataset: DatasetConfig
    num_workers: int
    predict_scene_splitting: PredictSceneSplitting = pydantic.Field(discriminator="strategy_name")
