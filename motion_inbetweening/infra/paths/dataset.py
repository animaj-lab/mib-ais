from motion_inbetweening.config.controller_keyframes_preprocessing import (
    ControllerKeyframeDetectionAlgorithmConfig,
    ControllerKeyframesPreprocessingConfig,
)
from motion_inbetweening.domain.data.hashing import GenericPath, get_hashed_dir_from_configs
from motion_inbetweening.domain.data.predict import PredictDataType
from motion_inbetweening.domain.data.split_type import SplitType
from shared.rig.trainable_controllers import TrainableController


def to_vectorized_block_keyframes_path(root_path: GenericPath, split_type: SplitType) -> GenericPath:
    return root_path / "vectorized_block_keyframes" / f"{split_type}.h5"


def to_ranges_path(root_path: GenericPath) -> GenericPath:
    return root_path / "ranges.csv"


def to_vectorized_controller_values_path(
    root_path: GenericPath, trainable_controllers: list[TrainableController], split_type: SplitType
) -> GenericPath:
    return (
        get_hashed_dir_from_configs(root_path / "vectorized_controller_values", [trainable_controllers])
        / f"{split_type}.h5"
    )


def to_vectorized_controller_keyframes_path(
    root_path: GenericPath,
    preprocessing_config: ControllerKeyframesPreprocessingConfig,
    trainable_controllers: list[TrainableController],
    split_type: SplitType,
) -> GenericPath:
    return (
        get_hashed_dir_from_configs(
            root_path / "vectorized_controller_keyframes", [preprocessing_config, trainable_controllers]
        )
        / f"{split_type}.h5"
    )


def to_predict_dir(root_path: GenericPath, predict_dataset_name: str) -> GenericPath:
    return root_path / "predict_datasets" / predict_dataset_name


def to_animation_files_dir(predict_dir: GenericPath) -> GenericPath:
    return predict_dir / "animation_files"


def to_predict_videos_dir(predict_dir: GenericPath, prefix: PredictDataType) -> GenericPath:
    return predict_dir / prefix / "videos"


def to_predict_video_path(predict_dir: GenericPath, prefix: PredictDataType, scene_name: str) -> GenericPath:
    return to_predict_videos_dir(predict_dir, prefix) / scene_name / f"{prefix}.mp4"


def to_controller_curves_dataframe_path(predict_dir: GenericPath) -> GenericPath:
    return predict_dir / "controller_curves.csv"


def to_predict_vectorized_controller_values_path(
    predict_dir: GenericPath, trainable_controllers: list[TrainableController]
) -> GenericPath:
    return (
        get_hashed_dir_from_configs(predict_dir / "vectorized_predict_data", [trainable_controllers])
        / f"{SplitType.PREDICT}.h5"
    )


def to_predict_controller_keyframes_path(
    predict_dir: GenericPath, algorithm_config: ControllerKeyframeDetectionAlgorithmConfig
) -> GenericPath:
    return (
        get_hashed_dir_from_configs(predict_dir / "controller_keyframes", [algorithm_config])
        / "dataset_controller_keyframes.csv"
    )
