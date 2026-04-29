import logging
from pathlib import Path

import h5py
import pandas as pd
import torch
from tqdm import tqdm

from motion_inbetweening.app_services.online_preprocessing.data_sample import (
    VectorizedControllersKeyframesDict,
    VectorizedControllersValuesDict,
    get_list_data_samples,
)
from motion_inbetweening.app_services.online_preprocessing.filtering import filter_scenes_with_high_ranges
from motion_inbetweening.app_services.online_preprocessing.relative_sequence import (
    get_indices_for_relative_controller_transformations,
    get_relative_sequence_vectors,
)
from motion_inbetweening.app_services.online_preprocessing.scene_splitting import PredictSceneSplitting, SceneSplitting
from motion_inbetweening.app_services.training.dataset.dataset import MotionInbetweeningDataset
from motion_inbetweening.app_services.training.dataset.predict_dataset import MotionInBetweeningPredictDataset
from motion_inbetweening.config.controller_keyframes_preprocessing import ControllerKeyframesPreprocessingConfig
from motion_inbetweening.config.data import DatasetConfig
from motion_inbetweening.domain.data.data_sample import AnimationSceneData, PredictDataSample, TrainDataSample
from motion_inbetweening.domain.data.predict import PredictDataType
from motion_inbetweening.domain.data.split_type import SplitType
from motion_inbetweening.infra.dataframe import SingleSceneDataFrame
from motion_inbetweening.infra.paths.dataset import (
    to_animation_files_dir,
    to_controller_curves_dataframe_path,
    to_predict_vectorized_controller_values_path,
    to_ranges_path,
    to_vectorized_block_keyframes_path,
    to_vectorized_controller_keyframes_path,
    to_vectorized_controller_values_path,
)
from shared.domain.entities.rig_controllers_values import SceneRigControllersValues
from shared.infrastructure.data_access import load_dataframe
from shared.rig.trainable_controllers import TrainableController


def load_vectorized_controller_values(path: Path) -> VectorizedControllersValuesDict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with h5py.File(path, "r") as f:
        data = {}
        n_samples = len(f)
        for i in tqdm(range(n_samples), desc="Loading vectorized controller values"):
            scene_group = f[str(i)]
            vectors = torch.from_numpy(scene_group["vectors"][()]).to(torch.float32)
            episode_id = scene_group["episode_id"][()]
            scene_id = scene_group["scene_id"][()]
            data[episode_id, scene_id] = {
                "vectors": vectors,
                "animation_keyframes": torch.from_numpy(scene_group["animation_keyframes"][()]).to(torch.bool),
            }
    return data


def load_vectorized_controller_keyframes(path: Path) -> VectorizedControllersKeyframesDict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with h5py.File(path, "r") as f:
        data = {}
        n_samples = len(f)
        for i in tqdm(range(n_samples), desc="Loading vectorized controller keyframes"):
            scene_group = f[str(i)]
            controller_keyframes_vectors = torch.from_numpy(scene_group["controller_keyframes_vectors"][()]).to(
                torch.float32
            )
            episode_id = scene_group["episode_id"][()]
            scene_id = scene_group["scene_id"][()]
            data[episode_id, scene_id] = controller_keyframes_vectors
    return data


def load_vectorized_block_keyframes(path: Path) -> VectorizedControllersKeyframesDict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with h5py.File(path, "r") as f:
        data = {}
        n_samples = len(f)
        for i in tqdm(range(n_samples), desc="Loading vectorized block keyframes"):
            scene_group = f[str(i)]
            block_keyframes_vectors = torch.from_numpy(scene_group["block_keyframes_vectors"][()]).to(torch.bool)
            episode_id = scene_group["episode_id"][()]
            scene_id = scene_group["scene_id"][()]
            data[episode_id, scene_id] = block_keyframes_vectors
    return data


def load_list_train_data_samples(
    dataset_dir: Path,
    trainable_controllers: list[TrainableController],
    controller_keyframes_preprocessing_config: ControllerKeyframesPreprocessingConfig | None,
    split_type: SplitType,
) -> list[TrainDataSample]:
    """Loads the data from the disk and returns a list of TrainDataSample objects.

    The loaded data includes:
        - Vectorized Controller values
        - Block keyframes
        - Optional: Vectorized Controller keyframes
    """
    vectorized_controller_values_path = to_vectorized_controller_values_path(
        dataset_dir, trainable_controllers, split_type
    )
    logging.info(f"Loading controller values from {vectorized_controller_values_path}")
    controllers_values_data = load_vectorized_controller_values(vectorized_controller_values_path)
    vectorized_block_keyframes_path = to_vectorized_block_keyframes_path(dataset_dir, split_type)
    block_keyframes_data = load_vectorized_block_keyframes(vectorized_block_keyframes_path)
    if controller_keyframes_preprocessing_config is not None:
        vectorized_controller_keyframes_path = to_vectorized_controller_keyframes_path(
            dataset_dir, controller_keyframes_preprocessing_config, trainable_controllers, split_type
        )
        logging.info(f"Loading controller keyframes from {vectorized_controller_keyframes_path}")
        controller_keyframes_data = load_vectorized_controller_keyframes(vectorized_controller_keyframes_path)
    else:
        controller_keyframes_data = None
    return get_list_data_samples(controllers_values_data, block_keyframes_data, controller_keyframes_data)


def load_data_and_apply_one_time_online_preprocessing(
    config: DatasetConfig,
    trainable_controllers: list[TrainableController],
    controller_keyframes_preprocessing_config: ControllerKeyframesPreprocessingConfig | None,
    split_type: SplitType,
    indices_for_relative_controller_transformations: list[int],
) -> list[TrainDataSample]:
    """Loads the data from the disk and applies online filtering and one-time online preprocessing."""
    data = load_list_train_data_samples(
        config.dataset_directory, trainable_controllers, controller_keyframes_preprocessing_config, split_type
    )
    logging.info(f"Filtering invalid scenes for {split_type}")
    ranges_df = load_dataframe(to_ranges_path(config.dataset_directory), "all")
    filtered_data = filter_scenes_with_high_ranges(data, config, ranges_df, split_type)
    logging.info(f"Number of scenes after filtering: {len(data)}")
    if config.relative_controller_transformations is not None:
        for scene_dict in filtered_data:
            scene_dict.vectors = get_relative_sequence_vectors(
                scene_dict.vectors, indices_for_relative_controller_transformations
            )
    return filtered_data


def load_and_instantiate_dataset(
    dataset_config: DatasetConfig,
    scene_splitting: SceneSplitting,
    trainable_controllers: list[TrainableController],
    split_type: SplitType,
    controller_keyframes_preprocessing_config: ControllerKeyframesPreprocessingConfig | None,
) -> MotionInbetweeningDataset:
    """Top-level function of the module (for train/test data).

    Loads the data from the disk, applies online filtering and one-time online preprocessing, and returns a
    MotionInbetweeningDataset object.
    """
    indices_for_relative_controller_transformations = get_indices_for_relative_controller_transformations(
        trainable_controllers, dataset_config.relative_controller_transformations
    )
    list_data_samples = load_data_and_apply_one_time_online_preprocessing(
        dataset_config,
        trainable_controllers,
        controller_keyframes_preprocessing_config,
        split_type,
        indices_for_relative_controller_transformations,
    )
    return MotionInbetweeningDataset(
        list_data_samples=list_data_samples,
        scene_splitting=scene_splitting,
        indices_for_relative_controller_transformations=indices_for_relative_controller_transformations,
        block_schedule_augmentation_config=dataset_config.block_schedule_augmentation,
    )


def controller_curve_df_to_rig_controller_values_df(df: pd.DataFrame) -> pd.DataFrame:
    """Go from controller curve dataframe to rig controller values dataframe."""
    columns_to_keep = [column for column in df.columns if PredictDataType.ground_truth in column] + ["frame_id", "name"]
    df = df[columns_to_keep]
    df.columns = [column.replace(f"{PredictDataType.ground_truth}_", "") for column in df.columns]
    return df


def load_ground_truth_rig_controllers_values(predict_dir: Path, scene_name: str) -> SceneRigControllersValues:
    df = load_dataframe(to_controller_curves_dataframe_path(predict_dir), "all")
    scene_df = df[df["name"] == scene_name]
    rig_controllers_values_df = controller_curve_df_to_rig_controller_values_df(scene_df).drop(columns=["name"])
    single_scene_df = SingleSceneDataFrame(rig_controllers_values_df)
    return single_scene_df.to_scene_rig_controllers_values()


def load_list_predict_data_samples(
    predict_dir: Path, trainable_controllers: list[TrainableController]
) -> list[PredictDataSample]:
    vectorized_controller_values_path = to_predict_vectorized_controller_values_path(predict_dir, trainable_controllers)
    if not vectorized_controller_values_path.exists():
        raise FileNotFoundError(f"File not found: {vectorized_controller_values_path}")
    with h5py.File(vectorized_controller_values_path, "r") as f:
        predict_data_samples = []
        for i in range(len(f)):
            scene_group = f[str(i)]
            pose_vectors = torch.from_numpy(scene_group["vectors"][()]).to(torch.float32)
            scene_name = scene_group["name"][()].decode("utf-8")
            scene_rig_controllers_values = load_ground_truth_rig_controllers_values(predict_dir, scene_name)
            unmasked_frames_id = scene_group["unmasked_frames"][()].tolist()
            ma_file_name = f"{scene_group['ma_file_name'][()].decode('utf-8')}"
            ma_file_path = to_animation_files_dir(predict_dir) / ma_file_name
            scene_data = AnimationSceneData(
                name=scene_name,
                ma_file_path=ma_file_path,
                frame_range=(scene_group["min_frame"][()], scene_group["max_frame"][()]),
                unmasked_frames_id=unmasked_frames_id,
                input_rig=scene_rig_controllers_values,
            )
            predict_data_sample = PredictDataSample(pose_vectors=pose_vectors, scene_data=scene_data)
            predict_data_samples.append(predict_data_sample)
    return predict_data_samples


def load_and_instantiate_predict_dataset(
    predict_dataset_config: DatasetConfig,
    scene_splitting: PredictSceneSplitting,
    trainable_controllers: list[TrainableController],
) -> MotionInBetweeningPredictDataset:
    """Top-level function of the module (for predict data).

    Loads the data from the disk and returns a MotionInBetweeningPredictDataset object.
    """
    indices_for_relative_controller_transformations = get_indices_for_relative_controller_transformations(
        trainable_controllers, predict_dataset_config.relative_controller_transformations
    )
    list_data_samples = load_list_predict_data_samples(predict_dataset_config.dataset_directory, trainable_controllers)
    if scene_splitting.strategy_name == "fixed_length":
        logging.info(f"Number of scenes before filtering for fixed length: {len(list_data_samples)}")
        max_length = scene_splitting.length
        filtered_data_samples = []
        for data_sample in list_data_samples:
            if data_sample.pose_vectors.size(0) <= max_length:
                filtered_data_samples.append(data_sample)
            else:
                logging.warning(
                    f"Scene {data_sample.scene_data.name} with length {data_sample.pose_vectors.size(0)} "
                    f"is longer than the fixed length {max_length} and will be skipped."
                )
        list_data_samples = filtered_data_samples
        logging.info(f"Number of scenes after filtering for fixed length: {len(list_data_samples)}")
    return MotionInBetweeningPredictDataset(
        list_predict_data_samples=list_data_samples,
        scene_splitting=scene_splitting,
        indices_for_relative_controller_transformations=indices_for_relative_controller_transformations,
        block_schedule_augmentation_config=predict_dataset_config.block_schedule_augmentation,
    )
