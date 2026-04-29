from pathlib import Path

import pandas as pd
from tqdm import tqdm

from motion_inbetweening.config.data import RelativeControllerTransformationConfig
from motion_inbetweening.config.post_processing import UnmaskedFramesPostProcessingStrategy
from motion_inbetweening.domain.controller_keyframes import RigControllersKeyframes, SceneRigControllersKeyframes
from motion_inbetweening.domain.data.data_sample import AnimationSceneData, get_all_frames_ids
from motion_inbetweening.domain.data.predict import PredictDataType
from motion_inbetweening.inference.post_processing import (
    get_scene_rig_controllers_keyframes,
    post_process_rig_prediction,
)
from shared.domain.entities.rig_controllers_values import RigControllersValues, SceneRigControllersValues
from shared.rig.trainable_controllers import TrainableController, transformation_to_attributes_names


def get_rig_csv_filepath(csv_rigs_dir: Path, scene: AnimationSceneData) -> Path:
    return csv_rigs_dir / scene.name / "controllers.csv"


def get_df_columns(trainable_controllers: list[TrainableController]) -> list[str]:
    df_columns = []
    for controller in trainable_controllers:
        for transformation in controller.transformations:
            transformation_names = transformation_to_attributes_names(transformation)
            for transformation_name in transformation_names:
                df_columns.append(f"{controller.name}/{transformation_name}")
    return df_columns


def get_df_from_scene_rig_controllers_values(
    scene_rig_controllers_values: SceneRigControllersValues, df_columns: list[str]
) -> pd.DataFrame:
    dataset_df = pd.DataFrame(columns=df_columns)
    total = len(scene_rig_controllers_values)
    for frame, rig in tqdm(
        scene_rig_controllers_values.items(), desc="Creating dataframe from scene rigs", total=total
    ):
        row = []
        for column in df_columns:
            controller_name, parameter = column.split("/")
            try:
                value = rig[controller_name][parameter]
            except Exception:
                raise Exception(f"Error when trying to access {controller_name}/{parameter} in frame {frame}") from None
            row.append(value)
        dataset_df.loc[frame] = row
    return dataset_df


def get_df_from_scene_rig_controllers_keyframes(
    scene_rig_controllers_keyframes: SceneRigControllersKeyframes, df_columns: list[str]
) -> pd.DataFrame:
    dataset_df = pd.DataFrame(columns=df_columns)
    total = len(scene_rig_controllers_keyframes)
    for frame, rig in tqdm(
        scene_rig_controllers_keyframes.items(), desc="Creating dataframe from scene rigs", total=total
    ):
        row = []
        for column in df_columns:
            controller_name, parameter = column.split("/")
            try:
                value = rig[controller_name][parameter]
            except Exception:
                raise Exception(f"Error when trying to access {controller_name}/{parameter} in frame {frame}") from None
            row.append(value)
        dataset_df.loc[frame] = row
    return dataset_df


def change_last_separator(string: str, old_separator: str, new_separator: str) -> str:
    return string[::-1].replace(old_separator, new_separator, 1)[::-1]


def process_df_predicted_rig_controllers_values(
    dataset_df: pd.DataFrame,
    scene: AnimationSceneData,
    predicted_rig_controllers_values: list[RigControllersValues],
    trainable_controllers: list[TrainableController],
    relative_controller_transformations: dict[str, list[RelativeControllerTransformationConfig]],
):
    scene_df = dataset_df[dataset_df["name"] == scene.name]
    output_rig_dict = post_process_rig_prediction(
        predicted_rig_controllers_values,
        scene=scene,
        relative_controller_transformations=relative_controller_transformations,
        do_controller_keyframes_reduction=False,
        predicted_controller_keyframes=None,
        threshold_controller_keyframes_reduction=None,
        unmasked_frames_post_processing_strategy=UnmaskedFramesPostProcessingStrategy.KEEP_PREDICTED,
        filter_plateau_tolerance_config=None,
    )
    df_columns = get_df_columns(trainable_controllers)
    predicted_df = get_df_from_scene_rig_controllers_values(output_rig_dict, df_columns)
    predicted_df.columns = [f"predicted_{col}" for col in predicted_df.columns]
    predicted_df["frame_id"] = get_all_frames_ids(scene)
    scene_df.columns = [change_last_separator(col, ":", "/") for col in scene_df.columns]
    ground_truth_df_columns = [f"{PredictDataType.ground_truth}_{col}" for col in df_columns]
    maya_baseline_df_columns = [f"{PredictDataType.maya_baseline}_{col}" for col in df_columns]
    columns_to_keep = ground_truth_df_columns + maya_baseline_df_columns + ["frame_id", "is_animation_keyframe"]
    columns_to_drop = [col for col in scene_df.columns if col not in columns_to_keep]
    scene_df = scene_df.drop(columns=columns_to_drop)
    dataset_df = pd.merge(scene_df, predicted_df, left_on="frame_id", right_on="frame_id")
    dataset_df["is_unmasked"] = False
    dataset_df.loc[dataset_df["frame_id"].isin(scene.unmasked_frames_id), "is_unmasked"] = True
    dataset_df.columns = [f"{col}_controller_values" if "/" in col else col for col in dataset_df.columns]
    return dataset_df


def process_df_predicted_rig_controllers_keyframes(
    dataset_df: pd.DataFrame,
    scene: AnimationSceneData,
    predicted_rig_controllers_keyframes: list[RigControllersKeyframes],
    trainable_controllers: list[TrainableController],
):
    scene_df = dataset_df[dataset_df["name"] == scene.name]
    scene_rig_controllers_keyframes = get_scene_rig_controllers_keyframes(predicted_rig_controllers_keyframes, scene)
    df_columns = get_df_columns(trainable_controllers)
    predicted_df = get_df_from_scene_rig_controllers_keyframes(scene_rig_controllers_keyframes, df_columns)
    predicted_df.columns = [f"predicted_{col}" for col in predicted_df.columns]
    predicted_df["frame_id"] = get_all_frames_ids(scene)
    scene_df.columns = [change_last_separator(col, ":", "/") for col in scene_df.columns]
    ground_truth_df_columns = [f"{PredictDataType.ground_truth}_{col}" for col in df_columns]
    columns_to_keep = ground_truth_df_columns + ["frame_id"]
    columns_to_drop = [col for col in scene_df.columns if col not in columns_to_keep]
    scene_df = scene_df.drop(columns=columns_to_drop)
    dataset_df = pd.merge(scene_df, predicted_df, left_on="frame_id", right_on="frame_id")
    dataset_df.columns = [f"{col}_controller_keyframes" if "/" in col else col for col in dataset_df.columns]
    return dataset_df


def write_controller_curves(
    dataset_df_controller_values: pd.DataFrame,
    dataset_df_controller_keyframes: pd.DataFrame | None,
    scene: AnimationSceneData,
    predicted_controller_values: list[RigControllersValues],
    predicted_controller_keyframes: list[RigControllersKeyframes] | None,
    trainable_controllers: list[TrainableController],
    relative_controller_transformations: dict[str, list[RelativeControllerTransformationConfig]],
    csv_rigs_dir: Path,
) -> None:
    dataset_df_controller_values = process_df_predicted_rig_controllers_values(
        dataset_df=dataset_df_controller_values,
        scene=scene,
        predicted_rig_controllers_values=predicted_controller_values,
        trainable_controllers=trainable_controllers,
        relative_controller_transformations=relative_controller_transformations,
    )
    if dataset_df_controller_keyframes is not None:
        if predicted_controller_keyframes is None:
            raise ValueError("Writing controller curves is enabled but no controller keyframes were predicted.")
        dataset_df_controller_keyframes = process_df_predicted_rig_controllers_keyframes(
            dataset_df=dataset_df_controller_keyframes,
            scene=scene,
            predicted_rig_controllers_keyframes=predicted_controller_keyframes,
            trainable_controllers=trainable_controllers,
        )
        dataset_df = pd.merge(dataset_df_controller_values, dataset_df_controller_keyframes, on="frame_id")
    else:
        dataset_df = dataset_df_controller_values
    output_filepath = get_rig_csv_filepath(csv_rigs_dir, scene)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    dataset_df.to_csv(output_filepath, index=False)
