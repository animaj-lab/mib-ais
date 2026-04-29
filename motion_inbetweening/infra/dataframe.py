from dataclasses import dataclass
from typing import Any, Self

import pandas as pd

from motion_inbetweening.domain.controller_keyframes import SceneRigControllersKeyframes
from motion_inbetweening.domain.rig_controllers_values import (
    scene_rig_controllers_values_to_controller_attribute_tuples,
)
from shared.domain.entities.animation import EpisodeID, FrameID, FrameID_type, SceneID
from shared.domain.entities.rig_controllers_values import (
    AttributeName,
    ControllerName,
    FullSceneRigControllersValues,
    SceneRigControllersValues,
)


@dataclass
class SingleSceneDataFrame:
    """
    A SingleSceneDataFrame encapsulates a dataframe that contains the keyframes for a single scene. We don't expect to
    have "episode_id" and "scene_id" columns in this dataframe.
    """

    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        self.validate(df)
        self.df = df

    @classmethod
    def new_empty(cls) -> Self:
        """Create a new empty SingleSceneDataFrame."""
        return cls(pd.DataFrame(columns=["frame_id"]))

    @classmethod
    def validate(cls, df: pd.DataFrame) -> None:
        """Validate the formatting of the columns of a SingleSceneDataFrame.

        This is a class method to validate dataframes without instantiating the class.

        Validation conditions are:
        - No "episode_id" or "scene_id" columns should be present.
        - "frame_id" column should be present.
        - All columns expect for "frame_id" should be in the format "ControllerName:AttributeName".
        """
        list_errors = []
        if "episode_id" in df.columns or "scene_id" in df.columns:
            list_errors.append("A SingleSceneDataFrame should not contain 'episode_id' or 'scene_id' columns.")
        if "frame_id" not in df.columns:
            list_errors.append("A SingleSceneDataFrame should contain a 'frame_id' column.")
        df["frame_id"] = df["frame_id"].astype(FrameID_type)
        wrong_format_columns = [col for col in df.columns if col not in get_non_controller_columns() and ":" not in col]
        if wrong_format_columns:
            list_errors.append(
                f"Columns {wrong_format_columns} should be in the format 'ControllerName:AttributeName'."
            )
        if list_errors:
            raise ValueError("Errors validating the SingleSceneDataFrame:\n" + "\n".join(list_errors))

    def get_columns(self) -> list[str]:
        return self.df.columns.tolist()

    def iterrows(self):
        return self.df.iterrows()

    def to_dataset_dataframe(self, episode_id: EpisodeID, scene_id: SceneID) -> "DatasetDataFrame":
        """Convert a single scene dataframe to a dataset dataframe."""
        dataset_df = self.df.copy()
        dataset_df["episode_id"] = episode_id
        dataset_df["scene_id"] = scene_id
        return DatasetDataFrame(dataset_df)

    def to_controller_attribute_tuples(self) -> list[tuple[ControllerName, AttributeName]]:
        return dataframe_to_controller_attribute_tuples(self)

    def get_dataframe(self) -> pd.DataFrame:
        return self.df

    @staticmethod
    def from_scene_rig_controllers_keyframes(scene_keyframes: SceneRigControllersKeyframes) -> "SingleSceneDataFrame":
        return scene_rig_controllers_keyframes_to_single_scene_dataframe(scene_keyframes)

    def to_scene_rig_controllers_keyframes(self) -> SceneRigControllersKeyframes:
        return single_scene_dataframe_to_scene_rig_controllers_keyframes(self)

    @staticmethod
    def from_scene_rig_controllers_values(
        scene_rig_controllers_values: SceneRigControllersValues,
    ) -> "SingleSceneDataFrame":
        return scene_rig_controllers_values_to_single_scene_dataframe(scene_rig_controllers_values)

    def to_full_scene_rig_controllers_values(self) -> FullSceneRigControllersValues:
        return single_scene_dataframe_to_full_scene_rig_controllers_values(self)

    def to_scene_rig_controllers_values(self) -> SceneRigControllersValues:
        return single_scene_dataframe_to_scene_rig_controllers_values(self)


@dataclass
class DatasetDataFrame:
    """
    A DatasetDataFrame encapsulates a dataframe that contains the keyframes for multiple scenes. We expect to have
    "episode_id" and "scene_id" columns in this dataframe.
    """

    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame) -> None:
        self.validate(df)
        self.df = df

    @classmethod
    def validate(cls, df: pd.DataFrame) -> None:
        """Validate the formatting of the columns of a DatasetDataFrame.

        This is a class method to validate dataframes without instantiating the class.

        Validation conditions are:
        - "episode_id" and "scene_id" columns should be present.
        - "frame_id" column should be present.
        - All columns expect for "frame_id", "episode_id" and "scene_id" should be in the format
            "ControllerName:AttributeName".
        """
        list_errors = []
        if "episode_id" not in df.columns or "scene_id" not in df.columns:
            list_errors.append("A DatasetDataFrame should contain 'episode_id' and 'scene_id' columns.")
        if "frame_id" not in df.columns:
            list_errors.append("A DatasetDataFrame should contain a 'frame_id' column.")
        df["frame_id"] = df["frame_id"].astype(FrameID_type)
        wrong_format_columns = [col for col in df.columns if col not in get_non_controller_columns() and ":" not in col]
        if wrong_format_columns:
            list_errors.append(
                f"Columns {wrong_format_columns} should be in the format 'ControllerName:AttributeName'."
            )
        if list_errors:
            raise ValueError("Errors validating the DatasetDataFrame:\n" + "\n".join(list_errors))

    def get_columns(self) -> list[str]:
        return self.df.columns.tolist()

    def iterrows(self):
        return self.df.iterrows()

    def to_single_scene_dataframe(self, episode_id: EpisodeID, scene_id: SceneID) -> SingleSceneDataFrame:
        sub_df = self.df[(self.df["episode_id"] == episode_id) & (self.df["scene_id"] == scene_id)]
        sub_df = sub_df.drop(columns=["episode_id", "scene_id"])
        return SingleSceneDataFrame(sub_df)

    def to_controller_attribute_tuples(self) -> list[tuple[ControllerName, AttributeName]]:
        return dataframe_to_controller_attribute_tuples(self)

    def get_dataframe(self) -> pd.DataFrame:
        return self.df

    def get_episode_id_scene_id_tuples(self) -> list[tuple[EpisodeID, SceneID]]:
        """Get a list of unique episode_id and scene_id tuples from the dataframe."""
        episode_scene_ids = self.df[["episode_id", "scene_id"]].drop_duplicates()
        episode_scene_ids = episode_scene_ids.to_records(index=False)
        return [(EpisodeID(episode_id), SceneID(scene_id)) for episode_id, scene_id in episode_scene_ids]

    @staticmethod
    def from_scene_rig_controllers_keyframes(
        scene_keyframes: SceneRigControllersKeyframes, episode_id: EpisodeID, scene_id: SceneID
    ) -> "DatasetDataFrame":
        return SingleSceneDataFrame.from_scene_rig_controllers_keyframes(scene_keyframes).to_dataset_dataframe(
            episode_id, scene_id
        )

    def to_scene_rig_controllers_keyframes(
        self, episode_id: EpisodeID, scene_id: SceneID
    ) -> SceneRigControllersKeyframes:
        return self.to_single_scene_dataframe(episode_id, scene_id).to_scene_rig_controllers_keyframes()

    @staticmethod
    def from_scene_rig_controllers_values(
        scene_rig_controllers_values: SceneRigControllersValues, episode_id: EpisodeID, scene_id: SceneID
    ) -> "DatasetDataFrame":
        return SingleSceneDataFrame.from_scene_rig_controllers_values(
            scene_rig_controllers_values
        ).to_dataset_dataframe(episode_id, scene_id)

    def to_full_scene_rig_controllers_values(
        self, episode_id: EpisodeID, scene_id: SceneID
    ) -> FullSceneRigControllersValues:
        return self.to_single_scene_dataframe(episode_id, scene_id).to_full_scene_rig_controllers_values()


def get_non_controller_columns() -> list[str]:
    return ["frame_id", "episode_id", "scene_id", "is_animation_keyframe"]


def dataframe_to_controller_attribute_tuples(
    df: SingleSceneDataFrame | DatasetDataFrame,
) -> list[tuple[ControllerName, AttributeName]]:
    """
    Given an input dataframe, returns a list of (ControllerName, AttributeName) tuples

    Current heuristic (maybe not the best):
        - Filter columns that contain ":"
        - Split the column name by ":"
        - Return the list of tuples

    """
    filtered_columns = [col for col in df.get_columns() if ":" in col]
    list_controller_columns = []
    for col in filtered_columns:
        if col.count(":") > 1:
            raise ValueError(f"Column {col} has more than one ':' character. Please check the column name.")
        controller, attribute = col.split(":")
        list_controller_columns.append((ControllerName(controller), AttributeName(attribute)))
    return list_controller_columns


def single_scene_dataframe_to_scene_rig_controllers_keyframes(df: SingleSceneDataFrame) -> SceneRigControllersKeyframes:
    controller_attribute_tuples = df.to_controller_attribute_tuples()
    scene_keyframes = {}
    for _, row in df.iterrows():
        frame_id = FrameID(row["frame_id"])
        scene_keyframes[frame_id] = {}
        for controller, attribute in controller_attribute_tuples:
            if controller not in scene_keyframes[frame_id]:
                scene_keyframes[frame_id][controller] = {}
            scene_keyframes[frame_id][controller][attribute] = row[f"{controller}:{attribute}"]
    return SceneRigControllersKeyframes.from_any(scene_keyframes)


def scene_rig_controllers_keyframes_to_single_scene_dataframe(
    scene_keyframes: SceneRigControllersKeyframes,
) -> SingleSceneDataFrame:
    """Convert a single scene keyframes object to a dataframe."""
    controller_attribute_tuples = scene_keyframes.to_controller_attribute_tuples()
    controller_columns = [f"{controller}:{attribute}" for controller, attribute in controller_attribute_tuples]
    records = []
    for frame_id, row in scene_keyframes.items():
        records.append(
            [frame_id] + [row[controller][attribute] for controller, attribute in controller_attribute_tuples]
        )
    df = pd.DataFrame(records, columns=["frame_id"] + controller_columns)
    df = SingleSceneDataFrame(df)
    return df


def single_scene_dataframe_to_dict(df: SingleSceneDataFrame) -> dict:
    controller_attribute_tuples = df.to_controller_attribute_tuples()
    scene_rig_controllers_values = {}
    for _, row in df.iterrows():
        frame_id = FrameID(row["frame_id"])
        scene_rig_controllers_values[frame_id] = {}
        for controller, attribute in controller_attribute_tuples:
            value = row[f"{controller}:{attribute}"]
            if pd.isna(value):
                continue
            if controller not in scene_rig_controllers_values[frame_id]:
                scene_rig_controllers_values[frame_id][controller] = {}
            scene_rig_controllers_values[frame_id][controller][attribute] = value
    return scene_rig_controllers_values


def single_scene_dataframe_to_full_scene_rig_controllers_values(
    df: SingleSceneDataFrame,
) -> FullSceneRigControllersValues:
    return FullSceneRigControllersValues.from_any(single_scene_dataframe_to_dict(df))


def single_scene_dataframe_to_scene_rig_controllers_values(df: SingleSceneDataFrame) -> SceneRigControllersValues:
    return SceneRigControllersValues.from_any(single_scene_dataframe_to_dict(df))


def scene_rig_controllers_values_to_single_scene_dataframe(scene: SceneRigControllersValues) -> SingleSceneDataFrame:
    """Convert a SceneRigControllersValues object to a SingleSceneDataFrame.

    Note: since columns can hold NaN values (which is how we allow for non-existent values in SceneRigControllersValues)
    int columns will be cast to float columns. This is a limitation of pandas, as NaN cannot be represented in integer
    columns. If you really need int values for a given controller attribute when is SceneRigControllersValues format,
    you can convert that specific controller attribute to int after the conversion to SceneRigControllersValues.
    """
    controller_attribute_tuples = scene_rig_controllers_values_to_controller_attribute_tuples(scene)
    controller_columns = [f"{controller}:{attribute}" for controller, attribute in controller_attribute_tuples]
    records = []
    for frame_id, row in scene.items():
        record: list[Any] = [frame_id]
        for controller, attribute in controller_attribute_tuples:
            if controller not in row or attribute not in row[controller]:
                record.append(None)
            else:
                record.append(row[controller][attribute])
        records.append(record)
    df = pd.DataFrame(records, columns=["frame_id"] + controller_columns)
    df = SingleSceneDataFrame(df)
    return df
