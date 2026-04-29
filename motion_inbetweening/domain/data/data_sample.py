from dataclasses import dataclass
from pathlib import Path

import torch

from shared.domain.entities.animation import EpisodeID, FrameID, FrameIndex, SceneID
from shared.domain.entities.rig_controllers_values import SceneRigControllersValues


@dataclass
class TrainDataSample:
    """
    A class to represent a training data sample for motion inbetweening.

    Attributes:
        episode_id (EpisodeID): The identifier for the episode.
        scene_id (SceneID): The identifier for the scene.
        vectors (torch.Tensor): The tensor containing motion vectors.
        animation_keyframes (torch.Tensor): The tensor containing animation keyframes.
        block_keyframes (torch.Tensor): The tensor containing block keyframes.
        controllers_animation_keyframes (torch.Tensor): The tensor containing animation keyframes
            for each controller's transformation or attribute depending on the aggregation strategy.
    """

    episode_id: EpisodeID
    scene_id: SceneID
    vectors: torch.Tensor
    animation_keyframes: torch.Tensor
    block_keyframes: torch.Tensor
    controller_keyframes: torch.Tensor


@dataclass
class AnimationSceneData:
    """
    A class to represent the data of an animation scene.

    Attributes:
        name (str): The name of the animation scene.
        ma_file_path (Path): The file path to the Maya animation file.
        input_rig (SceneRigDict): The dictionary containing the input rig data for the scene.
        frame_range (tuple[FrameID, FrameID]): A tuple representing the start and end frame IDs of the animation.
        unmasked_frames_id (list[FrameID]): A list of frame IDs that are not masked in the animation.
    """

    name: str
    ma_file_path: Path
    input_rig: SceneRigControllersValues
    frame_range: tuple[FrameID, FrameID]
    unmasked_frames_id: list[FrameID]


@dataclass
class PredictDataSample:
    """
    A class to represent a sample of prediction data for motion inbetweening predict dataset.

    Attributes:
        pose_vectors (torch.Tensor): A tensor containing the pose vectors for the sample.
        animation_keyframes (torch.Tensor): A tensor containing the animation keyframes for the sample.
        scene_data (AnimationSceneData): An instance of AnimationSceneData containing the scene data for the sample.
    """

    pose_vectors: torch.Tensor
    scene_data: AnimationSceneData


def get_start_frame_id(scene: AnimationSceneData) -> FrameID:
    return scene.frame_range[0]


def get_end_frame_id(scene: AnimationSceneData) -> FrameID:
    return scene.frame_range[1]


def get_unmasked_frames_ids(scene: AnimationSceneData) -> list[FrameID]:
    return scene.unmasked_frames_id


def get_unmasked_frames_indices(scene: AnimationSceneData) -> list[FrameIndex]:
    start_frame = get_start_frame_id(scene)
    return [frame_id - start_frame for frame_id in scene.unmasked_frames_id]


def get_all_frames_ids(scene: AnimationSceneData) -> list[FrameID]:
    start_frame = get_start_frame_id(scene)
    end_frame = get_end_frame_id(scene)
    return list(range(start_frame, end_frame + 1))
