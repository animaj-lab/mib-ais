import torch
from torch.utils.data import Dataset

from motion_inbetweening.app_services.online_preprocessing.block_keyframes_schedule_modifier import (
    apply_block_schedule_augmentation,
)
from motion_inbetweening.app_services.online_preprocessing.relative_sequence import get_relative_sequence_vectors
from motion_inbetweening.app_services.online_preprocessing.scene_splitting import SceneSplitting, truncate_motion
from motion_inbetweening.config.block_keyframes_schedule_modifier import BlockScheduleAugmentation
from motion_inbetweening.domain.data.data_sample import TrainDataSample


class MotionInbetweeningDataset(Dataset):
    def __init__(
        self,
        list_data_samples: list[TrainDataSample],
        scene_splitting: SceneSplitting,
        indices_for_relative_controller_transformations: list[int],
        block_schedule_augmentation_config: BlockScheduleAugmentation | None,
    ):
        self.list_data_samples = list_data_samples
        self.scene_splitting = scene_splitting
        self.indices_for_relative_controller_transformations = indices_for_relative_controller_transformations
        self.block_schedule_augmentation_config = block_schedule_augmentation_config

    def __len__(self) -> int:
        return len(self.list_data_samples)

    def __getitem__(self, i) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        train_data_sample: TrainDataSample = self.list_data_samples[i]
        sequence_vectors = train_data_sample.vectors.clone()
        animation_keyframes = train_data_sample.animation_keyframes.clone()
        block_keyframes = train_data_sample.block_keyframes.clone()
        controller_keyframes = train_data_sample.controller_keyframes.clone()
        (
            truncated_sequence_vectors,
            truncated_animation_keyframes,
            truncated_block_keyframes,
            truncated_controller_keyframes,
        ) = truncate_motion(
            sequence_vectors=sequence_vectors,
            animation_keyframes=animation_keyframes,
            block_keyframes=block_keyframes,
            controller_keyframes=controller_keyframes,
            scene_splitting=self.scene_splitting,
        )
        relative_sequence_vectors = get_relative_sequence_vectors(
            truncated_sequence_vectors, self.indices_for_relative_controller_transformations
        )
        if self.block_schedule_augmentation_config is not None:
            truncated_block_keyframes = apply_block_schedule_augmentation(
                block_keyframes=truncated_block_keyframes.clone(),
                block_schedule_augmentation=self.block_schedule_augmentation_config,
            )
        return (
            relative_sequence_vectors,
            truncated_animation_keyframes,
            truncated_block_keyframes,
            truncated_controller_keyframes,
        )
