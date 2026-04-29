import torch
from torch.utils.data import Dataset

from motion_inbetweening.app_services.online_preprocessing.block_keyframes_schedule_modifier import (
    apply_block_schedule_augmentation_frames_id,
)
from motion_inbetweening.app_services.online_preprocessing.predict import process_predict_data_sample
from motion_inbetweening.app_services.online_preprocessing.scene_splitting import PredictSceneSplitting
from motion_inbetweening.config.block_keyframes_schedule_modifier import BlockScheduleAugmentation
from motion_inbetweening.domain.data.data_sample import AnimationSceneData, PredictDataSample


class MotionInBetweeningPredictDataset(Dataset):
    def __init__(
        self,
        list_predict_data_samples: list[PredictDataSample],
        scene_splitting: PredictSceneSplitting,
        indices_for_relative_controller_transformations: list[int],
        block_schedule_augmentation_config: BlockScheduleAugmentation | None,
    ):
        self.list_predict_data_samples = list_predict_data_samples
        self.scene_splitting = scene_splitting
        self.indices_for_relative_controller_transformations = indices_for_relative_controller_transformations
        self.block_schedule_augmentation = block_schedule_augmentation_config

    def __len__(self) -> int:
        return len(self.list_predict_data_samples)

    def __getitem__(self, index) -> tuple[torch.Tensor, AnimationSceneData]:
        predict_data_sample: PredictDataSample = self.list_predict_data_samples[index]
        relative_sequence_vectors, scene_data = process_predict_data_sample(
            predict_data_sample, self.scene_splitting, self.indices_for_relative_controller_transformations
        )
        if self.block_schedule_augmentation is not None:
            scene_data.unmasked_frames_id = apply_block_schedule_augmentation_frames_id(
                scene_data.unmasked_frames_id, self.block_schedule_augmentation, scene_data.frame_range
            )
        return (relative_sequence_vectors, scene_data)
