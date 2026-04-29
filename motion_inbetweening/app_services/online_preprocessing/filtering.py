import logging

import pandas as pd

from motion_inbetweening.config.data import DatasetConfig
from motion_inbetweening.domain.data.data_sample import TrainDataSample
from motion_inbetweening.domain.data.split_type import SplitType

logger = logging.getLogger(__name__)


def filter_scenes_with_high_ranges(
    data: list[TrainDataSample], config: DatasetConfig, ranges_df: pd.DataFrame, split_type: SplitType
) -> list[TrainDataSample]:
    if config.relative_controller_transformations is None:
        return data
    list_filters: list[tuple[str, int]] = []
    for controller_name, controller_config in config.relative_controller_transformations.items():
        for transformation_config in controller_config:
            attribute_name = transformation_config.attribute
            full_controller_name = f"{controller_name}:{attribute_name}"
            threshold = (
                transformation_config.train_range_filter_threshold
                if split_type == SplitType.TRAIN
                else transformation_config.test_range_filter_threshold
            )
            if threshold is not None:
                list_filters.append((full_controller_name, threshold))
    if len(list_filters) == 0:
        return data
    logger.info(f"For split type {split_type}, len dataset before filter = {len(data)}")
    final_filter = pd.Series(data=False, index=ranges_df.index)
    for controller_name, threshold in list_filters:
        filter = ranges_df[controller_name] > threshold
        final_filter = final_filter | filter
    filtered_out_df = ranges_df[final_filter]
    filtered_out_df = filtered_out_df.set_index(["episode_id", "scene_id"])
    invalid_scenes = list(filtered_out_df.index.unique())
    filtered_data = [scene for scene in data if (scene.episode_id, scene.scene_id) not in invalid_scenes]
    logger.info(f"For split type {split_type}, len dataset after filter = {len(filtered_data)}")
    return filtered_data
