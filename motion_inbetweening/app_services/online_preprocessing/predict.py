import torch

from motion_inbetweening.app_services.online_preprocessing.relative_sequence import get_relative_sequence_vectors
from motion_inbetweening.app_services.online_preprocessing.scene_splitting import (
    FixedLengthSplitting,
    PredictSceneSplitting,
)
from motion_inbetweening.domain.data.data_sample import AnimationSceneData, PredictDataSample


def process_predict_data_sample(
    predict_data_sample: PredictDataSample,
    scene_splitting: PredictSceneSplitting,
    indices_for_relative_controller_transformations: list[int],
) -> tuple[torch.Tensor, AnimationSceneData]:
    """Online preprocessing of the predict data sample.

    Applied steps:
        - Scene splitting
        - Making the sequence relative to the first frame

    Args:
        predict_data_sample (PredictDataSample): Holds the sample data
        scene_splitting (PredictSceneSplitting): Scene splitting strategy
        indices_for_relative_controller_transformations (list[int]): Indices for which to compute the relative
            transformations

    Returns:
        tuple[torch.Tensor, AnimationSceneData]: Processed data sample
            (relative sequence, scene data)
    """
    sequence_vectors = predict_data_sample.pose_vectors.clone()
    if isinstance(scene_splitting, FixedLengthSplitting):
        if sequence_vectors.size(0) > scene_splitting.length:
            raise ValueError(
                f"The sequence length {sequence_vectors.size(0)} is greater than the split length "
                f"{scene_splitting.length} of the FixedLengthSplitting strategy. Please provide "
                f"predict scenes with a length less than or equal to the split length. "
                f"Or use the FullSceneSplitting strategy."
            )
    relative_sequence_vectors = get_relative_sequence_vectors(
        sequence_vectors, indices_for_relative_controller_transformations
    )
    scene_data = predict_data_sample.scene_data
    return (relative_sequence_vectors, scene_data)
