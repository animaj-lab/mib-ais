import random
from typing import Literal

import torch

from motion_inbetweening.config.base import BaseModel


class FullSceneSplitting(BaseModel):
    strategy_name: Literal["full_scene"]


class FixedLengthSplitting(BaseModel):
    strategy_name: Literal["fixed_length"]
    length: int


class RandomFixedLengthSplitting(BaseModel):
    strategy_name: Literal["random_fixed_length"]
    length: int


class RandomVariableLengthSplitting(BaseModel):
    strategy_name: Literal["random_variable_length"]
    min_length: int
    max_length: int


SceneSplitting = FullSceneSplitting | RandomFixedLengthSplitting | RandomVariableLengthSplitting | FixedLengthSplitting
PredictSceneSplitting = FullSceneSplitting | FixedLengthSplitting


def truncate_motion(
    sequence_vectors: torch.Tensor,
    animation_keyframes: torch.Tensor,
    block_keyframes: torch.Tensor,
    controller_keyframes: torch.Tensor,
    scene_splitting: SceneSplitting,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    match scene_splitting:
        case FixedLengthSplitting(length=split_length):
            (
                truncated_sequence_vectors,
                truncated_animation_keyframes,
                truncated_block_keyframes,
                truncated_controller_keyframes,
            ) = extract_subsequence(
                sequence_vectors=sequence_vectors,
                animation_keyframes=animation_keyframes,
                block_keyframes=block_keyframes,
                controller_keyframes=controller_keyframes,
                split_length=split_length,
                is_random=False,
            )
        case RandomFixedLengthSplitting(length=split_length):
            (
                truncated_sequence_vectors,
                truncated_animation_keyframes,
                truncated_block_keyframes,
                truncated_controller_keyframes,
            ) = extract_subsequence(
                sequence_vectors=sequence_vectors,
                animation_keyframes=animation_keyframes,
                block_keyframes=block_keyframes,
                controller_keyframes=controller_keyframes,
                split_length=split_length,
                is_random=True,
            )
        case RandomVariableLengthSplitting(min_length=min_length, max_length=max_length):
            max_seq_length = min(sequence_vectors.size(0), max_length)
            split_length = random.randint(min_length, max_seq_length) if max_seq_length > min_length else max_seq_length
            (
                truncated_sequence_vectors,
                truncated_animation_keyframes,
                truncated_block_keyframes,
                truncated_controller_keyframes,
            ) = extract_subsequence(
                sequence_vectors=sequence_vectors,
                animation_keyframes=animation_keyframes,
                block_keyframes=block_keyframes,
                controller_keyframes=controller_keyframes,
                split_length=split_length,
                is_random=True,
            )
        case FullSceneSplitting():
            truncated_sequence_vectors = sequence_vectors
            truncated_animation_keyframes = animation_keyframes
            truncated_block_keyframes = block_keyframes
            truncated_controller_keyframes = controller_keyframes
        case _:
            raise ValueError(f"Unknown {scene_splitting} type.")
    return (
        truncated_sequence_vectors,
        truncated_animation_keyframes,
        truncated_block_keyframes,
        truncated_controller_keyframes,
    )


def extract_subsequence(
    sequence_vectors: torch.Tensor,
    animation_keyframes: torch.Tensor,
    block_keyframes: torch.Tensor,
    controller_keyframes: torch.Tensor,
    split_length: int,
    is_random: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if sequence_vectors.size(0) <= split_length:
        return (sequence_vectors, animation_keyframes, block_keyframes, controller_keyframes)
    if is_random:
        start_index = torch.randint(0, sequence_vectors.size(0) - split_length, (1,)).item()
    else:
        start_index = 0
    end_index = start_index + split_length
    return (
        sequence_vectors[start_index:end_index],
        animation_keyframes[start_index:end_index],
        block_keyframes[start_index:end_index],
        controller_keyframes[start_index:end_index],
    )


def get_batch_final_sequence_length(lengths: list[int], scene_splitting: SceneSplitting) -> int:
    """
    This function is used to compute the final sequence length based on the scene splitting configuration.
    This indicates the length of the sequences after padding."""
    match scene_splitting:
        case RandomFixedLengthSplitting(length=split_length):
            max_sequence_length = split_length
        case FixedLengthSplitting(length=split_length):
            max_sequence_length = split_length
        case FullSceneSplitting():
            max_sequence_length = max(lengths)
        case RandomVariableLengthSplitting():
            max_sequence_length = max(lengths)
        case _:
            raise ValueError(f"Unknown {scene_splitting} type.")
    return max_sequence_length
