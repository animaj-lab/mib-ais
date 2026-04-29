import torch

from motion_inbetweening.app_services.online_preprocessing.padding import get_padded_batch_and_masks
from motion_inbetweening.app_services.online_preprocessing.scene_splitting import (
    PredictSceneSplitting,
    SceneSplitting,
    get_batch_final_sequence_length,
)
from motion_inbetweening.domain.data.data_sample import AnimationSceneData, get_unmasked_frames_indices
from shared.rig.normalizer import PoseNormalizer


def predict_collate_fn(
    batch: list[tuple[torch.Tensor, AnimationSceneData]], predict_scene_splitting: PredictSceneSplitting
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]], list[AnimationSceneData]]:
    """Given a batch of sequences, collates them a single tensor per data input.

    The sequences are of variable length, so they are padded to the same size.

    Args:
        batch (list[tuple[torch.Tensor, torch.Tensor, AnimationSceneData]]): List of
            (sequence, scene_data) tuples
        predict_scene_splitting (PredictSceneSplitting): Strategy used to split the scene

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]], list[AnimationSceneData]]:
            - padded_batch: tensor of shape (batch_size, max_sequence_length, pose_dim)
            - padding_mask: tensor of shape (batch_size, max_sequence_length, pose_dim)
            - lengths: tensor of shape (batch_size)
            - unmasked_frames_indices: list of lists of unmasked frames indices
            - scenes: list of AnimationSceneData
    """
    list_tensors = [sample[0] for sample in batch]
    scenes = [sample[1] for sample in batch]
    lengths = [len(sequence) for sequence in list_tensors]
    max_sequence_length = get_batch_final_sequence_length(lengths, predict_scene_splitting)
    torch_lengths = torch.IntTensor(lengths)
    padded_batch, _, padding_mask = get_padded_batch_and_masks(
        max_sequence_length=max_sequence_length, sequences=list_tensors, keyframes=[], sequence_lengths=torch_lengths
    )
    unmasked_frames_indices = [get_unmasked_frames_indices(scene) for scene in scenes]
    return (padded_batch, padding_mask, torch_lengths, unmasked_frames_indices, scenes)


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    scene_splitting: SceneSplitting,
    pose_normalizer: PoseNormalizer | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """This function is used to collate a batch of sequences for train/val/test steps.

    The sequences that the dataset returns are of variable length.
    The sequences are padded to the same size, then the sequences are stacked into a tensor.

    Args:
        batch: list of tuples containing sequences of poses, animation_keyframes and block_keyframes
    returns:
        padded_batch: tensor of shape (batch_size, max_sequence_length, pose_dim)
        padding_mask: tensor of shape (batch_size, max_sequence_length, pose_dim)
        lengths: tensor of shape (batch_size)
        padded_animation_keyframes: tensor of shape (batch_size, max_sequence_length)
        padded_block_keyframes: tensor of shape (batch_size, max_sequence_length)
    """
    sequences = [item[0] for item in batch]
    animation_keyframes = [item[1] for item in batch]
    block_keyframes = [item[2] for item in batch]
    controller_keyframes = [item[3] for item in batch]
    if pose_normalizer is not None:
        sequences = [pose_normalizer.normalize(sequence) for sequence in sequences]
    lengths = [len(sequence) for sequence in sequences]
    max_sequence_length = get_batch_final_sequence_length(lengths=lengths, scene_splitting=scene_splitting)
    torch_lengths = torch.IntTensor(lengths)
    keyframes = [animation_keyframes, block_keyframes, controller_keyframes]
    padded_sequences, padded_keyframes, padding_mask = get_padded_batch_and_masks(
        max_sequence_length=max_sequence_length,
        sequences=sequences,
        keyframes=keyframes,
        sequence_lengths=torch_lengths,
    )
    padded_animation_keyframes, padded_block_keyframes, padded_controller_keyframes = padded_keyframes
    return (
        padded_sequences,
        padded_animation_keyframes.bool(),
        padded_block_keyframes.bool(),
        padding_mask,
        padded_controller_keyframes,
        torch_lengths,
    )
