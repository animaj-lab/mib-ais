import random

import torch

from motion_inbetweening.config.block_keyframes_schedule_modifier import (
    BlockKeyframesScheduleAddition,
    BlockKeyframesScheduleDeletion,
    BlockKeyframesScheduleDeterministicShift,
    BlockKeyframesScheduleRandomShift,
    BlockKeyramesScheduleModifierConfig,
    BlockScheduleAugmentation,
)
from shared.domain.entities.animation import FrameID


def apply_block_schedule_augmentation(
    block_keyframes: torch.Tensor, block_schedule_augmentation: BlockScheduleAugmentation
) -> torch.Tensor:
    for block_schedule_probabilistic_augmentation in block_schedule_augmentation:
        if random.random() < block_schedule_probabilistic_augmentation.proba:
            block_keyframes = apply_block_keyframes_modification(
                block_keyframes, block_schedule_probabilistic_augmentation.modifier
            )
    block_keyframes = add_first_and_last_block_keyframes(block_keyframes)
    return block_keyframes


def apply_block_keyframes_modification(
    block_keyframes: torch.Tensor, block_keyframes_modifers_config: BlockKeyramesScheduleModifierConfig
) -> torch.Tensor:
    """
    Args:
        block_keyframes (torch.Tensor): torch.tensor of False and True, with True indicating a block keyframes shape
        (n_frames)
        block_keyframes_modifers_config(BlockKeyramesScheduleModifierConfig):
    """
    match block_keyframes_modifers_config:
        case BlockKeyframesScheduleDeterministicShift(shift_size=shift_size):
            modified_block_keyframes = deterministic_shift_block_keyframes(block_keyframes, shift_size)
        case BlockKeyframesScheduleRandomShift(min_shift_size=min_shift_size, max_shift_size=max_shift_size):
            modified_block_keyframes = random_shift_block_keyframes(
                block_keyframes=block_keyframes, min_shift_size=min_shift_size, max_shift_size=max_shift_size
            )
        case BlockKeyframesScheduleAddition(percentage=percentage):
            modified_block_keyframes = add_keyframes(block_keyframes, percentage)
        case BlockKeyframesScheduleDeletion(percentage=percentage):
            modified_block_keyframes = delete_keyframes(block_keyframes, percentage)
        case _:
            raise ValueError(f"Unsupported BlockKeyramesScheduleModifierConfig: {block_keyframes_modifers_config}")
    return modified_block_keyframes


def add_first_and_last_block_keyframes(keyframes: torch.Tensor) -> torch.Tensor:
    """
    Adds the first and last frame as block keyframes.

    Args:
        block_keyframes (torch.Tensor): A tensor of shape (n_frames) with boolean values.

    Returns:
        torch.Tensor: A tensor with the first and last frames set to True.
    """
    keyframes[0] = True
    keyframes[-1] = True
    return keyframes


def deterministic_shift_block_keyframes(block_keyframes: torch.Tensor, shift_size: int) -> torch.Tensor:
    """
    Shifts the block keyframes by a specified shift size. If a shift reaches beyond the tensor bounds,
    the values are clamped within the valid range.

    Args:
        block_keyframes (torch.Tensor): A tensor of shape (n_frames) with boolean values.
        shift_size (int): The number of frames to shift.

    Returns:
        torch.Tensor: A tensor with the shifted block keyframes.
    """
    n_frames = block_keyframes.shape[0]
    shifted_keyframes = torch.zeros_like(block_keyframes, dtype=torch.bool)
    for i in range(n_frames):
        if block_keyframes[i]:
            new_index = i + shift_size
            new_index = max(new_index, 0)
            new_index = min(new_index, n_frames - 1)
            shifted_keyframes[new_index] = True
    return shifted_keyframes


def random_shift_block_keyframes(
    block_keyframes: torch.Tensor, min_shift_size: int, max_shift_size: int
) -> torch.Tensor:
    """
    Shifts the block keyframes each by a random value between min_shift_size and max_shift_size.

    Args:
        block_keyframes (torch.Tensor): A tensor of shape (n_frames) with boolean values.
        min_shift_size (int): The minimum number of frames to shift.
        max_shift_size (int): The maximum number of frames to shift.

    Returns:
        torch.Tensor: A tensor with the shifted block keyframes.
    """
    n_frames = block_keyframes.shape[0]
    shifted_keyframes = torch.zeros_like(block_keyframes, dtype=torch.bool)
    for i in range(n_frames):
        if block_keyframes[i]:
            random_shift = random.randint(min_shift_size, max_shift_size)
            new_index = i + random_shift
            new_index = max(new_index, 0)
            new_index = min(new_index, n_frames - 1)
            shifted_keyframes[new_index] = True
    return shifted_keyframes


def add_keyframes(block_keyframes: torch.Tensor, percentage: int) -> torch.Tensor:
    """
    Adds keyframes to the schedule.
    The frames are added randomly in the schedule.
    The percentage of frames to add is given by the percentage attribute, based on the number of block keyframes.
    For instance, if percentage=10, and there are 80 block keyframes, 8 frames will be added.
    Note that this percentage can be greater than 100.

    Args:
        block_keyframes (torch.Tensor): A tensor of shape (n_frames) with boolean values.
        percentage (int): The percentage of frames to add.

    Returns:
        torch.Tensor: A tensor with the added keyframes.
    """
    modified_block_keyframes = block_keyframes.clone()
    n_keyframes_to_add = int(modified_block_keyframes.sum().item() * percentage / 100)
    available_indices = torch.where(~modified_block_keyframes)[0]
    if n_keyframes_to_add > len(available_indices):
        n_keyframes_to_add = len(available_indices)
    random_indices = random.sample(list(available_indices), n_keyframes_to_add)
    for index in random_indices:
        modified_block_keyframes[index] = True
    return modified_block_keyframes


def delete_keyframes(block_keyframes: torch.Tensor, percentage: int) -> torch.Tensor:
    """
    Deletes keyframes from the schedule.
    The frames are deleted randomly in the schedule.
    The percentage of frames to delete is given by the percentage attribute, based on the number of block keyframes.
    For instance, if percentage=10, and there are 80 block keyframes, 8 frames will be deleted.
    Note that this percentage can not be greater than 100.

    Args:
        block_keyframes (torch.Tensor): A tensor of shape (n_frames) with boolean values.
        percentage (int): The percentage of frames to delete.

    Returns:
        torch.Tensor: A tensor with the deleted keyframes.
    """
    modified_block_keyframes = block_keyframes.clone()
    n_keyframes_to_delete = int(modified_block_keyframes.sum().item() * percentage / 100)
    available_indices = torch.where(modified_block_keyframes)[0]
    if n_keyframes_to_delete > len(available_indices):
        n_keyframes_to_delete = len(available_indices)
    random_indices = random.sample(list(available_indices), n_keyframes_to_delete)
    for index in random_indices:
        modified_block_keyframes[index] = False
    return modified_block_keyframes


def apply_block_schedule_augmentation_frames_id(
    unmasked_frames_id: list[FrameID],
    block_schedule_augmentation: BlockScheduleAugmentation,
    frame_range: tuple[FrameID, FrameID],
) -> list[FrameID]:
    """
    Applies block schedule augmentations to a list of unmasked frame IDs.

    This function iterates through a list of probabilistic block schedule augmentations
    and applies modifications to the unmasked frame IDs based on the specified probability
    for each augmentation. After applying the modifications, it ensures that the first
    and last frame IDs in the given frame range are included, removes duplicates, and
    orders the frame IDs.

    Args:
        unmasked_frames_id (list[FrameID]): A list of frame IDs that are initially unmasked.
        block_schedule_augmentation (BlockScheduleAugmentation): A collection of probabilistic
            block schedule augmentations, each containing a probability and a modifier.
        frame_range (tuple[FrameID, FrameID]): A tuple specifying the range of frame IDs
            (start and end) to consider for augmentation.

    Returns:
        list[FrameID]: The modified list of unmasked frame IDs, with duplicates removed
        and ordered.
    """
    for block_schedule_probabilistic_augmentation in block_schedule_augmentation:
        if random.random() < block_schedule_probabilistic_augmentation.proba:
            unmasked_frames_id = apply_modification_unmasked_frames_id(
                unmasked_frames_id, block_schedule_probabilistic_augmentation.modifier, frame_range
            )
    unmasked_frames_id = add_first_and_last_unmasked_frames_id(
        unmasked_frames_id=unmasked_frames_id, frame_range=frame_range
    )
    unmasked_frames_id = remove_duplicate_and_order(unmasked_frames_id)
    return unmasked_frames_id


def apply_modification_unmasked_frames_id(
    unmasked_frames_id: list[FrameID],
    block_schedule_modification: BlockKeyramesScheduleModifierConfig,
    frame_range: tuple[FrameID, FrameID],
) -> list[FrameID]:
    """
    Modify the unmasked frame IDs based on the given block keyframes schedule modifier configuration.
    After the modification, the function reorders the frame IDs and removes duplicates.

    Args:
        unmasked_frames_id (list[FrameID]): A list of FrameID objects representing unmasked frames.
        block_schedule_modification (BlockKeyramesScheduleModifierConfig): Configuration for the block keyframes
            schedule modification.

    Returns:
        list[FrameID]: A list of FrameID objects with shifted frame IDs.
    """
    match block_schedule_modification:
        case BlockKeyframesScheduleDeterministicShift(shift_size=shift_size):
            modified_unmasked_frames_id = deterministic_shift_unmasked_frames_id(
                unmasked_frames_id=unmasked_frames_id, shift_size=shift_size, frame_range=frame_range
            )
        case BlockKeyframesScheduleRandomShift(min_shift_size=min_shift_size, max_shift_size=max_shift_size):
            modified_unmasked_frames_id = random_shift_unmasked_frames_id(
                unmasked_frames_id, min_shift_size, max_shift_size, frame_range=frame_range
            )
        case BlockKeyframesScheduleAddition(percentage=percentage):
            modified_unmasked_frames_id = add_unmasked_frames_id(
                unmasked_frames_id=unmasked_frames_id, percentage=percentage, frame_range=frame_range
            )
        case BlockKeyframesScheduleDeletion(percentage=percentage):
            modified_unmasked_frames_id = delete_unmasked_frames_id(
                unmasked_frames_id=unmasked_frames_id, percentage=percentage
            )
        case _:
            raise ValueError(f"Unsupported BlockKeyramesScheduleModifierConfig: {block_schedule_modification}")
    modified_unmasked_frames_id = remove_duplicate_and_order(modified_unmasked_frames_id)
    return modified_unmasked_frames_id


def add_first_and_last_unmasked_frames_id(
    unmasked_frames_id: list[FrameID], frame_range: tuple[FrameID, FrameID]
) -> list[FrameID]:
    """
    Adds the first and last frame IDs to the list of unmasked frames.

    Args:
        unmasked_frames_id (list[FrameID]): A list of FrameID objects representing unmasked frames.
        frame_range (tuple[FrameID, FrameID]): A tuple representing the range of valid frame IDs.

    Returns:
        list[FrameID]: A list of FrameID objects with the first and last frames added.
    """
    unmasked_frames_id.append(frame_range[0])
    unmasked_frames_id.append(frame_range[1])
    return unmasked_frames_id


def remove_duplicate_and_order(frames_id: list[FrameID]) -> list[FrameID]:
    """
    Removes duplicate FrameIDs and orders them in ascending order.

    Args:
        frames_id (list[FrameID]): A list of FrameID objects.

    Returns:
        list[FrameID]: A sorted list of unique FrameID objects.
    """
    return sorted(set(frames_id))


def deterministic_shift_unmasked_frames_id(
    unmasked_frames_id: list[FrameID], shift_size: int, frame_range: tuple[FrameID, FrameID]
) -> list[FrameID]:
    """
    Shifts the unmasked frame IDs by a specified shift size. If a shift reaches beyond the sequence bounds,
    the values are clamped within the valid range.

    Args:
        unmasked_frames_id (list[FrameID]): A list of FrameID objects representing unmasked frames.
        shift_size (int): The number of frames to shift.
        sequence_length (int): The total length of the sequence.

    Returns:
        list[FrameID]: A list of FrameID objects with shifted frame IDs.
    """
    shifted_frames_id = []
    for frame_id in unmasked_frames_id:
        new_frame_id = frame_id + shift_size
        new_frame_id = max(new_frame_id, frame_range[0])
        new_frame_id = min(new_frame_id, frame_range[1])
        shifted_frames_id.append(new_frame_id)
    return shifted_frames_id


def random_shift_unmasked_frames_id(
    unmasked_frames_id: list[FrameID], min_shift_size: int, max_shift_size: int, frame_range: tuple[FrameID, FrameID]
) -> list[FrameID]:
    """
    Shifts the unmasked frame IDs each by a random value between min_shift_size and max_shift_size.
    If a shift reaches beyond the sequence bounds, the values are clamped within the valid range.

    Args:
        unmasked_frames_id (list[FrameID]): A list of FrameID objects representing unmasked frames.
        min_shift_size (int): The minimum number of frames to shift.
        max_shift_size (int): The maximum number of frames to shift.
        max_frame_id (FrameID): The maximum valid frame ID.

    Returns:
        list[FrameID]: A list of FrameID objects with shifted frame IDs.
    """
    shifted_frames_id = []
    for frame_id in unmasked_frames_id:
        random_shift = random.randint(min_shift_size, max_shift_size)
        new_frame_id = frame_id + random_shift
        new_frame_id = max(new_frame_id, frame_range[0])
        new_frame_id = min(new_frame_id, frame_range[1])
        shifted_frames_id.append(new_frame_id)
    return shifted_frames_id


def add_unmasked_frames_id(
    unmasked_frames_id: list[FrameID], percentage: int, frame_range: tuple[FrameID, FrameID]
) -> list[FrameID]:
    """
    Adds unmasked frames to the schedule.
    The frames are added randomly in the schedule.
    The percentage of frames to add is given by the percentage attribute, based on the number of unmasked frames.
    For instance, if percentage=10, and there are 80 unmasked frames, 8 frames will be added.
    Note that this percentage can be greater than 100.

    Args:
        unmasked_frames_id (list[FrameID]): A list of FrameID objects representing unmasked frames.
        percentage (int): The percentage of frames to add.

    Returns:
        list[FrameID]: A list of FrameID objects with the added unmasked frames.
    """
    n_unmasked_frames = len(unmasked_frames_id)
    n_frames_to_add = int(n_unmasked_frames * percentage / 100)
    available_indices = set(range(frame_range[0], frame_range[1] + 1)) - set(unmasked_frames_id)
    if n_frames_to_add > len(available_indices):
        n_frames_to_add = len(available_indices)
    random_indices = random.sample(list(available_indices), n_frames_to_add)
    return unmasked_frames_id + random_indices


def delete_unmasked_frames_id(unmasked_frames_id: list[FrameID], percentage: int) -> list[FrameID]:
    """
    Deletes unmasked frames from the schedule.
    The frames are deleted randomly in the schedule.
    The percentage of frames to delete is given by the percentage attribute, based on the number of unmasked frames.
    For instance, if percentage=10, and there are 80 unmasked frames, 8 frames will be deleted.
    Note that this percentage can not be greater than 100.

    Args:
        unmasked_frames_id (list[FrameID]): A list of FrameID objects representing unmasked frames.
        percentage (int): The percentage of frames to delete.

    Returns:
        list[FrameID]: A list of FrameID objects with the deleted unmasked frames.
    """
    n_unmasked_frames = len(unmasked_frames_id)
    n_frames_to_delete = int(n_unmasked_frames * percentage / 100)
    if n_frames_to_delete > len(unmasked_frames_id):
        n_frames_to_delete = len(unmasked_frames_id)
    random_indices = random.sample(unmasked_frames_id, n_frames_to_delete)
    return [frame for frame in unmasked_frames_id if frame not in random_indices]
