from typing import Literal

import torch

from motion_inbetweening.config.base import BaseModel


class ManualMaskGeneratorConfig(BaseModel):
    strategy_name: Literal["manual"]


class RandomUniformMaskGeneratorConfig(BaseModel):
    strategy_name: Literal["random_uniform"]
    ratio_masked_frames: float


class CITLRandomMaskGeneratorConfig(BaseModel):
    strategy_name: Literal["citl_random"]


class DeltaInterpolatorMaskGeneratorConfig(BaseModel):
    strategy_name: Literal["delta_interpolator"]
    period: int


class CondMIRandomMaskGeneratorConfig(BaseModel):
    strategy_name: Literal["cond_mi_random"]


class RandomAnimationKeyframesMaskGeneratorConfig(BaseModel):
    strategy_name: Literal["random_animation_keyframes"]
    ratio_masked_animation_keyframes: float


class RandomBlockKeyframesMaskGeneratorConfig(BaseModel):
    strategy_name: Literal["random_block_keyframes"]
    ratio_masked_block_keyframes: float


MaskGeneratorConfig = (
    ManualMaskGeneratorConfig
    | RandomUniformMaskGeneratorConfig
    | RandomAnimationKeyframesMaskGeneratorConfig
    | RandomBlockKeyframesMaskGeneratorConfig
    | CITLRandomMaskGeneratorConfig
    | DeltaInterpolatorMaskGeneratorConfig
    | CondMIRandomMaskGeneratorConfig
)


def generate_mask(
    config: MaskGeneratorConfig,
    movement_torch: torch.Tensor,
    sequence_lengths: torch.IntTensor,
    animation_keyframes: torch.Tensor | None = None,
    block_keyframes: torch.Tensor | None = None,
    list_unmasked_frames: list[list[int]] | None = None,
) -> torch.Tensor:
    """
    This function generates a mask for a batch of movement sequences based on the provided configuration.
    Args:
        config (MaskGeneratorConfig): Masking configuration.
        movement_torch (torch.Tensor): Tensor of shape (n_batch, n_frames, n_features) containing the movement data.
        sequence_lengths (torch.IntTensor): Tensor of shape (n_batch,) containing the length of each sequence in
        the batch.
        keyframes (torch.Tensor): Tensor of shape (n_batch, n_frames) of bool indicating if the frame is a keyframe
        (True) or not (False).
        list_unmasked_frames (list[list[int]]): List of keyframes to keep unmasked for each sequence.
    Returns:
        mask: A boolean torch.Tensor of shape (n_batch, n_frames), where True indicates a masked frame and False
        indicates an unmasked frame."""
    match config:
        case RandomUniformMaskGeneratorConfig(ratio_masked_frames=ratio_masked_frames):
            return generate_random_uniform_mask(
                movement_data=movement_torch, sequence_lengths=sequence_lengths, ratio_masked_frames=ratio_masked_frames
            )
        case ManualMaskGeneratorConfig():
            if list_unmasked_frames is None:
                raise ValueError("list_unmasked_frames cannot be None for ManualMaskGeneratorConfig")
            return generate_manual_mask(movement_torch, sequence_lengths, list_unmasked_frames)
        case RandomAnimationKeyframesMaskGeneratorConfig(
            ratio_masked_animation_keyframes=ratio_masked_animation_keyframes
        ):
            if animation_keyframes is None:
                raise ValueError("Animation keyframes cannot be None for RandomAnimationKeyframesMaskGeneratorConfig")
            return generate_random_keyframes_mask(
                movement_data=movement_torch,
                keyframes=animation_keyframes,
                sequence_lengths=sequence_lengths,
                ratio_masked_keyframes=ratio_masked_animation_keyframes,
            )
        case RandomBlockKeyframesMaskGeneratorConfig(ratio_masked_block_keyframes=ratio_masked_block_keyframes):
            if block_keyframes is None:
                raise ValueError("Block keyframes cannot be None for RandomBlockKeyframesMaskGeneratorConfig")
            return generate_random_keyframes_mask(
                movement_data=movement_torch,
                keyframes=block_keyframes,
                sequence_lengths=sequence_lengths,
                ratio_masked_keyframes=ratio_masked_block_keyframes,
            )
        case CITLRandomMaskGeneratorConfig():
            return generate_ctil_random_mask(movement_data=movement_torch, sequence_lengths=sequence_lengths)
        case DeltaInterpolatorMaskGeneratorConfig(period=period):
            return generate_delta_interpolator_mask(
                movement_data=movement_torch, sequence_lengths=sequence_lengths, period=period
            )
        case CondMIRandomMaskGeneratorConfig():
            return generate_cond_mi_random_mask(movement_data=movement_torch, sequence_lengths=sequence_lengths)
        case _:
            raise ValueError(f"Masking strategy {type(config).__name__} not implemented.")


def generate_cond_mi_random_mask(movement_data: torch.Tensor, sequence_lengths: torch.IntTensor) -> torch.BoolTensor:
    """
    Generate a random mask for a batch of movement sequences.

    It samples randomly the number K of frames to mask, where K is a random number
    between 1 and the sequence length minus 2 (to avoid masking the first and last frames).
    It randomly selects K frames to mask, ensuring that the first and last frames are not masked.

    Args:
        movement_data (torch.Tensor): shape (B, T, F)
        sequence_lengths (torch.IntTensor): shape (B,) indicating the length of each sequence in the batch.

    Returns:
        mask (torch.BoolTensor): shape (B, T), where True = masked (to predict), False = keyframe
    """
    B, T, _ = movement_data.shape
    device = movement_data.device
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b in range(B):
        seq_len = sequence_lengths[b].item()
        if seq_len <= 2:
            continue
        num_mask = torch.randint(1, seq_len - 1, (1,)).item()
        possible_indices = torch.arange(1, seq_len - 1, device=device)
        masked_indices = possible_indices[torch.randperm(len(possible_indices))[:num_mask]]
        mask[b, masked_indices] = True
        if seq_len < T:
            mask[b, seq_len:] = True
    return mask


def generate_delta_interpolator_mask(
    movement_data: torch.Tensor, sequence_lengths: torch.IntTensor, period: int = 5
) -> torch.BoolTensor:
    """
    Generate a mask with a fixed number of keyframes (same for all samples).
    - Always includes first and last frame
    - Periodic keyframes (every `period`)
    - Adjusts by adding/removing random keyframes so each sample has same count

    Args:
        movement_data (torch.Tensor): shape (B, T, F)
        sequence_lengths (torch.IntTensor): shape (B,)
        period (int): Interval for periodic frames

    Returns:
        mask (torch.BoolTensor): shape (B, T), where True = masked (to predict), False = keyframe
    """
    B, T, _ = movement_data.shape
    device = movement_data.device
    max_length = sequence_lengths.max().item()
    min_length = sequence_lengths.min().item()
    min_keyframes = max(min_length // 5, 2)
    max_keyframes = min(max_length // 5, min_length)
    if min_keyframes > max_keyframes:
        num_keyframes = min_keyframes
    else:
        num_keyframes = torch.randint(low=min_keyframes, high=max_keyframes + 1, size=(1,)).item()
    time = torch.arange(T, device=device).unsqueeze(0)
    valid = time < sequence_lengths.view(B, 1)
    is_first = time == 0
    is_last = time == (sequence_lengths - 1).view(B, 1)
    is_periodic = time % period == 0
    initial_keyframes = (is_first | is_last | is_periodic) & valid
    mask = torch.ones((B, T), dtype=torch.bool, device=device)
    for i in range(B):
        seq_len = sequence_lengths[i].item()
        selected = initial_keyframes[i, :seq_len].nonzero(as_tuple=False).squeeze(1).tolist()
        selected = set(selected)
        current_kf = sorted(selected)
        delta = num_keyframes - len(current_kf)
        if delta > 0:
            all_indices = set(range(seq_len))
            available = list(all_indices - set(current_kf))
            if available:
                add = torch.randperm(len(available), device=device)[:delta]
                added_kf = [available[j.item()] for j in add]
                current_kf.extend(added_kf)
        elif delta < 0:
            removable = list(set(current_kf) - {0, seq_len - 1})
            if removable:
                remove = torch.randperm(len(removable), device=device)[:-delta]
                to_remove = set(removable[j.item()] for j in remove)
                current_kf = [kf for kf in current_kf if kf not in to_remove]
        current_kf = sorted(set(current_kf))
        mask[i, current_kf] = False
        mask[i, seq_len:] = True
    return mask


def generate_ctil_random_mask(movement_data: torch.Tensor, sequence_lengths: torch.IntTensor) -> torch.Tensor:
    """
    Generate a random CITL mask.

    Args:
        movement_data (torch.Tensor): shape (batch_size, n_frames, n_features)
        sequence_lengths (torch.IntTensor): shape (batch_size,)

    Returns:
        mask (torch.BoolTensor): shape (batch_size, n_frames), where
                                 True = masked (to predict), False = unmasked (keyframe)
    """
    batch_size, n_frames, _ = movement_data.shape
    device = movement_data.device
    max_len = sequence_lengths.max().item()
    min_len = sequence_lengths.min().item()
    min_kf = max(int(max_len // 24), 3)
    max_kf = min(int(max_len // 4), min_len, n_frames)
    if min_kf > max_kf:
        n_keyframes = min(min_len, n_frames)
    else:
        n_keyframes = torch.randint(low=min_kf, high=max_kf + 1, size=(1,)).item()
    mask = torch.ones((batch_size, n_frames), dtype=torch.bool, device=device)
    for i in range(batch_size):
        seq_len = sequence_lengths[i].item()
        keyframe_indices = {0, seq_len - 1}
        num_middle = n_keyframes - 2
        available_middle = seq_len - 2
        if num_middle > 0 and available_middle > 0:
            actual_num_middle = min(num_middle, available_middle)
            middle_indices = torch.randperm(available_middle, device=device)[:actual_num_middle] + 1
            keyframe_indices.update(middle_indices.tolist())
        mask[i, list(keyframe_indices)] = False
        mask[i, seq_len:] = True
    return mask


def generate_random_uniform_mask(
    movement_data: torch.Tensor, sequence_lengths: torch.IntTensor, ratio_masked_frames: float
) -> torch.Tensor:
    """
    Generate a random mask for a batch of movement sequences. The first and last frames
    of each sequence are not masked, and padding frames are always masked.

    Args:
        movement_data (torch.Tensor): Tensor of shape (n_batch, n_frames, n_features) containing the movement data.
        sequence_lengths (torch.IntTensor): Tensor of shape (n_batch,) containing the length of each sequence in
        the batch.
        ratio_masked_frames (float): Ratio of frames to mask.

    Returns:
        mask: A boolean torch.Tensor of shape (n_batch, n_frames), where True indicates a masked frame and False
        indicates an unmasked frame.
    """
    mask = generate_random_bernoulli_mask(movement_data, ratio_masked_frames)
    mask = unmask_first_and_last_frame(mask, sequence_lengths)
    padding_mask = generate_padding_mask(movement_data, sequence_lengths)
    mask = mask | padding_mask
    return mask


def generate_manual_mask(
    movement_torch: torch.Tensor, sequence_lengths: torch.IntTensor, list_unmasked_frames: list[list[int]]
) -> torch.Tensor:
    """Generate a mask indicating which frames to keep (0) and which to mask (1).
    The first and last frames of each sequence are always unmasked.

    Args:
        movement_torch (torch.Tensor): Input movement of shape (n_batch, n_frames, n_features).
        sequence_lengths (torch.IntTensor): Length of each sequence in the batch.
        list_unmasked_frames (list[list[int]]): List of keyframes to keep unmasked for each sequence.

    Returns:
        torch.Tensor: Output mask of shape (n_batch, n_frames) of bool.
    """
    n_batch, n_frames, _ = movement_torch.shape
    mask = torch.ones((n_batch, n_frames), dtype=torch.bool, device=movement_torch.device)
    for i, unmasked_frames in enumerate(list_unmasked_frames):
        for unmasked_index in unmasked_frames:
            mask[i, unmasked_index] = False
    mask[:, 0] = False
    for i in range(n_batch):
        mask[i, sequence_lengths[i] - 1] = False
    return mask


def generate_random_keyframes_mask(
    movement_data: torch.Tensor,
    keyframes: torch.Tensor,
    sequence_lengths: torch.IntTensor,
    ratio_masked_keyframes: float,
) -> torch.Tensor:
    """
    Generate a random mask for a batch of movement sequences. Only the first frame, the last frame, and a random subset
    of keyframes of proportion (1-ratio_masked_keyframes) are not masked, and padding frames are always masked.

    Args:
        movement_data (torch.Tensor): Tensor of shape (n_batch, n_frames, n_features) containing the movement data.
        keyframes (torch.Tensor): Tensor of shape (n_batch, n_frames) of bool indicating if the frame is a keyframe
        (True) or not (False).
        sequence_lengths (torch.IntTensor): Tensor of shape (n_batch,) containing the length of each sequence in
        the batch.
        ratio_masked_keyframes (float): Ratio of keyframes to mask.

    Returns:
        mask: A boolean torch.Tensor of shape (n_batch, n_frames), where True indicates a masked frame and False"""
    random_uniform_mask = generate_random_bernoulli_mask(movement_data, ratio_masked_keyframes)
    mask = random_uniform_mask | ~keyframes
    mask = unmask_first_and_last_frame(mask, sequence_lengths)
    return mask


def unmask_first_and_last_frame(mask: torch.Tensor, sequence_lengths: torch.IntTensor) -> torch.Tensor:
    """
    Ensure the first and last frames are not masked.

    Args:
        mask (torch.Tensor): Mask of shape (n_batch, n_frames) where True indicates a masked frame.
        sequence_lengths (torch.IntTensor): Length of each sequence in the batch.

    Returns:
        torch.Tensor: Mask with the first and last frames unmasked.
    """
    n_batch, n_frames = mask.shape
    mask[:, 0] = False
    mask[torch.arange(n_batch), sequence_lengths - 1] = False
    return mask


def generate_padding_mask(movement_data: torch.Tensor, sequence_lengths: torch.IntTensor) -> torch.Tensor:
    """
    Generate a padding mask for a batch of movement sequences.
    Args:
        movement_data (torch.Tensor): Tensor of shape (n_batch, n_frames, n_features) containing the movement data.
        sequence_lengths (torch.IntTensor): Tensor of shape (n_batch,) containing the length of each sequence in
        the batch.
    Returns:
        torch.Tensor: Padding mask of shape (n_batch, n_frames) where True indicates a padded frame.
    """
    n_batch, n_frames, _ = movement_data.shape
    padding_mask = torch.arange(n_frames, device=movement_data.device).expand(
        n_batch, n_frames
    ) >= sequence_lengths.unsqueeze(1)
    return padding_mask


def generate_random_bernoulli_mask(movement_data: torch.Tensor, bernoulli_prob: float) -> torch.Tensor:
    """
    Generate a random mask for a batch of movement sequences using Bernoulli distribution."""
    n_batch, n_frames, _ = movement_data.shape
    mask_probs = torch.full((n_batch, n_frames), bernoulli_prob, device=movement_data.device)
    mask = torch.bernoulli(mask_probs).bool()
    return mask
