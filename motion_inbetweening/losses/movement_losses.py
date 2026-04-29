from collections.abc import Callable

import torch


def compute_speed_loss(
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ground_truth_sequence: torch.Tensor,
    predicted_sequence: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the speed loss for the given ground truth and predicted sequences
    Args:
        loss_fn: loss function to use
        ground_truth_sequence: ground truth sequence of shape (batch_size, seq_len, pose_dim)
        predicted_sequence: predicted sequence of shape (batch_size, seq_len, pose_dim)
        sequence_lengths: sequence lengths tensor of shape (batch_size)
    Returns:
        speed_loss: loss_fn loss between the ground truth speed and predicted speed
    """
    ground_truth_speed = get_speed_from_sequence(ground_truth_sequence, sequence_lengths)
    predicted_speed = get_speed_from_sequence(predicted_sequence, sequence_lengths)
    speed_loss = loss_fn(ground_truth_speed, predicted_speed)
    return speed_loss


def get_speed_from_sequence(sequence: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
    """
    Compute speed from the given sequence. Final length considers the padding according to sequence_lengths
    Args:
        sequence: sequence of shape (batch_size, seq_len, pose_dim)
        sequence_lengths: sequence lengths tensor of shape (batch_size)
    Returns:
        speed: torch.Tensor of shape (batch_size, seq_len-1, pose_dim)
    """
    speed = sequence[:, 1:, :] - sequence[:, :-1, :]
    mask = get_padding_mask(sequence, sequence_lengths, 1)
    masked_speed = speed * mask.float()
    return masked_speed


def compute_acceleration_loss(
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ground_truth_sequence: torch.Tensor,
    predicted_sequence: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the acceleration loss for the given ground truth and predicted sequences
    Args:
        loss_fn: loss function to use
        ground_truth_sequence: ground truth sequence of shape (batch_size, seq_len, pose_dim)
        predicted_sequence: predicted sequence of shape (batch_size, seq_len, pose_dim)
        sequence_lengths: sequence lengths tensor of shape (batch_size)
    Returns:
        acceleration_loss: loss_fn loss between the ground truth acceleration and predicted acceleration
    """
    ground_truth_acceleration = get_acceleration_from_sequence(ground_truth_sequence, sequence_lengths)
    predicted_acceleration = get_acceleration_from_sequence(predicted_sequence, sequence_lengths)
    acceleration_loss = loss_fn(ground_truth_acceleration, predicted_acceleration)
    return acceleration_loss


def get_acceleration_from_sequence(sequence: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
    """
    Compute acceleration from the given sequence. Final length considers the padding according to sequence_lengths
    Args:
        sequence: sequence of shape (batch_size, seq_len, pose_dim)
        sequence_lengths: sequence lengths tensor of shape (batch_size)
    Returns:
        acceleration: torch.Tensor of shape (batch_size, seq_len-2, pose_dim)
    """
    acceleration = sequence[:, 2:, :] - 2 * sequence[:, 1:-1, :] + sequence[:, :-2, :]
    mask = get_padding_mask(sequence, sequence_lengths, 2)
    masked_acceleration = acceleration * mask.float()
    return masked_acceleration


def compute_jerk_loss(
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ground_truth_sequence: torch.Tensor,
    predicted_sequence: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the jerk loss for the given ground truth and predicted sequences
    Args:
        loss_fn: loss function to use
        ground_truth_sequence: ground truth sequence of shape (batch_size, seq_len, pose_dim)
        predicted_sequence: predicted sequence of shape (batch_size, seq_len, pose_dim)
        sequence_lengths: sequence lengths tensor of shape (batch_size)
    Returns:
        jerk_loss: loss_fn loss between the ground truth jerk and predicted jerk
    """
    ground_truth_jerk = get_jerk_from_sequence(ground_truth_sequence, sequence_lengths)
    predicted_jerk = get_jerk_from_sequence(predicted_sequence, sequence_lengths)
    jerk_loss = loss_fn(ground_truth_jerk, predicted_jerk)
    return jerk_loss


def get_jerk_from_sequence(sequence: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
    """
    Compute jerk from the given sequence. Final length considers the padding according to sequence_lengths
    Args:
        sequence: sequence of shape (batch_size, seq_len, pose_dim)
        sequence_lengths: sequence lengths tensor of shape (batch_size)
    Returns:
        jerk: torch.Tensor of shape (batch_size, seq_len-3, pose_dim)
    """
    jerk = sequence[:, 3:, :] - 3 * sequence[:, 2:-1, :] + 3 * sequence[:, 1:-2, :] - sequence[:, :-3, :]
    mask = get_padding_mask(sequence, sequence_lengths, 3)
    masked_jerk = jerk * mask.float()
    return masked_jerk


def get_padding_mask(sequence: torch.Tensor, sequence_lengths: torch.Tensor, frames_reduced: int) -> torch.Tensor:
    """
    Mask out the sequence values for the padded values in the sequence according to sequence_lengths
    Args:
        sequence: sequence tensor of shape (batch_size, seq_len, pose_dim)
        sequence_lengths: sequence lengths tensor of shape (batch_size)
        frames_reduced: number of frames reduced in the sequence (e.g. 1 for speed, 2 for acceleration, 3 for jerk)
    Returns:
        masked_sequence: sequence tensor of shape (batch_size, seq_len - frames_reduced, 1)
    """
    batch_size, current_length, _ = sequence.shape
    mask = torch.arange(current_length - frames_reduced, device=sequence.device).expand(
        batch_size, current_length - frames_reduced
    ).unsqueeze(-1) < (sequence_lengths - frames_reduced).unsqueeze(1).unsqueeze(-1)
    return mask
