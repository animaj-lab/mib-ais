import torch


def reshape_sequence(sequence: torch.Tensor) -> torch.Tensor:
    """
    Reshape the sequence tensor to have the shape (batch_size*seq_len, pose_dim)
    Args:
        sequence (torch.Tensor): The sequence tensor of shape (batch_size, seq_len, pose_dim)
    Returns:
        torch.Tensor: The reshaped tensor of shape (batch_size*seq_len, pose_dim)"""
    return sequence.view(-1, sequence.shape[-1])
