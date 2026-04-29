import torch


def get_padded_batch_and_masks(
    max_sequence_length: int,
    sequences: list[torch.Tensor],
    keyframes: list[list[torch.Tensor]],
    sequence_lengths: torch.IntTensor,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    padded_sequences = pad_tensor_to_length(sequences, max_sequence_length)
    padded_keyframes = [pad_tensor_to_length(kf, max_sequence_length) for kf in keyframes]
    batch_size, _, feature_dim = padded_sequences.size()
    padding_mask = torch.arange(max_sequence_length, device=padded_sequences.device).expand(
        batch_size, max_sequence_length
    ).unsqueeze(-1) >= sequence_lengths.unsqueeze(1).unsqueeze(-1)
    padding_mask = padding_mask.expand(-1, -1, feature_dim)
    return (padded_sequences, padded_keyframes, padding_mask)


def pad_tensor(list_tensor: list[torch.Tensor]) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(list_tensor, batch_first=True, padding_value=0.0)


def add_extra_padding_to_tensor(tensor: torch.Tensor, extra_padding: torch.Tensor):
    return torch.cat([tensor, extra_padding], dim=1)


def pad_tensor_to_length(list_tensor: list[torch.Tensor], length: int) -> torch.Tensor:
    """Pad a list of tensors to a given length.

    Args:
        list_tensor (list[torch.Tensor]): List of tensors to pad of shape (seq_len, pose_dim) or (seq_len)
        length (int): Length to pad the tensors to

    Returns:
        torch.Tensor: Padded tensor of shape (batch_size, length, pose_dim) or (batch_sizen, length)
    """
    padded_tensor = pad_tensor(list_tensor)
    if padded_tensor.size(1) < length:
        extra_padding = torch.zeros(
            padded_tensor.size(0),
            length - padded_tensor.size(1),
            *padded_tensor.size()[2:],
            device=padded_tensor.device,
        )
        padded_tensor = add_extra_padding_to_tensor(padded_tensor, extra_padding)
    return padded_tensor


def apply_padding_mask(batch_sequence: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Apply a padding mask to a batch of sequences.

    Args:
        batch_sequence (torch.Tensor): Batch of sequences of shape (batch_size, seq_len, pose_dim)
        padding_mask (torch.Tensor): Padding mask of shape (batch_size, seq_len)

    Returns:
        torch.Tensor: Batch of sequences with padding applied of shape (batch_size, seq_len, pose_dim)
    """
    return batch_sequence.masked_fill(padding_mask, 0.0)
