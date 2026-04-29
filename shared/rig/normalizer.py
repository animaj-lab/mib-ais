import torch


class PoseNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean
        self.std = std

    def normalize(self, vector: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.to(vector.device)
        self.std = self.std.to(vector.device)
        return (vector - self.mean) / self.std

    def denormalize(self, vector: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.to(vector.device)
        self.std = self.std.to(vector.device)
        return vector * self.std + self.mean


def initialize_pose_normalizer(vectors: torch.Tensor) -> PoseNormalizer:
    mean, std = calculate_mean_std(vectors)
    return PoseNormalizer(mean, std)


def calculate_mean_std(vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = vectors.mean(dim=0)
    std = vectors.std(dim=0)
    std[std == 0] = 1e-06
    return (mean, std)


def apply_normalization_padded_sequences(
    ground_truth_sequence: torch.Tensor, sequence_lengths: torch.Tensor, pose_normalizer: PoseNormalizer
) -> torch.Tensor:
    """
    Apply normalization to a padded sequence tensor.
    Args:
        ground_truth_sequence (torch.Tensor): The padded sequence tensor
        sequence_lengths (torch.Tensor): The lengths of the sequences in the batch
        pose_normalizer (PoseNormalizer): The pose normalizer
    Returns:
        torch.Tensor: The normalized padded sequence tensor
    """
    sequences = [ground_truth_sequence[i, :length] for i, length in enumerate(sequence_lengths)]
    normalized_sequences = [pose_normalizer.normalize(sequence) for sequence in sequences]
    max_length = ground_truth_sequence.size(1)
    padded_normalized_sequences = torch.nn.utils.rnn.pad_sequence(
        normalized_sequences, batch_first=True, padding_value=0.0
    )
    if padded_normalized_sequences.size(1) < max_length:
        padding = torch.zeros(
            (
                padded_normalized_sequences.size(0),
                max_length - padded_normalized_sequences.size(1),
                padded_normalized_sequences.size(2),
            ),
            device=ground_truth_sequence.device,
        )
        padded_normalized_sequences = torch.cat([padded_normalized_sequences, padding], dim=1)
    else:
        padded_normalized_sequences = padded_normalized_sequences[:, :max_length]
    return padded_normalized_sequences
