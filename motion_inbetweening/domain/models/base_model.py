import torch
from torch import nn


class BaseModel(nn.Module):
    def __init__(self, pose_dim: int) -> None:
        super().__init__()
        self.pose_dim = pose_dim

    def forward(
        self, masked_sequence: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            masked_sequence: [batch_size, sequence_length, pose_dim+1]
            mask: [batch_size, sequence_length]
        """
        raise NotImplementedError("Subclasses must implement this method")
