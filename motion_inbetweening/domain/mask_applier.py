from enum import StrEnum, auto

import torch

from motion_inbetweening.config.base import BaseModel
from motion_inbetweening.domain.slerp_interpolation import lerp_slerp_interpolation
from shared.rig.trainable_controllers import TrainableController

SPECIAL_VALUE = -10


class ConcatMode(StrEnum):
    CONCAT_DIM = auto()
    CONCAT_CHANNEL = auto()


class FillMode(StrEnum):
    ZEROS = auto()
    SPECIAL_VALUE = auto()
    PREVIOUS = auto()
    NEXT = auto()
    LINEAR = auto()
    CUBIC_SPLINE = auto()
    SLERP = auto()


class MaskApplierConfig(BaseModel):
    concat_mode: ConcatMode
    fill_mode: FillMode


class MaskApplier:
    def __init__(self, config: MaskApplierConfig):
        self.concat_mode = config.concat_mode
        self.fill_mode = config.fill_mode

    def apply_mask(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence: [batch_size, sequence_length, pose_dim]
            mask: [batch_size, sequence_length]
        """
        masked_sequence = self.fill_masked_sequence(sequence, mask)
        masked_sequence_concatenated = self.concat_masked_sequence(masked_sequence, mask)
        return masked_sequence_concatenated

    def check_index(self, unmasked: torch.Tensor, mode: FillMode) -> torch.Tensor:
        sequence_length = unmasked.size(1)
        sequence_indices = torch.arange(sequence_length, device=unmasked.device).unsqueeze(0)
        if mode == FillMode.PREVIOUS:
            masked_indices = torch.where(unmasked, sequence_indices, torch.full_like(sequence_indices, 1000))
            last_true_indices = masked_indices.min(dim=1).values
        elif mode == FillMode.NEXT:
            masked_indices = torch.where(unmasked, sequence_indices, torch.full_like(sequence_indices, -1))
            last_true_indices = masked_indices.max(dim=1).values
        return last_true_indices

    def fill_masked_sequence(
        self, sequence: torch.Tensor, mask: torch.Tensor, trainable_controllers: list[TrainableController] | None = None
    ) -> torch.Tensor:
        """
        Fills the masked parts of the sequence according to the fill mode.
            ZEROS: Fills the masked parts with zeros.
            SPECIAL_VALUE: Fills the masked parts with a special value.
            PREVIOUS: Fills the masked parts with the previous valid value.
                In the case the first elements are masked, we fall back to the "NEXT" behavior for these elements,
                until the first unmasked element.
            NEXT: Fills the masked parts with the next valid value.
                In the case the last elements are masked, we fall back to the "PREVIOUS" behavior for these elements,
                until the last unmasked element.
            LINEAR: Fills the masked parts with a linear interpolation.
            CUBIC_SPLINE: Fills the masked parts with a cubic spline interpolation.
        Args:
            sequence: [batch_size, sequence_length, pose_dim]
            mask: [batch_size, sequence_length]
        """
        if self.fill_mode == FillMode.ZEROS:
            masked_sequence = sequence.clone()
            masked_sequence[mask] = 0
        elif self.fill_mode == FillMode.SPECIAL_VALUE:
            masked_sequence = sequence.clone()
            masked_sequence[mask] = SPECIAL_VALUE
        elif self.fill_mode == FillMode.PREVIOUS:
            batch_size, seq_len, feature_dim = sequence.shape
            unmasked = ~mask
            smallest_index = self.check_index(unmasked, mode=self.fill_mode)
            indices = torch.arange(seq_len, device=sequence.device).unsqueeze(0).expand(batch_size, -1)
            masked_indices = torch.where(unmasked, indices, torch.full_like(indices, -1))
            masked_indices[:, 0] = smallest_index
            unmasked_idxs, _ = torch.cummax(masked_indices, dim=1)
            masked_sequence = torch.gather(sequence, 1, unmasked_idxs.unsqueeze(-1).expand(-1, -1, feature_dim))
        elif self.fill_mode == FillMode.NEXT:
            batch_size, seq_len, feature_dim = sequence.shape
            unmasked = ~mask
            largest_index = self.check_index(unmasked, mode=self.fill_mode)
            indices = torch.arange(seq_len, device=sequence.device).unsqueeze(0).expand(batch_size, -1)
            masked_indices = torch.where(unmasked, indices, torch.full_like(indices, seq_len))
            masked_indices[:, -1] = largest_index
            reversed_indices = masked_indices.flip(1)
            next_valid_idxs, _ = torch.cummin(reversed_indices, dim=1)
            next_valid_idxs = next_valid_idxs.flip(1)
            next_valid_idxs = torch.clamp(next_valid_idxs, max=seq_len - 1)
            masked_sequence = torch.gather(sequence, 1, next_valid_idxs.unsqueeze(-1).expand(-1, -1, feature_dim))
        elif self.fill_mode == FillMode.LINEAR:
            masked_sequence = linear_interpolation(sequence, mask)
        elif self.fill_mode == FillMode.CUBIC_SPLINE:
            raise NotImplementedError("Cubic spline fill mode is not implemented yet")
        elif self.fill_mode == FillMode.SLERP:
            masked_sequence_zero = sequence.clone()
            masked_sequence_zero[mask] = 0
            masked_sequence = lerp_slerp_interpolation(masked_sequence_zero, mask, trainable_controllers)
        else:
            raise ValueError(f"Unknown fill_mode: {self.fill_mode}")
        return masked_sequence

    def concat_masked_sequence(self, masked_sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            masked_sequence: [batch_size, sequence_length, pose_dim]
            mask: [batch_size, sequence_length]
        """
        if self.concat_mode == "concat_dim":
            masked_sequence_concatenated = torch.cat([masked_sequence, mask.unsqueeze(-1)], dim=-1)
        elif self.concat_mode == "concat_channel":
            masked_sequence_concatenated = torch.cat(
                [masked_sequence, mask.unsqueeze(-1).repeat(1, 1, masked_sequence.shape[-1])], dim=-1
            )
        else:
            raise NotImplementedError("Only concat_dim and concat_channel are supported for now")
        return masked_sequence_concatenated


def create_mask_appliers_for_explicit_residual_model() -> tuple[MaskApplier, MaskApplier, MaskApplier]:
    """
    Creates and returns three MaskApplier instances with different configurations.

    The configurations are as follows:
    - The first MaskApplier uses a configuration with CONCAT_DIM concat mode and ZEROS fill mode.
    - The second MaskApplier uses a configuration with CONCAT_DIM concat mode and PREVIOUS fill mode.
    - The third MaskApplier uses a configuration with CONCAT_DIM concat mode and NEXT fill mode.

    Returns:
        tuple[MaskApplier, MaskApplier, MaskApplier]: A tuple containing the three MaskApplier instances.
    """
    config_zero = MaskApplierConfig(concat_mode=ConcatMode.CONCAT_DIM, fill_mode=FillMode.ZEROS)
    config_previous = MaskApplierConfig(concat_mode=ConcatMode.CONCAT_DIM, fill_mode=FillMode.PREVIOUS)
    config_next = MaskApplierConfig(concat_mode=ConcatMode.CONCAT_DIM, fill_mode=FillMode.NEXT)
    mask_applier_zero = MaskApplier(config=config_zero)
    mask_applier_previous = MaskApplier(config=config_previous)
    mask_applier_next = MaskApplier(config=config_next)
    return (mask_applier_zero, mask_applier_previous, mask_applier_next)


def linear_interpolation(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Linearly interpolates the masked values in the sequence.

    Args:
        sequence: Tensor of shape [B, T, D]
        mask: Tensor of shape [B, T], with 0 = unmasked, 1 = masked

    Returns:
        Tensor of shape [B, T, D] with masked values linearly interpolated.
    """
    B, T, D = sequence.shape
    interpolated = sequence.clone()
    interpolated[mask] = 0
    for b in range(B):
        unmasked_indices = (~mask[b].bool()).nonzero(as_tuple=False).squeeze(1)
        for i in range(len(unmasked_indices) - 1):
            start = unmasked_indices[i]
            end = unmasked_indices[i + 1]
            if end - start > 1:
                start_val = sequence[b, start]
                end_val = sequence[b, end]
                num_steps = end - start
                for t in range(1, num_steps):
                    alpha = t / num_steps
                    interpolated[b, start + t] = (1 - alpha) * start_val + alpha * end_val
    return interpolated
