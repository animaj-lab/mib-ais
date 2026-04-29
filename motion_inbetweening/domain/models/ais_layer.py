import torch
import torch.nn as nn

from motion_inbetweening.domain.mask_applier import create_mask_appliers_for_explicit_residual_model


class AISLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layer_alpha = nn.Sequential(nn.Linear(hidden_size, self.output_size), nn.Sigmoid())
        self.layer_beta = nn.Sequential(nn.Linear(hidden_size, self.output_size), nn.Sigmoid())
        self.layer_synthesis = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, self.output_size)
        )
        self.mask_applier_zero, self.mask_applier_previous, self.mask_applier_next = (
            create_mask_appliers_for_explicit_residual_model()
        )

    def forward(self, hidden_vector: torch.Tensor, input_sequence: torch.Tensor, mask: torch.Tensor):
        sequence_previous = self.mask_applier_previous.fill_masked_sequence(input_sequence, mask)
        sequence_next = self.mask_applier_next.fill_masked_sequence(input_sequence, mask)
        alpha = self.layer_alpha(hidden_vector)
        sequence_interpolation = alpha * sequence_previous + (1 - alpha) * sequence_next
        sequence_synthesis = self.layer_synthesis(hidden_vector)
        beta = self.layer_beta(hidden_vector)
        predicted_sequence = beta * sequence_interpolation + (1 - beta) * sequence_synthesis
        return predicted_sequence
