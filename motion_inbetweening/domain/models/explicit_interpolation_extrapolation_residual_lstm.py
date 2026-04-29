import torch
import torch.nn as nn

from motion_inbetweening.config.model import LastLayerMode
from motion_inbetweening.domain.mask_applier import (
    ConcatMode,
    FillMode,
    MaskApplier,
    MaskApplierConfig,
    create_mask_appliers_for_explicit_residual_model,
)
from shared.rig.trainable_controllers import TrainableController


class ExplicitInterpolationExtrapolationResidualLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        dropout: float,
        last_layer_mode: LastLayerMode,
        trainable_controllers: list[TrainableController] | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.mask_applier_zero, self.mask_applier_previous, self.mask_applier_next = (
            create_mask_appliers_for_explicit_residual_model()
        )
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.layer_alpha = nn.Sequential(nn.Linear(hidden_size * 2, self.output_size), nn.Sigmoid())
        self.layer_beta = nn.Sequential(nn.Linear(hidden_size * 2, self.output_size), nn.Sigmoid())
        self.layer_synthesis = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2), nn.ReLU(), nn.Linear(hidden_size * 2, self.output_size)
        )
        self.last_layer_mode = last_layer_mode
        if self.last_layer_mode == LastLayerMode.SLERP:
            self.mask_applier_slerp = MaskApplier(
                config=MaskApplierConfig(concat_mode=ConcatMode.CONCAT_DIM, fill_mode=FillMode.SLERP)
            )
        elif self.last_layer_mode == LastLayerMode.LINEAR:
            self.mask_applier_linear = MaskApplier(
                config=MaskApplierConfig(concat_mode=ConcatMode.CONCAT_DIM, fill_mode=FillMode.LINEAR)
            )
        self.trainable_controllers = trainable_controllers

    def forward(
        self, sequence: torch.Tensor, mask: torch.Tensor, return_details=False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        sequence_zero = self.mask_applier_zero.apply_mask(sequence, mask)
        sequence_previous = self.mask_applier_previous.fill_masked_sequence(sequence, mask)
        sequence_next = self.mask_applier_next.fill_masked_sequence(sequence, mask)
        output_lstm, _ = self.lstm(sequence_zero)
        alpha = self.layer_alpha(output_lstm)
        sequence_interpolation = alpha * sequence_previous + (1 - alpha) * sequence_next
        sequence_synthesis = self.layer_synthesis(output_lstm)
        beta = self.layer_beta(output_lstm)
        match self.last_layer_mode:
            case LastLayerMode.DIRECT_SYNTHESIS:
                predicted_sequence = sequence_synthesis
            case LastLayerMode.LEARNED_INTERPOLATION_ONLY:
                predicted_sequence = sequence_interpolation
            case LastLayerMode.PREVIOUS:
                predicted_sequence = sequence_previous + sequence_synthesis
            case LastLayerMode.LINEAR:
                predicted_sequence = self.mask_applier_linear.fill_masked_sequence(sequence, mask) + sequence_synthesis
            case LastLayerMode.SLERP:
                predicted_sequence = (
                    self.mask_applier_slerp.fill_masked_sequence(
                        sequence, mask, trainable_controllers=self.trainable_controllers
                    )
                    + sequence_synthesis
                )
            case LastLayerMode.INTERP_PLUS_SYNTHESIS_NO_GATE:
                predicted_sequence = sequence_interpolation + sequence_synthesis
            case LastLayerMode.OFFSET:
                predicted_sequence = sequence_interpolation + beta * sequence_synthesis
            case LastLayerMode.FIXED_BETA:
                fixed_beta = 0.5
                predicted_sequence = fixed_beta * sequence_synthesis + (1 - fixed_beta) * sequence_interpolation
            case LastLayerMode.AIS:
                predicted_sequence = beta * sequence_synthesis + (1 - beta) * sequence_interpolation
        if not return_details:
            return (predicted_sequence, None)
        else:
            return (
                predicted_sequence,
                None,
                {
                    "alpha": alpha,
                    "beta": beta,
                    "sequence_interpolation": sequence_interpolation,
                    "sequence_synthesis": sequence_synthesis,
                    "sequence_predicted": predicted_sequence,
                    "sequence_previous": sequence_previous,
                    "sequence_next": sequence_next,
                    "ground_truth_sequence": sequence,
                    "mask": mask,
                },
            )
