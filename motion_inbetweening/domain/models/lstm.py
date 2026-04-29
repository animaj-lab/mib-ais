import torch
import torch.nn as nn

from motion_inbetweening.config.model import BiDirectionalLSTMModelConfig
from motion_inbetweening.domain.models.base_model import BaseModel


class BiDirectionalLSTMModel(BaseModel):
    def __init__(
        self,
        config: BiDirectionalLSTMModelConfig,
        pose_dim: int,
        do_controller_keyframes_prediction: bool,
        controller_keyframes_dim: int,
    ) -> None:
        super().__init__(pose_dim)
        self.input_size = self.pose_dim + 1
        self.output_size = self.pose_dim
        self.do_controller_keyframes_prediction = do_controller_keyframes_prediction
        self.controller_keyframes_dim = controller_keyframes_dim
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=True,
        )
        self.last_layer = nn.Linear(config.hidden_size * 2, self.output_size)
        self.do_controller_keyframes_prediction = do_controller_keyframes_prediction
        if self.do_controller_keyframes_prediction:
            self.controller_keyframes_dim = controller_keyframes_dim
            self.controller_keyframes_prediction_layers = nn.Linear(
                config.hidden_size * 2, self.controller_keyframes_dim
            )

    def forward(
        self, masked_sequence: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            masked_sequence: [batch_size, sequence_length, pose_dim+1]
        """
        outputs, _ = self.lstm(masked_sequence)
        pose_prediction = self.last_layer(outputs)
        if self.do_controller_keyframes_prediction:
            controller_keyframes_outputs = self.controller_keyframes_prediction_layers(outputs)
            return (pose_prediction, controller_keyframes_outputs)
        return (pose_prediction, None)
