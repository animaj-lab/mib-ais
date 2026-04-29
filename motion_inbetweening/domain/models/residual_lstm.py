import torch
import torch.nn as nn


class BidirectionalResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_layers = nn.ModuleList(
            [
                nn.LSTM(input_size if i == 0 else hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
                for i in range(num_layers)
            ]
        )
        self.input_projection = (
            nn.Linear(input_size, hidden_size * 2) if input_size != hidden_size * 2 else nn.Identity()
        )
        self.output_projection = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        residual = self.input_projection(x)
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            x = x + residual
            residual = x
        x = self.output_projection(x)
        return (x, None)
