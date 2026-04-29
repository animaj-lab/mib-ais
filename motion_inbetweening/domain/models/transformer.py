import math

import torch
import torch.nn as nn

from motion_inbetweening.config.model import TransformerModelConfig
from motion_inbetweening.domain.models.base_model import BaseModel


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \\text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \\text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \\text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        batch_size, seq_length, _ = x.size()
        x = x + self.pe[:, :seq_length, :]
        return self.dropout(x)


class TransformerModel(BaseModel):
    def __init__(self, config: TransformerModelConfig, pose_dim: int):
        super().__init__(pose_dim)
        self.positional_encoding = PositionalEncoding(d_model=config.d_model)
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.nheads = config.nhead
        self.input_projection = nn.Linear(self.pose_dim + 1, config.d_model)
        self.output_projection = nn.Linear(config.d_model, self.pose_dim)

    def forward(self, masked_pose_sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            masked_pose_seq: [batch_size, sequence_length, pose_dim+1]
            mask: [batch_size, sequence_length]
        """
        input_seq = self.input_projection(masked_pose_sequence)
        input_seq = self.positional_encoding(input_seq)
        sequence_mask = mask.bool()
        output = self.transformer(
            src=input_seq,
            tgt=input_seq,
            src_key_padding_mask=sequence_mask.squeeze(-1),
            tgt_key_padding_mask=sequence_mask.squeeze(-1),
        )
        output = self.output_projection(output)
        return output
