import torch

from motion_inbetweening.config.model import LastLayerMode
from motion_inbetweening.domain.models.ais_layer import AISLayer
from motion_inbetweening.domain.models.delta_interpolator.modules.interpolator import InbetweenInterpolator
from motion_inbetweening.domain.models.delta_interpolator.modules.layers import (
    Embedding,
    FCBlock,
    LayerNorm,
    MultiHeadAttention,
)


class ResidualBlock(torch.nn.Module):
    """Fully connected residual block"""

    def __init__(self, num_layers: int, layer_width: int, dropout: float, size_in: int):
        super().__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.fc_layers = [torch.nn.Linear(size_in, layer_width)]
        self.relu_layers = [torch.nn.LeakyReLU(inplace=True)]
        if dropout > 0.0:
            self.fc_layers.append(torch.nn.Dropout(p=dropout))
            self.relu_layers.append(torch.nn.Identity())
        self.fc_layers += [torch.nn.Linear(layer_width, layer_width) for _ in range(num_layers - 1)]
        self.relu_layers += [torch.nn.LeakyReLU(inplace=True) for _ in range(num_layers - 1)]
        self.fc_layers = torch.nn.ModuleList(self.fc_layers)
        self.relu_layers = torch.nn.ModuleList(self.relu_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x
        for layer, relu in zip(self.fc_layers, self.relu_layers, strict=False):
            h = relu(layer(h))
        return h + x


class DeltaInterpolatorInfillTransformer(torch.nn.Module):
    """
    Fully-connected residual architechture with many categorical inputs wrapped in embeddings
    """

    def __init__(
        self,
        pose_dim: int,
        num_blocks_enc: int,
        num_layers_enc: int,
        layer_width_enc,
        num_blocks_dec: int,
        num_layers_dec: int,
        layer_width_dec: int,
        dropout: float,
        embedding_dim: int,
        embedding_size: int,
        embedding_num: int,
        num_heads: int,
        delta_mode: str,
        input_delta_mode: str,
        last_layer_mode: LastLayerMode,
        layer_norm: bool = True,
        eps: float = 1e-06,
    ):
        super().__init__()
        self.layer_width_enc = layer_width_enc
        self.size_in = pose_dim
        self.size_out = pose_dim
        self.size_out_stage1 = pose_dim
        self.num_heads = num_heads
        self.num_blocks_dec = num_blocks_dec
        if delta_mode in ["interpolator", "last_pose", "none"]:
            self.delta_mode = delta_mode
        else:
            raise ValueError(f"Delta mode {delta_mode} is not implemented")
        if input_delta_mode in ["last_pose", "none"]:
            self.input_delta_mode = input_delta_mode
        else:
            raise ValueError(f"Input Delta mode {input_delta_mode} is not implemented")
        self.input_projection_src = torch.nn.Linear(self.size_in + embedding_dim * embedding_num, layer_width_enc)
        self.input_projection_tgt = torch.nn.Linear(self.size_in + embedding_dim * embedding_num, layer_width_enc)
        self.interpolator = InbetweenInterpolator()
        self.embeddings = [
            Embedding(num_embeddings=embedding_size, embedding_dim=embedding_dim) for _ in range(embedding_num)
        ]
        self.encoder_blocks = [
            ResidualBlock(
                num_layers=num_layers_enc, layer_width=layer_width_enc, dropout=dropout, size_in=layer_width_enc
            )
            for _ in range(num_blocks_enc)
        ]
        self.mha = [
            MultiHeadAttention(in_features=layer_width_enc, head_num=self.num_heads, activation=None)
            for _ in range(num_blocks_enc)
        ]
        self.layer_norm = [LayerNorm(num_features=layer_width_enc) for _ in range(num_blocks_enc)]
        self.last_layer_mode = last_layer_mode
        if last_layer_mode == LastLayerMode.ORIGINAL:
            self.last_layer_mode = LastLayerMode.ORIGINAL
            last_layer = None
        elif last_layer_mode == LastLayerMode.AIS:
            self.last_layer_mode = LastLayerMode.AIS
            self.ais_layer = AISLayer(layer_width_enc, pose_dim)
            last_layer = self.ais_layer
        self.stage1_blocks = FCBlock(
            num_layers=num_layers_dec,
            layer_width=layer_width_dec,
            dropout=dropout,
            size_in=layer_width_enc,
            size_out=self.size_out_stage1,
            last_layer_mode=last_layer_mode,
            last_layer=last_layer,
        )
        self.model = torch.nn.ModuleList(
            self.encoder_blocks
            + self.embeddings
            + self.mha
            + self.layer_norm
            + [self.input_projection_src]
            + [self.input_projection_tgt]
        )

    def decode(
        self, pose_embedding: torch.Tensor, input_sequence: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        inputs:
        pose_embedding torch.Tensor : BxTxC
        """
        batch, n_frames = pose_embedding.shape[:2]
        _, stage1_forecast = self.stage1_blocks(pose_embedding, input_sequence, mask)
        return stage1_forecast.view(batch, n_frames, self.size_out_stage1)

    def forward(
        self,
        input_poses: torch.Tensor,
        interpolated_sequence: torch.Tensor,
        target_frame_indices: torch.Tensor,
        past_frame_indices: torch.Tensor,
        future_frame_indices: torch.Tensor,
        key_frame_indices: torch.Tensor,
        input_sequence: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Performs a forward pass through the InfillTransformer module to interpolate and refine
        joint positions for motion inbetweening.

        Args:
            input_sequence (torch.Tensor): The input sequence of joint positions with
            shape (batch_size, num_keyposes, pose_dim).
            input_frame_indices (torch.Tensor): Indices of the input frames in the sequence
            for each sequence of the batch
            target_frame_indices (torch.Tensor): Indices of the target frames to be interpolated
            past_frame_indices (torch.Tensor): Indices of the past frames used as context with
            future_frame_indices (torch.Tensor): Indices of the future frames used as context.
            key_frame_indices (torch.Tensor): Indices of the key frames in the sequence.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - stage1 (torch.Tensor): The first stage output of the interpolated joint positions.
                - stage2 (torch.Tensor): The second stage output of the refined joint positions.
        """
        batch_size, num_keyposes, pose_dim = input_poses.shape
        num_missing_frames = target_frame_indices.shape[1]
        src_time = torch.cat([past_frame_indices, key_frame_indices, future_frame_indices], dim=1).to(dtype=torch.int64)
        tgt_time = target_frame_indices.to(dtype=torch.int64)
        reference_pose = input_poses[:, 0, :].unsqueeze(1)
        if self.input_delta_mode == "last_pose":
            x_src = input_poses - reference_pose
            x_tgt = torch.zeros((batch_size, num_missing_frames, pose_dim)).to(input_poses)
        src_time_batch = src_time
        length = tgt_time.shape[1] * torch.ones_like(src_time_batch)
        ee_src = [x_src]
        for i, v in enumerate([src_time_batch, length]):
            ee_src.append(self.embeddings[i](v))
        x_src = torch.cat(ee_src, dim=-1)
        ee_tgt = [x_tgt]
        tgt_time_batch = tgt_time
        length = tgt_time.shape[1] * torch.ones_like(tgt_time_batch)
        for i, v in enumerate([tgt_time_batch, length]):
            ee_tgt.append(self.embeddings[i](v))
        x_tgt = torch.cat(ee_tgt, dim=-1)
        x_src = self.input_projection_src(x_src)
        x_tgt = self.input_projection_tgt(x_tgt)
        for i, block in enumerate(self.encoder_blocks):
            shortcut_src = x_src
            x_src = self.mha[i](q=x_src, k=x_src, v=x_src)
            x_src = self.layer_norm[i](x_src + shortcut_src)
            x_src = torch.relu(x_src)
            x_src = block(x_src)
            shortcut_tgt = x_tgt
            x_tgt = self.mha[i](q=x_tgt, k=x_src, v=x_src)
            x_tgt = self.layer_norm[i](x_tgt + shortcut_tgt)
            x_tgt = torch.relu(x_tgt)
            x_tgt = block(x_tgt)
        pose_embedding = torch.zeros((x_src.shape[0], src_time.shape[1] + tgt_time.shape[1], x_src.shape[2])).to(x_src)
        pose_embedding[:, src_time, ...] = x_src
        pose_embedding[:, tgt_time, ...] = x_tgt
        stage1 = self.decode(pose_embedding, input_sequence, mask)
        if self.last_layer_mode == LastLayerMode.ORIGINAL:
            stage1 = stage1 + interpolated_sequence
        elif self.last_layer_mode == LastLayerMode.AIS:
            stage1 = stage1
        else:
            raise ValueError(f"Last layer mode {self.last_layer_mode} is not implemented")
        return stage1
