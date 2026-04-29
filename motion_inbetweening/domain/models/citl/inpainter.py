import torch
import torch.nn as nn

from motion_inbetweening.config.model import LastLayerMode
from motion_inbetweening.domain.models.ais_layer import AISLayer
from motion_inbetweening.domain.models.citl.transformers import Decoder, IntermediateEncoder, KeyframeEncoder


class Inpainter(nn.Module):
    def __init__(
        self,
        pose_dim,
        embed_size,
        max_length,
        heads,
        key_layers,
        interm_layers,
        dec_layers,
        dropout,
        last_layer_mode: LastLayerMode,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.pose_dim = pose_dim
        self.max_length = max_length
        self.heads = heads
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kf_encoder = KeyframeEncoder(embed_size, pose_dim, max_length, heads, key_layers, dropout).to(self.device)
        self.interm_encoder = IntermediateEncoder(embed_size, 16, max_length, heads, interm_layers, dropout).to(
            self.device
        )
        if last_layer_mode == LastLayerMode.ORIGINAL:
            self.last_layer_mode = LastLayerMode.ORIGINAL
            last_layer = None
        elif last_layer_mode == LastLayerMode.AIS:
            self.last_layer_mode = LastLayerMode.AIS
            self.ais_layer = AISLayer(embed_size, pose_dim).to(self.device)
            last_layer = self.ais_layer
        self.decoder = Decoder(embed_size, pose_dim, last_layer_mode, last_layer, heads, dec_layers, dropout).to(
            self.device
        )

    def _get_inverse_idx(self, idx, frames):
        inv_indices = []
        for b in range(idx.shape[0]):
            idx_ = torch.cat([torch.arange(frames), idx[b]], dim=0)
            unique, count = torch.unique(idx_, return_counts=True)
            inv_indices.append(unique[count == 1])
        for non_keyframes in inv_indices:
            assert len(non_keyframes) == frames - idx.shape[1], "Number of non-keyframes does not match expected count."
        inv_indices = torch.stack(inv_indices, dim=0)
        return inv_indices

    def forward(
        self,
        keyposes: torch.Tensor,
        keyframes: torch.Tensor,
        frames: int,
        ground_truth_sequence: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        keyframes = keyframes.cpu()
        if frames is None:
            frames = keyframes[0, -1] + 1
        input_data = torch.reshape(keyposes, (*keyposes.shape[:2], -1))
        interm_frames = self._get_inverse_idx(keyframes, frames)
        kf_enc = self.kf_encoder(input_data, keyframes)
        interm_enc = self.interm_encoder(interm_frames, kf_enc)
        output = self.decoder(
            interm_enc, interm_frames.to(self.device), kf_enc, keyframes.to(self.device), ground_truth_sequence, mask
        )
        return output
