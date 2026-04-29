import logging

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor

from motion_inbetweening.config.model import UNetTransformerModelConfig
from motion_inbetweening.domain.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8, zero=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            nn.Mish(),
        )
        if zero:
            nn.init.zeros_(self.block[0].weight)
            nn.init.zeros_(self.block[0].bias)

    def forward(self, x):
        """
        Args:
            x: [n, c, l]
        """
        return self.block(x)


class Conv1dAdaGNBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
        )
        self.block2 = nn.Mish()

    def forward(self, x):
        """
        Args:
            x: [n, nfeat, l]
            c: [n, ncond, 1]
        """
        x = self.block1(x)
        x = self.block2(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** (-0.5)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: einops.rearrange(t, "b (h c) d -> b h c d", h=self.heads), qkv)
        q = q * self.scale
        k = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = einops.rearrange(out, "b h c d -> b (h c) d")
        return self.to_out(out)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=5, adagn=False, zero=False):
        super().__init__()
        self.adagn = adagn
        self.blocks = nn.ModuleList(
            [
                Conv1dAdaGNBlock(inp_channels, out_channels, kernel_size)
                if adagn
                else Conv1dBlock(inp_channels, out_channels, kernel_size),
                Conv1dBlock(out_channels, out_channels, kernel_size, zero=zero),
            ]
        )
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) if inp_channels != out_channels else nn.Identity()

    def forward(self, x):
        """
        x : [ batch_size x inp_channels x horizon ]
        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class UNetTransformerModel(BaseModel):
    def __init__(self, config: UNetTransformerModelConfig, pose_dim: int):
        super().__init__(pose_dim)
        self.input_dim = self.pose_dim + 1
        self.output_dim = self.pose_dim
        dims = [self.input_dim, *map(lambda m: int(config.dim * m), config.dim_mults)]
        logger.debug(f"dims: {dims} mults: {config.dim_mults}")
        in_out = list(zip(dims[:-1], dims[1:], strict=True))
        logger.debug(f"[ models/temporal ] Channel dimensions: {in_out}")
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= num_resolutions - 1
            is_first = ind == 0
            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_in + config.added_input_channels * is_first,
                            dim_out,
                            embed_dim=config.dim,
                            adagn=config.adagn,
                            zero=config.zero,
                        ),
                        ResidualTemporalBlock(
                            dim_out, dim_out, embed_dim=config.dim, adagn=config.adagn, zero=config.zero
                        ),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))) if config.attention else nn.Identity(),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=config.dim, adagn=config.adagn, zero=config.zero
        )
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if config.attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=config.dim, adagn=config.adagn, zero=config.zero
        )
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= num_resolutions - 1
            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_out * 2, dim_in, embed_dim=config.dim, adagn=config.adagn, zero=config.zero
                        ),
                        ResidualTemporalBlock(
                            dim_in, dim_in, embed_dim=config.dim, adagn=config.adagn, zero=config.zero
                        ),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))) if config.attention else nn.Identity(),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim_in, dim_in, kernel_size=5), nn.Conv1d(dim_in, self.output_dim, 1)
        )
        if config.zero:
            nn.init.zeros_(self.final_conv[1].weight)
            nn.init.zeros_(self.final_conv[1].bias)

    def forward(self, masked_pose_sequence: Tensor, mask: Tensor = None) -> torch.Tensor:
        """
        Args:
            masked_sequence: [batch_size, sequence_length, pose_dim+1]
        """
        x = einops.rearrange(masked_pose_sequence, "b s d -> b d s")
        h = []
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x)
            x = resnet2(x)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)
        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x)
            x = resnet2(x)
            x = attn(x)
            x = upsample(x)
        x = self.final_conv(x)
        x = einops.rearrange(x, "b d s -> b s d")
        return x
