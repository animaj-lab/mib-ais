import logging
import math

import einops
import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from motion_inbetweening.domain.mask_applier import ConcatMode

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

    def forward(self, x, c):
        """
        Args:
            x: [n, nfeat, l]
            c: [n, ncond, 1]
        """
        scale, shift = c.chunk(2, dim=1)
        x = self.block1(x)
        x = ada_shift_scale(x, shift, scale)
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


def ada_shift_scale(x, shift, scale):
    return x * (1 + scale) + shift


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
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels * 2 if adagn else out_channels),
            Rearrange("batch t -> batch t 1"),
        )
        if adagn:
            nn.init.zeros_(self.time_mlp[1].weight)
            nn.init.zeros_(self.time_mlp[1].bias)
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        """
        x : [ batch_size x inp_channels x horizon ]
        t : [ batch_size x embed_dim ]
        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        cond = self.time_mlp(t)
        if self.adagn:
            out = self.blocks[0](x, cond)
        else:
            out = self.blocks[0](x) + cond
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        dim: int,
        dim_mults: tuple[int, int, int, int],
        attention: bool,
        adagn: bool,
        zero: bool,
        added_input_channels: int,
        added_output_channels: int,
    ):
        super().__init__()
        dims = [input_dim, *map(lambda m: int(dim * m), dim_mults)]
        logger.debug(f"dims: {dims} mults: {dim_mults}")
        in_out = list(zip(dims[:-1], dims[1:], strict=False))
        logger.debug(f"[ models/temporal ] Channel dimensions: {in_out}")
        time_dim = dim
        self.time_mlp = nn.Sequential(nn.Linear(cond_dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim))
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
                            dim_in + added_input_channels * is_first,
                            dim_out,
                            embed_dim=time_dim,
                            adagn=adagn,
                            zero=zero,
                        ),
                        ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, adagn=adagn, zero=zero),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, adagn=adagn, zero=zero)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, adagn=adagn, zero=zero)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= num_resolutions - 1
            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, adagn=adagn, zero=zero),
                        ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, adagn=adagn, zero=zero),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim_in, dim_in, kernel_size=5), nn.Conv1d(dim_in, input_dim + added_output_channels, 1)
        )
        if zero:
            nn.init.zeros_(self.final_conv[1].weight)
            nn.init.zeros_(self.final_conv[1].bias)

    def forward(self, x, cond):
        """
        x : [ seqlen x batch x dim ]
        cons: [ batch x cond_dim]
        """
        x = einops.rearrange(x, "s b d -> b d s")
        c = self.time_mlp(cond)
        h = []
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, c)
            x = resnet2(x, c)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, c)
        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, c)
            x = resnet2(x, c)
            x = attn(x)
            x = upsample(x)
        x = self.final_conv(x)
        x = einops.rearrange(x, "b d s -> s b d")
        return x


def cal_concat_multiple(in1, in2, multiple):
    """
    calculate the output channels of the concatenation of the two inputs while keeping the output channels a multiple of
    the given number
    """
    a = (in1 + in2) / multiple
    return int((1 - (a - math.floor(a))) * multiple + in1 + in2)


class TemporalUnetLarge(nn.Module):
    def __init__(
        self,
        input_dim,
        cond_dim,
        dim=256,
        dim_mults=(1, 2, 4, 8),
        out_mult=8,
        attention=False,
        adagn=False,
        zero=False,
        added_input_channels=0,
        final_type=1,
    ):
        super().__init__()
        dims = [input_dim, *map(lambda m: int(dim * m), dim_mults)]
        logger.debug(f"dims: {dims} mults: {dim_mults}")
        in_out = list(zip(dims[:-1], dims[1:], strict=False))
        logger.debug(f"[ models/temporal ] Channel dimensions: {in_out}")
        time_dim = dim
        self.time_mlp = nn.Sequential(nn.Linear(cond_dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim))
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
                            dim_in + added_input_channels * is_first,
                            dim_out,
                            embed_dim=time_dim,
                            adagn=adagn,
                            zero=zero,
                        ),
                        ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, adagn=adagn, zero=zero),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, adagn=adagn, zero=zero)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, adagn=adagn, zero=zero)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= num_resolutions - 1
            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, adagn=adagn, zero=zero),
                        ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, adagn=adagn, zero=zero),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
        final_in = cal_concat_multiple(dim_in, input_dim, out_mult)
        if final_type == 1:
            logger.debug("using final type 1")
            self.final_conv = nn.Sequential(
                nn.Conv1d(dim_in + input_dim, final_in, 1),
                nn.Conv1d(final_in, out_mult * input_dim, 5, padding=2, groups=out_mult),
                nn.Mish(),
                nn.Conv1d(out_mult * input_dim, input_dim, 1, groups=input_dim),
            )
        elif final_type == 2:
            logger.debug("using final type 2")
            self.final_conv = nn.Sequential(
                nn.Conv1d(dim_in + input_dim, final_in, 1),
                nn.Conv1d(final_in, out_mult * input_dim, 5, padding=2, groups=out_mult),
                nn.Mish(),
                nn.Conv1d(out_mult * input_dim, input_dim, 5, padding=2, groups=input_dim),
            )
        elif final_type == 3:
            logger.debug("using final type 3")
            self.final_conv = nn.Sequential(
                nn.Conv1d(dim_in + input_dim, final_in, 5, padding=2),
                nn.Conv1d(final_in, out_mult * input_dim, 5, padding=2, groups=out_mult),
                nn.Mish(),
                nn.Conv1d(out_mult * input_dim, input_dim, 5, padding=2, groups=input_dim),
            )
        else:
            raise NotImplementedError()
        if zero:
            nn.init.zeros_(self.final_conv[-1].weight)
            nn.init.zeros_(self.final_conv[-1].bias)

    def forward(self, x, cond):
        """
        x : [ seqlen x batch x dim ]
        cons: [ batch x cond_dim]
        """
        x = einops.rearrange(x, "s b d -> b d s")
        src = x
        c = self.time_mlp(cond)
        h = []
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, c)
            x = resnet2(x, c)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, c)
        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, c)
            x = resnet2(x, c)
            x = attn(x)
            x = upsample(x)
        x = torch.concat([x, src], dim=1)
        x = self.final_conv(x)
        x = einops.rearrange(x, "b d s -> s b d")
        return x


class MDM_UNET(nn.Module):
    """
    Diffuser's style UNET
    """

    def __init__(
        self,
        modeltype: str,
        pose_dim: int,
        latent_dim: int,
        dim_mults: tuple[int, int, int, int],
        attention: bool,
        ablation: dict | None,
        legacy: bool,
        emb_trans_dec: bool,
        adagn: bool,
        zero: bool,
        arch: str,
        unet_out_mult: int,
        keyframe_conditioned: bool,
        zero_keyframe_loss: bool,
        concat_mode: ConcatMode,
        learn_sigma: bool,
        final_type: int,
        **kwargs,
    ):
        super().__init__()
        self.legacy = legacy
        self.modeltype = modeltype
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.dim_mults = dim_mults
        self.attention = attention
        self.ablation = ablation
        self.action_emb = kwargs.get("action_emb", None)
        self.keyframe_conditioned = keyframe_conditioned
        self.zero_keyframe_loss = zero_keyframe_loss
        self.concat_mode = concat_mode
        if self.keyframe_conditioned:
            if self.concat_mode == ConcatMode.CONCAT_DIM:
                added_input_channels = 1
            elif self.concat_mode == ConcatMode.CONCAT_CHANNEL:
                added_input_channels = self.pose_dim
            elif self.concat_mode == ConcatMode.NONE:
                added_input_channels = 0
            else:
                raise NotImplementedError()
        else:
            added_input_channels = 0
        if learn_sigma:
            added_output_channels = self.pose_dim
        else:
            added_output_channels = 0
        self.normalize_output = kwargs.get("normalize_encoder_output", False)
        self.cond_mode = kwargs.get("cond_mode", "no_cond")
        self.cond_mask_prob = kwargs.get("cond_mask_prob", 0.0)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.emb_trans_dec = emb_trans_dec
        self.concat_mode = concat_mode
        logger.debug(f"Using UNET with latent dim: {self.latent_dim} and mults: {self.dim_mults}")
        if arch == "unet":
            self.unet = TemporalUnet(
                input_dim=self.pose_dim,
                cond_dim=self.latent_dim,
                dim=self.latent_dim,
                dim_mults=self.dim_mults,
                attention=self.attention,
                adagn=adagn,
                zero=zero,
                added_input_channels=added_input_channels,
                added_output_channels=added_output_channels,
            )
        elif arch == "unet_large":
            logger.debug(f"UNET large variation with output multiplier: {unet_out_mult}")
            self.unet = TemporalUnetLarge(
                input_dim=self.pose_dim,
                cond_dim=self.latent_dim,
                dim=self.latent_dim,
                dim_mults=self.dim_mults,
                attention=self.attention,
                adagn=adagn,
                zero=zero,
                out_mult=unet_out_mult,
                added_input_channels=added_input_channels,
                final_type=final_type,
            )
        else:
            raise NotImplementedError()
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith("clip_model.")]

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)
            return cond * (1.0 - mask)
        else:
            return cond

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y=None,
        obs_x0: torch.Tensor | None = None,
        obs_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            x: [batch_size, pose_dim, max_frames], denoted x_t in the paper
            timesteps: [batch_size] (int)
            y: dict of conditioning information
            obs_x0: [batch_size, pose_dim, max_frames], observed keyframes
            obs_mask: [batch_size, pose_dim, max_frames], mask for the observed keyframes. 1 for unmasked, 0 for masked

        Returns: [batch_size, pose_dim, max_frames]
        """
        assert (obs_x0 is None) == (
            obs_mask is None
        ), "with spatial-conditioning, both obs_x0 and obs_mask must be provided"
        if self.keyframe_conditioned:
            assert obs_x0 is not None, "obs_x0 must be provided for keyframe-conditioned model"
            assert obs_mask is not None, "obs_mask must be provided for keyframe-conditioned model"
            x = obs_x0 * obs_mask + x * ~obs_mask
            if self.concat_mode == ConcatMode.CONCAT_DIM:
                x = concatenate_dim(x, obs_mask)
            elif self.concat_mode == ConcatMode.CONCAT_CHANNEL:
                x = torch.cat([x, obs_mask], dim=1)
            elif self.concat_mode == ConcatMode.NONE:
                x = x
        return self.forward_core(x, timesteps, y)

    def forward_core(self, x: torch.Tensor, timesteps: list[int], y=None):
        """
        Args:
            x: [batch_size, pose_dim, max_frames], denoted x_t in the paper
            timesteps: [batch_size] (int)

        Returns: [batch_size, pose_dim, max_frames]
        """
        bs, pose_dim, nframes = x.shape
        emb = self.embed_timestep(timesteps)
        emb = emb.squeeze(0)
        x = x.permute((2, 0, 1))
        assert nframes == 224, f"the input should be 224 frames not {nframes}"
        x = self.unet(x, cond=emb)
        x = x.permute(1, 2, 0).float()
        return x


def concatenate_dim(x, obs_mask):
    assert x.shape == obs_mask.shape, "x and obs_mask must have the same shape"
    batch_size, pose_dim, num_frames = obs_mask.shape
    all_zeros = torch.all(obs_mask == 0, dim=1, keepdim=True)
    all_ones = torch.all(obs_mask == 1, dim=1, keepdim=True)
    new_vector = torch.where(
        all_zeros,
        torch.zeros_like(all_zeros, dtype=x.dtype),
        torch.where(all_ones, torch.ones_like(all_ones, dtype=x.dtype), torch.tensor(float("nan"))),
    )
    if torch.isnan(new_vector).any():
        raise ValueError("obs_mask contains frames with inconsistent values (not all 0s or all 1s along pose_dim)")
    final_output = torch.cat([x, new_vector], dim=1)
    return final_output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder
        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)
