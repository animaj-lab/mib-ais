from typing import Literal

from motion_inbetweening.config.base import BaseModel


class MDMConfig(BaseModel):
    pose_dim: int


class MDMUnetConfig(BaseModel):
    modeltype: str
    latent_dim: int
    dim_mults: tuple
    attention: bool
    ablation: str | None
    legacy: bool
    emb_trans_dec: bool
    adagn: bool
    zero: bool
    arch: Literal["unet", "unet_large"]
    unet_out_mult: int
    keyframe_conditioned: bool
    zero_keyframe_loss: bool
    final_type: int


class MDMDiTConfig(BaseModel):
    pose_dim: int


MotionDiffusionModelConfig = MDMConfig | MDMUnetConfig | MDMDiTConfig
