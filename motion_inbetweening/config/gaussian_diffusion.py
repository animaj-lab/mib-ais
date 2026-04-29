import enum
from typing import Literal

import numpy as np
import pydantic
import torch

from motion_inbetweening.config.base import MutableBaseModel


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class CosineNoiseSchedule(MutableBaseModel):
    name: Literal["cosine"] = "cosine"
    offset: float
    exponent: float


class LinearNoiseSchedule(MutableBaseModel):
    name: Literal["linear"] = "linear"
    scale_beta: float


class SquareRootNoiseSchedule(MutableBaseModel):
    name: Literal["sqrt"] = "sqrt"
    s: float


NoiseSchedule = CosineNoiseSchedule | LinearNoiseSchedule


class GaussianDiffusionConfig(MutableBaseModel):
    betas: np.ndarray
    model_mean_type: ModelMeanType
    model_var_type: ModelVarType
    loss_type: LossType
    pose_loss_function: torch.nn.Module
    speed_loss_function: torch.nn.Module
    rescale_timesteps: bool
    lambda_pose: float
    lambda_speed: float
    use_random_proj: bool
    fp16: bool
    apply_zero_mask: bool
    time_weighted_loss: bool
    train_x0_as_eps: bool
    train_keypoint_mask: str


class SpacedDiffusionConfig(MutableBaseModel):
    noise_schedule: LinearNoiseSchedule | CosineNoiseSchedule | SquareRootNoiseSchedule = pydantic.Field(
        discriminator="name"
    )
    steps: int
    use_ddim: bool
    predict_xstart: bool
    learn_sigma: bool
    sigma_small: bool
    rescale_timesteps: bool
    lambda_pose: float
    lambda_vel: float
    use_random_proj: bool
    fp16: bool
    apply_zero_mask: bool
    time_weighted_loss: bool
    train_x0_as_eps: bool
    train_keypoint_mask: str
