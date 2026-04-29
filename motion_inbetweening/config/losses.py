from collections.abc import Callable
from typing import Self

import pydantic
import torch

from motion_inbetweening.config.base import MutableBaseModel


class LossesFrameWeights(MutableBaseModel):
    unmasked_frames: float
    masked_animation_keyframes: float
    masked_non_animation_keyframes: float


class LossesControllerFrameWeights(MutableBaseModel):
    """
    Weight the pose loss based on the GT values of the controller keyframes.
    For each frame for each controller this value indicates the probability that the frame
    is a keyframe.
    We design a function taking as input this probability and outputing a weight
    """

    weight_function: Callable[[torch.Tensor], torch.Tensor]


class LossesWeights(MutableBaseModel):
    pose: float
    speed: float
    acceleration: float
    jerk: float
    controller_keyframes_prediction: float
    frame_weights: LossesFrameWeights | None = None
    controller_keyframes_weights: LossesControllerFrameWeights | None = None

    @pydantic.model_validator(mode="after")
    def validate_weights(self) -> Self:
        if self.speed == 0 and self.pose != 1:
            raise ValueError("If speed coeff is 0, pose coeff has to be 1")
        return self


class L1Loss(MutableBaseModel):
    reduction: str


class MSELoss(MutableBaseModel):
    reduction: str


class BCEWithLogitsLoss(MutableBaseModel):
    reduction: str
    pos_weight: float


TorchLossFunction = L1Loss | MSELoss | BCEWithLogitsLoss


class LossesConfig(MutableBaseModel):
    pose_loss: TorchLossFunction
    speed_loss: TorchLossFunction
    acceleration_loss: TorchLossFunction
    jerk_loss: TorchLossFunction
    controller_keyframes_prediction_loss: TorchLossFunction
    weights: LossesWeights
    auto_weighting_loss: bool
