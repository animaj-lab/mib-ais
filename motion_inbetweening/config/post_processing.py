from enum import StrEnum, auto

from motion_inbetweening.config.base import BaseModel


class UnmaskedFramesPostProcessingStrategy(StrEnum):
    """Strategy for post-processing unmasked frames.

    For FORCE_INPUT and KEEP_PREDICTED, the non-trainable controllers are inserted in the output rig.

    Options:
        FORCE_INPUT: Replace the predicted unmasked frames by the input ones.
        KEEP_PREDICTED: Keep the predictions.
        REMOVE: Remove the unmasked frames altogether.

    """

    FORCE_INPUT = auto()
    KEEP_PREDICTED = auto()
    REMOVE = auto()


class PostProcessingConfig(BaseModel):
    do_controller_keyframes_reduction: bool
    threshold_controller_keyframes_reduction: float | None
