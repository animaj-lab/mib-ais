from pydantic import Field

from motion_inbetweening.config.base import BaseModel


class BlockKeyframesScheduleDeterministicShift(BaseModel):
    """
    Class defining the parameters for the Deterministic Shift schedule modification.
    A deterministic shift means that all keyframes are shifted by the same fixed amount.


    Attributes:
        shift_size (int): The size of the shift to apply to the keyframes.
            A positive value indicates a right shift, while a negative value
            indicates a left shift.
    """

    shift_size: int


class BlockKeyframesScheduleRandomShift(BaseModel):
    """
    Class defining the parameters for the Random Shift schedule modification.
    A random shift means that each keyframe is shifted by an amount sampled uniformly between a min and a max value
    The values of the shifts can be positive (shift to the right) or negative (shift to the left).

    Attributes:
        min_shift_size (int): The minimum amount by which keyframes can be shifted.
        max_shift_size (int): The maximum amount by which keyframes can be shifted.
    """

    min_shift_size: int
    max_shift_size: int


class BlockKeyframesScheduleAddition(BaseModel):
    """
    Class defining the parameters for the Addition schedule modification.
    This modification adds keyframes to the schedule.
    The frames are added randomly in the schedule.
    The percentage of frames to add is given by the percentage attribute, based on the number of block keyframes.
    For instance, if percentage=10, and there are 80 block keyframes, 8 frames will be added.
    Note that this percentage can be greater than 100
    """

    percentage: int = Field(ge=0)


class BlockKeyframesScheduleDeletion(BaseModel):
    """
    Class defining the parameters for the Deletion schedule modification.
    This modification deletes keyframes in the schedule.
    The frames are deleted randomly in the schedule.
    The percentage of frames to delete is given by the percentage attribute, based on the number of block keyframes.
    For instance, if percentage=10, and there are 80 block keyframes, 8 frames will be deleted.
    Note that this percentage can not be greater than 100
    """

    percentage: int = Field(ge=0, le=100)


BlockKeyramesScheduleModifierConfig = (
    BlockKeyframesScheduleDeterministicShift
    | BlockKeyframesScheduleRandomShift
    | BlockKeyframesScheduleAddition
    | BlockKeyframesScheduleDeletion
)


class BlockScheduleProbabilisticAugmentation(BaseModel):
    """
    Represents a probabilistic augmentation configuration for block keyframe schedules.
    Attributes:
        modifier (BlockKeyramesScheduleModifierConfig): The configuration for modifying block keyframe schedules.
        proba (float): The probability of applying the augmentation. Must be a value between 0 and 1 (inclusive).
    """

    modifier: BlockKeyramesScheduleModifierConfig
    proba: float = Field(ge=0, le=1)


BlockScheduleAugmentation = list[BlockScheduleProbabilisticAugmentation]
