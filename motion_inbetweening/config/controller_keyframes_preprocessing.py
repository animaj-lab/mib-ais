from enum import Enum, StrEnum, auto

from motion_inbetweening.config.base import BaseModel
from motion_inbetweening.domain.controller_keyframe_detectors import ErrorToleranceConfig
from shared.rig.trainable_controllers import Transformation


class ControllerKeyframeDetectionMethod(StrEnum):
    RDP = auto()
    MAYA_KEY_REDUCER = auto()


class RDPControllerKeyframeConfig(BaseModel):
    """Config to run RDP controller keyframe detection algorithm

    Attributes:
        method: SceneKeyframeDetectionMethod
        max_sequence_length: We apply the algorithm per chunk of fixed length (to avoid being dependent on the total
            length of the scene). This parameter defines the length of the chunks.
        error_tolerance: float. Used to specify how much error is allowed in the approximation.
    """

    method: ControllerKeyframeDetectionMethod = ControllerKeyframeDetectionMethod.RDP
    max_sequence_length: int
    error_tolerance: ErrorToleranceConfig


class PrecisionMode(Enum):
    ABSOLUTE = 0
    PERCENTAGE = 1


class MayaKeyReducerConfig(BaseModel):
    """Config to run maya key reducer

    Attributes:
        method: SceneKeyframeDetectionMethod
        precision_mode: PrecisionMode
        error_tolerance: float. In Absolute mode, it is the maximum error allowed. In percentage mode, it is the maximum
            percentage error allowed (we expect a value between 0 and 100 in relative mode).
    """

    method: ControllerKeyframeDetectionMethod = ControllerKeyframeDetectionMethod.MAYA_KEY_REDUCER
    precision_mode: PrecisionMode
    error_tolerance: ErrorToleranceConfig


ControllerKeyframeDetectionAlgorithmConfig = RDPControllerKeyframeConfig | MayaKeyReducerConfig


class AttributeKeyframeDivisionStategy(StrEnum):
    """
    AttributeKeyframeDivisionStategy is an enumeration that defines the strategies for dividing keyframes in animation.

    Attributes:
        ONE_FOR_EACH_ATTRIBUTE (str): Each attribute of the transformation has its own keyframe.
        ONE_FOR_ALL_ATTRIBUTES (str): A single keyframe is used for all attributes of the transformation.
    """

    ONE_FOR_EACH_ATTRIBUTE = "one_for_each_attribute"
    ONE_FOR_ALL_ATTRIBUTES = "one_for_all_attributes"


class TransformationName(StrEnum):
    ROTATION1D = "Rotation1D"
    ROTATION3D = "Rotation3D"
    TRANSLATION = "Translation"
    SCALE = "Scale"


def get_transformation_name(transformation: Transformation) -> TransformationName:
    return TransformationName(transformation.to_transformation_name())


TransformationsKeyframesDivisionStrategy = dict[TransformationName, AttributeKeyframeDivisionStategy]


class ControllerKeyframesPreprocessingConfig(BaseModel):
    """Config to run controller keyframe detection and vectorization on the whole dataset"""

    detection_algorithm: ControllerKeyframeDetectionAlgorithmConfig
    transformations_keyframe_division_strategy: TransformationsKeyframesDivisionStrategy
