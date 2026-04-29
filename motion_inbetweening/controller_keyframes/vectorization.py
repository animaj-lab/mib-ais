from collections import defaultdict

import numpy as np

from motion_inbetweening.config.controller_keyframes_preprocessing import (
    AttributeKeyframeDivisionStategy,
    TransformationsKeyframesDivisionStrategy,
    get_transformation_name,
)
from motion_inbetweening.domain.controller_keyframes import RigControllersKeyframes
from shared.rig.trainable_controllers import TrainableController, transformation_to_attributes_names


def controller_keyframes_from_vector(
    trainable_controllers: list[TrainableController],
    transformations_keyframe_division_strategy: TransformationsKeyframesDivisionStrategy,
    vec: np.ndarray,
) -> RigControllersKeyframes:
    rig_controllers_keyframes_dict = defaultdict(dict)
    index = 0
    for trainable_controller in trainable_controllers:
        rig_controllers_keyframes_dict[trainable_controller.name] = {}
        for transformation in trainable_controller.transformations:
            if get_transformation_name(transformation) not in transformations_keyframe_division_strategy:
                raise ValueError(f"Missing keyframe division strategy for {transformation}")
            keyframe_division_strategy = transformations_keyframe_division_strategy[
                get_transformation_name(transformation)
            ]
            attribute_keys = transformation_to_attributes_names(transformation)
            if keyframe_division_strategy == AttributeKeyframeDivisionStategy.ONE_FOR_EACH_ATTRIBUTE:
                for key in attribute_keys:
                    rig_controllers_keyframes_dict[trainable_controller.name][key] = vec[index]
                    index += 1
            elif keyframe_division_strategy == AttributeKeyframeDivisionStategy.ONE_FOR_ALL_ATTRIBUTES:
                value = vec[index]
                for key in attribute_keys:
                    rig_controllers_keyframes_dict[trainable_controller.name][key] = value
                index += 1
    return RigControllersKeyframes.from_any(rig_controllers_keyframes_dict)
