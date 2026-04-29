from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from shared.domain.entities.animation import FrameID
from shared.domain.entities.rig_controllers_values import AttributeName, ControllerName, SceneRigControllersValues
from shared.domain.services.animation import validate_list_frame_ids


class AnimationCurveType(StrEnum):
    TRANSLATION = "translate"
    ROTATION = "rotate"
    SCALE = "scale"


NumpyFloatType = np.float16
NumpyAnimationCurve = np.ndarray


@dataclass
class AnimationCurve:
    curve: dict[FrameID, float]
    transformation: AnimationCurveType

    def get_first_frame_id(self) -> FrameID:
        return min(self.curve.keys())

    def get_last_frame_id(self) -> FrameID:
        return max(self.curve.keys())

    def to_numpy(self) -> NumpyAnimationCurve:
        return np.array(list(self.curve.values()), dtype=NumpyFloatType)


def validate_and_convert_to_animation_curve(
    animation_curve: Any, animation_curve_type: AnimationCurveType
) -> AnimationCurve:
    """Validate and convert a raw animation curve to a proper AnimationCurve object.

    Validation includes:
    - Checking that the input is a dictionary.
    - Checking that the keys are castable to FrameID.
    - Checking that the sorted keys are consecutive.
    - Checking that the values are castable to float, and cast and round them off.

    Args:
        animation_curve (Any)

    Returns:
        AnimationCurve
    """
    if not isinstance(animation_curve, dict):
        raise ValueError("The animation curve must be a dictionary.")
    validate_list_frame_ids(list(animation_curve.keys()), True)
    for frame_id, value in animation_curve.items():
        try:
            animation_curve[FrameID(frame_id)] = float(value)
        except Exception as e:
            raise ValueError(f"Value could not be casted to float: {value}") from e
    for frame_id, value in animation_curve.items():
        animation_curve[frame_id] = round(value, 6)
    return AnimationCurve(animation_curve, animation_curve_type)


def get_animation_curve_type(attribute_name: AttributeName) -> AnimationCurveType:
    if attribute_name.startswith("translate"):
        return AnimationCurveType.TRANSLATION
    if attribute_name.startswith("rotate"):
        return AnimationCurveType.ROTATION
    if attribute_name.startswith("scale"):
        return AnimationCurveType.SCALE
    raise ValueError(f"Cannot determine the animation curve type for attribute: {attribute_name}")


def get_animation_curve_from_scene_rig_controllers_values(
    scene_rig_controllers_values: SceneRigControllersValues,
    controller_name: ControllerName,
    attribute_name: AttributeName,
) -> AnimationCurve:
    """
    Get the animation curve for a given controller and attribute from a scene rig.
    """
    animation_curve = {}
    for frame_id, rig_controllers_values in scene_rig_controllers_values.items():
        if controller_name in rig_controllers_values.keys():
            controller = rig_controllers_values[controller_name]
            if attribute_name in controller.keys():
                animation_curve[frame_id] = controller[attribute_name]
    animation_curve_type = get_animation_curve_type(attribute_name)
    return validate_and_convert_to_animation_curve(animation_curve, animation_curve_type)


def select_non_plateau_frames_ids_from_curve(animation_curve: AnimationCurve, tolerance: float) -> list[FrameID]:
    """
    Identifies the FrameIDs of non-plateau frames in an animation curve.

    A plateau is defined as a sequence of consecutive frames where the values remain within a specified tolerance.
    This function returns the FrameIDs of frames that are not part of such plateaus, ensuring that significant
    changes in the curve are preserved.

    Args:
        animation_curve (AnimationCurve): The animation curve to analyze.
        tolerance (float): The tolerance value to determine plateaus.

    Returns:
        list[FrameID]: A list of FrameIDs corresponding to the non-plateau frames.
    """
    np_curve = animation_curve.to_numpy()
    frame_ids = list(animation_curve.curve.keys())
    non_plateau_frame_ids = [frame_ids[0]]
    for i in range(1, len(np_curve) - 1):
        if not (
            abs(np_curve[i] - np_curve[i - 1]) <= tolerance
            and abs(np_curve[i] - np_curve[i + 1]) <= tolerance
            and (abs(np_curve[i - 1] - np_curve[i + 1]) <= tolerance)
        ):
            non_plateau_frame_ids.append(frame_ids[i])
    non_plateau_frame_ids.append(frame_ids[-1])
    return non_plateau_frame_ids
