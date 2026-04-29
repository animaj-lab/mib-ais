import copy
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

from shared.domain.entities.animation import FrameID
from shared.domain.entities.rig_controllers_values import AttributeName, ControllerName
from shared.domain.services.animation import validate_list_frame_ids

IsControllerKeyframeType = float


@dataclass
class ControllerKeyframes:
    _data: dict[AttributeName, IsControllerKeyframeType]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: AttributeName) -> IsControllerKeyframeType:
        return self._data[key]

    def __contains__(self, key: AttributeName) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def to_deepcopy_dict(self) -> dict:
        return copy.deepcopy(self._data)

    @classmethod
    def from_any(cls, input: Any) -> Self:
        """Validate and convert an input to a proper ControllerKeyframes object.

        Validation includes:
            - Checking that the input is a dictionary.
            - Checking that the keys are castable to AttributeName.
            - Checking that the values are castable to a type included in AttributeValue.
        """
        if not isinstance(input, dict):
            raise ValueError(f"The input must be a dictionary. Got {type(input)}")
        data = {}
        for key, value in input.items():
            try:
                casted_key = AttributeName(key)
            except Exception as e:
                raise ValueError(f"Couldn't cast key to AttributeName: {key}") from e
            if type(value) is bool:
                pass
            elif type(value) is float and 0 <= value <= 1:
                pass
            elif type(value) in (np.float32, np.float64):
                value = float(value)
            else:
                raise ValueError(
                    f"Invalid value for is_keyframe: {value} (type {type(value)}). "
                    f"Must be either a boolean or a float between 0 and 1."
                )
            data[casted_key] = value
        return cls(data)

    def to_dict(self) -> dict:
        return self._data


@dataclass
class RigControllersKeyframes:
    _data: dict[ControllerName, ControllerKeyframes]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: ControllerName) -> ControllerKeyframes:
        return self._data[key]

    def __contains__(self, key: ControllerName) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def to_deepcopy_dict(self) -> dict:
        return {
            controller_name: controller_keyframes.to_deepcopy_dict()
            for controller_name, controller_keyframes in self._data.items()
        }

    def to_controller_attribute_tuples(self) -> list[tuple[ControllerName, AttributeName]]:
        """Get a list of tuples containing the controller names and attribute names."""
        return [
            (controller_name, attribute_name)
            for controller_name in self.keys()
            for attribute_name in self[controller_name].keys()
        ]

    @classmethod
    def from_any(cls, input: Any) -> Self:
        """Validate and convert an input to a proper RigControllersKeyframes object.

        Validation includes:
            - Checking that the input is a dictionary.
            - Checking that the keys are castable to ControllerName.
            - Validating the values as ControllerKeyframes.
        """
        if not isinstance(input, dict):
            raise ValueError(f"The input must be a dictionary. Got: {type(input)}")
        data = {}
        for key, value in input.items():
            try:
                casted_key = ControllerName(key)
            except Exception as e:
                raise ValueError(f"Couldn't cast key to ControllerName: {key}") from e
            if not isinstance(value, ControllerKeyframes):
                value = ControllerKeyframes.from_any(value)
            data[casted_key] = value
        return cls(data)

    def to_dict(self) -> dict:
        return {
            controller_name: controller_keyframes.to_dict()
            for controller_name, controller_keyframes in self._data.items()
        }


@dataclass
class SceneRigControllersKeyframes:
    _data: dict[FrameID, RigControllersKeyframes]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: FrameID) -> RigControllersKeyframes:
        return self._data[key]

    def __contains__(self, key: FrameID) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def to_deepcopy_dict(self) -> dict:
        return {
            frame_id: rig_controllers_keyframes.to_deepcopy_dict()
            for frame_id, rig_controllers_keyframes in self.items()
        }

    def get_first_frame_id(self) -> FrameID:
        return min(self.keys())

    def get_last_frame_id(self) -> FrameID:
        return max(self.keys())

    def get_first_frame_rig_controller_keyframes(self) -> RigControllersKeyframes:
        return self._data[self.get_first_frame_id()]

    def to_controller_attribute_tuples(self) -> list[tuple[ControllerName, AttributeName]]:
        """Get a list of tuples containing the controller names and attribute names."""
        return self.get_first_frame_rig_controller_keyframes().to_controller_attribute_tuples()

    @classmethod
    def from_any(cls, input: Any) -> Self:
        """Validate and convert a raw scene to a proper SceneRigControllersKeyframes object.

        Validation includes:
            - Checking that the input is a dictionary.
            - Checking that the keys are castable to FrameID.
            - Checking that the sorted keys are consecutive.
            - Validating the values as RigControllersKeyframes.
        """
        if not isinstance(input, dict):
            raise ValueError("The scene must be a dictionary.")
        validate_list_frame_ids(list(input.keys()), True)
        data = {}
        for frame_id, value in input.items():
            if not isinstance(value, RigControllersKeyframes):
                value = RigControllersKeyframes.from_any(value)
            data[FrameID(frame_id)] = value
        return cls(data)

    def to_dict(self) -> dict:
        return {
            frame_id: rig_controllers_keyframes.to_dict() for frame_id, rig_controllers_keyframes in self._data.items()
        }
