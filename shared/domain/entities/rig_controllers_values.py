import copy
from dataclasses import dataclass
from typing import Any, Self

from shared.domain.entities.animation import FrameID
from shared.domain.services.animation import validate_list_frame_ids

AttributeName = str
AttributeValue = float | int | bool | str
ControllerName = str


@dataclass
class ControllerValues:
    _data: dict[AttributeName, AttributeValue]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: AttributeName) -> AttributeValue:
        return self._data[item]

    @classmethod
    def set(cls, instance: Self, attribute_name: AttributeName, value: AttributeValue) -> Self:
        """Returns a new instance of ControllerValues with the updated value. The original instance is not modified.

        You can call this using the class name or the instance name:

        updated_controller_values = ControllerValues.set(controller_values, attribute_name, value)

        OR

        updated_controller_values = controller_values.set(controller_values, attribute_name, value)
        """
        data_copy = instance._data.copy()
        data_copy[attribute_name] = value
        return cls(data_copy)

    @classmethod
    def remove_attribute(cls, instance: Self, attribute_name: AttributeName) -> Self:
        """Removes the specified attribute from the instance and returns a new instance. The original instance is not
        modified.

        You can call this using the class name or the instance name:

        updated_controller_values = ControllerValues.remove_attribute(controller_values, attribute_name)

        OR

        updated_controller_values = controller_values.remove_attribute(controller_values, attribute_name)
        """
        data_copy = instance._data.copy()
        del data_copy[attribute_name]
        return cls(data_copy)

    def __contains__(self, item: AttributeName) -> bool:
        return item in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def to_deepcopy_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def get_attribute_names(self) -> list[AttributeName]:
        return list(self.keys())

    @classmethod
    def from_any(cls, input: Any) -> Self:
        """Validate and convert an input to a proper ControllerValues object.

        Validation includes:
            - Checking that the input is a dictionary.
            - Checking that the keys are castable to AttributeName.
            - Checking that the values are castable to a type included in AttributeValue.
        """
        if not isinstance(input, dict):
            raise ValueError("The controller values must be a dictionary.")
        data = {}
        for key, value in input.items():
            try:
                casted_key = AttributeName(key)
            except Exception as e:
                raise ValueError(f"Couldn't cast key to AttributeName: {key}") from e
            if not isinstance(value, AttributeValue.__args__):
                raise ValueError(f"Value for key '{key}' must be of type AttributeValue, got {type(value)}")
            data[casted_key] = value
        return cls(data)


@dataclass
class RigControllersValues:
    _data: dict[ControllerName, ControllerValues]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: ControllerName) -> ControllerValues:
        return self._data[item]

    @classmethod
    def set(
        cls, instance: Self, controller_name: ControllerName, attribute_name: AttributeName, value: AttributeValue
    ) -> Self:
        """Returns a new instance of RigControllersValues with the updated value. The original instance is not modified.

        You can call this using the class name or the instance name:

        updated_rig_controllers_values = RigControllersValues.set(
            rig_controllers_values, controller_name, attribute_name, value
        )

        OR

        updated_rig_controllers_values = rig_controllers_values.set(
            rig_controllers_values, controller_name, attribute_name, value
        )
        """
        data_copy = instance._data.copy()
        data_copy[controller_name] = ControllerValues.set(instance._data[controller_name], attribute_name, value)
        return cls(data_copy)

    @classmethod
    def fill_missing_with_other(cls, instance: Self, other: Self) -> Self:
        """Returns a new instance of RigControllersValues with the missing values filled from another instance.
        The original instance is not modified.

        Args:
            instance (Self): The instance to fill.
            other (Self): The other instance to fill from.

        Returns:
            Self: A new instance with the missing values filled.
        """
        data_copy = instance._data.copy()
        for controller_name, controller_values in other.items():
            if controller_name not in data_copy:
                data_copy[controller_name] = controller_values
                continue
            for attribute_name, attribute_value in controller_values.items():
                if attribute_name not in data_copy[controller_name]:
                    data_copy[controller_name] = ControllerValues.set(
                        data_copy[controller_name], attribute_name, attribute_value
                    )
                    continue
        return cls(data_copy)

    @classmethod
    def remove_attribute(cls, instance: Self, controller_name: ControllerName, attribute_name: AttributeName) -> Self:
        """Returns a new instance of RigControllersValues with the specified attribute removed. The original instance is
        not modified.

        You can call this using the class name or the instance name:

        updated_rig_controllers_values = RigControllersValues.remove_attribute(
            rig_controllers_values, controller_name, attribute_name
        )

        OR

        updated_rig_controllers_values = rig_controllers_values.remove_attribute(
            rig_controllers_values, controller_name, attribute_name
        )
        """
        data_copy = instance._data.copy()
        data_copy[controller_name] = ControllerValues.remove_attribute(data_copy[controller_name], attribute_name)
        return cls(data_copy)

    @classmethod
    def remove_controller(cls, instance: Self, controller_name: ControllerName) -> Self:
        """Returns a new instance of RigControllersValues with the specified controller removed. The original instance
        is not modified.

        You can call this using the class name or the instance name:

        updated_rig_controllers_values = RigControllersValues.remove_controller(
            rig_controllers_values, controller_name
        )

        OR

        updated_rig_controllers_values = rig_controllers_values.remove_controller(
            rig_controllers_values, controller_name
        )
        """
        data_copy = instance._data.copy()
        del data_copy[controller_name]
        return cls(data_copy)

    def __contains__(self, item: ControllerName) -> bool:
        return item in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def to_deepcopy_dict(self) -> dict:
        return {controller: controller_values.to_deepcopy_dict() for controller, controller_values in self.items()}

    def get_controller_names(self) -> list[ControllerName]:
        return list(self.keys())

    def to_controller_attribute_tuples(self) -> list[tuple[ControllerName, AttributeName]]:
        return [
            (controller, attribute)
            for controller in self._data.keys()
            for attribute in self._data[controller].get_attribute_names()
        ]

    @classmethod
    def from_any(cls, input: Any) -> Self:
        """Validate and convert an input to a proper AllRigControllersValues object.

        Validation includes:
            - Checking that the input is a dictionary.
            - Checking that the keys are castable to ControllerName.
            - Validating the values as ControllerValues
        """
        if not isinstance(input, dict):
            raise ValueError("The rig controllers values must be a dictionary.")
        data = {}
        for key, value in input.items():
            try:
                casted_key = ControllerName(key)
            except Exception as e:
                raise ValueError(f"Couldn't cast key to ControllerName: {key}") from e
            if not isinstance(value, ControllerValues):
                value = ControllerValues.from_any(value)
            data[casted_key] = value
        return cls(data)


@dataclass
class SceneRigControllersValues:
    """
    Class to hold the values of some controllers in the character rig for some frames in a scene.

    Use cases (not exhaustive):
        - Inference (we don't expect to have data for all the frames, and we don't expect to have data for all the
        rig controllers, only the ones that were learned by the model)
    """

    _data: dict[FrameID, RigControllersValues]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: FrameID) -> RigControllersValues:
        return self._data[item]

    @classmethod
    def set(
        cls,
        instance: Self,
        frame_id: FrameID,
        controller_name: ControllerName,
        attribute_name: AttributeName,
        value: AttributeValue,
    ) -> Self:
        """Returns a new instance of SceneRigControllersValues with the updated value. The original instance is not
        modified.

        You can call this using the class name or the instance name:

        updated_scene_rig_controllers_values = SceneRigControllersValues.set(
            scene_rig_controllers_values, frame_id, controller_name, attribute_name, value
        )

        OR

        updated_scene_rig_controllers_values = scene_rig_controllers_values.set(
            scene_rig_controllers_values, frame_id, controller_name, attribute_name, value
        )
        """
        data_copy = instance._data.copy()
        data_copy[frame_id] = RigControllersValues.set(instance._data[frame_id], controller_name, attribute_name, value)
        return cls(data_copy)

    @classmethod
    def fill_with_default_rig_controllers_values(
        cls, instance: Self, default_rig_controllers_values: RigControllersValues
    ) -> Self:
        """Returns a new instance of SceneRigControllersValues with the missing values filled from a default
        RigControllersValues instance. The original instance is not modified.

        You can call this using the class name or the instance name:

        updated_scene_rig_controllers_values = SceneRigControllersValues.fill_with_default_rig_controllers_values(
            scene_rig_controllers_values, default_rig_controllers_values
        )

        OR

        updated_scene_rig_controllers_values = scene_rig_controllers_values.fill_with_default_rig_controllers_values(
            scene_rig_controllers_values, default_rig_controllers_values
        )

        """
        filled_scene = {}
        for frame_id, rig_controllers_values in instance.items():
            filled_scene[frame_id] = RigControllersValues.fill_missing_with_other(
                rig_controllers_values, default_rig_controllers_values
            )
        return cls(filled_scene)

    def __contains__(self, item: FrameID) -> bool:
        return item in self._data

    def to_deepcopy_dict(self) -> dict:
        return {
            frame_id: rig_controllers_values.to_deepcopy_dict() for frame_id, rig_controllers_values in self.items()
        }

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def get_first_frame_id(self) -> FrameID:
        return min(self.keys())

    def get_last_frame_id(self) -> FrameID:
        return max(self.keys())

    def get_first_frame_rig_controllers_values(self) -> RigControllersValues:
        return self._data[self.get_first_frame_id()]

    @classmethod
    def from_any(cls, input: Any) -> Self:
        """Validate and convert a raw scene to a proper SceneRigControllersValues object.

        Validation includes:
            - Checking that the input is a dictionary.
            - Checking that the keys are castable to FrameID.
            - Checking that the sorted keys are consecutive.
            - Validating the values as AllRigControllersValues.

        """
        if not isinstance(input, dict):
            raise ValueError("The scene must be a dictionary.")
        validate_list_frame_ids(list(input.keys()), False)
        data = {}
        for frame_id, value in input.items():
            if not isinstance(value, RigControllersValues):
                value = RigControllersValues.from_any(value)
            data[FrameID(frame_id)] = value
        return cls(data)


class FullSceneRigControllersValues(SceneRigControllersValues):
    """
    Objects of this class are expected to have the controller values of the full character rig for all the frames

    Use case: holding ground truth animation data (we expect to have dense data)
    """

    def get_controller_names(self) -> list[ControllerName]:
        return self.get_first_frame_rig_controllers_values().get_controller_names()

    def to_controller_attribute_tuples(self) -> list[tuple[ControllerName, AttributeName]]:
        return self.get_first_frame_rig_controllers_values().to_controller_attribute_tuples()

    @classmethod
    def from_any(cls, input: Any) -> Self:
        """Validate that all the RigControllerValues have the same keys"""
        scene_rig_controllers_values = SceneRigControllersValues.from_any(input)
        validate_list_frame_ids(list(input.keys()), True)
        expected_keys = None
        for _, rig_controllers_values in scene_rig_controllers_values.items():
            if expected_keys is None:
                expected_keys = rig_controllers_values.keys()
                continue
            if rig_controllers_values.keys() != expected_keys:
                raise ValueError("All RigControllersValues must have the same keys.")
        return cls(scene_rig_controllers_values._data)
