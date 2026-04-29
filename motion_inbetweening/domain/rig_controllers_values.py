from shared.domain.entities.animation import FrameID
from shared.domain.entities.rig_controllers_values import (
    AttributeName,
    AttributeValue,
    ControllerName,
    ControllerValues,
    FullSceneRigControllersValues,
    RigControllersValues,
    SceneRigControllersValues,
)


def controller_values_to_dict(controllers_values: ControllerValues) -> dict[AttributeName, AttributeValue]:
    """Returns a dictionary representation of the ControllerValues object.

    CAUTION: The returned dictionary is a shallow copy of the original data. Modifying the returned dictionary will
    affect the original ControllerValues object. Use this function only in one of the following cases:
    - You won't be using the original object anymore.
    - You can guarantee that the returned dictionary will not be modified.

    Args:
        controllers_values (ControllerValues): The ControllerValues object to convert.

    Returns:
        dict: A dictionary representation of the ControllerValues object.
    """
    return controllers_values._data


def rig_controllers_values_to_dict(
    rig_controllers_values: RigControllersValues,
) -> dict[ControllerName, dict[AttributeName, AttributeValue]]:
    """Returns a dictionary representation of the RigControllersValues object.

    CAUTION: The returned dictionary is a shallow copy of the original data. Modifying the returned dictionary will
    affect the original RigControllersValues object. Use this function only in one of the following cases:
    - You won't be using the original object anymore.
    - You can guarantee that the returned dictionary will not be modified.

    Args:
        rig_controllers_values (RigControllersValues): The RigControllersValues object to convert.

    Returns:
        dict: A dictionary representation of the RigControllersValues object.
    """
    return {
        controller_name: controller_values_to_dict(controller_values)
        for controller_name, controller_values in rig_controllers_values.items()
    }


def scene_rig_controllers_values_to_dict(
    scene_rig_controllers_values: SceneRigControllersValues,
) -> dict[FrameID, dict[ControllerName, dict[AttributeName, AttributeValue]]]:
    """Returns a dictionary representation of the SceneRigControllersValues object.

    CAUTION: The returned dictionary is a shallow copy of the original data. Modifying the returned dictionary will
    affect the original SceneRigControllersValues object. Use this function only in one of the following cases:
    - You won't be using the original object anymore.
    - You can guarantee that the returned dictionary will not be modified.

    Args:
        scene_rig_controllers_values (SceneRigControllersValues): The SceneRigControllersValues object to convert.

    Returns:
        dict: A dictionary representation of the SceneRigControllersValues object.
    """
    return {
        scene_name: rig_controllers_values_to_dict(rig_controllers_values)
        for scene_name, rig_controllers_values in scene_rig_controllers_values.items()
    }


def remove_frames_from_scene_rig_controllers_values(
    scene: SceneRigControllersValues, frame_ids: list[FrameID]
) -> SceneRigControllersValues:
    """Returns a new SceneRigControllersValues object with the specified frames removed.

    Args:
        scene (SceneRigControllersValues): The original SceneRigControllersValues object.
        frame_ids (list[FrameID]): The list of frame IDs to remove.

    Returns:
        SceneRigControllersValues: A new SceneRigControllersValues object with the specified frames removed.
    """
    return SceneRigControllersValues(
        {frame_id: scene[frame_id] for frame_id in scene.keys() if frame_id not in frame_ids}
    )


def scene_rig_controllers_values_to_controller_attribute_tuples(
    scene: SceneRigControllersValues,
) -> list[tuple[ControllerName, AttributeName]]:
    """Given a SceneRigControllersValues, returns a list of tuples (ControllerName, AttributeName) that represent
    the controller names and attribute names present in the scene.

    Note that non-dense scenes (i.e., scenes where not each frame has the same controller attributes) will work; we will
    just return the union of all the controller attributes present in the scene.

    Args:
        scene (SceneRigControllersValues): A SceneRigControllersValues object.
    """
    if isinstance(scene, FullSceneRigControllersValues):
        return scene.to_controller_attribute_tuples()
    if len(scene) == 0:
        return []
    set_controller_attribute_tuples = set()
    for rig_controllers_values in scene.values():
        if len(set_controller_attribute_tuples) == 0:
            set_controller_attribute_tuples = set(rig_controllers_values.to_controller_attribute_tuples())
            continue
        frame_controller_attribute_tuples = rig_controllers_values.to_controller_attribute_tuples()
        for controller_attribute_tuple in frame_controller_attribute_tuples:
            set_controller_attribute_tuples.add(controller_attribute_tuple)
    return sorted(set_controller_attribute_tuples)
