"""Read and write the Pocoyo rig controller values in the Maya scene."""

from __future__ import annotations

import math

import maya.OpenMaya as om  # type: ignore
import maya.OpenMayaAnim as oma  # type: ignore
from maya import cmds, mel  # type: ignore

RIG_ROOT_NAME = "Pocoyo"
SELECT_CONTROLLER_INSTRUCTION = "Select a controller of the Pocoyo rig."


def get_selected_rig_namespace() -> str:
    """Return the namespace of the Pocoyo rig that contains the selected node.

    Returns:
        str: The namespace (for example "py_ch0001_1"), or "" if the rig has no namespace

    Raises:
        RuntimeError: If no node is selected, or if the namespace of the selected node has no Pocoyo rig
    """
    selection = cmds.ls(selection=True)
    if not selection:
        raise RuntimeError(f"Nothing is selected. {SELECT_CONTROLLER_INSTRUCTION}")

    namespace = selection[0].rpartition(":")[0]
    rig_root = _with_namespace(namespace, RIG_ROOT_NAME)
    if not cmds.objExists(rig_root):
        raise RuntimeError(
            f"The selected node '{selection[0]}' is not in a Pocoyo rig. {SELECT_CONTROLLER_INSTRUCTION}"
        )
    return namespace


def get_selection_time_range() -> tuple[int, int]:
    """Return the (start, end) frames of the range selected in the timeline, end included.

    If no range is selected, return the playback range.
    """
    playback_slider = mel.eval("$mibTmpPlaybackSlider = $gPlayBackSlider")
    if cmds.timeControl(playback_slider, query=True, rangeVisible=True):
        start, end = cmds.timeControl(playback_slider, query=True, rangeArray=True)
        end -= 1  # the end of rangeArray is excluded
    else:
        start = cmds.playbackOptions(query=True, minTime=True)
        end = cmds.playbackOptions(query=True, maxTime=True)

    start, end = int(round(start)), int(round(end))
    if start >= end:
        raise RuntimeError(f"Select a time range of at least 2 frames in the timeline. Got {start} -> {end}.")
    return start, end


def get_keyed_frames_rig_controllers_values(namespace: str, start: int, end: int) -> dict[int, dict]:
    """Return the rig controller values at each frame of [start, end] that has a key on a controller.

    Args:
        namespace (str): Namespace of the rig, "" if the rig has no namespace
        start (int): First frame of the range
        end (int): Last frame of the range, included

    Returns:
        dict[int, dict]: Frame -> controller (without namespace) -> attribute -> value. The rotations are in degrees.
    """
    controllers_attributes = _get_controllers_attributes(namespace)
    plugs = [
        f"{controller}.{attribute}"
        for controller, attributes in controllers_attributes.items()
        for attribute in attributes
    ]
    key_times = cmds.keyframe(plugs, query=True, timeChange=True, time=(start, end)) or []
    keyed_frames = sorted({int(round(key_time)) for key_time in key_times})

    current_time = oma.MAnimControl.currentTime()
    try:
        scene_rig_controllers_values = {}
        for frame in keyed_frames:
            oma.MAnimControl.setCurrentTime(om.MTime(frame, om.MTime.uiUnit()))
            scene_rig_controllers_values[frame] = _read_rig_controllers_values(controllers_attributes)
    finally:
        oma.MAnimControl.setCurrentTime(current_time)
    return scene_rig_controllers_values


def set_breakdown_keys(namespace: str, scene_rig_controllers_values: dict[int, dict]) -> None:
    """Write the values as breakdown keys on the rig controllers.

    Args:
        namespace (str): Namespace of the rig, "" if the rig has no namespace
        scene_rig_controllers_values (dict[int, dict]): Frame -> controller (without namespace) -> attribute -> value.
            The rotations are in degrees.

    Raises:
        RuntimeError: If a controller attribute does not exist in the scene. The function then writes no key.
    """
    _validate_plugs_exist(namespace, scene_rig_controllers_values)

    current_time = oma.MAnimControl.currentTime()
    try:
        for frame, rig_controllers_values in scene_rig_controllers_values.items():
            plugs = _set_frame_keys(namespace, frame, rig_controllers_values)
            cmds.keyframe(plugs, time=(frame, frame), breakdown=True)
    finally:
        oma.MAnimControl.setCurrentTime(current_time)


def cut_keys(namespace: str, scene_rig_controllers_values: dict[int, dict]) -> None:
    """Remove the keys of the given controller attributes at the given frames."""
    for frame, rig_controllers_values in scene_rig_controllers_values.items():
        plugs = [
            f"{_with_namespace(namespace, controller)}.{attribute}"
            for controller, attributes in rig_controllers_values.items()
            for attribute in attributes
        ]
        existing_plugs = [plug for plug in plugs if cmds.objExists(plug)]
        if existing_plugs:
            cmds.cutKey(existing_plugs, time=(frame, frame), clear=True)


def _with_namespace(namespace: str, name: str) -> str:
    return f"{namespace}:{name}" if namespace else name


def _get_controllers_attributes(namespace: str) -> dict[str, list[str]]:
    """Return the keyable attributes of each controller of the rig. A controller is a transform with a nurbsCurve
    shape. The shading nodes are skipped."""
    rig_root = _with_namespace(namespace, RIG_ROOT_NAME)
    shapes = cmds.listRelatives(rig_root, allDescendents=True, fullPath=True, type="nurbsCurve") or []

    controllers_attributes = {}
    for shape in shapes:
        if "shd" in shape.lower():
            continue
        controller = cmds.listRelatives(shape, parent=True, fullPath=True, type="transform")[0]
        controllers_attributes[controller] = cmds.listAttr(controller, keyable=True) or []

    if not controllers_attributes:
        raise RuntimeError(f"No controller found under '{rig_root}'.")
    return controllers_attributes


def _read_rig_controllers_values(controllers_attributes: dict[str, list[str]]) -> dict[str, dict]:
    """Read the controller values at the current time. Keep the numeric and boolean values only."""
    rig_controllers_values = {}
    for controller, attributes in controllers_attributes.items():
        controller_values = {}
        for attribute in attributes:
            value = cmds.getAttr(f"{controller}.{attribute}")
            if isinstance(value, bool | int | float):
                controller_values[attribute] = value
        short_name = controller.split("|")[-1].split(":")[-1]
        rig_controllers_values[short_name] = controller_values
    return rig_controllers_values


def _validate_plugs_exist(namespace: str, scene_rig_controllers_values: dict[int, dict]) -> None:
    missing_plugs = set()
    for rig_controllers_values in scene_rig_controllers_values.values():
        for controller, attributes in rig_controllers_values.items():
            for attribute in attributes:
                plug = f"{_with_namespace(namespace, controller)}.{attribute}"
                if not cmds.objExists(plug):
                    missing_plugs.add(plug)
    if missing_plugs:
        raise RuntimeError(f"These controller attributes do not exist in the scene: {sorted(missing_plugs)}")


def _set_frame_keys(namespace: str, frame: int, rig_controllers_values: dict[str, dict]) -> list[str]:
    """Add a key on each controller attribute at the frame, with the Maya API for speed. Return the keyed plugs."""
    time = om.MTime(frame, om.MTime.uiUnit())
    oma.MAnimControl.setCurrentTime(time)

    keyed_plugs = []
    for controller, attributes in rig_controllers_values.items():
        node = _with_namespace(namespace, controller)
        selection_list = om.MSelectionList()
        selection_list.add(node)
        node_object = om.MObject()
        selection_list.getDependNode(0, node_object)
        dependency_node = om.MFnDependencyNode(node_object)

        for attribute, value in attributes.items():
            # MFnAnimCurve takes values in internal units: the angles are in radians
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            elif "rotate" in attribute:
                value = math.radians(value)

            plug = dependency_node.findPlug(attribute, False)
            anim_curve = _find_or_create_anim_curve(plug)
            anim_curve.addKey(time, float(value), oma.MFnAnimCurve.kTangentGlobal, oma.MFnAnimCurve.kTangentGlobal)
            keyed_plugs.append(f"{node}.{attribute}")
    return keyed_plugs


def _find_or_create_anim_curve(plug: om.MPlug) -> oma.MFnAnimCurve:
    connected_plugs = om.MPlugArray()
    plug.connectedTo(connected_plugs, True, False)  # sources only
    for i in range(connected_plugs.length()):
        connected_node = connected_plugs[i].node()
        if connected_node.hasFn(om.MFn.kAnimCurve):
            return oma.MFnAnimCurve(connected_node)

    anim_curve = oma.MFnAnimCurve()
    anim_curve.create(plug)
    return anim_curve
