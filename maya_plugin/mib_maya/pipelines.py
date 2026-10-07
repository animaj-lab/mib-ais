from __future__ import annotations

from mib_maya import client, history, maya_scene
from mib_maya.history import InferenceRecord
from mib_maya.progress import MayaProgressTracker

MIN_KEYED_FRAMES = 2


def do_inference_pipeline(progress_tracker: MayaProgressTracker) -> None:
    """Predict the in-between frames of the selected Pocoyo rig in the timeline range, and key them."""
    progress_tracker.update_progress(1, "Reading the keyed frames...")
    namespace = maya_scene.get_selected_rig_namespace()
    start, end = maya_scene.get_selection_time_range()
    input_rig_controllers_values = maya_scene.get_keyed_frames_rig_controllers_values(namespace, start, end)
    _validate_keyed_frames(sorted(input_rig_controllers_values), start, end)

    progress_tracker.update_progress(2, "Running the inference...")
    output_rig_controllers_values = client.request_inbetween(input_rig_controllers_values)

    progress_tracker.update_progress(3, "Writing the keys...")
    maya_scene.set_breakdown_keys(namespace, output_rig_controllers_values)
    history.push(InferenceRecord(namespace, output_rig_controllers_values))

    progress_tracker.update_progress(4, "Done")


def undo_inference_pipeline() -> bool:
    """Remove the keys of the last in-betweening. Return False if there is nothing to undo."""
    record = history.pop()
    if record is None:
        return False
    maya_scene.cut_keys(record.namespace, record.scene_rig_controllers_values)
    return True


def _validate_keyed_frames(keyed_frames: list[int], start: int, end: int) -> None:
    """Make sure that the keyed frames leave at least one frame to predict. Any key on a controller attribute,
    breakdowns included, makes a keyed frame."""
    if len(keyed_frames) < MIN_KEYED_FRAMES:
        raise RuntimeError(
            f"The range {start} -> {end} must contain at least {MIN_KEYED_FRAMES} keyed frames. "
            f"Keyed frames found: {keyed_frames}."
        )
    if keyed_frames[-1] - keyed_frames[0] + 1 == len(keyed_frames):
        raise RuntimeError(
            f"There is no frame to predict in the range {start} -> {end}: all the frames between the first and the "
            f"last keys have a key. Keyed frames found: {keyed_frames}.\n"
            "A key on any controller attribute, breakdowns included, makes a keyed frame."
        )
