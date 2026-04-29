from typing import Any

from shared.domain.entities.animation import FrameID, FrameInterval


def are_frame_intervals_equal(interval1: FrameInterval, interval2: FrameInterval) -> bool:
    return abs(interval1 - interval2) < 1e-06


def check_frames_consecutive(list_frame_ids: list[FrameID]) -> None:
    """Check if the list of frame ids are consecutive, ie if the difference between frame ids is always the same.

    Args:
        list_frame_ids (list[FrameID]): List of frame ids to check
    """
    sorted_frame_ids = sorted(list_frame_ids)
    first_frame_interval: FrameInterval = sorted_frame_ids[1] - sorted_frame_ids[0]
    for i, frame_id in enumerate(sorted_frame_ids[1:], start=1):
        previous_frame_id = sorted_frame_ids[i - 1]
        frame_interval: FrameInterval = frame_id - previous_frame_id
        if not are_frame_intervals_equal(frame_interval, first_frame_interval):
            raise ValueError(
                f"Frames are not consecutive: {previous_frame_id} -> {frame_id} "
                f"have a difference of {frame_interval} instead of {first_frame_interval}"
            )


def validate_list_frame_ids(list_unvalidated_frame_ids: list[Any], check_consecutive: bool) -> None:
    """Validate that a list of frame ids is valid.

    Validation includes:
        - Checking that the keys are castable to FrameID.
        - Checking that the sorted keys are consecutive (= that the difference between consecutive frame ids is always
          the same).

    Args:
        list_unvalidated_frame_ids (list[Any]): List of frame ids to validate
    """
    if not isinstance(list_unvalidated_frame_ids, list):
        raise ValueError("The frame ids must be a list.")
    frame_ids = [FrameID(frame_id) for frame_id in list_unvalidated_frame_ids]
    if check_consecutive:
        check_frames_consecutive(frame_ids)
