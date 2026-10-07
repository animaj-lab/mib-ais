"""History of the in-betweening results, so that the undo button can remove the predicted keys.

The history is a module global. It lives as long as the Maya session, because Maya imports the module one time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InferenceRecord:
    """The keys that one in-betweening wrote in the scene.

    Attributes:
        namespace (str): Namespace of the rig, "" if the rig has no namespace
        scene_rig_controllers_values (dict[int, dict]): Frame -> controller (without namespace) -> attribute -> value
    """

    namespace: str
    scene_rig_controllers_values: dict[int, dict]


_HISTORY: list[InferenceRecord] = []


def push(record: InferenceRecord) -> None:
    _HISTORY.append(record)


def pop() -> InferenceRecord | None:
    """Remove and return the last record, or return None if the history is empty."""
    if not _HISTORY:
        return None
    return _HISTORY.pop()
