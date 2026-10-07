"""Entry points of the shelf buttons."""

from __future__ import annotations

import traceback

from maya import cmds  # type: ignore

from mib_maya.pipelines import do_inference_pipeline, undo_inference_pipeline
from mib_maya.progress import MayaProgressTracker

INFERENCE_STEPS_NUMBER = 4


def do_inference() -> None:
    progress_tracker = MayaProgressTracker(INFERENCE_STEPS_NUMBER)
    try:
        do_inference_pipeline(progress_tracker)
    except Exception as e:
        traceback.print_exc()
        _show_error_dialog(str(e))
    finally:
        progress_tracker.close()


def undo_inference() -> None:
    try:
        if not undo_inference_pipeline():
            cmds.inViewMessage(
                assistMessage="Motion in-betweening: there is nothing to undo.", position="midCenter", fade=True
            )
    except Exception as e:
        traceback.print_exc()
        _show_error_dialog(str(e))


def _show_error_dialog(message: str) -> None:
    cmds.confirmDialog(
        title="Motion in-betweening", message=message, button=["OK"], defaultButton="OK", icon="critical"
    )
