from __future__ import annotations

from maya import cmds  # type: ignore


class MayaProgressTracker:
    """A progress bar window for a task in a fixed number of steps.

    Args:
        steps_number (int): The number of steps of the task. It must be greater than 0.
    """

    def __init__(self, steps_number: int) -> None:
        self.steps_number = steps_number
        self.window_id = "mibProgressWindow"
        self.close()  # no more than one instance

        cmds.window(self.window_id, title="Motion in-betweening", resizeToFitChildren=True)
        cmds.columnLayout(adjustableColumn=True)
        self.progress_control = cmds.progressBar(maxValue=steps_number, width=280)
        self.text = cmds.text(label="Please wait...", align="center")
        cmds.showWindow(self.window_id)

    def update_progress(self, step_index: int, step_label: str) -> None:
        """Show the step. The window closes when step_index reaches steps_number."""
        if not cmds.window(self.window_id, exists=True):
            return

        cmds.progressBar(self.progress_control, edit=True, progress=step_index)
        cmds.text(self.text, edit=True, label=step_label)
        cmds.refresh()

        if step_index >= self.steps_number:
            self.close()

    def close(self) -> None:
        if cmds.window(self.window_id, exists=True):
            cmds.deleteUI(self.window_id)
