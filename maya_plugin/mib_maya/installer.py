"""Add the motion in-betweening buttons to the current Maya shelf.

`install.mel` runs this file when you drop it in the Maya viewport.
"""

from __future__ import annotations

import sys
from pathlib import Path

from maya import cmds, mel  # type: ignore

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
ICONS_DIRECTORY = PLUGIN_ROOT / "icons"
PLUGIN_COMMAND_MARKER = "import mib_maya.commands"


def install() -> None:
    shelf_top_level = mel.eval("$mibTmpShelfTopLevel = $gShelfTopLevel")
    shelf = cmds.tabLayout(shelf_top_level, query=True, selectTab=True)

    installed_buttons = _find_installed_buttons(shelf_top_level)
    for button in installed_buttons:
        cmds.deleteUI(button)
    _unload_plugin_modules()

    _create_shelf_button(
        shelf,
        "do_inference",
        "In-between Pocoyo",
        "MOTION_IN_BETWEENING_POCOYO.png",
        "Motion in-betweening: predict the Pocoyo frames between the keys of the selected time range.\n"
        "Select a Pocoyo controller first. The local inference server must run.",
    )
    _create_shelf_button(
        shelf,
        "undo_inference",
        "Undo",
        "MOTION_IN_BETWEENING_UNDO.png",
        "Motion in-betweening: remove the keys of the last in-betweening.",
    )
    _show_install_panel(is_update=bool(installed_buttons), shelf=shelf)


def _find_installed_buttons(shelf_top_level: str) -> list[str]:
    """Return the full paths of the plugin buttons on all the shelves."""
    installed_buttons = []
    for shelf in cmds.shelfTabLayout(shelf_top_level, query=True, childArray=True) or []:
        shelf_path = f"{shelf_top_level}|{shelf}"
        for child in cmds.shelfLayout(shelf_path, query=True, childArray=True) or []:
            button = f"{shelf_path}|{child}"
            if cmds.shelfButton(button, exists=True) and PLUGIN_COMMAND_MARKER in (
                cmds.shelfButton(button, query=True, command=True) or ""
            ):
                installed_buttons.append(button)
    return installed_buttons


def _unload_plugin_modules() -> None:
    """Remove the plugin modules from the Python session, so that the next click imports the new code. This also
    clears the undo history of the plugin."""
    for module_name in [name for name in sys.modules if name == "mib_maya" or name.startswith("mib_maya.")]:
        del sys.modules[module_name]


def _show_install_panel(is_update: bool, shelf: str) -> None:
    title = "Plugin updated" if is_update else "Plugin installed"
    message = f"Motion in-betweening: the shelf buttons are on the '{shelf}' shelf."
    if is_update:
        message += "\nThe buttons now use the new code. The undo history of the plugin is cleared."
    print(f"{title}. {message}")
    cmds.confirmDialog(title=title, message=message, button=["OK"], defaultButton="OK", icon="information")


def _create_shelf_button(shelf: str, command_name: str, label: str, icon_file_name: str, annotation: str) -> None:
    """Create a shelf button that calls mib_maya.commands.<command_name>."""
    command = (
        "import sys\n"
        f"plugin_root = r'{PLUGIN_ROOT}'\n"
        "if plugin_root not in sys.path:\n"
        "    sys.path.insert(0, plugin_root)\n"
        f"{PLUGIN_COMMAND_MARKER}\n"
        f"mib_maya.commands.{command_name}()\n"
    )
    icon_path = ICONS_DIRECTORY / icon_file_name
    image = str(icon_path) if icon_path.exists() else "commandButton.png"  # Maya default icon

    cmds.shelfButton(
        parent=shelf,
        command=command,
        sourceType="python",
        image=image,
        image1=image,
        label=label,
        annotation=annotation,
        useAlpha=True,
    )


install()
