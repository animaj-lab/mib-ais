"""HTTP client for the local inference server (`motion_inbetweening.scripts.serve`).

Uses the standard library only, so that Maya's Python needs no extra package.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_SERVER_URL = "http://127.0.0.1:8765"
SERVER_URL_ENV_VAR = "MIB_SERVER_URL"
REQUEST_TIMEOUT_SECONDS = 120
START_SERVER_COMMAND = "uv run python -m motion_inbetweening.scripts.serve"


def get_server_url() -> str:
    """Return the server URL from the MIB_SERVER_URL environment variable, or the default URL."""
    return os.environ.get(SERVER_URL_ENV_VAR, DEFAULT_SERVER_URL).rstrip("/")


def request_inbetween(scene_rig_controllers_values: dict[int, dict]) -> dict[int, dict]:
    """Send the keyed frames to the server and return the predicted in-between frames.

    Args:
        scene_rig_controllers_values (dict[int, dict]): Frame -> controller (without namespace) -> attribute -> value

    Returns:
        dict[int, dict]: The predicted frames, in the same format. The keyed frames are not in the output.
    """
    url = f"{get_server_url()}/v1/inbetween"
    data = json.dumps({str(frame): rig for frame, rig in scene_rig_controllers_values.items()}).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"The inference server returned an error ({e.code}): {_read_error(e)}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach the inference server at {get_server_url()} ({e.reason}).\n"
            f"Start it from the mib-ais repository root with:\n{START_SERVER_COMMAND}"
        ) from e
    except TimeoutError as e:
        raise RuntimeError(f"The inference server did not answer in {REQUEST_TIMEOUT_SECONDS} seconds.") from e

    return {int(frame): rig for frame, rig in body.items()}


def _read_error(error: urllib.error.HTTPError) -> str:
    raw_body = error.read().decode("utf-8")
    try:
        return json.loads(raw_body)["error"]
    except (ValueError, KeyError, TypeError):
        return raw_body
