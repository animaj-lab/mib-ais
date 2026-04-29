from pathlib import Path

import pydantic


class ControllerConfig(pydantic.BaseModel):
    default_controllers_path: Path
    trainable_controllers_path: Path
    simplify_classes_path: Path | None
