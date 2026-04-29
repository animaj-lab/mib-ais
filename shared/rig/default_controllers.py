from pathlib import Path

import yaml


def load_default_controllers(yaml_file_path: Path) -> dict[str, dict[str, float | bool]]:
    with yaml_file_path.open() as f:
        default_controllers = yaml.safe_load(f)
    if not isinstance(default_controllers, dict):
        raise ValueError(f"Default controllers should be a dictionary. Got {type(default_controllers)}")
    for k, v in default_controllers.items():
        if not isinstance(k, str):
            raise ValueError(f"Controller name should be a string. Got {type(k)}")
        if not isinstance(v, dict):
            raise ValueError(f"Controller {k} should be a dictionary. Got {type(v)}")
        for k2, v2 in v.items():
            if not isinstance(k2, str):
                raise ValueError(f"Controller attribute name should be a string. Got {type(k2)}")
            if not (isinstance(v2, float) or isinstance(v2, bool) or isinstance(v2, int)):
                raise ValueError(f"Controller attribute value should be a float or a boolean. Got {type(v2)}")
    return default_controllers
