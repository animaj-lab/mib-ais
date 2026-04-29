import hashlib
from pathlib import Path
from typing import Any, TypeVar

from motion_inbetweening.config.base import BaseModel
from shared.domain.services.data_access import ReadPath, ReadWritePath

GenericPath = TypeVar("GenericPath", Path, ReadPath, ReadWritePath)


def get_hash(string: str) -> str:
    return hashlib.sha256(string.encode("utf-8")).hexdigest()


def get_hashed_dir_from_configs(root_path: GenericPath, list_configs: list[Any]) -> GenericPath:
    """Given a list of configs, return a directory path

    Note: The order of the configs is important in the hashing process.
    """
    full_str_to_hash = ""
    for config in list_configs:
        match config:
            case BaseModel():
                str_to_hash = config.model_dump_json()
            case _:
                try:
                    str_to_hash = str(config)
                except Exception as e:
                    raise ValueError(f"Could not cast config to str: {config}") from e
        full_str_to_hash += str_to_hash
    return root_path / get_hash(full_str_to_hash)
