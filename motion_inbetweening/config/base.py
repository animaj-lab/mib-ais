from pathlib import Path
from typing import Self

from pydantic import BaseModel as PydanticBaseModel

from shared.utils import load_yaml


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True
        frozen = False
        extra = "forbid"

    @classmethod
    def from_yaml(cls, config_path: Path) -> Self:
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        config = load_yaml(config_path)
        return cls(**config)


class MutableBaseModel(BaseModel):
    class Config:
        frozen = False
