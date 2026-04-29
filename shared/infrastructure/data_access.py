import os
import pathlib
import shutil
import tempfile
from collections.abc import Iterable
from typing import Literal, Self

import pandas as pd

from shared.domain.services.data_access import ReadPath, ReadWritePath


class ReadOnlyFSPath(ReadPath):
    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)

    def get_name(self) -> str:
        return self.path.name

    def relative_to(self, parent_path: ReadPath) -> str:
        if not isinstance(parent_path, ReadOnlyFSPath):
            raise ValueError("parent_path must be a ReadOnlyFSPath")
        return str(self.path.relative_to(parent_path.path))

    def get_parent(self) -> Self:
        return self.__class__(self.path.parent)

    def get_parents(self, up_to_level: int | None) -> list[Self]:
        if up_to_level is not None and up_to_level < 1:
            raise ValueError("up_to_level must be a positive integer or None.")
        return [self.__class__(str(p)) for p in self.path.parents][:up_to_level]

    def get_stem(self) -> str:
        return self.path.stem

    def get_suffix(self) -> str:
        return self.path.suffix

    def get_parts(self) -> tuple[str, ...]:
        return self.path.parts

    def read_text(self) -> str:
        return self.path.read_text()

    def find_by_suffix(self, suffix: str) -> Iterable[Self]:
        return (self.__class__(p) for p in self.path.rglob(f"*{suffix}"))

    def find_by_suffixes(self, suffixes: list[str]) -> Iterable[Self]:
        return (self.__class__(p) for suffix in suffixes for p in self.path.rglob(f"*{suffix}"))

    def listdir(self, recursive: bool) -> Iterable[Self]:
        dir_path = self.path
        if recursive:
            return (self.__class__(p) for p in dir_path.rglob("*"))
        else:
            return (self.__class__(p) for p in dir_path.iterdir())

    def __truediv__(self, other: str) -> Self:
        return self.__class__(self.path / other)

    def find_by_prefix(self, prefix: str) -> Iterable[Self]:
        return (self.__class__(p) for p in self.path.glob(f"{prefix}*"))

    def find_by_prefixes(self, prefixes: list[str]) -> Iterable[Self]:
        return (self.__class__(p) for prefix in prefixes for p in self.path.glob(f"{prefix}*"))

    def exists(self) -> bool:
        return self.path.exists()

    def is_dir(self) -> bool:
        return self.path.is_dir()

    def is_file(self) -> bool:
        return self.path.is_file()

    def download_to(self, destination: pathlib.Path) -> None:
        """Download the file to the destination path.

        Note that it is the responsibility of the caller to ensure that the destination path is valid (i.e. the parent
        directory exists), and that the file does not already exist at the destination path.

        He also needs to ensure that the file is removed after use.
        """
        shutil.copyfile(self.path, destination)

    def download_directory_to(self, local_directory: pathlib.Path, override: bool) -> None:
        if not self.path.is_dir():
            raise NotADirectoryError("Source path must be a directory")
        if local_directory.is_file():
            raise NotADirectoryError("Destination path must be a directory")
        if local_directory.exists() and (not override):
            raise FileExistsError(f"Destination directory {local_directory} already exists")
        shutil.copytree(self.path, local_directory, dirs_exist_ok=True)

    def __repr__(self) -> str:
        return str(self.path)


class FSPath(ReadOnlyFSPath, ReadWritePath):
    def write_text(self, utf8_data: str, override: bool) -> None:
        if not override and os.path.exists(self.path):
            raise FileExistsError(f"File '{self.path}' already exists. Use override=True to overwrite.")
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(utf8_data)

    def write_bytes(self, data: bytes, override: bool) -> None:
        if not override and os.path.exists(self.path):
            raise FileExistsError(f"File '{self.path}' already exists. Use override=True to overwrite.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            f.write(data)

    def upload_from(self, path: pathlib.Path, override: bool) -> None:
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        if path.is_file():
            if self.path.exists() and (not override):
                raise FileExistsError(f"File '{self.path}' already exists. Use override=True to overwrite.")
            shutil.copyfile(path, self.path)
        else:
            raise ValueError("Use upload_directory_from to upload directories")

    def upload_directory_from(self, local_directory: pathlib.Path, override: bool) -> None:
        if local_directory.is_file():
            raise NotADirectoryError("Source path must be a directory")
        if self.path.is_file():
            raise NotADirectoryError("Destination path must be a directory")
        if self.path.exists() and (not override):
            raise FileExistsError(f"The directory {self.path} already exists")
        shutil.copytree(local_directory, self.path)

    def mkdir(self, parents: bool, exist_ok: bool) -> None:
        self.path.mkdir(parents=parents, exist_ok=exist_ok)


def load_dataframe(path: ReadPath | pathlib.Path, columns: list[str] | Literal["all"]) -> pd.DataFrame:
    """Load a pandas DataFrame

    The file format is inferred from the suffix.

    Args:
        path (Path): Path to load the DataFrame from

    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    if isinstance(path, pathlib.Path):
        return load_dataframe(FSPath(path), columns)
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = pathlib.Path(tmp_dir) / path.get_name()
        path.download_to(local_path)
        match path.get_suffix():
            case ".csv":
                if columns == "all":
                    return pd.read_csv(local_path)
                else:
                    return pd.read_csv(local_path, usecols=columns)
            case ".parquet":
                if columns == "all":
                    return pd.read_parquet(local_path)
                else:
                    return pd.read_parquet(local_path, columns=columns)
            case _:
                raise ValueError(f"Unsupported extension {path.get_suffix()}")
