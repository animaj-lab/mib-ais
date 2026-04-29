import pathlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Self


class ReadPath(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def relative_to(self, parent_path: "ReadPath") -> str:
        pass

    @abstractmethod
    def get_parent(self) -> Self:
        pass

    @abstractmethod
    def get_stem(self) -> str:
        pass

    @abstractmethod
    def get_parts(self) -> tuple[str, ...]:
        pass

    @abstractmethod
    def read_text(self) -> str:
        pass

    @abstractmethod
    def find_by_suffix(self, suffix: str) -> Iterable[Self]:
        pass

    @abstractmethod
    def find_by_suffixes(self, suffixes: list[str]) -> Iterable[Self]:
        pass

    @abstractmethod
    def find_by_prefix(self, prefix: str) -> Iterable[Self]:
        pass

    @abstractmethod
    def find_by_prefixes(self, prefixes: list[str]) -> Iterable[Self]:
        pass

    @abstractmethod
    def listdir(self, recursive: bool) -> Iterable[Self]:
        pass

    @abstractmethod
    def __truediv__(self, other: str | Self) -> Self:
        pass

    @abstractmethod
    def exists(self) -> bool:
        pass

    @abstractmethod
    def is_dir(self) -> bool:
        pass

    @abstractmethod
    def is_file(self) -> bool:
        pass

    @abstractmethod
    def download_to(self, destination: pathlib.Path) -> None:
        pass

    @abstractmethod
    def download_directory_to(self, local_directory: pathlib.Path, override: bool) -> None:
        pass

    @abstractmethod
    def get_suffix(self) -> str:
        pass


class ReadWritePath(ReadPath):
    @abstractmethod
    def write_text(self, utf8_data: str, override: bool) -> None:
        pass

    @abstractmethod
    def write_bytes(self, data: bytes, override: bool) -> None:
        pass

    @abstractmethod
    def upload_from(self, path: pathlib.Path, override: bool) -> None:
        pass

    @abstractmethod
    def upload_directory_from(self, local_directory: pathlib.Path, override: bool) -> None:
        pass

    @abstractmethod
    def mkdir(self, parents: bool, exist_ok: bool) -> None:
        pass
