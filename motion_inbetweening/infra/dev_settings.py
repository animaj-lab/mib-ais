from functools import lru_cache
from pathlib import Path

import pydantic_core
import pydantic_settings

from shared.domain.entities.ip import CharacterName


class DevSettings(pydantic_settings.BaseSettings):
    """Settings dependent on the environment you run the scripts in.

    Some ways to set the variables are:
        - via environment variables (e.g. MIB_DATASET_DIR)
        - via a .env file (! environment variables take precedence over this method)
    See https://docs.pydantic.dev/latest/concepts/pydantic_settings/ for more details.

    Note: we use the "mib_<character_name>_" prefix for all environment variables to avoid conflicts with other projects
    For example, to set the dataset_dir attribute for pocoyo, you must set "MIB_POCOYO_DATASET_DIR" in your environment
    variables or in the .env file. The same applies to all other attributes of this class. Note that the case is not
    important, so you can use "mib_pocoyo_dataset_dir" or "MIB_POCOYO_DATASET_DIR" or any other case variation.

    Attributes:
        - mib_dataset_dir: The directory where the dataset is stored. We expect a local path here. Interaction with
            a remote repository is handled via the "upload_dataset" and "download_dataset" scripts.
        - python_env: The Python environment to use. See the docstring of PythonEnv to set this correctly (if you're
            using the dev docker environment, you can set this value to "single").
        - num_workers: The number of workers to use for parallel processing, for the relevant steps.
    """

    model_config = pydantic_settings.SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    dataset_dir: Path
    experiment_dir: Path


@lru_cache(maxsize=1)
def get_dev_settings(character_name: CharacterName) -> DevSettings:
    try:
        dev_settings = DevSettings(_env_prefix=f"mib_{character_name.value.lower()}_")
    except pydantic_core._pydantic_core.ValidationError as e:
        raise ValueError(
            f"Failed to load dev settings for character '{character_name.value}'. Check your environment variables or "
            f".env file. Expected prefix: MIB_{character_name.value.upper()}_. "
        ) from e
    return dev_settings


def get_dataset_directory(character_name: CharacterName) -> Path:
    return get_dev_settings(character_name).dataset_dir


def get_experiment_directory(character_name: CharacterName) -> Path:
    return get_dev_settings(character_name).experiment_dir
