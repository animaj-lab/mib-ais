from pathlib import Path

from shared.domain.entities.ip import CharacterName, PocoyoCharacterName


def get_controllers_config_directory() -> Path:
    return Path("motion_inbetweening/infra/configs/controllers_configs")


def get_pcy_default_controllers_path(character_name: CharacterName) -> Path:
    match character_name:
        case PocoyoCharacterName.POCOYO:
            return get_controllers_config_directory() / "pocoyo" / "pcy_default_controllers.yaml"
        case PocoyoCharacterName.PATO:
            return get_controllers_config_directory() / "pato" / "pcy_default_controllers.yaml"
        case _:
            raise NotImplementedError(
                f"Default pcy controllers for {character_name} does not exist, create it and update this function"
            )


def get_pcy_trainable_controllers_path(character_name: CharacterName) -> Path:
    match character_name:
        case PocoyoCharacterName.POCOYO:
            return get_controllers_config_directory() / "pocoyo" / "pcy_trainable_controllers_without_coeff.yaml"
        case PocoyoCharacterName.PATO:
            return get_controllers_config_directory() / "pato" / "pcy_trainable_controllers.yaml"
        case _:
            raise NotImplementedError(
                f"Trainable controllers for {character_name} does not exist, create it and update this function"
            )
