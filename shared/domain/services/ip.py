from shared.domain.entities.ip import CharacterName


def check_for_conflicting_character_names() -> None:
    character_names = {}
    for subclass in CharacterName.__subclasses__():
        for name in subclass:
            if name in character_names:
                conflicting_class = character_names[name]
                raise ValueError(
                    f"Conflicting character name found: {name} in subclasses {conflicting_class} and {subclass}"
                )
            character_names[name] = subclass


def validate_character_name(character_name: str) -> CharacterName:
    character_name = character_name.lower()
    check_for_conflicting_character_names()
    for subclass in CharacterName.__subclasses__():
        if character_name in [member.value for member in subclass]:
            return subclass(character_name)
    possible_values = [member.value for subclass in CharacterName.__subclasses__() for member in subclass]
    raise ValueError(
        f"Character {character_name} not found in any subclass of CharacterName. Possible values: {possible_values}"
    )
