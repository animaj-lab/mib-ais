from enum import StrEnum, auto


class CharacterName(StrEnum):
    """Abstract class, to be implemented for each IP"""

    pass


class PocoyoCharacterName(CharacterName):
    POCOYO = auto()
    PATO = auto()
    ELLY = auto()
    LOULA = auto()
    CHR_BEA = auto()
