from dataclasses import dataclass, field
from pathlib import Path
from os import environ

from koregraph.models.types import MusicGenre
from koregraph.params import AUDIO_DIRECTORY


@dataclass
class AISTFile:
    """
    Representation of an AIST++ file
    ex: gBR_sBM_cAll_d04_mBR0_ch01
        - g = genre
        - d = danser
        - m = music
        - ch = choregraphy_number
    """

    name: str
    choregraphy: Path = field(default=None, init=False)
    music: Path = field(default=None, init=False)
    genre: str = field(default=None, init=False)

    def __post_init__(self):
        genre, _, _, _, music, _ = self.name.split("_")
        self.music = AUDIO_DIRECTORY / f"{music}.mp3"
        self.genre = genre[1:]


if __name__ == "__main__":
    print(AISTFile("gBR_sBM_cAll_d04_mBR0_ch01"))
