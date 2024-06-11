from dataclasses import dataclass, field
from pathlib import Path


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

    choregraphy_file: Path
    name: str = field(default=None, init=False)
    music: str = field(default=None, init=False)
    genre: str = field(default=None, init=False)

    def __post_init__(self):
        genre, _, _, _, music, _ = self.choregraphy_file.stem.split("_")
        self.music = music
        self.genre = genre[1:]
        self.name = self.choregraphy_file.stem


if __name__ == "__main__":
    print(AISTFile("gBR_sBM_cAll_d04_mBR0_ch01"))
