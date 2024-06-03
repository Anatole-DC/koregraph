from pathlib import Path
from os import environ

from mutagen.mp3 import MP3

from koregraph.models.choregraphy import Choregraphy


AUDIO_DIRECTORY: Path = Path(
    environ.get(
        "AUDIO_DIRECTORY",
        "data/musics/mp3",
    )
)
if not AUDIO_DIRECTORY.exists():
    raise FileNotFoundError(
        f"Could not find music directory at '{AUDIO_DIRECTORY.absolute()}'"
    )


def load_choregraphy_audio(name: str) -> MP3:
    with open(AUDIO_DIRECTORY / f"{name}.mp3") as audio_file:
        audio_content = MP3(audio_file)
    return audio_content
