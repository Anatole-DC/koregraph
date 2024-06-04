from pathlib import Path
from os import environ
import numpy as np
import soundfile as sf
from mutagen.mp3 import MP3
from koregraph.params import AUDIO_DIRECTORY


if not AUDIO_DIRECTORY.exists():
    raise FileNotFoundError(
        f"Could not find music directory at '{AUDIO_DIRECTORY.absolute()}'"
    )


def load_choregraphy_audio(name: str) -> MP3:
    with open(AUDIO_DIRECTORY / f"{name}.mp3") as audio_file:
        audio_content = MP3(audio_file)
    return audio_content


def save_audio_chunk(audio: np.ndarray, sample_rate: int, fullpath: Path) -> None:
    sf.write(fullpath, audio, samplerate=sample_rate)
