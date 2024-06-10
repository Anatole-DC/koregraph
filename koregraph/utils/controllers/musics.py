from pathlib import Path

from numpy import float16, float32, int16, ndarray
import soundfile as sf
from mutagen.mp3 import MP3
from librosa import load

from koregraph.config.params import AUDIO_DIRECTORY


if not AUDIO_DIRECTORY.exists():
    raise FileNotFoundError(
        f"Could not find music directory at '{AUDIO_DIRECTORY.absolute()}'"
    )


def load_choregraphy_audio(name: str) -> MP3:
    with open(AUDIO_DIRECTORY / f"{name}.mp3") as audio_file:
        audio_content = MP3(audio_file)
    return audio_content


def save_audio_chunk(audio: ndarray, sample_rate: int, fullpath: Path) -> None:
    sf.write(fullpath, audio, samplerate=sample_rate)


def load_music(music_path: Path) -> ndarray:
    if not music_path.exists():
        raise FileNotFoundError(f"File {music_path.absolute()} does not exist.")
    return load(music_path, sr=None)
