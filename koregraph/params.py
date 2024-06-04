from os import environ, listdir
from pathlib import Path

AUDIO_DIRECTORY: Path = Path(
    environ.get(
        "AUDIO_DIRECTORY",
        "data/music/mp3",
    )
)

KEYPOINTS_DIRECTORY: Path = Path(
    environ.get(
        "KEYPOINTS_DIRECTORY",
        "data/keypoints2d",
    )
)

GENERATED_KEYPOINTS_DIRECTORY: Path = Path(
    environ.get(
        "GENERATED_KEYPOINTS_DIRECTORY",
        "generated/chunks/keypoints2d",
    )
)
GENERATED_AUDIO_DIRECTORY: Path = Path(
    environ.get(
        "GENERATED_AUDIO_DIRECTORY",
        "generated/chunks/music",
    )
)

GENERATED_PICKLE_DIRECTORY: Path = Path(
    environ.get(
        "GENERATED_PICKLE_DIRECTORY",
        "generated/data/pickles",
    )
)

ALL_ADVANCED_MOVE_NAMES = [
    name for name in listdir(KEYPOINTS_DIRECTORY) if "sFM" in name
]

ALL_BASIC_MOVE_NAMES = [name for name in listdir(KEYPOINTS_DIRECTORY) if "sBM" in name]

ALL_MUSIC_NAMES = [
    name for name in listdir(AUDIO_DIRECTORY) if name.endswith(".mp3")
    ]
