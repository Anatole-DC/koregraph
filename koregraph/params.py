from os import environ, listdir
from pathlib import Path

from koregraph.models.constants import LAST_CHUNK_TYPE

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

ALL_ADVANCED_MOVE_NAMES = [
    name for name in listdir(KEYPOINTS_DIRECTORY) if "sFM" in name
]

ALL_BASIC_MOVE_NAMES = [name for name in listdir(KEYPOINTS_DIRECTORY) if "sBM" in name]

LAST_CHUNK_TYPE_STRATEGY = LAST_CHUNK_TYPE.ROLLING
