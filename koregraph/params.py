from os import environ
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
