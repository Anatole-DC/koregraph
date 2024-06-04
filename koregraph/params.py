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

WEIGHTS_BACKUP_DIRECTORY: Path = Path(
    environ.get("WEIGHTS_BACKUP_DIRECTORY", "temp/backup")
)
WEIGHTS_BACKUP_DIRECTORY.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_DIRECTORY: Path = Path(environ.get("OUT_DIRECTORY", "out"))
MODEL_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
