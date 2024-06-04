from os import environ, path, listdir
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()
print("dotenv loaded")


PROJECT_ROOT = Path(path.dirname(path.dirname(__file__)))

AUDIO_DIRECTORY: Path = PROJECT_ROOT.joinpath(
    environ.get("AUDIO_DIRECTORY", "data/music/mp3")
)

IMAGE_DIRECTORY: Path = PROJECT_ROOT.joinpath(
    environ.get("IMAGE_DIRECTORY", "data/images")
)

KEYPOINTS_DIRECTORY: Path = PROJECT_ROOT.joinpath(
    environ.get("KEYPOINTS_DIRECTORY", "data/keypoints2d")
)

GENERATED_KEYPOINTS_DIRECTORY: Path = PROJECT_ROOT.joinpath(
    environ.get("GENERATED_KEYPOINTS_DIRECTORY", "generated/chunks/keypoints2d")
)

GENERATED_AUDIO_DIRECTORY: Path = PROJECT_ROOT.joinpath(
    environ.get("GENERATED_AUDIO_DIRECTORY", "generated/chunks/music")
)

CHUNK_SIZE: int = int(environ.get("CHUNK_SIZE", "10"))

if not AUDIO_DIRECTORY.exists():
    raise FileNotFoundError(
        f"Could not find audio directory at '{AUDIO_DIRECTORY.absolute()}'"
    )

if not KEYPOINTS_DIRECTORY.exists():
    raise FileNotFoundError(
        f"Could not find keypoints directory at '{KEYPOINTS_DIRECTORY.absolute()}'"
    )

if not GENERATED_KEYPOINTS_DIRECTORY.exists():
    GENERATED_KEYPOINTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    print(f"Created directory'{GENERATED_KEYPOINTS_DIRECTORY}'")

if not GENERATED_AUDIO_DIRECTORY.exists():
    GENERATED_AUDIO_DIRECTORY.mkdir(parents=True, exist_ok=True)
    print(f"Created directory'{GENERATED_AUDIO_DIRECTORY}'")

if not IMAGE_DIRECTORY.exists():
    IMAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    print(f"Created directory'{IMAGE_DIRECTORY}'")

WEIGHTS_BACKUP_DIRECTORY: Path = Path(
    environ.get("WEIGHTS_BACKUP_DIRECTORY", "temp/backup")
)
WEIGHTS_BACKUP_DIRECTORY.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_DIRECTORY: Path = Path(environ.get("OUT_DIRECTORY", "out"))
MODEL_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

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
