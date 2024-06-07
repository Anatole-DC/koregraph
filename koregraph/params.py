from os import environ, path, listdir
from pathlib import Path
from dotenv import load_dotenv
from pickle import load as load_pickle

from koregraph.models.constants import LAST_CHUNK_TYPE

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

CHUNK_SIZE: int = int(environ.get("CHUNK_SIZE", "5"))

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

KEYPOINTS_BUILDER_TEMP_DIRECTORY = PROJECT_ROOT.joinpath(
    environ.get("KEYPOINTS_BUILDER_TEMP_DIRECTORY", "generated/videos")
)
KEYPOINTS_BUILDER_TEMP_DIRECTORY.mkdir(parents=True, exist_ok=True)


WEIGHTS_BACKUP_DIRECTORY: Path = PROJECT_ROOT.joinpath(
    environ.get("WEIGHTS_BACKUP_DIRECTORY", "temp/backup")
)
WEIGHTS_BACKUP_DIRECTORY.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_DIRECTORY: Path = PROJECT_ROOT.joinpath(
    environ.get("OUT_DIRECTORY", "generated/models")
)
MODEL_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

PREDICTION_OUTPUT_DIRECTORY: Path = PROJECT_ROOT.joinpath(
    environ.get("PREDICTION_OUTPUT_DIRECTORY", "generated/predictions/")
)
PREDICTION_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

GENERATED_PICKLE_DIRECTORY: Path = PROJECT_ROOT.joinpath(
    environ.get(
        "GENERATED_PICKLE_DIRECTORY",
        "generated/inputs",
    )
)

ALL_ADVANCED_MOVE_NAMES = [
    name for name in listdir(KEYPOINTS_DIRECTORY) if "sFM" in name
]

ALL_BASIC_MOVE_NAMES = [name for name in listdir(KEYPOINTS_DIRECTORY) if "sBM" in name]

LAST_CHUNK_TYPE_STRATEGY = LAST_CHUNK_TYPE.ROLLING

ALL_MUSIC_NAMES = [name for name in listdir(AUDIO_DIRECTORY) if name.endswith(".mp3")]


FRAME_FORMAT = (1920, 1080)

X_MIN = -80
X_MAX = 0

PERCENTAGE_CUT = 0.2

GENERATED_FEATURES_DIRECTORY: Path = PROJECT_ROOT.joinpath(
    environ.get(
        "GENERATED_FEATURES_DIRECTORY",
        "generated/features",
    )
)
GENERATED_FEATURES_DIRECTORY.mkdir(parents=True, exist_ok=True)

if len(ALL_ADVANCED_MOVE_NAMES) == 0:
    try:
        with open('data/all_advanced_move_names.pkl', 'rb') as f:
            ALL_ADVANCED_MOVE_NAMES =  load_pickle.load(f)
    except Exception as e:
        print('Could not load all_advanced_move_names variable')
