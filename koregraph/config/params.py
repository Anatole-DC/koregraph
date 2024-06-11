from pathlib import Path
from dotenv import load_dotenv

from koregraph.models.aist_file import AISTFile
from koregraph.models.constants import LAST_CHUNK_TYPE
from koregraph.utils.params import get_env_or_default


load_dotenv()
print("dotenv loaded")


PROJECT_ROOT = get_env_or_default("PROJECT_ROOT", ".", Path)


# PATH RELATED TO THE DATASET

DATA_PATH = get_env_or_default("DATA_PATH", PROJECT_ROOT / "data", Path)

AUDIO_DIRECTORY: Path = get_env_or_default(
    "AUDIO_DIRECTORY", DATA_PATH / "music/mp3/", Path
)

IMAGE_DIRECTORY: Path = get_env_or_default(
    "IMAGE_DIRECTORY", DATA_PATH / "images", Path
)
IMAGE_DIRECTORY.mkdir(parents=True, exist_ok=True)

KEYPOINTS_DIRECTORY: Path = get_env_or_default(
    "KEYPOINTS_DIRECTORY", DATA_PATH / "keypoints2d", Path
)

if not AUDIO_DIRECTORY.exists():
    raise FileNotFoundError(
        f"Could not find audio directory at '{AUDIO_DIRECTORY.absolute()}'"
    )

if not KEYPOINTS_DIRECTORY.exists():
    raise FileNotFoundError(
        f"Could not find keypoints directory at '{KEYPOINTS_DIRECTORY.absolute()}'"
    )


# PATH RELATED TO THE PREPROCESSING, TRAING AND PREDICTION OUTPUTS

GENERATED_OUTPUT_PATH: Path = get_env_or_default(
    "GENERATED_OUTPUT_PATH", PROJECT_ROOT / "generated", Path
)

GENERATED_KEYPOINTS_DIRECTORY: Path = get_env_or_default(
    "GENERATED_KEYPOINTS_DIRECTORY", GENERATED_OUTPUT_PATH / "chunks/keypoints2d", Path
)
GENERATED_KEYPOINTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

GENERATED_AUDIO_DIRECTORY: Path = get_env_or_default(
    "GENERATED_AUDIO_DIRECTORY", GENERATED_OUTPUT_PATH / "chunks/music", Path
)
GENERATED_AUDIO_DIRECTORY.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_DIRECTORY: Path = get_env_or_default(
    "OUT_DIRECTORY", GENERATED_OUTPUT_PATH / "models", Path
)
MODEL_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

GENERATED_PICKLE_DIRECTORY: Path = get_env_or_default(
    "GENERATED_PICKLE_DIRECTORY", GENERATED_OUTPUT_PATH / "data", Path
)

GENERATED_FEATURES_DIRECTORY: Path = get_env_or_default(
    "GENERATED_FEATURES_DIRECTORY", GENERATED_OUTPUT_PATH / "features", Path
)
GENERATED_FEATURES_DIRECTORY.mkdir(parents=True, exist_ok=True)

GENERATED_LOSS_DIRECTORY: Path = get_env_or_default(
    "GENERATED_LOSS_DIRECTORY", GENERATED_OUTPUT_PATH / "loss/", Path
)

GENERATED_LOSS_BACKUP_DIRECTORY: Path = get_env_or_default(
    "GENERATED_LOSS_BACKUP_DIRECTORY", GENERATED_LOSS_DIRECTORY / "backup/", Path
)
GENERATED_LOSS_BACKUP_DIRECTORY.mkdir(parents=True, exist_ok=True)

GENERATED_AUDIO_SILENCE_DIRECTORY: Path = get_env_or_default(
    "GENERATED_AUDIO_SILENCE_DIRECTORY", GENERATED_OUTPUT_PATH / "chunks/silence", Path
)
GENERATED_AUDIO_SILENCE_DIRECTORY.mkdir(parents=True, exist_ok=True)

# VARIABLES RELATED TO TEMPORARY CACHE

KEYPOINTS_BUILDER_TEMP_DIRECTORY = get_env_or_default(
    "KEYPOINTS_BUILDER_TEMP_DIRECTORY", GENERATED_OUTPUT_PATH / "videos", Path
)
KEYPOINTS_BUILDER_TEMP_DIRECTORY.mkdir(parents=True, exist_ok=True)

WEIGHTS_BACKUP_DIRECTORY: Path = get_env_or_default(
    "WEIGHTS_BACKUP_DIRECTORY", "temp/backup", Path
)
WEIGHTS_BACKUP_DIRECTORY.mkdir(parents=True, exist_ok=True)

PREDICTION_OUTPUT_DIRECTORY: Path = get_env_or_default(
    "PREDICTION_OUTPUT_DIRECTORY", GENERATED_OUTPUT_PATH / "predictions/", Path
)
PREDICTION_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

# MLFLOW RELATED VARIABLES

MLFLOW_TRACKING_URI = get_env_or_default("MLFLOW_TRACKING_URI", "http://localhost:8000")

GCLOUD_AUTHENTICATION = get_env_or_default(
    "GCLOUD_AUTHENTICATION",
    PROJECT_ROOT / "secrets/le-wagon-420414-c20b739bfbba.json",
    Path,
)

# PREPROCESSING RELATED VARIABLES

ALL_ADVANCED_MOVE_NAMES = [
    AISTFile(advanced_move_path)
    for advanced_move_path in KEYPOINTS_DIRECTORY.glob("*sFM*")
]

ALL_BASIC_MOVE_NAMES = [
    basic_move_path.name for basic_move_path in KEYPOINTS_DIRECTORY.glob("*sBM*")
]

ALL_MUSIC_NAMES = [music_path.name for music_path in AUDIO_DIRECTORY.glob("*.mp3")]

LAST_CHUNK_TYPE_STRATEGY = LAST_CHUNK_TYPE.ROLLING

CHUNK_SIZE: int = get_env_or_default("CHUNK_SIZE", 5, int)

FRAME_FORMAT = get_env_or_default("FRAME_FORMAT", (1920, 1080), tuple)

X_MIN = -80
X_MAX = 0

PERCENTAGE_CUT = 0.05
