from pathlib import Path
from os import environ

from dotenv import load_dotenv


load_dotenv()

DATASET_PATH: Path = Path(environ.get("DATASET_PATH", "/data/koregraph/"))
DATASET_PATH.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"Test scope of the {__file__} file.\n")

    print(DATASET_PATH.absolute())
