from argparse import ArgumentParser
from pathlib import Path
import os, shutil

from koregraph.tools.choregraphy_to_chunks import split_sequence
from koregraph.managers.choregraphy import save_choregaphy_chunk


parser = ArgumentParser(
    "Koregraph chunk",
    description="Use this create keypoints + audio chunks",
)

parser.add_argument(
    "-c", "--choregraphy", dest="choregraphy", required=True, help="Choregraphy name"
)
parser.add_argument(
    "-s", "--size", dest="chunk_size", required=True, help="Size of chunks in seconds"
)


def main():
    GENERATED_KEYPOINTS_DIRECTORY: Path = Path(os.environ.get(
        "GENERATED_KEYPOINTS_DIRECTORY",
        "generated/data/keypoints2d",
    ))
    GENERATED_KEYPOINTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    arguments = parser.parse_args()

    choregraphy_name = Path(arguments.choregraphy).stem
    chunk_size = arguments.chunk_size

    destination_path = GENERATED_KEYPOINTS_DIRECTORY / choregraphy_name / chunk_size
    if os.path.exists(destination_path) and not os.path.isfile(destination_path):
        try:
            shutil.rmtree(destination_path)
        except Exception as e:
            print(f'Failed to delete {destination_path}. Reason: {e}')
    destination_path.mkdir(parents=True, exist_ok=True)


    chores = split_sequence(choregraphy_name, int(chunk_size))
    for chore in chores:
        save_choregaphy_chunk(chore, destination_path)


if __name__ == "__main__":
    main()
