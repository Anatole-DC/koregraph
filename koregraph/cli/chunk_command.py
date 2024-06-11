from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from koregraph.config.params import CHUNK_SIZE

from koregraph.api.preprocessing.chunks_api import generate_chunk

parser = ArgumentParser(
    "Koregraph chunk",
    description="Use this create keypoints + audio chunks",
)

parser.add_argument(
    "-c", "--choregraphy", dest="choregraphy", required=True, help="Choregraphy name"
)

parser.add_argument(
    "-s",
    "--size",
    dest="chunk_size",
    required=True,
    help="Size of chunks in seconds",
    default=CHUNK_SIZE,
)

parser.add_argument(
    "--reload-music",
    dest="reload_music",
    required=False,
    action=BooleanOptionalAction,
    help="The music will be split again for this chunk size if it already exists",
)


def main():
    arguments = parser.parse_args()

    choregraphy_name = Path(arguments.choregraphy).stem
    chunk_size = arguments.chunk_size
    reload_music = arguments.reload_music

    generate_chunk(choregraphy_name, chunk_size, reload_music)


if __name__ == "__main__":
    main()
