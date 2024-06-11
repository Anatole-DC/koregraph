from argparse import ArgumentParser
from pathlib import Path

from koregraph.models.aist_file import AISTFile
from koregraph.tools.video_builder import keypoints_video_audio_builder

parser = ArgumentParser(
    "Koregraph Viewer",
    description="Use this viewer to export drawn keypoints",
)

parser.add_argument(
    "-c",
    "--choregraphy",
    dest="choregraphy",
    required=True,
    help="Path to choregraphy.",
)


def main():
    arguments = parser.parse_args()

    choregraphy_file = Path(arguments.choregraphy)
    choregraphy_file = AISTFile(choregraphy_file)
    keypoints_video_audio_builder(choregraphy_file.name, choregraphy_file.music)


if __name__ == "__main__":
    main()
