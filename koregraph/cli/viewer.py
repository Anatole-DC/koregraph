from argparse import ArgumentParser
from pathlib import Path

from koregraph.tools.video_builder import keypoints_video_audio_builder

parser = ArgumentParser(
    "Koregraph Viewer",
    description="Use this viewer to export drawn keypoints",
)

parser.add_argument(
    "-c", "--choregraphy", dest="choregraphy", required=True, help="Choregraphy name"
)


def main():
    arguments = parser.parse_args()

    choregraphy_name = Path(arguments.choregraphy).stem
    keypoints_video_audio_builder(choregraphy_name)


if __name__ == "__main__":
    main()
