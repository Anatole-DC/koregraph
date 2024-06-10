from argparse import ArgumentParser
from pathlib import Path

from koregraph.api.preprocessing.dataset_preprocessing import generate_training_pickles
from koregraph.config.params import GENERATED_PICKLE_DIRECTORY


parser = ArgumentParser(
    "Koregraph preprocessing",
    description="""Use this cli to preprocess the AIST++ dataset into training files.

The preprocessing takes a mode depending on the model used. The preprocessed training files will be generated inside
a 'mode' directory, at the 'output' path.
    """,
)

parser.add_argument(
    "-m",
    "--mode",
    dest="mode",
    default="barbarie",
    choices=["barbarie", "chunks"],
    help="The preprocessing mode, depending on the model you use (default 'barbarie')",
)

parser.add_argument(
    "-o",
    "--output",
    dest="output",
    default=GENERATED_PICKLE_DIRECTORY,
    help="The path where the 'mode' directory will be created.",
)


def main():
    arguments = parser.parse_args()
    mode = arguments.mode
    output = Path(arguments.output)

    generate_training_pickles(mode, output)


if __name__ == "__main__":
    main()
