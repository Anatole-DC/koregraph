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

parser.add_argument(
    "-i",
    "--interpolation-mode",
    dest="interpolation_mode",
    choices=[None, "silence", "blend"],
    default=None,
    help="Interpolation mode to use for preprocessing (default None).",
)


parser.add_argument(
    "-d",
    "--downsample",
    dest="downsample",
    default=None,
    help="Downsampling fps (default. no downsampling)",
)


def main():
    arguments = parser.parse_args()
    mode = arguments.mode
    interpolation_mode = str(arguments.interpolation_mode)
    downsample = int(arguments.downsample) if arguments.downsample is not None else None
    output = Path(arguments.output)

    generate_training_pickles(mode, output, interpolation_mode, downsaple_fps=downsample)


if __name__ == "__main__":
    main()
