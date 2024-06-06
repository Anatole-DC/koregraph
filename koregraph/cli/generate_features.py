<<<<<<< HEAD
from argparse import ArgumentParser

from koregraph.api.machine_learning.pickle_creation import (
    generate_pickle_files,
    generate_all_chunks,
)

parser = ArgumentParser(
    "Generate",
    description="Use this generate features pickle files",
)

parser.add_argument("--chunks", dest="chunks", action="store_true")


def main():
    args = parser.parse_args()
    if args.chunks:
        generate_all_chunks()
    else:
        generate_pickle_files()
=======
from koregraph.api.machine_learning.pickle_creation import generate_pickle_files


def main():
    generate_pickle_files()
>>>>>>> 2b3865a67773b463ee9dbded945f4c623e35febe


if __name__ == "__main__":
    main()
