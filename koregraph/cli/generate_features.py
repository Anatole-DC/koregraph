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


if __name__ == "__main__":
    main()
