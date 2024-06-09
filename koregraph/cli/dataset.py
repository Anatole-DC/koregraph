from argparse import ArgumentParser

from koregraph.utils.cloud.load_dataset import download_dataset_from_bucket

parser = ArgumentParser(
    "Koregraph prediction",
    description="Use this to predict a chore from audio",
)

parser.add_argument(
    "-d",
    "--download",
    dest="download",
    action="store_true",
    help="Download the full dataset",
)


def main():
    arguments = parser.parse_args()

    download = arguments.download

    if download:
        download_dataset_from_bucket()


if __name__ == "__main__":
    main()
