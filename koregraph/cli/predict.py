from argparse import ArgumentParser
from pathlib import Path

from koregraph.api.machine_learning.prediction_workflow import predict as predict_api

parser = ArgumentParser(
    "Koregraph prediction",
    description="Use this to predict a chore from audio",
)

parser.add_argument(
    "-a", "--audio", dest="audio_name", required=True, help="Audio name"
)

parser.add_argument(
    "-m",
    "--model",
    dest="model_name",
    required=False,
    help="Model name",
    default="model",
)

parser.add_argument(
    "-i",
    "--chore-id",
    dest="chore_id",
    required=False,
    default="01",
    help="Id of the chore to be created. Ex: '01'",
)


def main():
    arguments = parser.parse_args()

    audio_name = arguments.audio_name
    # model_name = arguments.model_name
    chore_id = arguments.chore_id

    predict_api(audio_name=audio_name, chore_id=chore_id)


if __name__ == "__main__":
    main()
