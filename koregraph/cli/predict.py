from argparse import ArgumentParser

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

parser.add_argument("--chunks", dest="is_chunks", action="store_true")


def main():
    arguments = parser.parse_args()

    audio_name = arguments.audio_name
    model_name = arguments.model_name
    is_chunks = arguments.is_chunks

    predict_api(audio_name=audio_name, model_name=model_name, chunk=is_chunks)


if __name__ == "__main__":
    main()
