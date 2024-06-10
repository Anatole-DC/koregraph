from argparse import ArgumentParser

from koregraph.api.machine_learning.prediction_workflow import predict as predict_api
from koregraph.utils.cloud.cloud_bucket import download_model_history_from_bucket

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
    help="Model name",
    default="model",
)

parser.add_argument(
    "--from-cloud",
    dest="from_cloud",
    action="store_true",
    help="When passed, will attempt to download the model from gcloud storage",
)


def main():
    arguments = parser.parse_args()

    audio_name = arguments.audio_name
    model_name = arguments.model_name
    from_cloud = arguments.from_cloud

    if from_cloud:
        print("Downloading the model from google cloud storage...")
        download_model_history_from_bucket(model_name)
        print("Model downloaded !")

    predict_api(audio_name=audio_name, model_name=model_name)


if __name__ == "__main__":
    main()
