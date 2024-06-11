from argparse import ArgumentParser

from koregraph.api.machine_learning.prediction_workflow import (
    predict as predict_api,
    predict_next_move,
)
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
    "-c", "--choregraphy", dest="choregraphy", required=False, help="Choregraphy name"
)

parser.add_argument(
    "--from-cloud",
    dest="from_cloud",
    action="store_true",
    help="When passed, will attempt to download the model from gcloud storage",
)


parser.add_argument(
    "-b",
    "--backup",
    dest="backup",
    action="store_true",
    help="When passed, will use the backup model.",
)

parser.add_argument(
    "-i", "--chunk-id", dest="chunk_id", required=False, help="Choregraphy name"
)

parser.add_argument("--chunks", dest="is_chunks", action="store_true")

parser.add_argument("--predict-next", dest="predict_next", action="store_true")


def main():
    arguments = parser.parse_args()

    audio_name = arguments.audio_name
    model_name = arguments.model_name
    from_cloud = arguments.from_cloud
    backup = arguments.backup

    if from_cloud:
        print("Downloading the model from google cloud storage...")
        download_model_history_from_bucket(model_name)
        print("Model downloaded !")

    if backup:
        print("Using backup model")

    is_chunks = arguments.is_chunks
    choregraphy = arguments.choregraphy
    chunk_id = arguments.chunk_id

    if is_chunks:
        predict_api(audio_name=audio_name, model_name=model_name, chunk=is_chunks)
    elif arguments.predict_next:
        assert choregraphy is not None
        assert chunk_id is not None
        predict_next_move(
            audio_name=audio_name,
            model_name=model_name,
            chore_chunk_name=choregraphy,
            chunk_id=chunk_id,
        )
    else:
        predict_api(
            audio_name=audio_name, model_name=model_name, backup=backup, chunk=is_chunks
        )


if __name__ == "__main__":
    main()
