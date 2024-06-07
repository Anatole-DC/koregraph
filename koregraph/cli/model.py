from pathlib import Path
from argparse import ArgumentParser

import numpy as np

from tensorflow.python.keras.models import Model

from koregraph.api.training.train_workflow import train_workflow
from koregraph.api.environment.training_cloud import run_mlflow_pipeline
from koregraph.config.params import MODEL_OUTPUT_DIRECTORY, AUDIO_DIRECTORY


parser = ArgumentParser(
    "Koregraph prediction",
    description="Use this to predict a chore from audio",
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
    "-c",
    "--with-cloud",
    dest="with_cloud",
    action="store_true",
    help="When passed, run the train workflow on google cloud.",
)

parser.add_argument(
    "-d",
    "--dataset-size",
    dest="dataset_size",
    default=1.0,
    help="Size ratio of the dataset to be used (default to 1)",
)


def main():
    arguments = parser.parse_args()
    model_name = str(arguments.model_name)
    with_cloud = bool(arguments.with_cloud)
    dataset_size = float(arguments.dataset_size)

    if with_cloud:
        print("Running training with google cloud")
        run_mlflow_pipeline(model_name, dataset_size)
        return

    print("Running training locally")
    train_workflow(model_name, dataset_size)


if __name__ == "__main__":
    main()
