from pathlib import Path
from argparse import ArgumentParser

import numpy as np

from tensorflow.python.keras.models import Model

from koregraph.api.machine_learning.workflow import train_workflow
from koregraph.api.environment.training_cloud import run_mlflow_pipeline
from koregraph.params import MODEL_OUTPUT_DIRECTORY, AUDIO_DIRECTORY


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


def main():
    arguments = parser.parse_args()
    model_name = arguments.model_name
    with_cloud = arguments.with_cloud

    if with_cloud:
        print("Running training with google cloud")

        run_mlflow_pipeline(model_name)
        return

    print("Running training locally")
    train_workflow(model_name=model_name)


if __name__ == "__main__":
    main()
