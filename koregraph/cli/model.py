from pathlib import Path
from argparse import ArgumentParser

import numpy as np

from tensorflow.python.keras.models import Model

from koregraph.api.machine_learning.train_workflow.workflow import train_workflow
from koregraph.api.machine_learning.train_workflow.chunks_workflow import (
    train_chunks_workflow,
)
from koregraph.api.machine_learning.train_workflow.next_chunks_workflow import (
    train_workflow as train_pred_next_workflow,
)
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

parser.add_argument("--chunks", dest="chunks", action="store_true")

parser.add_argument("--next-chunks", dest="predict_next", action="store_true")


def main():
    arguments = parser.parse_args()
    model_name = arguments.model_name

    if arguments.chunks:
        print("Training with chunks")
        train_chunks_workflow(model_name=model_name)
    elif arguments.predict_next:
        print("Training with chunks: predicting next X seconds")
        train_pred_next_workflow(model_name=model_name)
    else:
        train_workflow(model_name=model_name)


if __name__ == "__main__":
    main()
