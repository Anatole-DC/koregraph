from difflib import restore
from pathlib import Path
from argparse import ArgumentParser

import numpy as np

from tensorflow.keras.models import Model, load_model

from koregraph.utils.controllers.pickles import load_pickle_object
from koregraph.api.training.train_workflow import train_workflow
from koregraph.api.environment.training_cloud import run_mlflow_pipeline
from koregraph.config.params import WEIGHTS_BACKUP_DIRECTORY


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
    "-e",
    "--epochs",
    dest="epochs",
    default=20,
    help="Number of epochs to perform.",
)

parser.add_argument(
    "-d",
    "--dataset-size",
    dest="dataset_size",
    default=1.0,
    help="Size ratio of the dataset to be used (default to 1)",
)

parser.add_argument(
    "-b",
    "--batch-size",
    dest="batch_size",
    default=16,
    help="Size ratio of the batch size (default to 16)",
)

parser.add_argument(
    "-r",
    "--restore-backup",
    dest="restore_backup",
    action="store_true",
    help="If a model backup is found, load the backup and start training from it.",
)

parser.add_argument(
    "--no-early-stopping",
    dest="with_early_stopping",
    action="store_false",
    help="When passed, disable the early stopping callback.",
)


def main():
    arguments = parser.parse_args()
    model_name = str(arguments.model_name)
    with_cloud = bool(arguments.with_cloud)
    dataset_size = float(arguments.dataset_size)
    batch_size = int(arguments.batch_size)
    restore_backup = bool(arguments.restore_backup)
    epochs = int(arguments.epochs)

    model = None
    initial_epoch = 0

    if restore_backup:
        # Check for existing model backup
        model_backup_path = WEIGHTS_BACKUP_DIRECTORY / f"{model_name}_backup.keras"
        if model_backup_path.exists():
            model = load_model(model_backup_path)

        # Check for existing history backup
        history_backup_path = WEIGHTS_BACKUP_DIRECTORY / f"{model_name}_history.pkl"
        if history_backup_path.exists():
            history = load_pickle_object(history_backup_path)
            initial_epoch = len(history["loss"])
    else:
        model_backup_path = WEIGHTS_BACKUP_DIRECTORY / f"{model_name}_backup.keras"
        if model_backup_path.exists():
            model_backup_path.unlink()

        history_backup_path = WEIGHTS_BACKUP_DIRECTORY / f"{model_name}_history.pkl"
        if history_backup_path.exists():
            history_backup_path.unlink()

    if model is not None:
        print(f"Using backup for model {model_name} at epoch {initial_epoch}")

    if with_cloud:
        # @TODO: change for a real tensorflow cloud
        print("Running training with google cloud")
        run_mlflow_pipeline(model_name, dataset_size, initial_epoch)
        return

    print("Running training locally")
    train_workflow(model_name, epochs, batch_size, dataset_size, model, initial_epoch)


if __name__ == "__main__":
    main()
