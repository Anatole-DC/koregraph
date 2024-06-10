from pickle import load as load_pickle
import matplotlib.pyplot as plt

from koregraph.config.params import (
    MODEL_OUTPUT_DIRECTORY,
    GENERATED_LOSS_DIRECTORY,
    GENERATED_LOSS_BACKUP_DIRECTORY,
)


def plot_loss(model_name: str, backup: bool = False):
    path = MODEL_OUTPUT_DIRECTORY / f"{model_name}_history.pkl"
    with open(path, "rb") as f:
        model_history = load_pickle(f)

    if backup:
        plt.plot(model_history["loss"], label="train")
        plt.plot(model_history["val_loss"], label="test")
        plt.legend()
        plt.savefig(GENERATED_LOSS_BACKUP_DIRECTORY / f"{model_name}.png")
    else:
        plt.plot(model_history.history["loss"], label="train")
        plt.plot(model_history.history["val_loss"], label="test")
        plt.legend()
        plt.savefig(GENERATED_LOSS_DIRECTORY / f"{model_name}.png")
