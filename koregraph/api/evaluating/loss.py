from pickle import load as load_pickle
import matplotlib.pyplot as plt

from koregraph.config.params import MODEL_OUTPUT_DIRECTORY, GENERATED_LOSS_DIRECTORY


def plot_loss(model_name: str):
    path = MODEL_OUTPUT_DIRECTORY / f"{model_name}_history.pkl"
    with open(path, "rb") as f:
        model_history = load_pickle(f)

    plt.plot(model_history.history["loss"], label="train")
    plt.plot(model_history.history["val_loss"], label="test")
    plt.legend()
    plt.savefig(GENERATED_LOSS_DIRECTORY / f"{model_name}.png")
