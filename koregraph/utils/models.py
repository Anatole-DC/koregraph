from pathlib import Path

from tensorflow.keras.saving import load_model

from koregraph.utils.controllers.pickles import save_object_pickle
from koregraph.config.params import WEIGHTS_BACKUP_DIRECTORY


def check_model_name_increment(model_name: str) -> str:
    """When called, check if the model already exists and adds an increment at the end.

    Args:
        model_name (str): The initial model name.

    Returns:
        str: The model name, with the increment if the name was found.
    """
    ...


def load_keras_model(model_path: Path):
    model = load_model(model_path)
    return model


if __name__ == "__main__":
    model = load_keras_model(WEIGHTS_BACKUP_DIRECTORY / "full_scale-1_backup.keras")
    save_object_pickle(model, "full_scale-1")
