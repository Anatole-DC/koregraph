from pickle import dump as pickle_dump, load as pickle_load, HIGHEST_PROTOCOL
from pathlib import Path
from typing import Any

from koregraph.config.params import MODEL_OUTPUT_DIRECTORY


def save_object_pickle(obj: Any, obj_name: str = None, obj_path: Path = None):
    """Save a python object inside a pickle object.

    Args:
        obj (Any): The object to save
        obj_name (str): The name under which the object must be saved.

    Returns:
        Path: The path where the object was saved.
    """
    assert obj_name is not None or obj_path is not None

    saved_object_path = MODEL_OUTPUT_DIRECTORY / f"{obj_name}.pkl"
    if obj_path is not None:
        saved_object_path = obj_path
    with open(saved_object_path, "wb") as pickle_file:
        pickle_dump(obj, pickle_file, protocol=HIGHEST_PROTOCOL)
    return saved_object_path


def load_pickle_object(obj_path: Path):
    """Load a python object from a pickle object.

    Args:
        obj_path (Path): The path to the pickle object.

    Returns:
        Any: The loaded object.
    """
    with open(obj_path, "rb") as pickle_file:
        loaded_object = pickle_load(pickle_file)
    return loaded_object
