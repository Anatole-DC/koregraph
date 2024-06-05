"""
    All utilities functions to load the train dataset.
"""

from typing import Any

from numpy import ndarray

from koregraph.utils.pickle import load_pickle_object
from koregraph.params import GENERATED_PICKLE_DIRECTORY


def load_preprocess_dataset() -> tuple[ndarray, ndarray]:
    """
    Load and preprocess the dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The dataset.
    """
    y, X = load_pickle_object(
        GENERATED_PICKLE_DIRECTORY / "generated_gBR_sFM_cAll_d04_mBR0_ch01.pkl"
    )

    return X, y


def check_dataset_format(): ...
