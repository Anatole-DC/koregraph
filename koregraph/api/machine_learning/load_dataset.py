"""
    All utilities functions to load the train dataset.
"""

from typing import Any

from numpy import ndarray, append

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

    files = [
        "generated_gBR_sFM_cAll_d04_mBR1_ch02.pkl"
    ]

    for file in files:
        y_tmp, X_tmp = load_pickle_object(GENERATED_PICKLE_DIRECTORY / file)
        X = append(X, X_tmp, axis=0)
        y = append(y, y_tmp, axis=0)

    return X, y


def check_dataset_format(): ...
