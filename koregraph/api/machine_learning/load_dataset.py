"""
    All utilities functions to load the train dataset.
"""

from typing import Any

from numpy import ndarray, append

from koregraph.utils.controllers.pickles import load_pickle_object
from koregraph.config.params import GENERATED_PICKLE_DIRECTORY


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
        "generated_gBR_sFM_cAll_d04_mBR0_ch01.pkl",
        # "generated_gBR_sFM_cAll_d04_mBR1_ch02.pkl",
        # "generated_gBR_sFM_cAll_d04_mBR2_ch03.pkl",
        # "generated_gBR_sFM_cAll_d04_mBR3_ch04.pkl",
        # "generated_gBR_sFM_cAll_d04_mBR4_ch05.pkl",
        # "generated_gBR_sFM_cAll_d04_mBR4_ch07.pkl",
        # "generated_gBR_sFM_cAll_d04_mBR5_ch06.pkl",
        # "generated_gBR_sFM_cAll_d05_mBR2_ch09.pkl",
        # "generated_gBR_sFM_cAll_d05_mBR3_ch10.pkl",
        # "generated_gBR_sFM_cAll_d05_mBR4_ch11.pkl",
        # "generated_gBR_sFM_cAll_d05_mBR4_ch13.pkl",
        # "generated_gKR_sFM_cAll_d28_mKR0_ch01.pkl",
    ]

    for file in GENERATED_PICKLE_DIRECTORY.glob("*.pkl"):
        y_tmp, X_tmp = load_pickle_object(GENERATED_PICKLE_DIRECTORY / file)
        X = append(X, X_tmp, axis=0)
        y = append(y, y_tmp, axis=0)

    return X, y


def check_dataset_format(): ...
