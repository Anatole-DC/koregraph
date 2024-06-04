from numpy import array
from koregraph.params import KEYPOINTS_DIRECTORY
from pickle import load as load_pickle

from typing import Tuple


def generate_pickle_and_cut_audio(choregraphy_name: str) -> Tuple[array, int]:
    """Create a numpy array with 34 columns

    Args:
        name (str): The choregraphy file's name.

    Returns:
        Array of positions: The postures 34 columns N rows.
        Number of frame to use in the audio
    """
    with open(KEYPOINTS_DIRECTORY / (choregraphy_name + ".pkl"), "rb") as f:
        data = load_pickle(f)
        postures = data["keypoints2d"][0, :, :, :2]

    return postures.reshape(-1, 34), len(data["timestamps"])
