from pickle import load as load_pickle
from typing import Tuple

from numpy import ndarray, nan_to_num

from koregraph.config.params import KEYPOINTS_DIRECTORY, FRAME_FORMAT


def generate_posture_array(
    choregraphy_name: str, frame_format: Tuple = FRAME_FORMAT
) -> ndarray:
    """Create a numpy array with 34 columns

    Args:
        name (str): The choregraphy file's name.

    Returns:
        Array of positions: The postures 34 columns N rows.
    """

    with open(KEYPOINTS_DIRECTORY / (choregraphy_name), "rb") as f:
        data = load_pickle(f)
        postures = data["keypoints2d"][0, :, :, :2]
        postures = nan_to_num(postures, 0)
        postures[:, :, 0] = postures[:, :, 0] / frame_format[0]
        postures[:, :, 1] = postures[:, :, 1] / frame_format[1]

    return postures.reshape(-1, 34)


# @TODO: verifier le format d'entree de prediction
def upscale_posture_pred(
    prediction: ndarray, frame_format: Tuple = FRAME_FORMAT
) -> ndarray:
    """Create a numpy array with 34 columns

    Args:
        name (str): The choregraphy file's name.

    Returns:
        Array of positions: The postures 34 columns N rows.
    """

    prediction = prediction.reshape(-1, 17, 2)
    prediction[:, :, 0] = prediction[:, :, 0] * frame_format[0]
    prediction[:, :, 1] = prediction[:, :, 1] * frame_format[1]

    return prediction
