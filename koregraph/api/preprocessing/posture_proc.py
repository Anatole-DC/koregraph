from pickle import load as load_pickle
from typing import Tuple

from numpy import array, ndarray, nan_to_num, asarray, isnan

from koregraph.config.params import KEYPOINTS_DIRECTORY, FRAME_FORMAT


def fill_forward(arr):
    """
    Fill NaN values in the array with the previous row's values for each column.

    Parameters:
    arr (numpy.ndarray): Input array with possible NaN values.

    Returns:
    numpy.ndarray: Array with NaN values filled forward.
    """
    arr = asarray(arr, dtype=float)

    for i in range(1, arr.shape[0]):
        mask = isnan(arr[i, :])
        arr[i, mask] = arr[i - 1, mask]

    return arr


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
        postures = fill_forward(postures)
        postures = nan_to_num(postures, 0)

    return postures.reshape(-1, 34)


def scale_posture_pred(
    prediction: ndarray, frame_format: tuple = FRAME_FORMAT
) -> array:
    """Create a numpy array with 34 columns

    Args:
        name (str): The choregraphy file's name.

    Returns:
        Array of positions: The postures 34 columns N rows.
    """

    prediction = prediction.reshape(-1, 17, 2)
    prediction[:, :, 0] = prediction[:, :, 0] / frame_format[0]
    prediction[:, :, 1] = prediction[:, :, 1] / frame_format[1]

    return prediction


def upscale_posture_pred(
    prediction: ndarray, frame_format: Tuple = FRAME_FORMAT
) -> ndarray:
    """Upscale postures according to the frame format wanted.

    The model outputs downscaled predictions between 0 and 1 to reduce the value range.
    This function takes the outputs and upscale them in order to draw them in the final viewer.

    @TODO: Implement smaller formats for faster video building time

    Args:
        prediction (ndarray): The downscaled predictions (between 0 and 1)
        frame_format (Tuple, optional): The scaling image dimensions. Defaults to FRAME_FORMAT.

    Returns:
        ndarray: The upscaled postures
    """

    prediction = prediction.reshape(-1, 17, 2)
    prediction[:, :, 0] = prediction[:, :, 0] * frame_format[0]
    prediction[:, :, 1] = prediction[:, :, 1] * frame_format[1]

    return prediction
