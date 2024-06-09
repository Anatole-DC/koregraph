from pickle import load as load_pickle
from typing import Tuple

from numpy import ndarray, nan_to_num

from koregraph.config.params import KEYPOINTS_DIRECTORY, FRAME_FORMAT
from koregraph.models.choregraphy import Choregraphy


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


def array_from_keypoints(keypoints: ndarray) -> ndarray:
    """Flatten keypoints (x, y pairs) into a one dimensional array.

    Args:
        keypoints (ndarray): The (x,y) keypoint pairs.

    Returns:
        ndarray: The 34 points array.
    """

    return keypoints.reshape(-1, 34)


def downscale_posture(
    posture_array: ndarray, frame_format: Tuple = FRAME_FORMAT
) -> ndarray:
    """Downscale keypoints between 0 and 1.

    The keypoints are pairs of coordinates on the original videos. The downscale
    projects the points from the original coordinates format to a format between
    0 and 1.

    Args:
        posture_array (ndarray): The (x,y) keypoint pairs
        frame_format (Tuple, optional): The original frame resolution. Defaults to FRAME_FORMAT.

    Returns:
        ndarray: The downscale (x,y) keypoint pairs.
    """

    posture_array[:, :, 0] = posture_array[:, :, 0] / frame_format[0]
    posture_array[:, :, 1] = posture_array[:, :, 1] / frame_format[1]
    return posture_array


def posture_array_to_keypoints(posture_array: ndarray) -> ndarray:
    """Reshape a flatten points into a (x,y) keypoint pairs array.

    Args:
        posture_array (ndarray): The 34 points array.

    Returns:
        ndarray: The (x,y) keypoint pairs.
    """

    return posture_array.reshape(-1, 17, 2)


def convert_to_train_posture(choregraphy: Choregraphy) -> ndarray:
    """Convert a choregraphy posture into a training array.

    The output is a (34,) dimension array, corresponding to the
    flatten and downscaled original keypoints.

    Args:
        choregraphy (Choregraphy): The choregraphy to convert.

    Returns:
        ndarray: The training array.
    """

    interpolated = nan_to_num(choregraphy.keypoints2d, 0)
    downscale = downscale_posture(interpolated)
    posture_array = array_from_keypoints(downscale)
    return posture_array


def upscale_posture_pred(
    keypoints: ndarray, frame_format: Tuple = FRAME_FORMAT
) -> ndarray:
    """Upscale postures according to the frame format wanted.

    The model outputs downscaled predictions between 0 and 1 to reduce the value range.
    This function takes the outputs and upscale them in order to draw them in the final viewer.

    @TODO: Implement smaller formats for faster video building time

    Args:
        keypoints (ndarray): The downscaled predictions (between 0 and 1)
        frame_format (Tuple, optional): The scaling image dimensions. Defaults to FRAME_FORMAT.

    Returns:
        ndarray: The upscaled postures
    """

    keypoints[:, :, 0] = keypoints[:, :, 0] * frame_format[0]
    keypoints[:, :, 1] = keypoints[:, :, 1] * frame_format[1]

    return keypoints
