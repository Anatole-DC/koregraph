from pickle import load as load_pickle
from typing import Tuple

from numpy import ndarray, nan_to_num, ones, zeros

from koregraph.api.preprocessing.interpolation import add_transition
from koregraph.config.params import (
    ALL_ADVANCED_MOVE_NAMES,
    KEYPOINTS_DIRECTORY,
    FRAME_FORMAT,
)
from koregraph.models.choregraphy import Choregraphy
from koregraph.tools.video_builder import export_choregraphy_keypoints
from koregraph.utils.controllers.choregraphies import load_choregraphy


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

    filled = nan_to_num(choregraphy.keypoints2d, 0)
    interpolated = add_transition(filled)
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


def generate_and_export_choreography(posture_file_2):
    """Generate and export a choreography with interpolated posture arrays.
    Args:
        posture_file_2 (str): The name of the posture file where you want to interpolate the posture arrays.
        Returns:
        np.array: The final array of postures."""

    posture_file_1 = "gPO_sBM_cAll_d10_mPO1_ch01.pkl"

    # Définition des arrays de posture
    posture_array_1 = generate_posture_array(posture_file_1)
    posture_array_2 = generate_posture_array(posture_file_2)

    # Nombre de lignes pour chaque partie de l'array final
    n_rows_part1 = 60
    n_rows_part2 = len(posture_array_2)
    n_rows_part3 = n_rows_part1

    # Initialisation de l'array final
    final_array = zeros(
        (n_rows_part1 + n_rows_part2 + n_rows_part3, posture_array_1.shape[1])
    )

    # Ajout de la première partie (transition de posture_array_1 à posture_array_2)
    for i in range(n_rows_part1):
        final_array[i] = posture_array_1[0] + (
            posture_array_2[0] - posture_array_1[0]
        ) * (i / n_rows_part1)

    # Ajout de la deuxième partie (toutes les lignes de posture_array_2)
    final_array[n_rows_part1 : n_rows_part1 + n_rows_part2] = posture_array_2

    # Ajout de la troisième partie (transition de la dernière posture de posture_array_2 à la première posture de posture_array_1)
    for i in range(n_rows_part3):
        final_array[n_rows_part1 + n_rows_part2 + i] = posture_array_2[-1] + (
            posture_array_1[0] - posture_array_2[-1]
        ) * (i / n_rows_part3)

    return final_array


if __name__ == "__main__":
    export_choregraphy_keypoints(
        Choregraphy(
            "test",
            upscale_posture_pred(
                posture_array_to_keypoints(
                    convert_to_train_posture(
                        load_choregraphy(ALL_ADVANCED_MOVE_NAMES[0])
                    )
                )
            ),
        ),
        "test",
    )
