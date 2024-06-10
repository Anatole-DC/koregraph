"""
    All functions to interpolate data for the barbarie method.

    When concatenating all the audios, the interpolation functions add smooth transition
    between each music/choregraphies.
"""

from numpy import ndarray
from koregraph.models.choregraphy import Choregraphy


def interpolate_posture(choregraphy: Choregraphy, nb_frame: int = 60) -> Choregraphy:
    """Adds interpolation posture on nb_frame before and after the given choregraphy.

    Args:
        choregraphy (Choregraphy): The choregraphy to interpolates
        nb_frame (int, optional): The number of frame for the interpolation duration. Defaults to 60.

    Returns:
        Choregraphy: The interpolated choregraphy.
    """
    ...


def interpolate_audio(audio_array: ndarray, nb_frame: int = 60) -> ndarray:
    """Adds interpolation audio on nb_frame duration before and after the given audio.

    Args:
        audio_array (ndarray): The audio to interpolate
        nb_frame (int, optional): The number of frame for the interpolation duration. Defaults to 60.

    Returns:
        ndarray: The interpolated audio
    """
    ...


def interpolate_training_data(
    choregraphy: Choregraphy, audio: ndarray, nb_frame: int = 60
):
    return (
        interpolate_posture(choregraphy, nb_frame),
        interpolate_audio(audio, nb_frame),
    )
