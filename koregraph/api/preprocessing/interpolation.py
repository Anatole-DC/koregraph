"""
    All functions to interpolate data for the barbarie method.

    When concatenating all the audios, the interpolation functions add smooth transition
    between each music/choregraphies.
"""

from numpy import ndarray, array, concatenate

from koregraph.models.constants import StandingPosture
from koregraph.models.posture import Posture
from koregraph.models.choregraphy import Choregraphy


def add_transition(
    choregraphy_keypoints: ndarray, nb_seconds: int = 1, fps: int = 60
) -> ndarray:
    frames = nb_seconds * fps
    standing_posture = StandingPosture.keypoints

    def interpolate(start, end, num_frames):
        return array(
            [start + (end - start) * t / (num_frames - 1) for t in range(num_frames)]
        )

    start_transition = interpolate(standing_posture, choregraphy_keypoints[0], frames)
    end_transition = interpolate(choregraphy_keypoints[-1], standing_posture, frames)

    new_choregraphy_keypoints = concatenate(
        [start_transition, choregraphy_keypoints, end_transition], axis=0
    )

    return new_choregraphy_keypoints


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
