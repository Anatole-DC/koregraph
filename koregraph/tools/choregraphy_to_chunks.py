import os
from typing import List, Tuple
from pathlib import Path
import pickle
import librosa
from numpy import ndarray, pad

from koregraph.models.choregraphy import Choregraphy
from koregraph.params import KEYPOINTS_DIRECTORY, AUDIO_DIRECTORY


def split_sequence(sequence_file: str, chunk_size_sec: int) -> List[Choregraphy]:
    """Splits a sequence file into N chunks of size chunk_size
    (last posture can be shorter)
    Args:
        sequence_file (str): The name of the keypoints path (without .pkl).
        chunk_size_sec (int): Size in seconds for each chunks

    Returns:
        List of Choregraphy corresponding to each chunks

    """

    with open(KEYPOINTS_DIRECTORY / (sequence_file + ".pkl"), "rb") as f:
        data = pickle.load(f)
        postures = data["keypoints2d"]
        timestamps = data["timestamps"]
        output = []
        for j, i in enumerate(range(0, len(timestamps), chunk_size_sec * 60)):
            output.append(
                Choregraphy(
                    name=sequence_file + f"_{j}",
                    keypoints2d=postures[:, i : i + 60 * chunk_size_sec, :, :],
                    timestamps=timestamps[i : i + 60 * chunk_size_sec],
                )
            )

    return output


def split_audio(
    sequence_file: str, chunk_size_sec: int
) -> Tuple[List[Tuple[ndarray, int]], int]:
    """Splits the audio file associated with a sequence file into N chunks of size chunk_size.
    Last chunk is padded

    Args:
        sequence_file (str): The name of the keypoints path (without .pkl).
        chunk_size_sec (int): Size in seconds for each chunks

    Returns:
        - List of audio corresponding to each chunks (audio, chunk id)
        - The sample rate

    """
    _, _, _, _, music, _ = sequence_file.split("_")
    y, sr = librosa.load(AUDIO_DIRECTORY / (music + ".mp3"))
    chunk_length = chunk_size_sec * sr
    output = []
    for chunk_id, music_index in enumerate(range(0, len(y), chunk_length)):
        output.append((y[music_index : music_index + chunk_length], chunk_id))

    output[-1] = (
        pad(output[-1][0], (0, chunk_length - len(output[-1])), constant_values=0),
        output[-1][1],
    )
    return output, sr
