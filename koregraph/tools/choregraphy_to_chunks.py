import os
from typing import List, Tuple
from pathlib import Path
import pickle
import librosa
from numpy import ndarray, pad, array, concatenate, tile

from koregraph.models.choregraphy import Choregraphy
from koregraph.params import KEYPOINTS_DIRECTORY, AUDIO_DIRECTORY
from koregraph.models.constants import default_2d


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
                    keypoints2d=postures[0, i : i + 60 * chunk_size_sec, :, :2],
                    timestamps=timestamps[i : i + 60 * chunk_size_sec],
                )
            )

        # TODO pad last sequence
        last_chore = output[-1]
        chunk_final_size = 60 * chunk_size_sec

        last_chore.keypoints2d = concatenate(
            (
                last_chore.keypoints2d,
                tile(default_2d, (chunk_final_size - len(last_chore.timestamps), 1, 1)),
            )
        )

        # Timestamp
        last_tp = last_chore.timestamps[-1]
        padding_ts = array(
            [
                last_tp + (i * round(5000 / 3))
                for i in range(1, chunk_final_size - len(last_chore.timestamps) + 1)
            ]
        )
        last_chore.timestamps = concatenate((last_chore.timestamps, padding_ts))

        output[-1] = last_chore

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
        pad(output[-1][0], (0, chunk_length - len(output[-1][0])), constant_values=0),
        output[-1][1],
    )
    return output, sr
