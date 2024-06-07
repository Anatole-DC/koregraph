from typing import List, Tuple
import pickle
import librosa
from numpy import ndarray, pad, array, concatenate, tile

from koregraph.models.choregraphy import Choregraphy
from koregraph.config.params import (
    KEYPOINTS_DIRECTORY,
    AUDIO_DIRECTORY,
    LAST_CHUNK_TYPE_STRATEGY,
)
from koregraph.models.constants import default_2d, LAST_CHUNK_TYPE


def split_sequence(
    sequence_file: str,
    chunk_size_sec: int,
    last_chunk_type: LAST_CHUNK_TYPE = LAST_CHUNK_TYPE_STRATEGY,
) -> List[Choregraphy]:
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

        output[-1] = last_chunk_sequence(
            output[-1],
            chunk_size_sec * 60,
            type=last_chunk_type,
            previous_chore=output[-2],
        )

    return output


def last_chunk_sequence(
    last_chore: Choregraphy,
    chunk_length: int,
    type: LAST_CHUNK_TYPE = LAST_CHUNK_TYPE_STRATEGY,
    previous_chore: Choregraphy = None,
):
    if type == LAST_CHUNK_TYPE.PADDING:
        # Postures
        last_chore.keypoints2d = concatenate(
            (
                last_chore.keypoints2d,
                tile(default_2d, (chunk_length - len(last_chore.timestamps), 1, 1)),
            )
        )

        # Timestamp
        last_tp = last_chore.timestamps[-1]
        padding_ts = array(
            [
                last_tp + (i * round(5000 / 3))
                for i in range(1, chunk_length - len(last_chore.timestamps) + 1)
            ]
        )
        last_chore.timestamps = concatenate((last_chore.timestamps, padding_ts))
        return last_chore

    if type == LAST_CHUNK_TYPE.ROLLING:
        assert previous_chore is not None
        last_chore.keypoints2d = concatenate(
            (
                previous_chore.keypoints2d[len(last_chore.keypoints2d) :, :, :],
                last_chore.keypoints2d,
            )
        )
        last_tp = last_chore.timestamps[-1]
        padding_ts = array(
            [
                last_tp + (i * round(5000 / 3))
                for i in range(1, chunk_length - len(last_chore.timestamps) + 1)
            ]
        )
        last_chore.timestamps = concatenate((last_chore.timestamps, padding_ts))
        return last_chore


def split_audio(
    sequence_file: str,
    chunk_size_sec: int,
    last_chunk_type: LAST_CHUNK_TYPE = LAST_CHUNK_TYPE_STRATEGY,
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

    # pad last sequence
    output[-1] = (
        last_chunk_audio(
            output[-1][0],
            chunk_length,
            type=last_chunk_type,
            previous_chunk=output[-2][0],
        ),
        output[-1][1],
    )

    print(output[-1][0].shape)
    print(
        last_chunk_audio(
            output[-1][0],
            chunk_length,
            type=LAST_CHUNK_TYPE.PADDING,
            previous_chunk=output[-2][0],
        ).shape
    )
    return output, sr


def last_chunk_audio(
    last_chunk: ndarray,
    chunk_length: int,
    type: LAST_CHUNK_TYPE = LAST_CHUNK_TYPE_STRATEGY,
    previous_chunk: ndarray = None,
):
    if type == LAST_CHUNK_TYPE.PADDING:
        return pad(last_chunk, (0, chunk_length - len(last_chunk)), constant_values=0)
    if type == LAST_CHUNK_TYPE.ROLLING:
        assert previous_chunk is not None
        return concatenate((previous_chunk[len(last_chunk) :], last_chunk))
