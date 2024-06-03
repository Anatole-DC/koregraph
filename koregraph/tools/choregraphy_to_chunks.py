import os
from typing import List
from pathlib import Path
import pickle

from koregraph.models.choregraphy import Choregraphy


KEYPOINTS_DIRECTORY: Path = Path(
    os.environ.get(
        "KEYPOINTS_DIRECTORY",
        "data/keypoints2d",
    )
)

def split_sequence(sequence_file:str, chunk_size_sec:int) -> List[Choregraphy]:
    '''Splits a sequence file into N chunks of size chunk_size
    (last posture can be shorter)
    Args:
        sequence_file (str): The name of the keypoints path (without .pkl).
        chunk_size_sec (int): Size in seconds for each chunks

    Returns:
        List of Choregraphy corresponding to each chunks

    '''

    with open(KEYPOINTS_DIRECTORY / (sequence_file + '.pkl'), 'rb') as f:
        data = pickle.load(f)
        postures = data['keypoints2d']
        timestamps = data['timestamps']
        output = []
        for i in range(0, len(timestamps), chunk_size_sec*60):
            output.append(Choregraphy(
                name=sequence_file + f'_{i}',
                keypoints2d=postures[:,i:i + 60*chunk_size_sec, :, :],
                timestamps=timestamps[i:i+60*chunk_size_sec]
            ))

    return output
