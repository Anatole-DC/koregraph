from numpy import array
from enum import Enum

from koregraph.models.posture import Posture

EmptyPosture: Posture = Posture(
    [
        (-1, -1),
        (-1, -1),
        (-1, -1),
        (-1, -1),
        (-1, -1),
        (-1, -1),
        (-1, -1),
        (-1, -1),
        (-1, -1),
        (-1, -1),
        (-1, -1),
        (-1, -1),
        (-1, -1),
        (-1, -1),
        (-1, -1),
        (-1, -1),
        (-1, -1),
    ]
)

default_2d: array = EmptyPosture.keypoints

LAST_CHUNK_TYPE = Enum("LastChunkType", ["PADDING", "ROLLING"])
