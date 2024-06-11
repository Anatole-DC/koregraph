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

StandingPosture: Posture = Posture(
    [
        (933, 519),
        (939, 511),
        (927, 511),
        (950, 514),
        (921, 514),
        (963, 554),
        (915, 554),
        (980, 592),
        (896, 592),
        (982, 609),
        (890, 609),
        (957, 677),
        (924, 677),
        (959, 764),
        (913, 760),
        (965, 843),
        (911, 841),
    ]
)

default_2d: array = EmptyPosture.keypoints

LAST_CHUNK_TYPE = Enum("LastChunkType", ["PADDING", "ROLLING"])
