from numpy import array
from enum import Enum

from koregraph.models.posture import Posture

EmptyPosture: Posture = Posture(
    nose=(-1, -1),
    left_eye=(-1, -1),
    right_eye=(-1, -1),
    left_ear=(-1, -1),
    right_ear=(-1, -1),
    left_shoulder=(-1, -1),
    right_shoulder=(-1, -1),
    left_elbow=(-1, -1),
    right_elbow=(-1, -1),
    left_wrist=(-1, -1),
    right_wrist=(-1, -1),
    left_hip=(-1, -1),
    right_hip=(-1, -1),
    left_knee=(-1, -1),
    right_knee=(-1, -1),
    left_ankle=(-1, -1),
    right_ankle=(-1, -1),
)

default_2d: array = array(
    [
        EmptyPosture.nose,
        EmptyPosture.left_eye,
        EmptyPosture.right_eye,
        EmptyPosture.left_ear,
        EmptyPosture.right_ear,
        EmptyPosture.left_shoulder,
        EmptyPosture.right_shoulder,
        EmptyPosture.left_elbow,
        EmptyPosture.right_elbow,
        EmptyPosture.left_wrist,
        EmptyPosture.right_wrist,
        EmptyPosture.left_hip,
        EmptyPosture.right_hip,
        EmptyPosture.left_knee,
        EmptyPosture.right_knee,
        EmptyPosture.left_ankle,
        EmptyPosture.right_ankle,
    ]
)

LAST_CHUNK_TYPE = Enum("LastChunkType", ["PADDING", "ROLLING"])
