from dataclasses import dataclass, field
from typing import List, Tuple

from numpy import ndarray

from koregraph.models.types import Point


@dataclass
class Posture:
    """Represents a single posture within a choregraphy sequence"""

    keypoints: ndarray

    nose: Point = field(init=False)
    left_eye: Point = field(init=False)
    right_eye: Point = field(init=False)
    left_ear: Point = field(init=False)
    right_ear: Point = field(init=False)
    left_shoulder: Point = field(init=False)
    right_shoulder: Point = field(init=False)
    left_elbow: Point = field(init=False)
    right_elbow: Point = field(init=False)
    left_wrist: Point = field(init=False)
    right_wrist: Point = field(init=False)
    left_hip: Point = field(init=False)
    right_hip: Point = field(init=False)
    left_knee: Point = field(init=False)
    right_knee: Point = field(init=False)
    left_ankle: Point = field(init=False)
    right_ankle: Point = field(init=False)

    def __post_init__(self):
        (
            # Face
            self.nose,
            self.left_eye,
            self.right_eye,
            self.left_ear,
            self.right_ear,
            # Trunc
            self.left_shoulder,
            self.right_shoulder,
            self.left_elbow,
            self.right_elbow,
            self.left_wrist,
            self.right_wrist,
            # Legs
            self.left_hip,
            self.right_hip,
            self.left_knee,
            self.right_knee,
            self.left_ankle,
            self.right_ankle,
        ) = self.keypoints

    def bones(self) -> List[Tuple[Point, Point]]:
        return [
            # Face
            (self.nose, self.left_eye),
            (self.nose, self.right_eye),
            (self.right_eye, self.right_ear),
            (self.left_eye, self.left_ear),
            # Trunc
            (self.left_shoulder, self.left_elbow),
            (self.right_shoulder, self.right_elbow),
            (self.left_elbow, self.left_wrist),
            (self.right_elbow, self.right_wrist),
            # Legs
            (self.left_hip, self.left_knee),
            (self.right_hip, self.right_knee),
            (self.left_knee, self.left_ankle),
            (self.right_knee, self.right_ankle),
        ]
