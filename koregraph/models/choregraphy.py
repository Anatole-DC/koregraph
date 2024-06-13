from dataclasses import dataclass
from typing import List
from datetime import timedelta

from numpy import ndarray

from koregraph.models.posture import Posture


@dataclass
class Choregraphy:
    name: str
    keypoints: ndarray

    def postures(self) -> List[Posture]:
        return [Posture(*keypoints) for keypoints in self.keypoints2d]

    @property
    def duration(self):
        return int(len(self.keypoints) / 60)

    def __str__(self):
        return (
            f"{self.name}: {len(self.keypoints)} postures ({self.duration} seconds)"
        )
