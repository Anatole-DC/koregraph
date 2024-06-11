from dataclasses import dataclass
from typing import List
from datetime import timedelta

from numpy import ndarray

from koregraph.models.posture import Posture


@dataclass
class Choregraphy:
    name: str
    keypoints2d: ndarray
    timestamps: ndarray = None

    def postures(self) -> List[Posture]:
        return [Posture(*keypoints) for keypoints in self.keypoints2d]

    @property
    def duration(self):
        if self.timestamps is None:
            return int(len(self.keypoints2d) / 60)
        return timedelta(milliseconds=int(self.timestamps[-1]))

    def __str__(self):
        return (
            f"{self.name}: {len(self.keypoints2d)} postures ({self.duration} seconds)"
        )
