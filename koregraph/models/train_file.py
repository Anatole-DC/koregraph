from dataclasses import dataclass

from numpy import ndarray


@dataclass
class TrainFile:
    X: ndarray
    y: ndarray
    music: str
    name: str  # Choregraphy original name

    def __post_init__(self):
        assert (
            self.X.shape[0] == self.y.shape[0]
        ), f"The number of observations and predictions does not match X={self.X.shape} and y={self.y.shape}"
