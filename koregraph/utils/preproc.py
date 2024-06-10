from typing import Tuple
from numpy import ndarray


def cut_percentage(x: ndarray, perc: float) -> Tuple[ndarray, ndarray]:
    idx = len(x) - int(len(x) * perc)
    return x[:idx], x[idx:]
