from koregraph.params import KEYPOINTS_DIRECTORY, FRAME_FORMAT, X_MIN, X_MAX
from numpy import ndarray


def scale_audio(X: ndarray, X_min: float = X_MIN, X_max: float = X_MAX) -> ndarray:
    X_std = (X - X_min) / (X_max - X_min)
    return X_std
