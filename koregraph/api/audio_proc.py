from koregraph.params import KEYPOINTS_DIRECTORY, FRAME_FORMAT, X_MIN, X_MAX


def scale(X, X_min: float = X_MIN, X_max: float = X_MAX):
    X_std = (X - X_min) / (X_max - X_min)
    return X_std
