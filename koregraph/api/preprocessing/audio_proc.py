from numpy import ndarray

from koregraph.config.params import X_MIN, X_MAX


def scale_audio(X: ndarray, X_min: float = X_MIN, X_max: float = X_MAX) -> ndarray:
    """Apply a MinMax scaler to an audio.

    Args:
        X (ndarray): The audio to scale.
        X_min (float, optional): The minimum value within x. Defaults to X_MIN.
        X_max (float, optional): The maximum value within x. Defaults to X_MAX.

    Returns:
        ndarray: The scaled audio.
    """

    X_std = (X - X_min) / (X_max - X_min)
    return X_std
