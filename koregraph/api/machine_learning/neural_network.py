"""
    Utilities functions to initialize a model.
"""

from typing import List

from tensorflow.keras.layers import Dense, LSTM, Normalization, Dropout
from tensorflow.keras.models import Sequential, Model


def prepare_model(X, y) -> Model:
    """Initilizer a non compiled model with layers within.

    Args:
        input_shape (List[int]): The shape of the first layer (audio input for Koregraph)
        nb_features_out (int): The number of output (34 features for each Koregraph postures)

    Returns:
        Model: The non-compiled model.
    """

    normalization_layer = Normalization()
    normalization_layer.adapt(X)

    return Sequential(
        [
            normalization_layer,
            LSTM(256),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dropout(rate=0.5),
            Dense(64, activation="relu"),
            Dropout(rate=0.5),
            Dense(y.shape[1], activation="linear"),
        ]
    )


def compile_model(model: Model) -> Model:
    """Takes a non-compiled model and return the model compiled.

    Args:
        model (Model): The non-compiled model.

    Returns:
        Model: The compiled model.
    """

    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    return model


def initialize_model(X, y) -> Model:
    """Initialize a compiled model.

        @TODO: Add the input parameters to initialize the model.

    Returns:
        Model: The compiled model.
    """
    new_model = prepare_model(X, y)
    compiled_model = compile_model(new_model)
    return compiled_model


if __name__ == "__main__":
    initialize_model()