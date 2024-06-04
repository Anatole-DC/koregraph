"""
    Utilities functions to initialize a model.
"""

from typing import List

from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras.models import Sequential, Model


def prepare_model(input_shape: List[int], nb_features_out: int) -> Model:
    """Initilizer a non compiled model with layers within.

    Args:
        input_shape (List[int]): The shape of the first layer (audio input for Koregraph)
        nb_features_out (int): The number of output (34 features for each Koregraph postures)

    Returns:
        Model: The non-compiled model.
    """

    return Sequential(
        [
            LSTM(32, input_shape=input_shape),
            Dense(nb_features_out, activation="softmax"),
        ]
    )


def compile_model(model: Model) -> Model:
    """Takes a non-compiled model and return the model compiled.

    Args:
        model (Model): The non-compiled model.

    Returns:
        Model: The compiled model.
    """

    model.compile(loss="mse", optimizer="adam", metrics="mae")
    return model


def initialize_model() -> Model:
    """Initialize a compiled model.

        @TODO: Add the input parameters to initialize the model.

    Returns:
        Model: The compiled model.
    """
    new_model = prepare_model([5], 2)
    compiled_model = compile_model(new_model)
    return compiled_model


if __name__ == "__main__":
    initialize_model()
