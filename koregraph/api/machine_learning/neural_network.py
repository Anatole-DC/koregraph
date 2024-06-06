"""
    Utilities functions to initialize a model.
"""

from typing import List

from keras.layers import (
    Dense,
    LSTM,
    Normalization,
    Dropout,
    Bidirectional,
    BatchNormalization,
)
from keras.models import Sequential, Model
from keras.optimizers import Adam


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
            Bidirectional(LSTM(256, activation="relu", return_sequences=True)),
            Bidirectional(LSTM(128, activation="relu", return_sequences=True)),
            Bidirectional(LSTM(64, activation="relu")),
            Dense(128, activation="relu"),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dropout(rate=0.2),
            Dense(64, activation="relu"),
            Dropout(rate=0.2),
            Dense(32, activation="relu"),
            Dropout(rate=0.2),
            Dense(y.shape[1], activation="sigmoid"),
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

    Returns:
        Model: The compiled model.
    """
    new_model = prepare_model(X, y)
    compiled_model = compile_model(new_model)
    return compiled_model


def initialize_model_chunks(X, y) -> Model:
    """Initialize a compiled model.

    Returns:
        Model: The compiled model.
    """

    normalization_layer = Normalization()
    normalization_layer.adapt(X)

    new_model = Sequential(
        [
            normalization_layer,
            Bidirectional(
                LSTM(
                    128, activation="tanh", return_sequences=True, recurrent_dropout=0.2
                )
            ),
            BatchNormalization(),
            Bidirectional(LSTM(64, activation="tanh", recurrent_dropout=0.2)),
            Dense(256, activation="relu", activity_regularizer="l2"),
            Dense(128, activation="relu", activity_regularizer="l2"),
            Dense(64, activation="relu", activity_regularizer="l2"),
            Dropout(rate=0.2),
            Dense(64, activation="relu", activity_regularizer="l2"),
            Dropout(rate=0.2),
            Dense(y.shape[1], activation="sigmoid"),
        ]
    )

    adam = Adam(learning_rate=0.00001, clipvalue=0.01)
    new_model.compile(loss="mse", optimizer=adam, metrics=["mae"])
    return new_model


if __name__ == "__main__":
    initialize_model()
