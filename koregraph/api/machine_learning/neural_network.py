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
    Conv1D,
    Conv2D,
    Conv2DTranspose,
    MaxPooling1D,
    MaxPooling2D,
    Flatten,
    TimeDistributed,
    Input,
)
from keras.initializers import glorot_uniform
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop

from koregraph.api.machine_learning.loss import my_mse


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
            Bidirectional(LSTM(512, activation="relu", return_sequences=True)),
            # Bidirectional(LSTM(512, activation="relu", return_sequences=True)),
            Bidirectional(LSTM(256, activation="relu")),
            # Bidirectional(LSTM(128, activation="relu")),
            Dense(128, activation="relu"),
            Dense(128, activation="relu"),
            Dense(128, activation="relu"),
            Dropout(rate=0.2),
            Dense(64, activation="relu"),
            Dropout(rate=0.2),
            Dense(y.shape[1], activation="relu"),
        ]
    )


def compile_model(model: Model) -> Model:
    """Takes a non-compiled model and return the model compiled.

    Args:
        model (Model): The non-compiled model.

    Returns:
        Model: The compiled model.
    """

    model.compile(
        loss="mae",
        optimizer=RMSprop(
            learning_rate=0.005,
        ),
        metrics=["mae"],
    )
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

    # normalization_layer = Normalization()
    # normalization_layer.adapt(X)
    print("x 0 shape", X[0].shape)
    new_model = Sequential(
        [
            # normalization_layer,
            Bidirectional(
                LSTM(
                    256,
                    activation="tanh",
                    kernel_initializer=glorot_uniform(),
                    return_sequences=True,
                ),
            ),
            # # BatchNormalization(),
            LSTM(128, activation="tanh", recurrent_dropout=0.2),
            Dense(256, activation="relu"),
            Dense(256, activation="relu", activity_regularizer="l2"),
            Dense(128, activation="relu", activity_regularizer="l2"),
            Dropout(rate=0.2),
            Dense(64, activation="relu", activity_regularizer="l2"),
            Dropout(rate=0.2),
            Dense(y.shape[1], activation="linear"),
        ]
    )

    adam = Adam(learning_rate=0.00001)  # , clipvalue=0.01)
    new_model.compile(loss="mse", optimizer=adam, metrics=["mae"])
    return new_model


def initialize_model_next_chunks(X, y) -> Model:
    """Initialize a compiled model.

    Returns:
        Model: The compiled model.
    """

    new_model = Sequential(
        [
            Input(X[0].shape),
            Conv2D(
                512,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
            ),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(
                256,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
            ),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(
                128,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
            ),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(
                64,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
            ),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(
                128,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
            ),
            MaxPooling2D((2, 2), padding="same"),
            Conv2DTranspose(
                32, (3, 3), strides=(2, 2), padding="same", activation="relu"
            ),
            Conv2DTranspose(
                16, (3, 3), strides=(2, 2), padding="same", activation="relu"
            ),
            Conv2DTranspose(
                1, (3, 3), strides=(2, 2), padding="same", activation="relu"
            ),
            TimeDistributed(Flatten()),
            Bidirectional(
                LSTM(
                    512,
                    activation="tanh",
                    kernel_initializer=glorot_uniform(),
                    return_sequences=True,
                ),
            ),
            Bidirectional(
                LSTM(
                    256,
                    activation="tanh",
                    kernel_initializer=glorot_uniform(),
                    return_sequences=False,
                ),
            ),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dropout(rate=0.2),
            Dense(64, activation="relu"),
            Dropout(rate=0.2),
            Dense(y.shape[1], activation="relu"),
        ]
    )

    new_model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    return new_model


if __name__ == "__main__":
    initialize_model()
