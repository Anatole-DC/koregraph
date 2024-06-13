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
    concatenate,
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


def initialize_model_next_chunks(X, X_audio, y) -> Model:
    """Initialize a compiled model.

    Returns:
        Model: The compiled model.
    """

    inputs = Input(shape=X[0].shape)
    conv2d_1 = Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same")(
        inputs
    )
    mp1 = MaxPooling2D((2, 2), padding="same")(conv2d_1)
    conv2d_2 = Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same")(mp1)
    mp2 = MaxPooling2D((2, 2), padding="same")(conv2d_2)
    conv2d_3 = Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")(mp2)
    mp3 = MaxPooling2D((2, 2), padding="same")(conv2d_3)
    conv2d_4 = Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(mp3)
    mp4 = MaxPooling2D((2, 2), padding="same")(conv2d_4)
    conv2d_5 = Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")(mp4)
    mp5 = MaxPooling2D((2, 2), padding="same")(conv2d_5)

    conv2d_transpose_1 = Conv2DTranspose(
        64, (3, 3), strides=(2, 2), padding="same", activation="relu"
    )(mp5)
    conv2d_transpose_2 = Conv2DTranspose(
        32, (3, 3), strides=(2, 2), padding="same", activation="relu"
    )(conv2d_transpose_1)
    conv2d_transpose_3 = Conv2DTranspose(
        16, (3, 3), strides=(2, 2), padding="same", activation="relu"
    )(conv2d_transpose_2)
    conv2d_transpose_4 = Conv2DTranspose(
        1, (3, 3), strides=(2, 2), padding="same", activation="relu"
    )(conv2d_transpose_3)

    time_dist = TimeDistributed(Flatten())(conv2d_transpose_4)

    lstm1 = Bidirectional(
        LSTM(
            512,
            activation="tanh",
            kernel_initializer=glorot_uniform(),
            return_sequences=True,
        ),
    )(time_dist)
    lstm2 = Bidirectional(
        LSTM(
            256,
            activation="tanh",
            kernel_initializer=glorot_uniform(),
            return_sequences=False,
        ),
    )(lstm1)
    dense1 = Dense(256, activation="relu")(lstm2)
    dense2 = Dense(128, activation="relu")(dense1)
    dropout = Dropout(rate=0.2)(dense2)
    dense3 = Dense(64, activation="relu")(dropout)
    dropout2 = Dropout(rate=0.2)(dense3)
    dense_ouput = Dense(32, activation="relu")(dropout2)

    input_audio = Input(shape=X_audio[0].shape)
    dense_audio_1 = Dense(256, activation="relu")(input_audio)
    dense_audio_2 = Dense(128, activation="relu")(dense_audio_1)
    dropout_audio = Dropout(rate=0.2)(dense_audio_2)
    dense_audio_3 = Dense(64, activation="relu")(dropout_audio)
    dropout_audio_2 = Dropout(rate=0.2)(dense_audio_3)
    flatten = Flatten()(dropout_audio_2)
    dense_audio_ouput = Dense(32, activation="relu")(flatten)

    joined = concatenate([dense_ouput, dense_audio_ouput])

    dense_all_1 = Dense(256, activation="relu")(joined)
    dense_all_2 = Dense(128, activation="relu")(dense_all_1)
    dropou_all_t = Dropout(rate=0.2)(dense_all_2)
    dense_all_3 = Dense(64, activation="relu")(dropou_all_t)
    dropout_all_2 = Dropout(rate=0.2)(dense_all_3)
    chore_outputs = Dense(y.shape[1], activation="relu")(dropout_all_2)

    new_model = Model(
        inputs=[inputs, input_audio], outputs=chore_outputs, name="chore_model"
    )

    new_model.summary()

    new_model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    return new_model

def initialize_model_seventeen_output(Xs, y) -> Model:
    """Initialize a compiled model.

    Returns:
        Model: The compiled model.
    """

    # inputs = Input(shape=X[0].shape)
    models = []
    outputs = []
    inputs = []
    print('len Xs', len(Xs))
    for i in range(len(Xs)):
        print(f'articulation {i}')
        articulation = Input(shape=Xs[i][0].shape)
        inputs.append(articulation)
        lstm1 = Bidirectional(
            LSTM(
                512,
                activation="tanh",
                kernel_initializer=glorot_uniform(),
                return_sequences=True,
            ),
        )(articulation)
        lstm2 = Bidirectional(
            LSTM(
                256,
                activation="tanh",
                kernel_initializer=glorot_uniform(),
                return_sequences=False,
            ),
        )(lstm1)
        dense1 = Dense(256, activation="relu")(lstm2)
        dense2 = Dense(128, activation="relu")(dense1)
        dropout = Dropout(rate=0.2)(dense2)
        dense3 = Dense(64, activation="relu")(dropout)
        dropout2 = Dropout(rate=0.2)(dense3)
        dense_ouput = Dense(32, activation="relu")(dropout2)
        chore_outputs = Dense(2, activation="relu")(dense_ouput)

        outputs.append(chore_outputs)
        models.append(Model(
            inputs=articulation, outputs=chore_outputs, name=f"{i}_model"
        ))

    print(type(outputs[0]))

    # outputs = concatenate([m.outputs for m in models])
    outputs_layer = concatenate(outputs)

    dense_all_1 = Dense(256, activation="relu")(outputs_layer)
    dense_all_2 = Dense(128, activation="relu")(dense_all_1)
    dropout_all_t = Dropout(rate=0.2)(dense_all_2)
    dense_all_3 = Dense(64, activation="relu")(dropout_all_t)
    dropout_all_2 = Dropout(rate=0.2)(dense_all_3)

    chore_output = Dense(y.shape[-1], activation="relu")(dropout_all_2)
    new_model = Model(
        inputs=inputs, outputs=chore_output, name="seventeen_model"
    )

    new_model.summary()

    new_model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    return new_model


if __name__ == "__main__":
    initialize_model()
