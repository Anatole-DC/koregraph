from tensorflow.keras.layers import Dense, LSTM, Normalization, Dropout, Bidirectional
from tensorflow.keras.models import Sequential, Model

MODEL_LIST = {
    "random-shit-put-together-1": Sequential(
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
    ),
    "robin-1": Sequential(
        [
            normalization_layer,
            Bidirectional(LSTM(256, activation="relu", return_sequences=True)),
            Bidirectional(LSTM(128, activation="relu")),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dropout(rate=0.2),
            Dense(64, activation="relu"),
            Dropout(rate=0.2),
            Dense(y.shape[1], activation="sigmoid"),
        ]
    ),
    "anatole-1": Sequential(
        [
            normalization_layer,
            Bidirectional(LSTM(256, activation="tanh", return_sequences=True)),
            Bidirectional(LSTM(128, activation="tanh", return_sequences=True)),
            Bidirectional(LSTM(64, activation="tanh")),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dropout(rate=0.2),
            Dense(64, activation="relu"),
            Dropout(rate=0.2),
            Dense(y.shape[1], activation="sigmoid"),
        ]
    ),
}
