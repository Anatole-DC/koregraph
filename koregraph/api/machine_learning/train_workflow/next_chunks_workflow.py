from numpy import expand_dims, float32, ndarray, isnan, any, isinf

from koregraph.api.machine_learning.neural_network import initialize_model_next_chunks
from koregraph.api.machine_learning.load_dataset import (
    load_next_chunks_preprocess_dataset as load_preprocess_dataset,
)
from koregraph.api.machine_learning.callbacks import BackupCallback
from koregraph.utils.pickle import save_object_pickle, load_pickle_object
from sklearn.preprocessing import MinMaxScaler
from koregraph.params import GENERATED_FEATURES_DIRECTORY, CHUNK_SIZE, PERCENTAGE_CUT
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from koregraph.params import WEIGHTS_BACKUP_DIRECTORY


def train_workflow(model_name: str = "model"):

    X, y = load_preprocess_dataset()
    # X = load_pickle_object(GENERATED_FEATURES_DIRECTORY / "x.pkl")
    # y = load_pickle_object(GENERATED_FEATURES_DIRECTORY / "y.pkl")
    y = y.astype(float32)

    print("y has nan", isnan(y).any())
    print("X has nan", isnan(X).any())
    print("y has inf", isinf(y).any())
    print("X has inf", isinf(X).any())
    print("Y min", y.min())
    print("Y max", y.max())

    X = X.reshape(-1, int((CHUNK_SIZE * (1 - PERCENTAGE_CUT)) * 60), 17, 2)
    y = y.reshape(-1, int(CHUNK_SIZE * PERCENTAGE_CUT * 60 * 17 * 2))

    print("Model X shape:", X.shape)
    print("Model y shape:", y.shape)

    model = initialize_model_next_chunks(X, y)

    history = model.fit(
        x=X,
        y=y,
        validation_split=0.2,
        batch_size=16,
        epochs=1000,
        callbacks=[
            ModelCheckpoint(
                WEIGHTS_BACKUP_DIRECTORY / f"{model_name}_backup.keras",
                monitor="val_loss",
                verbose=0,
                save_best_only=False,
                save_weights_only=False,
                mode="auto",
                save_freq="epoch",
                initial_value_threshold=None,
            ),
            EarlyStopping(
                monitor="val_loss", patience=7, verbose=0, restore_best_weights=True
            ),
        ],
    )

    save_object_pickle(model, model_name)
    save_object_pickle(history, model_name + "_history")


if __name__ == "__main__":
    train_workflow()
