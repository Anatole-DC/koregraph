from numpy import expand_dims, float32, ndarray

from koregraph.api.machine_learning.neural_network import initialize_model
from koregraph.api.machine_learning.load_dataset import (
    load_preprocess_dataset,
    check_dataset_format,
)
from koregraph.api.machine_learning.callbacks import BackupCallback, StoppingCallback
from koregraph.utils.pickle import save_object_pickle
from sklearn.preprocessing import MinMaxScaler
from koregraph.api.audio_proc import scale_audio
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from koregraph.params import WEIGHTS_BACKUP_DIRECTORY


def train_workflow(model_name: str = "model"):

    X, y = load_preprocess_dataset()

    X_scaled = scale_audio(X)
    X_scaled = X_scaled.reshape((-1, 1, 128))

    y = y.astype(float32)

    print(X_scaled.shape)
    model = initialize_model(X, y)

    history = model.fit(
        x=X_scaled,
        y=y,
        validation_split=0.2,
        epochs=20,
        batch_size=16,
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
