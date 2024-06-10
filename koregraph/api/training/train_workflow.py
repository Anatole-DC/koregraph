from gc import callbacks
from numpy import float32

from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from koregraph.config.params import MODEL_OUTPUT_DIRECTORY, WEIGHTS_BACKUP_DIRECTORY
from koregraph.api.machine_learning.neural_network import initialize_model
from koregraph.api.machine_learning.load_dataset import (
    load_preprocess_dataset,
)
from koregraph.api.machine_learning.callbacks import GCSCallback, HistorySaver
from koregraph.utils.controllers.pickles import save_object_pickle

# from koregraph.api.preprocessing.audio_proc import scale_audio


def train_workflow(
    model_name: str = "model",
    epochs: int = 16,
    batch_size: int = 16,
    dataset_size: float = 1.0,
    backup_model: Model = None,
    initial_epoch: int = 0,
    patience: int = 20,
    with_cloud: bool = False,
):
    print(f"Run training in {'cloud' if with_cloud else 'local'} mode")

    X, y = load_preprocess_dataset(dataset_size)

    # X_scaled = scale_audio(X)
    X_scaled = X
    X_scaled = X_scaled.reshape((-1, 1, 128))

    y = y.astype(float32)

    print(X_scaled.shape)
    model = initialize_model(X, y) if backup_model is None else backup_model

    model_backup_path = WEIGHTS_BACKUP_DIRECTORY / model_name
    model_backup_path.mkdir(parents=True, exist_ok=True)

    model_callbacks = [
        ModelCheckpoint(
            model_backup_path / f"{model_name}_backup.keras",
            monitor="val_loss",
            verbose=0,
            save_best_only=False,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
            initial_value_threshold=None,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            verbose=0,
            restore_best_weights=True,
        ),
        HistorySaver(model_backup_path / f"{model_name}_history.pkl"),
    ]

    if with_cloud:
        model_callbacks.append(GCSCallback(model_backup_path, "koregraph"))

    history = model.fit(
        x=X_scaled,
        y=y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        initial_epoch=initial_epoch,
        callbacks=model_callbacks,
    )

    model.save(MODEL_OUTPUT_DIRECTORY / f"{model_name}.keras")
    save_object_pickle(model, model_name)
    save_object_pickle(history, model_name + "_history")


if __name__ == "__main__":
    train_workflow()
