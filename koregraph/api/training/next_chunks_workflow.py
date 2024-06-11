from numpy import expand_dims, float32, ndarray, isnan, any, isinf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from koregraph.api.machine_learning.neural_network import initialize_model_next_chunks
from koregraph.api.machine_learning.load_dataset import (
    load_next_chunks_preprocess_dataset as load_preprocess_dataset,
)
from koregraph.utils.controllers.pickles import save_object_pickle, load_pickle_object
from koregraph.config.params import (
    GENERATED_FEATURES_DIRECTORY,
    CHUNK_SIZE,
    PERCENTAGE_CUT,
    WEIGHTS_BACKUP_DIRECTORY,
    MODEL_OUTPUT_DIRECTORY,
    BUCKET_NAME,
)
from koregraph.api.machine_learning.callbacks import GCSCallback, HistorySaver


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

    X, X_audio, y = load_preprocess_dataset(dataset_size=dataset_size)
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
    X_audio = X_audio.reshape(-1, int(CHUNK_SIZE * 60), 128)
    y = y.reshape(-1, int(CHUNK_SIZE * PERCENTAGE_CUT * 60 * 17 * 2))

    print("Model X shape:", X.shape)
    print("Model X audio shape:", X_audio.shape)
    print("Model y shape:", y.shape)

    model = (
        initialize_model_next_chunks(X, X_audio, y)
        if backup_model is None
        else backup_model
    )

    model_backup_path = MODEL_OUTPUT_DIRECTORY / model_name
    model_backup_path.mkdir(parents=True, exist_ok=True)

    model_callbacks = [
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
            monitor="val_loss",
            patience=patience,
            verbose=0,
            restore_best_weights=True,
        ),
    ]

    model.summary()

    if with_cloud:
        model_callbacks.append(GCSCallback(model_backup_path, BUCKET_NAME))

    history = model.fit(
        x=[X, X_audio],
        y=y,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=model_callbacks,
    )

    print("Exporting model locally")
    (MODEL_OUTPUT_DIRECTORY / model_name).mkdir(exist_ok=True, parents=True)
    model.save(MODEL_OUTPUT_DIRECTORY / model_name / f"{model_name}.keras")
    model.save(MODEL_OUTPUT_DIRECTORY / model_name / f"{model_name}.h5")
    save_object_pickle(
        model,
        model_name,
        MODEL_OUTPUT_DIRECTORY / model_name / f"{model_name}.pkl",
    )
    save_object_pickle(
        history,
        model_name + "_history",
        MODEL_OUTPUT_DIRECTORY / model_name / f"{model_name}_history.pkl",
    )

    if with_cloud:
        print("Exporting model to google cloud storage")
        GCSCallback(model_backup_path, BUCKET_NAME).upload_file_to_gcs(
            MODEL_OUTPUT_DIRECTORY / model_name / f"{model_name}.keras",
            f"generated/models/{model_name}/",
        )
        GCSCallback(model_backup_path, BUCKET_NAME).upload_file_to_gcs(
            MODEL_OUTPUT_DIRECTORY / model_name / f"{model_name}.pkl",
            f"generated/models/{model_name}/",
        )
        GCSCallback(model_backup_path, BUCKET_NAME).upload_file_to_gcs(
            MODEL_OUTPUT_DIRECTORY / model_name / f"{model_name}.h5",
            f"generated/models/{model_name}/",
        )


if __name__ == "__main__":
    train_workflow()
