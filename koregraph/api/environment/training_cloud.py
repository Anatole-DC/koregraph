from numpy import float32
from mlflow import (
    create_experiment,
    get_experiment,
    get_experiment_by_name,
    start_run,
    tensorflow as mlflow_tensorflow,
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from koregraph.config.params import WEIGHTS_BACKUP_DIRECTORY
from koregraph.api.machine_learning.callbacks import BackupCallback, StoppingCallback
from koregraph.api.machine_learning.neural_network import initialize_model
from koregraph.api.machine_learning.load_dataset import load_preprocess_dataset
from koregraph.config.params import MODEL_OUTPUT_DIRECTORY
from koregraph.utils.controllers.pickles import save_object_pickle


def load_experiment(name: str):
    experiment = get_experiment_by_name(name)

    # Return the experiment if it exists
    if experiment is not None:
        print(f"Found already existing experiment {name}")
        return experiment

    # Create experiment if it does not exist
    print(f"Creating new experiment {name}")
    new_experiment_id = create_experiment(name)
    experiment = get_experiment(new_experiment_id)
    if experiment is None:
        raise Exception("An error occured while creating the experiment")

    return experiment


def run_mlflow_pipeline(model_name: str, dataset_size: float = 1.0):

    X, y = load_preprocess_dataset(dataset_size)

    # scaler = MinMaxScaler()
    # X_scaled = scaler.fit_transform(X)

    X = X.reshape((-1, 1, 128))

    y = y.astype(float32)

    print(X.shape)
    model = initialize_model(X, y)

    # experiment_id = create_experiment("test_cloud_run")
    experiment = load_experiment(model_name)

    with start_run(experiment_id=experiment.experiment_id) as mlflow_run:
        model.fit(
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
                # EarlyStopping(
                #     monitor="val_loss", patience=7, verbose=0, restore_best_weights=True
                # ),
            ],
        )

        print("Exporting model to pickle...")
        save_object_pickle(model, model_name)

        print("Exporting model to mlflow...")
        mlflow_tensorflow.log_model(
            model, artifact_path="test", registered_model_name=model_name
        )
        mlflow_tensorflow.save_model(
            model, path=MODEL_OUTPUT_DIRECTORY / f"{mlflow_run.info.run_name}"
        )


if __name__ == "__main__":
    run_mlflow_pipeline("test_with_gcloud")
