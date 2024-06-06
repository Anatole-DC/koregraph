from numpy import float32

from koregraph.api.machine_learning.neural_network import initialize_model
from koregraph.api.machine_learning.load_dataset import (
    load_preprocess_dataset,
)
from koregraph.api.machine_learning.callbacks import BackupCallback, StoppingCallback
from koregraph.utils.pickles import save_object_pickle

# from koregraph.api.preprocessing.audio_proc import scale_audio


def train_workflow(model_name: str = "model"):

    X, y = load_preprocess_dataset()

    # X_scaled = scale_audio(X)
    X_scaled = X
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
        callbacks=[BackupCallback, StoppingCallback],
    )

    save_object_pickle(model, model_name)
    save_object_pickle(history, model_name + "_history")


if __name__ == "__main__":
    train_workflow()
