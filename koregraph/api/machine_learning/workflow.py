from numpy import expand_dims, float32

from koregraph.api.machine_learning.neural_network import initialize_model
from koregraph.api.machine_learning.load_dataset import (
    load_preprocess_dataset,
    check_dataset_format,
)
from koregraph.api.machine_learning.callbacks import BackupCallback
from koregraph.utils.pickle import save_object_pickle
from sklearn.preprocessing import MinMaxScaler

def train_workflow():

    X, y = load_preprocess_dataset()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X = X_scaled.reshape((-1, 1, 128))

    y = y.astype(float32)

    print(X.shape)
    model = initialize_model(X, y)

    history = model.fit(
        x=X,
        y=y,
        validation_split=0.2,
        batch_size=16,
        epochs=20,
        # callbacks=[BackupCallback],
    )

    save_object_pickle(model, "model")
    save_object_pickle(history, "model_history")


if __name__ == "__main__":
    train_workflow()
