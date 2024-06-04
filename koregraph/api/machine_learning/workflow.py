from .neural_network import initialize_model
from .load_dataset import load_preprocess_dataset, check_dataset_format
from .callbacks import BackupCallback
from koregraph.utils.pickle import save_object_pickle


def train_workflow():
    X, y = load_preprocess_dataset()

    check_dataset_format(X, y)

    model = initialize_model()

    history = model.fit(
        x=X,
        y=y,
        validation_data=0.2,
        batch_size=16,
        epochs=20,
        callbacks=[BackupCallback],
    )

    save_object_pickle(model, "model")
    save_object_pickle(history, "model_history")


if __name__ == "__main__":
    train_workflow()
