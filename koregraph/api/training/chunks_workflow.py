from numpy import expand_dims, float32, ndarray, isnan, any, isinf

from koregraph.api.machine_learning.neural_network import initialize_model_chunks
from koregraph.api.machine_learning.load_dataset import (
    load_chunk_preprocess_dataset,
)
from koregraph.utils.controllers.pickles import save_object_pickle


def train_chunks_workflow(model_name: str = "model"):

    X, y = load_chunk_preprocess_dataset()

    y = y.astype(float32)
    print("y has nan", isnan(y).any())
    print("X has nan", isnan(X).any())
    print("y has inf", isinf(y).any())
    print("X has inf", isinf(X).any())
    print("Y min", y.min())
    print("Y max", y.max())

    print("Model X shape:", X.shape)
    print("Model y shape:", y.shape)

    model = initialize_model_chunks(X, y)

    history = model.fit(
        x=X,
        y=y,
        validation_split=0.2,
        batch_size=16,
        epochs=50,
    )

    save_object_pickle(model, model_name)
    save_object_pickle(history, model_name + "_history")


if __name__ == "__main__":
    train_chunks_workflow()
