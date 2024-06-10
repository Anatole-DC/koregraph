"""
    All utilities functions to load the train dataset.
"""

from random import sample

from numpy import ndarray, append

from koregraph.utils.controllers.pickles import load_pickle_object
from koregraph.config.params import GENERATED_PICKLE_DIRECTORY


def load_preprocess_dataset(
    dataset_size: float = 1.0, mode: str = "barbarie"
) -> tuple[ndarray, ndarray]:
    """
    Load and preprocess the dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The dataset.
    """

    # Retrieve the training files
    train_file_path = GENERATED_PICKLE_DIRECTORY / mode
    all_files = list(train_file_path.glob("*.pkl"))

    # Compute the random sample size
    sample_rate = int(len(all_files) * dataset_size)

    # Load the first file to determine the shape of the final dataset
    base_shape_file = all_files[0]
    X, y = load_pickle_object(base_shape_file)

    # Sample size based on the dataset_size
    files = sample(all_files, sample_rate)

    # Load all files and concatenate them
    for file in files:
        if file == base_shape_file:
            continue
        X_tmp, y_tmp = load_pickle_object(file)
        X = append(X, X_tmp, axis=0)
        y = append(y, y_tmp, axis=0)

    print(f"Preprocess dataset used {len(files)} files")

    return X, y


def check_dataset_format(): ...


if __name__ == "__main__":
    load_preprocess_dataset(0.5)
