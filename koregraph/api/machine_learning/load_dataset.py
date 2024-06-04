"""
    All utilities functions to load the train dataset.
"""
from koregraph.utils.pickle import load_pickle_object

from koregraph.params import GENERATED_PICKLE_DIRECTORY

def load_preprocess_dataset():
    """
    Load and preprocess the dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The dataset.
    """
    y, X = load_pickle_object(GENERATED_PICKLE_DIRECTORY)

    return X, y


def check_dataset_format(): ...
