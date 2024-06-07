from pathlib import Path
from pickle import dump as pickle_dump

from tensorflow.keras.callbacks import Callback

from koregraph.utils.controllers.pickles import load_pickle_object


class HistorySaver(Callback):
    def __init__(self, history_path):
        super(HistorySaver, self).__init__()
        self.history_path: Path = history_path

    def on_epoch_end(self, epoch: int, logs=None):
        if self.history_path.exists():
            history = load_pickle_object(self.history_path)
            for key, value in logs.items():
                history[key].append(value)
        else:
            history = {key: [value] for key, value in logs.items()}
        with open(self.history_path, "wb") as f:
            pickle_dump(history, f)
