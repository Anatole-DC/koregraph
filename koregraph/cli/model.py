from pathlib import Path

import numpy as np

from tensorflow.python.keras.models import Model

from koregraph.utils.pickle import load_pickle_object, save_object_pickle
from koregraph.api.music_to_numpy import music_to_numpy
from koregraph.params import MODEL_OUTPUT_DIRECTORY, AUDIO_DIRECTORY

def main():
    model: Model = load_pickle_object(MODEL_OUTPUT_DIRECTORY / "model.pkl")

    music_array = music_to_numpy(AUDIO_DIRECTORY / "mBR0.mp3")

    predictions = model.predict(music_array.reshape(-1, 1, 128))

    predictions_reshaped = predictions.reshape(-1, 17, 2)

    print(predictions_reshaped.shape)

    # save_object_pickle({"keypoints2d": predictions_reshaped, "timestamps": np.ones(predictions_reshaped.shape)}, "predictions")

if __name__ == "__main__":
    main()
