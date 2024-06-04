from pickle import dump as dump_pickle
from os import makedirs

from koregraph.api.music_to_numpy import music_to_numpy
from koregraph.api.posture_proc import generate_posture_array
from koregraph.params import (
    AUDIO_DIRECTORY,
    GENERATED_PICKLE_DIRECTORY,
    ALL_ADVANCED_MOVE_NAMES,
)


def generate_pickle_files():
    for move_name in ALL_ADVANCED_MOVE_NAMES:
        try:
            # Extract the music name from the move name
            audio_name = move_name.split("_")[4] + ".mp3"

            # Path to the pkl file
            pkl_file = move_name

            # Path to the corresponding audio file
            audio_file = AUDIO_DIRECTORY / audio_name

            # Generate posture array from pkl file
            posture_array = generate_posture_array(pkl_file)

            # Convert audio file to numpy array
            audio_array = music_to_numpy(audio_file)

            if len(posture_array) != len(audio_array):
                audio_array = audio_array[: len(posture_array)]

            # Check if the directory exists, create it if it doesn't
            makedirs(GENERATED_PICKLE_DIRECTORY, exist_ok=True)

            with open(
                GENERATED_PICKLE_DIRECTORY / ("generated_" + move_name), "wb"
            ) as f:
                dump_pickle([posture_array, audio_array], f)

        except Exception as e:
            print(f"Error in {move_name}: {e}")
            continue


if __name__ == "__main__":
    generate_pickle_files()
