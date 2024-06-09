"""
    All functions to generate the training dataset.
"""

from pathlib import Path
from pickle import dump as dump_pickle

from koregraph.api.preprocessing.audio_preprocessing import music_to_numpy
from koregraph.config.params import (
    ALL_ADVANCED_MOVE_NAMES,
    AUDIO_DIRECTORY,
    GENERATED_PICKLE_DIRECTORY,
)
from koregraph.utils.controllers.choregraphies import load_choregraphy
from koregraph.api.preprocessing.posture_preprocessing import convert_to_train_posture


def generate_training_pickles(
    mode: str = "barbarie", output_path: Path = GENERATED_PICKLE_DIRECTORY
) -> Path:
    """Generate the files for the model training.

    Files will be generated at the 'output_path/mode/' directory.

    Args:
        mode (str, optional): The preprocessing technic according to the model used. Defaults to "barbarie".
        output_path (Path, optional): The generated files path. Defaults to GENERATED_PICKLE_DIRECTORY.

    Returns:
        Path: The Path to the output directory.
    """

    # Ensure the output path exists
    generated_pickles_path = output_path / mode
    generated_pickles_path.mkdir(exist_ok=True, parents=True)

    # Pickle generation
    for move_file in ALL_ADVANCED_MOVE_NAMES:
        # Load the music and the choregraphy
        music = music_to_numpy(AUDIO_DIRECTORY / move_file.music)
        choregraphy = load_choregraphy(move_file)

        # Preprocess the choregraphy
        train_choregraphy = None
        if mode == "barbarie":
            train_choregraphy = convert_to_train_posture(choregraphy)
        elif mode == "chunks":
            # @TODO: implement chunk conversion for posture and music
            train_choregraphy = convert_to_train_posture(choregraphy)

        # Ensure X and y have the same length
        if len(train_choregraphy) != len(music):
            music = music[: len(train_choregraphy)]

        assert len(train_choregraphy) == len(
            music
        ), f"In {move_file.name}, preprocessed choregraphy and music don't have the same length ({len(train_choregraphy)=}, {len(music)=})"

        # Save the training file
        with open(
            generated_pickles_path / f"train_{move_file.name}.pkl", "wb"
        ) as train_file:
            dump_pickle([music, train_choregraphy], train_file)

    print(
        f"{len(list(generated_pickles_path.glob('*.pkl')))} training files were generated in {generated_pickles_path.absolute()}"
    )
    return generated_pickles_path


if __name__ == "__main__":
    generate_training_pickles()
