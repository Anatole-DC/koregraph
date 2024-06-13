"""
    All functions to generate the training dataset.
"""

from pathlib import Path
from pickle import dump as dump_pickle

from numpy import ndarray
from tqdm import trange

from koregraph.api.preprocessing.audio_preprocessing import (
    convert_music_array_to_train_audio,
    music_to_numpy,
)
from koregraph.config.params import (
    ALL_ADVANCED_MOVE_NAMES,
    ALL_ADVANCED_MOVE_NAMES_3D,
    AUDIO_DIRECTORY,
    GENERATED_AUDIO_SILENCE_DIRECTORY,
    GENERATED_PICKLE_DIRECTORY,
)
from koregraph.models.aist_file import AISTFile
from koregraph.models.choregraphy import Choregraphy
from koregraph.tools.video_builder import (
    keypoints_video_audio_builder_from_choreography,
)
from koregraph.utils.controllers.choregraphies import load_choregraphy
from koregraph.api.preprocessing.posture_preprocessing import (
    convert_to_train_posture,
    generate_and_export_choreography,
    posture_array_to_keypoints,
    upscale_posture_pred,
)
from koregraph.utils.controllers.musics import load_music, save_audio_chunk


def downsample_array(array: ndarray, target_fps: int = 30, original_fps: int = 60):
    """Downsample an array from original_fps to target_fps."""
    factor = original_fps // target_fps
    return array[::factor]


def generate_training_pickles(
    mode: str = "barbarie",
    output_path: Path = GENERATED_PICKLE_DIRECTORY,
    output_preprocessed_audio: bool = False,
    interpolation_mode: str = None,
    downsaple_fps: int = None,
    dimension: int = 2
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
    for move_file in [ALL_ADVANCED_MOVE_NAMES, ALL_ADVANCED_MOVE_NAMES_3D][dimension - 2]:

        music_path = (
            GENERATED_AUDIO_SILENCE_DIRECTORY / f"silence_{move_file.music}.mp3"
            if interpolation_mode is not None
            else AUDIO_DIRECTORY / f"{move_file.music}.mp3"
        )

        music, _ = load_music(music_path)
        choregraphy = load_choregraphy(move_file, dimension)

        # Preprocess the choregraphy
        train_choregraphy = None
        raw_music = None
        if mode == "barbarie":
            train_choregraphy = convert_to_train_posture(
                choregraphy, (interpolation_mode is not None), dimension
            )
            train_audio, raw_music = convert_music_array_to_train_audio(music)

        elif mode == "chunks":
            # @TODO: implement chunk conversion for posture and music
            train_choregraphy = convert_to_train_posture(choregraphy)

        if output_preprocessed_audio:
            save_audio_chunk(
                raw_music,
                44100,
                GENERATED_AUDIO_SILENCE_DIRECTORY / f"train_{move_file.music}.mp3",
            )

        # Ensure X and y have the same length
        if len(train_choregraphy) != len(train_audio):
            train_audio = train_audio[: len(train_choregraphy)]

        if downsaple_fps is not None:
            train_audio = downsample_array(train_audio, downsaple_fps)
            train_choregraphy = downsample_array(train_choregraphy, downsaple_fps)

        assert len(train_choregraphy) == len(
            train_audio
        ), f"In {move_file.name}, preprocessed choregraphy and music don't have the same length ({len(train_choregraphy)=}, {len(train_audio)=})"

        # Save the training file
        with open(
            generated_pickles_path / f"train_{move_file.name}.pkl", "wb"
        ) as train_file:
            dump_pickle([train_audio, train_choregraphy], train_file)

    print(
        f"{len(list(generated_pickles_path.glob('*.pkl')))} training files were generated in {generated_pickles_path.absolute()}"
    )
    return generated_pickles_path


if __name__ == "__main__":
    test_file = ALL_ADVANCED_MOVE_NAMES[0]

    chore = load_choregraphy(test_file)

    posture = convert_to_train_posture(chore)

    from shutil import copyfile

    copyfile(
        (GENERATED_AUDIO_SILENCE_DIRECTORY / f"silence_{test_file.music}.mp3"),
        (AUDIO_DIRECTORY / f"silence_{test_file.music}.mp3"),
    )

    keypoints_video_audio_builder_from_choreography(
        Choregraphy(
            "test_audio", upscale_posture_pred(posture_array_to_keypoints(posture))
        ),
        f"silence_{test_file.music}",
    )
