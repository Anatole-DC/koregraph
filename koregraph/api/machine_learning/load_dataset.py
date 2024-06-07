"""
    All utilities functions to load the train dataset.
"""

from typing import Any, Tuple
from os import listdir

from numpy import ndarray, append, isnan, any, nan_to_num, concatenate, split, delete

from koregraph.utils.pickle import load_pickle_object, save_object_pickle
from koregraph.api.music_to_numpy import music_to_numpy
from koregraph.api.posture_proc import fill_forward
from koregraph.params import (
    GENERATED_KEYPOINTS_DIRECTORY,
    GENERATED_AUDIO_DIRECTORY,
    GENERATED_PICKLE_DIRECTORY,
    ALL_ADVANCED_MOVE_NAMES,
    CHUNK_SIZE,
    FRAME_FORMAT,
    GENERATED_FEATURES_DIRECTORY,
    PERCENTAGE_CUT,
)
from koregraph.utils.preproc import cut_percentage


def load_preprocess_dataset() -> Tuple[ndarray, ndarray]:
    """
    Load and preprocess the dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The dataset.
    """
    y, X = load_pickle_object(
        GENERATED_PICKLE_DIRECTORY / "generated_gBR_sFM_cAll_d04_mBR0_ch01.pkl"
    )

    files = [
        "generated_gBR_sFM_cAll_d04_mBR1_ch02.pkl",
        "generated_gBR_sFM_cAll_d04_mBR2_ch03.pkl",
        "generated_gBR_sFM_cAll_d04_mBR3_ch04.pkl",
        "generated_gBR_sFM_cAll_d04_mBR4_ch05.pkl",
        "generated_gBR_sFM_cAll_d04_mBR4_ch07.pkl",
        "generated_gBR_sFM_cAll_d04_mBR5_ch06.pkl",
        "generated_gBR_sFM_cAll_d05_mBR2_ch09.pkl",
        "generated_gBR_sFM_cAll_d05_mBR3_ch10.pkl",
        "generated_gBR_sFM_cAll_d05_mBR4_ch11.pkl",
        "generated_gBR_sFM_cAll_d05_mBR4_ch13.pkl",
        "generated_gKR_sFM_cAll_d28_mKR0_ch01.pkl",
    ]

    for file in files:
        try:
            y_tmp, X_tmp = load_pickle_object(GENERATED_PICKLE_DIRECTORY / file)
            X = append(X, X_tmp, axis=0)
            y = append(y, y_tmp, axis=0)
        except Exception as e:
            print(f"Error {e}")

    return X, y


def load_chunk_preprocess_dataset() -> Tuple[ndarray, ndarray]:
    chore_names = ALL_ADVANCED_MOVE_NAMES[:100]
    X = None
    y = None
    for chore_name in chore_names:
        chore_name = chore_name.replace(".pkl", "")
        _, _, _, _, music_name, _ = chore_name.split("_")

        chore_path = GENERATED_KEYPOINTS_DIRECTORY / chore_name / str(CHUNK_SIZE)
        music_path = GENERATED_AUDIO_DIRECTORY / music_name / str(CHUNK_SIZE)
        print(f"Parsing {chore_name} chunks")

        for file in listdir(chore_path):
            chunk_id = file.replace(".pkl", "").split("_")[-1]
            chore_filepath = chore_path / file
            music_filepath = music_path / f"{music_name}_{chunk_id}.mp3"

            X_tmp = music_to_numpy(music_filepath)
            y_tmp = load_pickle_object(chore_filepath)["keypoints2d"]

            y_tmp = fill_forward(y_tmp)
            if isnan(y_tmp).any():
                print(f"Fill forward failed for chunk {chunk_id}. Filling with 0")
                y_tmp = nan_to_num(y_tmp, 0)
            # y_tmp[:, :, 0] = y_tmp[:, :, 0] / FRAME_FORMAT[0]
            # y_tmp[:, :, 1] = y_tmp[:, :, 1] / FRAME_FORMAT[1]

            if y is None:
                y = y_tmp
            else:
                y = append(y, y_tmp, axis=0)
            if X is None:
                X = X_tmp
            else:
                X = append(X, X_tmp, axis=0)

    return X.reshape(-1, CHUNK_SIZE * 60, 128), y.reshape(-1, CHUNK_SIZE * 60 * 34)


def load_next_chunks_preprocess_dataset(perc_cut: float = PERCENTAGE_CUT):
    chore_names = ALL_ADVANCED_MOVE_NAMES
    X = None
    y = None
    for chore_name in chore_names:
        chore_name = chore_name.replace(".pkl", "")
        _, _, _, _, music_name, _ = chore_name.split("_")

        chore_path = GENERATED_KEYPOINTS_DIRECTORY / chore_name / str(CHUNK_SIZE)
        music_path = GENERATED_AUDIO_DIRECTORY / music_name / str(CHUNK_SIZE)
        print(f"Parsing {chore_name} chunks")

        for file in listdir(chore_path):
            chunk_id = file.replace(".pkl", "").split("_")[-1]
            chore_filepath = chore_path / file
            music_filepath = music_path / f"{music_name}_{chunk_id}.mp3"

            audio_tmp = music_to_numpy(music_filepath)
            chore_tmp = load_pickle_object(chore_filepath)["keypoints2d"]

            chore_tmp = fill_forward(chore_tmp)
            if isnan(chore_tmp).any():
                print(f"Fill forward failed for chunk {chunk_id}. Filling with 0")
                chore_tmp = nan_to_num(chore_tmp, 0)
            chore_tmp[:, :, 0] = chore_tmp[:, :, 0] / FRAME_FORMAT[0]
            chore_tmp[:, :, 1] = chore_tmp[:, :, 1] / FRAME_FORMAT[1]

            chore_X, y_tmp = cut_percentage(chore_tmp.reshape(-1, 34), perc_cut)
            audio_X, _ = cut_percentage(audio_tmp, perc_cut)
            # X_tmp = concatenate((chore_X, audio_X), axis=1)
            X_tmp = chore_X

            # print('X_tmp shape', X_tmp.shape)
            # print('y_tmp shape', y_tmp.shape)

            if y is None:
                y = y_tmp
            else:
                y = append(y, y_tmp, axis=0)

            if X is None:
                X = X_tmp
            else:
                X = append(X, X_tmp, axis=0)

    X = X.reshape(-1, int(CHUNK_SIZE * (1 - perc_cut)) * 60, X.shape[-1])
    y = y.reshape(-1, int(CHUNK_SIZE * perc_cut) * 60 * 34)

    print("X final shape", X.shape)
    print("y final shape", y.shape)
    save_object_pickle(X, GENERATED_FEATURES_DIRECTORY / "x")
    save_object_pickle(y, GENERATED_FEATURES_DIRECTORY / "y")

    y = delete(y, [60, 63], axis=0)
    X = delete(X, [60, 63], axis=0)
    return X, y


def check_dataset_format(): ...
