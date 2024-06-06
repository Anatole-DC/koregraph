from os import path
from os import environ
from pathlib import Path
from librosa import load
from librosa import display
from librosa import power_to_db
from librosa import feature
from koregraph.config.params import IMAGE_DIRECTORY

import numpy as np
import matplotlib.pyplot as plt


def save_images_log_power_spectogram(file_path: Path):
    """Save log power spectogram image of the audio file

    Args:
        file_path (Path) : path to the audio file

    Returns:
        Path: path to the saved image
    """

    y, sr = load(file_path, duration=10)  # load the first 10 seconds of the audio file

    fig, ax = plt.subplots(figsize=(50, 10))
    melspectrogram = feature.melspectrogram(
        y=y, sr=sr, n_fft=512, n_mels=128, hop_length=128
    )
    imgdb = display.specshow(
        power_to_db(melspectrogram, ref=np.max),
        sr=sr,
        y_axis="mel",
        x_axis="time",
        ax=ax,
        cmap="gray_r",
    )
    ax.set(title="Log-Power spectrogram")
    fig.colorbar(imgdb, ax=ax, format="%+2.0f dB")
    image_path = IMAGE_DIRECTORY / f"{file_path.stem}_log_power_spectrogram.png"
    plt.savefig(image_path)
    plt.close()

    return Path(
        f"{IMAGE_DIRECTORY}/{path.basename(file_path)}_log_power_spectrogram.png"
    )
