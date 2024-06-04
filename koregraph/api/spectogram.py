import os
import pathlib
import librosa
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

os.environ["IMAGE_FOLDER"] = '../data/images'

def save_images_log_power_spectogram(file_path: Path):
    '''Save log power spectogram image of the audio file

    Args:
        file_path (Path) : path to the audio file

    Returns:
        Path: path to the saved image
    '''

    image_folder = os.environ.get("IMAGE_FOLDER")
    pathlib.Path(image_folder).mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(file_path, duration=10) # load the first 10 seconds of the audio file

    fig, ax = plt.subplots(figsize=(50, 10))
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=512, n_mels=128, hop_length=128)
    imgdb = librosa.display.specshow(librosa.power_to_db(melspectrogram, ref=np.max), sr=sr, y_axis='mel', x_axis='time', ax=ax, cmap='gray_r')
    ax.set(title='Log-Power spectrogram')
    fig.colorbar(imgdb, ax=ax, format="%+2.0f dB")
    image_path = pathlib.Path(image_folder) / f"{os.path.basename(file_path)}_log_power_spectrogram.png"
    plt.savefig(image_path)
    plt.close()

    return pathlib.Path(f"{image_folder}/{os.path.basename(file_path)}_log_power_spectrogram.png")

# save_images_log_power_spectogram("../data/music/mp3/mBR0.mp3")
