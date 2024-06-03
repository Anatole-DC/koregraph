import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

def save_images_log_power_spectogram(file_path):
    '''Save log power spectogram image of the audio file
    Args:
        file_path (str): path to the audio file

    Returns:
        str: path to the saved image
    '''

    image_folder = '../data/images'
    os.makedirs(image_folder, exist_ok=True)

    y, sr = librosa.load(file_path, duration=10)
    S = np.abs(librosa.stft(y))

    fig, ax = plt.subplots(figsize=(50, 10))
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=512, n_mels=128, hop_length=128)
    imgdb = librosa.display.specshow(librosa.power_to_db(melspectrogram, ref=np.max), sr=sr, y_axis='mel', x_axis='time', ax=ax, cmap='gray_r')
    ax.set(title='Log-Power spectrogram')
    fig.colorbar(imgdb, ax=ax, format="%+2.0f dB")
    plt.savefig(f"{image_folder}/{os.path.basename(file_path)}_log_power_spectrogram.png")
    plt.close()

    return f"{image_folder}/{os.path.basename(file_path)}_log_power_spectrogram.png"

# save_images_log_power_spectogram("../data/music/mp3/mBR0.mp3")
