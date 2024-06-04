from librosa import load
from librosa.feature import melspectrogram
from librosa import power_to_db
import numpy as np

def music_to_numpy(audio_file_path, fps=60, sr=44100, n_mels=128, n_fft=2048):
    """
    Compute the mel spectrogram of the audio for each frame of a video.

    Parameters:
    - audio_file_path: Path, path to the audio file.
    - fps: int, frames per second of the video.
    - sr: int, sample rate for the audio.
    - n_mels: int, number of mel bands to generate.
    - n_fft: int, length of the FFT window.

    Returns:
    - Numpy array with shape (n_frames, n_mels), where n_frames = duration * fps.
    """

    # Charger le fichier audio
    y, sr = load(audio_file_path, sr=sr)

    # Calculer la durée d'une frame
    duration_per_frame = 1 / fps

    # Calculer le hop_length
    hop_length = int(sr * duration_per_frame)

    # Calculer le spectrogramme Mel
    S = melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Convertir le spectrogramme en dB
    S_db = power_to_db(S, ref=np.max)

    # Transposer pour obtenir le bon format
    S_db_T = S_db.T

    # Calculer le nombre de frames attendu
    n_frames = int(len(y) / hop_length)

    # Si le nombre de frames est supérieur à celui attendu, couper le numpy array
    if len(S_db_T) > n_frames:
        S_db_T = S_db_T[:n_frames, :]

    return S_db_T
