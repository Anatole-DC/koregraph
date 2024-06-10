import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from librosa.feature import melspectrogram
from librosa import power_to_db
from scipy.ndimage import median_filter


def reduce_noise(
    y,
    sr,
    median_filter_size=(4, 16),
    threshold_multiplier=2.5,
    n_fft=2048,
    hop_length=512,
):
    """
    Reduce noise from an audio signal using median filtering.

    Parameters:
    - y: np.ndarray, audio time series.
    - sr: int, sample rate.
    - median_filter_size: tuple, size of the median filter window.
    - threshold_multiplier: float, multiplier for setting the threshold for noise reduction.
    - n_fft: int, FFT window size.
    - hop_length: int, hop length for STFT.

    Returns:
    - y_filtered: np.ndarray, noise-reduced audio time series.
    """
    # Compute STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(D)

    # Apply median filtering
    filtered_magnitude = median_filter(magnitude, size=median_filter_size)

    # Compute residual
    residual = magnitude - filtered_magnitude

    # Set threshold for noise reduction
    threshold = np.median(residual) * threshold_multiplier

    # Apply noise reduction
    denoised_magnitude = np.where(residual > threshold, magnitude, 0)

    # Reconstruct denoised STFT
    denoised_stft = denoised_magnitude * phase
    y_filtered = librosa.istft(denoised_stft, hop_length=hop_length)

    return y_filtered


def music_to_numpy(audio_file_path, fps=60, sr=44100, n_mels=128, n_fft=2048):
    """
    Compute the mel spectrogram of the noise-reduced audio for each frame of a video.

    Parameters:
    - audio_file_path: Path, path to the audio file.
    - fps: int, frames per second of the video.
    - sr: int, sample rate for the audio.
    - n_mels: int, number of mel bands to generate.
    - n_fft: int, length of the FFT window.
    - noise_duration: float, duration in seconds to estimate the noise profile.

    Returns:
    - Numpy array with shape (n_frames, n_mels), where n_frames = duration * fps.
    """

    # Charger le fichier audio
    y, sr = librosa.load(audio_file_path, sr=sr)

    # Réduire le bruit du fichier audio
    y_filtered = reduce_noise(y, sr, n_fft=n_fft, hop_length=int(sr * (1 / fps)))

    # Calculer la durée d'une frame
    duration_per_frame = 1 / fps

    # Calculer le hop_length
    hop_length = int(sr * duration_per_frame)

    # Calculer le spectrogramme Mel
    S = melspectrogram(
        y=y_filtered, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    # Convertir le spectrogramme en dB
    S_db = power_to_db(S, ref=np.max)

    # Transposer pour obtenir le bon format
    S_db_T = S_db.T

    # Calculer le nombre de frames attendu
    n_frames = int(len(y_filtered) / hop_length)

    # Si le nombre de frames est supérieur à celui attendu, couper le numpy array
    if len(S_db_T) > n_frames:
        S_db_T = S_db_T[:n_frames, :]

    # Mettre les 60 premières lignes à -80 dB
    if len(S_db_T) > 60:
        S_db_T[:60, :] = np.full((60, n_mels), -80)

    # Mettre les 60 dernières lignes à -80 dB
    if len(S_db_T) > 120:  # Assurer qu'il y a au moins 120 lignes
        S_db_T[-60:, :] = full((60, nb_mels), -80)

    return S_db_T
