from pathlib import Path
from random import sample

from librosa import db_to_power, griffinlim, load, stft, magphase, istft
from librosa.feature import melspectrogram
from librosa.feature.inverse import mel_to_stft
from librosa import power_to_db
from numpy import (
    append,
    clip,
    concatenate,
    float16,
    max,
    ndarray,
    where,
    median,
    full,
    int16,
    zeros,
)
from scipy.ndimage import median_filter
from soundfile import write
from pydub import AudioSegment

from koregraph.config.params import (
    AUDIO_DIRECTORY,
    GENERATED_AUDIO_SILENCE_DIRECTORY,
    X_MIN,
    X_MAX,
)
from koregraph.utils.controllers.musics import load_music, save_audio_chunk


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
    D = stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = magphase(D)

    # Apply median filtering
    filtered_magnitude = median_filter(magnitude, size=median_filter_size)

    # Compute residual
    residual = magnitude - filtered_magnitude

    # Set threshold for noise reduction
    threshold = median(residual) * threshold_multiplier

    # Apply noise reduction
    denoised_magnitude = where(residual > threshold, magnitude, 0)

    # Reconstruct denoised STFT
    denoised_stft = denoised_magnitude * phase
    y_filtered = istft(denoised_stft, hop_length=hop_length)

    return y_filtered


def music_to_numpy(
    audio_file_path: Path,
    fps: int = 60,
    sample_rate: int = 44100,
    nb_mels: int = 128,
    nb_fft: int = 2048,
) -> ndarray:
    """Compute the mel spectogram of the audio for each frame of an audio file.

    Args:
        audio_file_path (Path): Path to the audio file
        fps (int, optional): Frame rate of the video. Defaults to 60 fps.
        sample_rate (int, optional): Sample rate for the audio output. Defaults to 44100.
        nb_mels (int, optional): Number of mel bands to generate. Defaults to 128.
        nb_fft (int, optional): Length of the fft window. Defaults to 2048.

    Returns:
        ndarray: Array with shape (audio_duration * fps, nb_mels)
    """

    # Charger le fichier audio
    y, _ = load(audio_file_path, sr=sample_rate)

    # Réduire le bruit du fichier audio
    y_filtered = reduce_noise(
        y, sample_rate, n_fft=nb_fft, hop_length=int(sample_rate * (1 / fps))
    )

    # Calculer la durée d'une frame
    duration_per_frame = 1 / fps

    # Calculer le hop_length
    hop_length = int(sample_rate * duration_per_frame)

    # Calculer le spectrogramme Mel
    S = melspectrogram(
        y=y_filtered,
        sr=sample_rate,
        n_fft=nb_fft,
        hop_length=hop_length,
        n_mels=nb_mels,
    )

    # Convertir le spectrogramme en dB
    S_db = power_to_db(S, ref=max)

    # Transposer pour obtenir le bon format
    S_db_T = S_db.T

    # Calculer le nombre de frames attendu
    n_frames = int(len(y) / hop_length)

    # Si le nombre de frames est supérieur à celui attendu, couper le numpy array
    if len(S_db_T) > n_frames:
        S_db_T = S_db_T[:n_frames, :]

    # Mettre les 60 premières lignes à -80 dB
    if len(S_db_T) > 60:
        S_db_T[:60, :] = full((60, nb_mels), -80)

    # Mettre les 60 dernières lignes à -80 dB
    if len(S_db_T) > 120:  # Assurer qu'il y a au moins 120 lignes
        S_db_T[-60:, :] = full((60, nb_mels), -80)

    return S_db_T


def scale_audio(X: ndarray, X_min: float = X_MIN, X_max: float = X_MAX) -> ndarray:
    """Apply a MinMax scaler to an audio.

    Args:
        X (ndarray): The audio to scale.
        X_min (float, optional): The minimum value within x. Defaults to X_MIN.
        X_max (float, optional): The maximum value within x. Defaults to X_MAX.

    Returns:
        ndarray: The scaled audio.
    """

    X_std = (X - X_min) / (X_max - X_min)
    return X_std


def add_silence(
    music: ndarray,
    nb_seconds: int = 1,
    sample_rate: int = 44100,
) -> ndarray:

    num_silence_samples = sample_rate * nb_seconds

    # Create a new array with the total size (silence + music + silence)
    y_silence_added = zeros(num_silence_samples * 2 + music.shape[0], dtype=float16)

    # Place the original music in the middle of the array
    y_silence_added[num_silence_samples : num_silence_samples + music.shape[0]] = music

    return y_silence_added


def convert_music_array_to_train_audio(
    music_array: ndarray,
    sample_rate: int = 44100,
    fps: int = 60,
    nb_mels: int = 128,
    nb_fft: int = 2048,
) -> tuple[ndarray, ndarray]:

    # interpolate silence
    y = music_array  # add_silence(music_array, sample_rate)

    # reduce noise
    y_filtered = reduce_noise(
        y, sample_rate, n_fft=nb_fft, hop_length=int(sample_rate * (1 / fps))
    )

    # Calculer la durée d'une frame
    duration_per_frame = 1 / fps

    # Calculer le hop_length
    hop_length = int(sample_rate * duration_per_frame)

    # Calculer le spectrogramme Mel
    S = melspectrogram(
        y=y_filtered,
        sr=sample_rate,
        n_fft=nb_fft,
        hop_length=hop_length,
        n_mels=nb_mels,
    )

    # Convertir le spectrogramme en dB
    S_db = power_to_db(S, ref=max)

    # Transposer pour obtenir le bon format
    S_db_T = S_db.T

    # Calculer le nombre de frames attendu
    n_frames = int(len(y) / hop_length)

    # Si le nombre de frames est supérieur à celui attendu, couper le numpy array
    if len(S_db_T) > n_frames:
        S_db_T = S_db_T[:n_frames, :]

    return S_db_T, y_filtered


if __name__ == "__main__":
    convert_music_array_to_train_audio(load_music(AUDIO_DIRECTORY / "mBR0.mp3")[0])
