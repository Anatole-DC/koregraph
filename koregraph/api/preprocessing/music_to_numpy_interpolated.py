from pathlib import Path
from librosa import load
from librosa.feature import melspectrogram
from librosa import power_to_db
from numpy import max, ndarray, full


def music_to_numpy_interpolated(
    audio_file_path: Path,
    fps: int = 60,
    sample_rate: int = 44100,
    nb_mels: int = 128,
    nb_fft: int = 2048,
) -> ndarray:
    """Compute the mel spectogram of the audio for each frame of an audio file,
    with the first and last 60 frames set to -80 dB.

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
    y, sample_rate = load(audio_file_path, sr=sample_rate)

    # Calculer la durée d'une frame
    duration_per_frame = 1 / fps

    # Calculer le hop_length
    hop_length = int(sample_rate * duration_per_frame)

    # Calculer le spectrogramme Mel
    S = melspectrogram(
        y=y,
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
