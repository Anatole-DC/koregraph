import librosa
import numpy as np
import scipy.signal
from soundfile import write

from koregraph.config.params import AUDIO_DIRECTORY


def reduce_echo(y, sr, frame_length=2048, hop_length=512):
    # Perform Short-Time Fourier Transform (STFT)
    D = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)

    # Convert to magnitude and phase
    magnitude, phase = librosa.magphase(D)

    # Create a simple echo reduction filter (this is a very basic approach)
    filter_size = 5
    filter = np.ones(filter_size) / filter_size

    # Apply the filter to the magnitude spectrogram
    magnitude_filtered = scipy.signal.convolve2d(
        magnitude, filter[:, None], mode="same"
    )

    # Recombine the filtered magnitude with the original phase
    D_filtered = magnitude_filtered * phase

    # Inverse STFT to get the time-domain signal
    y_filtered = librosa.istft(D_filtered, hop_length=hop_length)

    # Normalize the audio to avoid clipping
    y_filtered = y_filtered / np.max(np.abs(y_filtered))

    return y_filtered


def reduce_drum_volume(y, sr, percussive_reduction=0.5):
    # Perform harmonic-percussive source separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Reduce the volume of the percussive component
    y_percussive *= percussive_reduction

    # Recombine the harmonic and percussive components
    y_reduced_drums = y_harmonic + y_percussive

    # Normalize the recombined audio to avoid clipping
    y_reduced_drums = y_reduced_drums / np.max(np.abs(y_reduced_drums))

    return y_reduced_drums


def reduce_noise_and_saturation(y, sr):
    # Noise reduction using a Wiener filter
    def reduce_noise(y):
        # Apply a Wiener filter
        y_denoised = scipy.signal.wiener(y)
        return y_denoised

    # Saturation reduction by normalizing the audio
    def reduce_saturation(y):
        # Normalize the audio to be within the range [-1, 1]
        y_normalized = y / np.max(np.abs(y))
        return y_normalized

    # Reduce noise
    y_denoised = reduce_noise(y)
    # Reduce saturation
    y_processed = reduce_saturation(y_denoised)

    return y_processed


# Load an audio file
y, sr = librosa.load(AUDIO_DIRECTORY / "mBR0.mp3")

write("no_processed_audio.wav", y, sr)

# Reduce noise and saturation
y_processed = reduce_echo(y, sr)
# y_processed = reduce_drum_volume(y, sr, 0)
# Save the processed audio to a file
write("processed_audio.wav", y_processed, sr)
