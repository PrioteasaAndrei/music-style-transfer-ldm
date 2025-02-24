from io import BytesIO
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import librosa
from PIL import Image
import matplotlib.pyplot as plt


class AudioPreprocessor:
    def __init__(self, target_sr=22050):
        """
        Initialize AudioPreprocessor.
        :param target_sr: Target sampling rate for audio.
        """
        self.target_sr = target_sr

    def load_audio(self, filepath):
        """
        Loads an audio file using librosa.
        :param filepath: Path to the audio file.
        :return: Tuple of (audio time series, sampling rate).
        """
        audio, sr = librosa.load(filepath, sr=self.target_sr, mono=True)
        return audio, sr

    def trim_silence(self, audio, top_db=20):
        """
        Trims the silence from the beginning and end of an audio signal.
        :param audio: Audio time series.
        :param top_db: Threshold (in decibels) below reference to consider as silence.
        :return: The trimmed audio.
        """
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed_audio

    def extract_features(self, audio, sr):
        """
        Extracts a Mel spectrogram from the audio.
        :param audio: Audio time series.
        :param sr: Sampling rate.
        :return: Log-scaled Mel spectrogram.
        """
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec

    def spectogram_to_grayscale_image(self, spectogram, max_db=80):
        """
        Converts a log-scaled Mel spectrogram to an image.
        :param spectogram: Log-scaled Mel spectrogram.
        :param max_db: Maximum decibel value for clipping.
        :return: Image of the Mel spectrogram.
        """
        # Shift to positive values
        spectogram = spectogram + max_db
        # Scale to 0-255 (grayscale)
        spectogram = spectogram * (255.0 / max_db)
        # Clip out of bounds
        spectogram = np.clip(spectogram, 0, 255)
        # Do rounding trick and convert to uint8
        spectogram = (spectogram + 0.5).astype(np.uint8)
        
        # Create an image
        image = Image.fromarray(spectogram)
        return image

    def get_raw_image_bytes(self, image):
        with BytesIO() as output:
            image.save(output, format="PNG")
            image = output.getvalue()
        return image

    def plot_audio(self, audio, sr):
        """
        Plots the audio time series.
        :param audio: Audio time series.
        :param sr: Sampling rate.
        """
        # Create a time array in seconds
        time = np.linspace(0, len(audio) / sr, num=len(audio))
        plt.figure(figsize=(14, 5))
        plt.plot(time, audio)
        plt.title("Audio Waveform")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

    def plot_mel_spectrogram(self, mel_spec):
        """
        Plots the Mel spectrogram.
        :param mel_spec: Log-scaled Mel spectrogram.
        """
        fig, ax = plt.subplots(figsize=(14, 5))
        img = librosa.display.specshow(mel_spec, x_axis="time", y_axis="log", sr=self.target_sr, ax=ax)
        ax.set_title("Mel spectrogram")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.tight_layout()
        plt.show()
