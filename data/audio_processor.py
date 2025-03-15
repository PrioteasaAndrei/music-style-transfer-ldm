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

    def normalize_audio(self, audio):
        # ! TODO: Implement normalization
        pass

    def get_mel_spectogram(self, audio, sr, n_mels=256):
        """
        Extracts a Mel spectrogram from the audio.
        :param audio: Audio time series.
        :param sr: Sampling rate.
        :return: Log-scaled Mel spectrogram.
        """
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec

    def mel_spectogram_to_grayscale_image(self, spectogram, max_db=80):
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
    
    def grayscale_mel_spectogram_image_to_audio(self, image, sr, im_height, im_width, max_db=80):
        """
        ! TODO: Not tested yet!  
        Converts a grayscale image of a Mel spectrogram back to audio
        :param image: Grayscale image.
        :param sr: Sampling rate.
        :param im_height: Image height (number of frequency bins).
        :param im_width: Image width (number of time frames).
        :param max_db: Maximum decibel value used during the conversion.
        :return: Reconstructed audio time series.
        """
        
        # To raw bytes
        image_bytes = np.frombuffer(image.tobytes(), dtype=np.uint8)
        # Reshape
        image_bytes = image_bytes.reshape(im_height, im_width)
        log_mel_spec = image_bytes.astype(np.float32) * (max_db / 255.0) - max_db
        mel_spec = librosa.db_to_power(log_mel_spec)
        audio = librosa.feature.inverse.mel_to_audio(mel_spec, sr=sr)
        return audio

    def get_spectogram(self, audio):
        """
        Extracts a spectrogram from the audio.
        :param audio: Audio time series.
        :return: Spectrogram.
        """
        stft = librosa.stft(audio)
        spectogram = np.abs(stft)
        spectogram = librosa.amplitude_to_db(spectogram, ref=np.max)
        return spectogram
    
    def spectogram_to_grayscale_image(self, spectogram, max_db=80):
        """
        ! TODO: Not tested yet!  
        Converts a log-scaled spectrogram to a grayscale image.
        :param spectogram: Log-scaled spectrogram.
        :param max_db: Maximum decibel value for clipping.
        :return: Grayscale image of the spectrogram.
        """
        # Shift to positive values
        spectogram = spectogram + max_db
        # Scale values to the 0-255 range
        spectogram = spectogram * (255.0 / max_db)
        # Clip values outside [0, 255]
        spectogram = np.clip(spectogram, 0, 255)
        # Round and convert to uint8
        spectogram = (spectogram + 0.5).astype(np.uint8)
        image = Image.fromarray(spectogram)
        return image

    def grayscale_spectogram_image_to_audio(self, image, im_height, im_width, max_db=80, n_iter=32, hop_length=None, win_length=None):
        """
        ! TODO: Not tested yet! 
        Converts a grayscale image of a spectrogram back to audio
        :param image: Grayscale image.
        :param im_height: Image height (number of frequency bins).
        :param im_width: Image width (number of time frames).
        :param max_db: Maximum decibel value used during the conversion.
        :param n_iter: Number of iterations for  Griffin-Lim.
        :param hop_length: Hop length for STFT (if None, librosa's default is used).
        :param win_length: Window length for STFT (if None, librosa's default is used).
        :return: Reconstructed audio time series.
        """
        # Convert image to raw bytes and reshape
        image_bytes = np.frombuffer(image.tobytes(), dtype=np.uint8)
        image_bytes = image_bytes.reshape(im_height, im_width)
        # Convert back to a log-scaled spectrogram
        log_spec = image_bytes.astype(np.float32) * (max_db / 255.0) - max_db
        # Convert from decibels to amplitude
        amplitude_spec = librosa.db_to_amplitude(log_spec)
        # Reconstruct audio using the Griffin-Lim algorithm
        audio = librosa.griffinlim(amplitude_spec, n_iter=n_iter, hop_length=hop_length, win_length=win_length)
        return audio            

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
