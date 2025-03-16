import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf  # Use soundfile to write wav files

# Ensure the root directory is in sys.path so that data module can be imported
sys.path.append(str(Path(__file__).parent.parent))

from data.audio_processor import AudioPreprocessor


def main():
    # Create output directory for test reconstruction
    output_dir = Path('tests/downloads/reconstruction_test')
    output_dir.mkdir(parents=True, exist_ok=True)

    mp3_files = list(Path('downloads').rglob('*.mp3'))
    if not mp3_files:
        print('No mp3 files found in downloads directory.')
        sys.exit(1)
    test_file = mp3_files[0]
    print(f'Processing file: {test_file}')
    proc = AudioPreprocessor()
    
    audio, sr = proc.load_audio(test_file)
    audio = proc.trim_silence(audio)
    CHUNK_SIZE = 5  # seconds
    chunk_size = int(CHUNK_SIZE * sr)
    chunk = audio[:chunk_size]

    # Save original audio chunk to wav format
    # Convert audio from float [-1, 1] to int16
    original_audio_path = output_dir / 'original_chunk.wav'
    sf.write(str(original_audio_path), np.int16(chunk * 32767), sr)
    print(f'Saved original audio chunk to {original_audio_path}')
    
    # Extract mel spectrogram
    mel_spec = proc.get_mel_spectogram(chunk, sr, n_mels=128)

    # Convert mel spectrogram to grayscale image
    mel_img = proc.mel_spectogram_to_grayscale_image(mel_spec)
    original_mel_spec_path = output_dir / 'original_mel_spec.png'
    mel_img.save(str(original_mel_spec_path))
    print(f'Saved original mel spectrogram to {original_mel_spec_path}')
    
    # Extract normal spectrogram and save as grayscale image
    norm_spec = proc.get_spectogram(chunk)
    norm_img = proc.spectogram_to_grayscale_image(norm_spec)
    original_norm_spec_path = output_dir / 'original_normal_spec.png'
    norm_img.save(str(original_norm_spec_path))
    print(f'Saved original normal spectrogram to {original_norm_spec_path}')

    # Convert spectrogram to grayscale image
    image = proc.mel_spectogram_to_grayscale_image(mel_spec)

    # Reconstruct audio from mel mel spectrogram grayscale image
    recon_audio = proc.grayscale_mel_spectogram_image_to_audio(image, sr, im_height=mel_spec.shape[0], im_width=mel_spec.shape[1])

    # Save reconstructed audio chunk using soundfile.write
    recon_audio_path = output_dir / 'reconstructed_mel_audio.wav'
    sf.write(str(recon_audio_path), np.int16(recon_audio * 32767), sr)
    print(f'Saved reconstructed audio chunk to {recon_audio_path}')
    
    # Extract mel spectrogram from reconstructed audio
    recon_mel_spec = proc.get_mel_spectogram(recon_audio, sr)
    recon_mel_img = proc.mel_spectogram_to_grayscale_image(recon_mel_spec)
    reconstructed_mel_spec_path = output_dir / 'reconstructed_mel_spec.png'
    recon_mel_img.save(str(reconstructed_mel_spec_path))
    print(f'Saved reconstructed mel spectrogram to {reconstructed_mel_spec_path}')
    
    # Also extract normal spectrogram from reconstructed audio and save as grayscale image
    recon_norm_spec = proc.get_spectogram(recon_audio)
    recon_norm_img = proc.spectogram_to_grayscale_image(recon_norm_spec)
    reconstructed_norm_spec_path = output_dir / 'reconstructed_normal_spec.png'
    recon_norm_img.save(str(reconstructed_norm_spec_path))
    print(f'Saved reconstructed normal spectrogram to {reconstructed_norm_spec_path}')

    # Reconstruct audio from normal spectrogram grayscale image
    recon_normal_audio = proc.grayscale_spectogram_image_to_audio(recon_norm_img, recon_norm_spec.shape[0], recon_norm_spec.shape[1])
    reconstructed_normal_audio_path = output_dir / 'reconstructed_normal_audio.wav'
    sf.write(str(reconstructed_normal_audio_path), np.int16(recon_normal_audio * 32767), sr)
    print(f'Saved reconstructed normal audio to {reconstructed_normal_audio_path}')

    print('\nTest reconstruction complete.')


if __name__ == '__main__':
    main()
