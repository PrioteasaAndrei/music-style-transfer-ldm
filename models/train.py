from model import *
from loss import *
from torch.utils.data import DataLoader
import torch.optim as optim
from loss import *


def train_autoencoder():
    # Initialize components
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load autoencoder
    encoder = SpectrogramEncoder().to(device)
    decoder = SpectrogramDecoder().to(device)

    # Feature extractor for perceptual loss (e.g., pretrained CNN on spectrograms)
    feature_extractor = None  # Set this to a pretrained model if needed

    # Optimizer
    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

    # Training loop
    num_epochs = 50  # Adjust as needed
    for epoch in range(num_epochs):
        for spectrogram in train_loader:  # Load spectrogram batches
            spectrogram = spectrogram.to(device)

            # Forward pass (Encode -> Decode)
            latent = encoder(spectrogram)
            reconstructed = decoder(latent)

            # Compute loss
            loss = compression_loss(spectrogram, reconstructed, latent, feature_extractor)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")



def main():
    train_autoencoder()

if __name__ == "__main__":
    main()