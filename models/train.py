from model import *
from loss import *
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import config

def train_autoencoder(config):
    # Initialize components
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load autoencoder
    encoder = SpectrogramEncoder().to(device)
    decoder = SpectrogramDecoder().to(device)

    # Feature extractor for perceptual loss (e.g., pretrained CNN on spectrograms)
    feature_extractor = None  # Set this to a pretrained model if needed

    # Optimizer
    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['learning_rate_factor'], patience=config['learning_rate_patience'], min_lr=config['learning_rate_min'])

    # Prepare dataset
    train_loader, test_loader = prepare_dataset(config)

    # Training loop
    num_epochs = config['num_epochs']  # Adjust as needed
    for epoch in range(num_epochs):
        running_loss = 0.0
        encoder.train()
        decoder.train()

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
        avg_loss = running_loss / len(train_loader)
        scheduler.step(avg_loss)

        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch}')
        print(f'Average Loss: {avg_loss:.6f}')
        print(f'Learning Rate: {current_lr:.6f}')


    # Save model
    torch.save(encoder.state_dict(), 'pretrained/encoder.pth')
    torch.save(decoder.state_dict(), 'pretrained/decoder.pth')



def main():
    train_autoencoder(config)

if __name__ == "__main__":
    main()