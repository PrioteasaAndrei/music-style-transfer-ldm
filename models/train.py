import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple
import math

from model import *
from loss import *
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import config
import matplotlib.pyplot as plt
import argparse
import yaml
from tqdm import tqdm
from loss import style_loss

# cuda or error
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print("Using GPU")
else:
    raise RuntimeError("GPU not available, please check your setup.")

def train_autoencoder(config):
    # Initialize components
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print("Using GPU")
    else:
        raise RuntimeError("GPU not available, please check your setup.")
    
    # Load autoencoder
    encoder = SpectrogramEncoder(config['latent_dim_encoder']).to(device)
    decoder = SpectrogramDecoder(config['latent_dim_encoder']).to(device)

    # Feature extractor for perceptual loss (e.g., pretrained CNN on spectrograms)
    # feature_extractor = None  # Set this to a pretrained model if needed
    feature_extractor = VGGishFeatureLoss().to(device)

    # Optimizer
    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['learning_rate_factor'], patience=config['learning_rate_patience'], min_lr=config['learning_rate_min'])

    # Prepare dataset
    train_loader, test_loader = prepare_dataset(config)

    # Training loop
    num_epochs = config['num_epochs']  # Adjust as needed
    train_losses = []  # Track losses for plotting
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        running_train_loss = 0.0
        encoder.train()
        decoder.train()

        # Create progress bar for batches
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for spectrogram in pbar:  # Load spectrogram batches
            spectrogram = spectrogram[0]
            spectrogram = spectrogram.to(device)

            # Forward pass (Encode -> Decode)
            latent = encoder(spectrogram)
            reconstructed = decoder(latent)

            # Compute loss
            loss = compression_loss(spectrogram, reconstructed, latent, feature_extractor)
            running_train_loss += loss.item()

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({'train_loss': f'{loss.item():.4f}'})

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)


        encoder.eval()
        decoder.eval()

        running_val_loss = 0.0
        with torch.no_grad():
            for spectrogram in test_loader:
                spectrogram = spectrogram[0]
                spectrogram = spectrogram.to(device)

                latent = encoder(spectrogram)
                reconstructed = decoder(latent)

                val_loss = compression_loss(spectrogram, reconstructed, latent, feature_extractor)
                running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(encoder.state_dict(), 'models/pretrained/encoder.pth')
            torch.save(decoder.state_dict(), 'models/pretrained/decoder.pth')
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch}')
        print(f'Average Train Loss: {avg_train_loss:.6f}')
        print(f'Average Val Loss: {avg_val_loss:.6f}')
        print(f'Learning Rate: {current_lr:.6f}')

    # Plot loss curve
 
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('models/plots/autoencoder_loss.png')
    plt.close()

    # Save model
    torch.save(encoder.state_dict(), 'models/pretrained/encoder.pth')
    torch.save(decoder.state_dict(), 'models/pretrained/decoder.pth')


class LDMTrainer:
    def __init__(
        self,
        model,
        train_loader,
        device,
        lr=1e-4,
        style_loss_weight=0.1,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.style_loss_weight = style_loss_weight
        # Initialize optimizer (only for trainable parameters)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=lr)
        self.scaler = torch.amp.GradScaler('cuda')
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10,
        )

    def train_step(self, content_spec, style_spec):
        """Single training step"""
        self.optimizer.zero_grad()
        content_spec = content_spec.float()
        style_spec = style_spec.float()
        
        # Sample random timesteps (learns to denoise at different timesteps)
        batch_size = content_spec.shape[0]

        t = torch.randint(0, self.model.num_timesteps, (batch_size,), device=self.device)
        
        with torch.autocast(device_type=self.device.type):
            outputs = self.model(content_spec, style_spec, t)
            noise_pred = outputs['noise_pred']
            noise = outputs['noise']
            z_0 = outputs['z_0']
            reconstructed = outputs['reconstructed']
            z_t = outputs['z_t']

            denoisinsg_loss = diffusion_loss(noise_pred, noise)
            compression_loss_ = compression_loss(content_spec, reconstructed, z_0, self.model.feature_loss_net)
            style_loss_ = style_loss(reconstructed, style_spec, self.model.feature_loss_net)
            
            total_loss = compression_loss_ + denoisinsg_loss + self.style_loss_weight * style_loss_
            

        self.scaler.scale(total_loss).backward()

        # # Add gradient norm logging for style encoder
        # total_norm = 0
        # for p in self.model.style_encoder.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** 0.5
        # print(f"Style encoder gradient norm: {total_norm}")

        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {
            'compression_loss': compression_loss_.item(),
            'denoisinsg_loss': denoisinsg_loss.item(),
            'style_loss': style_loss_.item(),
            'total_loss': total_loss.item()
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_compression_loss = 0
        total_denoisinsg_loss = 0
        total_style_loss = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, element in enumerate(pbar):
                # Correct unpacking of the batch
                (content_spec, content_label), (style_spec, style_label) = element
                # print(content_spec.shape)  # You can keep this if needed
                
                # Move data to device
                content_spec = content_spec.to(self.device)
                style_spec = style_spec.to(self.device)
                
                # Training step
                # for _ in range(config['training_iteration_noise']):
                losses = self.train_step(content_spec, style_spec)
            
                # Update progress bar
                total_loss += losses['total_loss']
                total_compression_loss += losses['compression_loss']
                total_denoisinsg_loss += losses['denoisinsg_loss']
                total_style_loss += losses['style_loss']
                pbar.set_postfix({'loss': losses['total_loss']})
        
        avg_loss = total_loss / num_batches * config['training_iteration_noise']
        avg_compression_loss = total_compression_loss / num_batches * config['training_iteration_noise']
        avg_denoisinsg_loss = total_denoisinsg_loss / num_batches * config['training_iteration_noise']
        avg_style_loss = total_style_loss / num_batches * config['training_iteration_noise']
        return avg_loss, avg_compression_loss, avg_denoisinsg_loss, avg_style_loss
    
    def train(self, num_epochs):
        """Full training loop"""
        best_loss = float('inf')

        train_losses = []
        compression_losses = []
        denoisinsg_losses = []
        style_losses = []
        
        for epoch in range(num_epochs):
            # Training epoch
            train_loss, compression_loss, denoisinsg_loss, style_loss = self.train_epoch(epoch)
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}')
            self.scheduler.step(train_loss)

            train_losses.append(train_loss)
            compression_losses.append(compression_loss)
            denoisinsg_losses.append(denoisinsg_loss)
            style_losses.append(style_loss)
            # print other losses
            print(f'Compression Loss: {compression_loss:.4f}')
            print(f'Denoisinsg Loss: {denoisinsg_loss:.4f}')
            print(f'Style Loss: {style_loss:.4f}')

            if epoch % 100 == 0:
                # Save model
                torch.save(self.model.state_dict(), f'models/pretrained/ldm_{epoch}.pth')

                # Plot loss curves
                plt.figure(figsize=(10, 5))
                plt.plot(train_losses, label='Train Loss (Total)')
                plt.plot(compression_losses, label='Compression Loss')
                plt.plot(denoisinsg_losses, label='Denoisinsg Loss')
                plt.plot(style_losses, label='Style Loss')
                plt.legend()
                plt.savefig(f'models/plots/ldm_loss_{epoch}.png')
                plt.close()

                # Plot with log scale y-axis
                plt.figure(figsize=(10, 5))
                plt.plot(train_losses, label='Train Loss (Total)')
                plt.plot(compression_losses, label='Compression Loss')
                plt.plot(denoisinsg_losses, label='Denoisinsg Loss')
                plt.plot(style_losses, label='Style Loss')
                plt.yscale('log')
                plt.legend()
                plt.savefig(f'models/plots/ldm_loss_log_{epoch}.png')
                plt.close()


def train_ldm(config):
    # Initialize components
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print("Using GPU")
    else:
        raise RuntimeError("GPU not available, please check your setup.")


    # will automatically load pretrained weights
    model = LDM(latent_dim=config['latent_dim_encoder'], load_full_model=False).to(device)

    style_dataset = SpectrogramPairDataset(config["processed_spectograms_dataset_folderpath"], config["pairing_file_path"])
    train_dataset, test_dataset = torch.utils.data.random_split(style_dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)

    trainer = LDMTrainer(model, train_loader, device, lr=config['learning_rate'], 
                          style_loss_weight=config['style_loss_weight'])
    trainer.train(config['num_epochs'])

def main():
    parser = argparse.ArgumentParser(description='Train models')
    parser.add_argument('--model', type=str, required=True, choices=['autoencoder', 'ldm'],
                        help='Which model to train (autoencoder or ldm)')

    args = parser.parse_args()

    
    if args.model == 'autoencoder':
        train_autoencoder(config)
    elif args.model == 'ldm':
        train_ldm(config)

if __name__ == "__main__":
    main()