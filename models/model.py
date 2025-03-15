import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple
import math


class SpectrogramEncoder(nn.Module):
    def __init__(self, latent_dim=4):
        super(SpectrogramEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=4, padding=0),  # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=4, padding=0),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            nn.Conv2d(128, latent_dim, kernel_size=2, stride=2, padding=0),  # 8x8xlatent_dim
            nn.BatchNorm2d(latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)
    
class SpectrogramDecoder(nn.Module):
    def __init__(self, latent_dim=4):
        super(SpectrogramDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=2, stride=2, padding=0),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4, padding=0),  # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=4, padding=0),  # 256x256
            nn.Tanh()  # Output normalized to [-1, 1]
        )

    def forward(self, z):
        return self.decoder(z)

class ForwardDiffusion(nn.Module):
    def __init__(self, num_timesteps=1000):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # Pre-compute values
        beta_start = 0.0001
        beta_end = 0.02
        self.beta_t = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alpha_t = 1 - self.beta_t
        self.alpha_bar_t = torch.cumprod(self.alpha_t, dim=0)
    
    def forward(self, x_0, t):
        device = x_0.device
        
        # Move pre-computed values to the correct device
        self.beta_t = self.beta_t.to(device)
        self.alpha_t = self.alpha_t.to(device)
        self.alpha_bar_t = self.alpha_bar_t.to(device)
        
        # Move t to correct device and ensure it's long type for indexing
        t = t.to(device)
        
        # Get alpha_bar_t for the batch
        alpha_bar_t = self.alpha_bar_t[t].view(-1, 1, 1, 1)  # Shape: (batch, 1, 1, 1)
        
        # Sample noise
        eps = torch.randn_like(x_0)
        
        # Apply forward process
        z_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * eps
        
        return z_t, eps


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_filters=64, num_timesteps=1000):
        super(UNet, self).__init__()

        
        # Define the channel dimensions used in your UNet
        time_emb_dim = 128  # This should match the channel dimension where you add the time embedding
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),  # Match the channel dimension
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Downsampling path
        self.enc1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, stride=2, padding=1)  # 128x128
        self.enc3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, stride=2, padding=1)  # 64x64

        # Cross-attention mechanism for style transfer
        # TODO
        self.cross_attention = nn.MultiheadAttention(embed_dim=num_filters * 4, num_heads=4)

        # Bottleneck
        self.bottleneck = nn.Conv2d(num_filters * 4, num_filters * 4, kernel_size=3, stride=1, padding=1)

        # Upsampling path
        self.dec3 = nn.ConvTranspose2d(num_filters * 4, num_filters * 2, kernel_size=3, stride=2, padding=1, output_padding=1)  # 128x128
        self.dec2 = nn.ConvTranspose2d(num_filters * 2, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1)  # 256x256
        self.dec1 = nn.Conv2d(num_filters, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, z, t, style_embedding=None):
        """
        z: Noisy latent spectrogram
        t: Diffusion timestep (for time conditioning)
        style_embedding: Style embedding for cross-attention (in latent space)
        """
        # Process time embedding
        t_embedding = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)  # Make it broadcastable

        if style_embedding is not None:
            style_embedding = self.encoder(style_embedding)

        # Encoder
        z1 = F.relu(self.enc1(z))
        z2 = F.relu(self.enc2(z1)) + t_embedding  # Inject time conditioning
        z3 = F.relu(self.enc3(z2))

        if style_embedding is not None:
            batch_size, c, h, w = z3.shape
            z3_flat = z3.view(batch_size, c, h * w).permute(2, 0, 1)  # Reshape for attention
            style_embedding = style_embedding.unsqueeze(0).repeat(h * w, 1, 1)  # Match dimensions
            z3_flat, _ = self.cross_attention(z3_flat, style_embedding, style_embedding)
            z3 = z3_flat.permute(1, 2, 0).view(batch_size, c, h, w)  # Reshape back

        # Bottleneck
        bottleneck = F.relu(self.bottleneck(z3))

        # Decoder - these are skip connections
        z3_up = F.relu(self.dec3(bottleneck)) + z2
        z2_up = F.relu(self.dec2(z3_up)) + z1
        output = self.dec1(z2_up)

        return output
    

# TODO: I still doubt the formulas are right here
def ddim_sample(z_T, model, alpha_bar_t, beta_t, eta=0.0, timesteps=100):
    """
    DDIM Reverse Sampling Process

    z_T: Initial noisy latent (from Gaussian prior)
    model: Trained denoising U-Net
    alpha_bar_t: Precomputed cumulative noise schedule
    beta_t: Noise variance schedule
    num_steps: Number of denoising steps (less = faster)
    eta: Controls stochasticity (0 = deterministic DDIM)

    Returns: Denoised latent z_0 (final spectrogram latent)
    """
    device = z_T.device
    batch_size = z_T.shape[0]
    
    # Move alpha_bar_t and beta_t to the same device as z_T
    alpha_bar_t = alpha_bar_t.to(device)
    beta_t = beta_t.to(device)
    
    for i in range(timesteps - 1, -1, -1):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        t_float = t.float()
        
        with torch.no_grad():
            noise_pred = model(z_T, t_float)
            
            # Get alpha values and reshape for broadcasting
            alpha_bar_t_prev = alpha_bar_t[t].view(-1, 1, 1, 1)  # [B, 1, 1, 1]
            alpha_bar_t_curr = alpha_bar_t[t+1].view(-1, 1, 1, 1) if i+1 < timesteps else torch.ones(batch_size, 1, 1, 1, device=device)

            # Compute sigma (variance term)
            sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) * (1 - alpha_bar_t_curr) / alpha_bar_t_curr)

            # Compute predicted x_0 (denoised sample)
            pred_x0 = (z_T - torch.sqrt(1 - alpha_bar_t_prev) * noise_pred) / torch.sqrt(alpha_bar_t_prev)

            # Compute direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_bar_t_curr - sigma_t**2) * noise_pred

            # Sample z_{t-1} using the DDIM formula
            z_T = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + sigma_t * torch.randn_like(z_T)

    return z_T

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    


class LatentDiffusionModel(nn.Module):
    def __init__(
        self,
        latent_dim=4,
        num_timesteps=100,
        device=None
    ):
        super().__init__()
        if device is None:
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = device

        # Initialize components
        self.encoder = SpectrogramEncoder(latent_dim=latent_dim).to(self.device)
        self.decoder = SpectrogramDecoder(latent_dim=latent_dim).to(self.device)
        self.unet = UNet(num_timesteps=num_timesteps).to(self.device)
        self.noise_scheduler = ForwardDiffusion(num_timesteps=num_timesteps)

    def encode(self, x):
        """Encode spectrogram to latent space"""
        return self.encoder(x)

    def decode(self, z):
        """Decode latent representation back to spectrogram"""
        return self.decoder(z)

    def diffuse(self, z_0, t):
        """Apply forward diffusion to latent"""
        return self.noise_scheduler(z_0, t)

    def denoise(self, z_t, t):
        """Single denoising step"""
        return self.unet(z_t, t)

    def sample(self, z_T, num_steps=50, eta=0.0):
        """Sample from noise using DDIM"""
        return ddim_sample(
            z_T, 
            self.unet,
            self.noise_scheduler.alpha_bar_t,
            self.noise_scheduler.beta_t,
            timesteps=num_steps,
            eta=eta
        )

    def forward(self, x, t):
        """
        Forward pass through the full model
        x: input spectrogram
        t: timesteps for diffusion
        """
        # Encode to latent space
        z_0 = self.encode(x)
        
        # Apply forward diffusion
        z_t, noise = self.diffuse(z_0, t)
        
        # Predict noise
        noise_pred = self.denoise(z_t, t)
        
        return z_t, noise, noise_pred