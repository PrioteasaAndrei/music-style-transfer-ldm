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
    

class StyleEncoder(nn.Module):
    """
    Encodes the style spectrogram into multiple resolution embeddings
    so that it can be injected into different layers of the U-Net.
    """
    def __init__(self, in_channels=1, num_filters=64):
        super().__init__()

        # Downsampling path
        self.enc1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, stride=2, padding=1)  # 128x128
        self.enc3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, stride=2, padding=1)  # 64x64
        self.enc4 = nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=3, stride=2, padding=1)  # 32x32

    def forward(self, style_spectrogram):
        """
        Returns a dictionary of different resolution style embeddings.
        """
        s1 = F.relu(self.enc1(style_spectrogram))  # 256x256
        s2 = F.relu(self.enc2(s1))  # 128x128
        s3 = F.relu(self.enc3(s2))  # 64x64
        s4 = F.relu(self.enc4(s3))  # 32x32

        return { "s1": s1, "s2": s2, "s3": s3, "s4": s4 }  # Return different resolutions

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
    


class CrossAttention(nn.Module):
    """
    Implements cross-attention for injecting style spectrogram information into the U-Net.
    """
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, unet_features, style_embedding):
        """
        unet_features: Feature map from U-Net (Q)
        style_embedding: Encoded style spectrogram (K, V)
        """
        batch_size, c, h, w = unet_features.shape

        # Reshape feature maps for attention
        unet_features = unet_features.view(batch_size, c, h * w).permute(2, 0, 1)  # [H*W, B, C]
        
        # Reshape style_embedding from [1, 2, 256, 64, 64] to [B, C, H, W] format
        style_embedding = style_embedding.squeeze(0)  # Remove first dim
        style_embedding = style_embedding.view(batch_size, -1, h, w)  # Combine channels
        
        # Reshape for attention
        style_embedding = style_embedding.view(batch_size, -1, h * w).permute(2, 0, 1)  # [H*W, B, C]
        
        # Apply cross-attention
        attended_features, _ = self.multihead_attn(unet_features, style_embedding, style_embedding)

        # Reshape back to feature map
        attended_features = attended_features.permute(1, 2, 0).view(batch_size, c, h, w)

        return attended_features


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_filters=64):
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
        self.enc4 = nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=3, stride=2, padding=1)  # 32x32

        self.cross_attention1 = CrossAttention(embed_dim=num_filters * 4, num_heads=4)
        self.cross_attention2 = CrossAttention(embed_dim=num_filters * 8, num_heads=4)

        # Bottleneck
        self.bottleneck = nn.Conv2d(num_filters * 8, num_filters * 8, kernel_size=3, stride=1, padding=1)

        # Upsampling path
        self.dec4 = nn.ConvTranspose2d(num_filters * 8, num_filters * 4, kernel_size=3, stride=2, padding=1, output_padding=1)  # 64x64
        self.dec3 = nn.ConvTranspose2d(num_filters * 4, num_filters * 2, kernel_size=3, stride=2, padding=1, output_padding=1)  # 128x128
        self.dec2 = nn.ConvTranspose2d(num_filters * 2, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1)  # 256x256
        self.dec1 = nn.Conv2d(num_filters, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, z, t, style_embedding: dict =None):
        """
        z: Noisy latent spectrogram
        t: Diffusion timestep (for time conditioning)
        style_embedding: Style embedding for cross-attention (in latent space)
        """
        # Process time embedding
        t_embedding = self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)  # Make it broadcastable

        # Encoder
        z1 = F.relu(self.enc1(z))
        z2 = F.relu(self.enc2(z1)) + t_embedding  # Inject time conditioning
        z3 = F.relu(self.enc3(z2))
        z3 = self.cross_attention1(z3, style_embedding["s3"])
        z4 = F.relu(self.enc4(z3))
        z4 = self.cross_attention2(z4, style_embedding["s4"])

        # Bottleneck
        bottleneck = F.relu(self.bottleneck(z4))

        # Decoder - these are skip connections
        z4_up = F.relu(self.dec4(bottleneck)) + z3
        z3_up = F.relu(self.dec3(z4_up)) + z2
        z2_up = F.relu(self.dec2(z3_up)) + z1
        output = self.dec1(z2_up)

        return output

# TODO: I still doubt the formulas are right here
def ddim_sample(z_T, model, alpha_bar_t, beta_t, style_embedding: dict =None, eta=0.0, timesteps=100):
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
            noise_pred = model(z_T, t_float, style_embedding)
            
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


class LDM(nn.Module):
    def __init__(self, num_filters=64, num_timesteps=100):
        super(LDM, self).__init__()
        self.encoder = SpectrogramEncoder(latent_dim=num_filters * 8) # NOTE: needs to be freezed after pretraining
        self.decoder = SpectrogramDecoder(latent_dim=num_filters * 8) # NOTE: train with the unet starting from the pretrained
        self.unet = UNet(in_channels=1, out_channels=1, num_filters=num_filters)
        self.noise_scheduler = ForwardDiffusion(num_timesteps=1000)
        self.style_encoder = StyleEncoder(in_channels=1, num_filters=num_filters)

        # Diffusion parameters
        self.num_timesteps = num_timesteps
        self.beta_t, self.alpha_t, self.alpha_bar_t = self.get_noise_schedule(num_timesteps)


    '''
    TODO: these are not right I just copied them from chat gpt, but I need to adjust them to my code.
    
    '''
    def get_noise_schedule(self, num_timesteps, beta_start=1e-4, beta_end=0.02):
        """Creates the noise schedule used in DDIM."""
        beta_t = torch.linspace(beta_start, beta_end, num_timesteps)
        alpha_t = 1 - beta_t
        alpha_bar_t = torch.cumprod(alpha_t, dim=0)
        return beta_t, alpha_t, alpha_bar_t

    def forward(self, spectrogram, style_spectrogram, t):
        """
        Full forward pass of the LDM.
        
        spectrogram: Input spectrogram to transform.
        style_spectrogram: Style reference.
        t: Diffusion timestep.
        """
        with torch.no_grad():
            # Encode spectrograms into latent space
            z_0 = self.encoder(spectrogram)
            style_embedding = self.style_encoder(style_spectrogram)

        # Sample Gaussian noise
        noise = torch.randn_like(z_0)

        # Forward diffusion: Add noise to latent at timestep t
        noisy_latent = (
            torch.sqrt(self.alpha_bar_t[t]).view(-1, 1, 1, 1) * z_0 +
            torch.sqrt(1 - self.alpha_bar_t[t]).view(-1, 1, 1, 1) * noise
        )

        # Denoising step using U-Net (with cross-attention style conditioning)
        noise_pred = self.unet(noisy_latent, style_embedding)

        return noise_pred, noise