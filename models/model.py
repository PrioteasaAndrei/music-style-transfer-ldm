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
        self.embed_dim = embed_dim

    def forward(self, unet_features, style_embedding):
        """
        unet_features: Feature map from U-Net (Q) [B, C, H, W]
        style_embedding: Encoded style spectrogram (K, V) [B, C, H, W]
        """
        batch_size, c, h, w = unet_features.shape

        # Reshape feature maps for attention
        # [B, C, H, W] -> [H*W, B, C]
        unet_features = unet_features.permute(2, 3, 0, 1)  # [H, W, B, C]
        unet_features = unet_features.reshape(h * w, batch_size, c)  # [H*W, B, C]
        
        # Reshape style_embedding
        # [B, C, H, W] -> [H*W, B, C]
        style_embedding = style_embedding.permute(2, 3, 0, 1)  # [H, W, B, C]
        style_embedding = style_embedding.reshape(h * w, batch_size, c)  # [H*W, B, C]
 
        # Apply cross-attention
        attended_features, _ = self.multihead_attn(unet_features, style_embedding, style_embedding)
        
        # Reshape back to feature map
        # [H*W, B, C] -> [B, C, H, W]
        attended_features = attended_features.reshape(h, w, batch_size, c)  # [H, W, B, C]
        attended_features = attended_features.permute(2, 3, 0, 1)  # [B, C, H, W]
        
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

        # Downsampling path with proper padding to maintain spatial dimensions
        self.enc1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, stride=2, padding=1)  # 128x128
        self.enc3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, stride=2, padding=1)  # 64x64
        self.enc4 = nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=3, stride=2, padding=1)  # 32x32

        # Cross attention layers with correct embedding dimensions
        self.cross_attention1 = CrossAttention(embed_dim=num_filters * 4, num_heads=4)  # For 64x64 feature maps
        self.cross_attention2 = CrossAttention(embed_dim=num_filters * 8, num_heads=4)  # For 32x32 feature maps

        # Bottleneck
        self.bottleneck = nn.Conv2d(num_filters * 8, num_filters * 8, kernel_size=3, stride=1, padding=1)

        # Upsampling path with proper padding to maintain spatial dimensions
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
        z1 = F.relu(self.enc1(z))  # [B, 64, H, W]
        z2 = F.relu(self.enc2(z1)) + t_embedding  # [B, 128, H/2, W/2]
        z3 = F.relu(self.enc3(z2))  # [B, 256, H/4, W/4]

        # Store original z3 for skip connection
        z3_orig = z3
        
        # Ensure spatial dimensions match before cross-attention
        if z3.shape[-1] != style_embedding['s3'].shape[-1] or z3.shape[-2] != style_embedding['s3'].shape[-2]:
            z3 = F.interpolate(z3, size=style_embedding['s3'].shape[-2:], mode='bilinear', align_corners=False)
        
        z3 = self.cross_attention1(z3, style_embedding["s3"])
        z4 = F.relu(self.enc4(z3))  # [B, 512, H/8, W/8]
        
        # Store original z4 for skip connection
        z4_orig = z4
        
        # Ensure spatial dimensions match before cross-attention
        if z4.shape[-1] != style_embedding['s4'].shape[-1] or z4.shape[-2] != style_embedding['s4'].shape[-2]:
            z4 = F.interpolate(z4, size=style_embedding['s4'].shape[-2:], mode='bilinear', align_corners=False)
            
        z4 = self.cross_attention2(z4, style_embedding["s4"])

        # Bottleneck
        bottleneck = F.relu(self.bottleneck(z4))

        # Decoder - these are skip connections
        z4_up = F.relu(self.dec4(bottleneck))
        # Ensure z4_up matches z3_orig dimensions for skip connection
        if z4_up.shape != z3_orig.shape:
            z4_up = F.interpolate(z4_up, size=z3_orig.shape[-2:], mode='bilinear', align_corners=False)
        z4_up = z4_up + z3_orig
        
        z3_up = F.relu(self.dec3(z4_up))
        # Ensure z3_up matches z2 dimensions for skip connection
        if z3_up.shape != z2.shape:
            z3_up = F.interpolate(z3_up, size=z2.shape[-2:], mode='bilinear', align_corners=False)
        z3_up = z3_up + z2
        
        z2_up = F.relu(self.dec2(z3_up))
        # Ensure z2_up matches z1 dimensions for skip connection
        if z2_up.shape != z1.shape:
            z2_up = F.interpolate(z2_up, size=z1.shape[-2:], mode='bilinear', align_corners=False)
        z2_up = z2_up + z1
        
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
    def __init__(self, latent_dim, pretrained_path: str = 'models/pretrained/', num_timesteps=100):
        super(LDM, self).__init__()
        self.encoder = SpectrogramEncoder(latent_dim=latent_dim) # NOTE: needs to be freezed after pretraining
        self.decoder = SpectrogramDecoder(latent_dim=latent_dim) # NOTE: train with the unet starting from the pretrained

        if pretrained_path:
            # load pretrained weights
            self.encoder.load_state_dict(torch.load(pretrained_path + 'encoder.pth'))
            self.decoder.load_state_dict(torch.load(pretrained_path + 'decoder.pth'))
          
            # freeze pretrained weights
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            for param in self.decoder.parameters():
                param.requires_grad = True

            self.encoder.eval()
            self.decoder.train()
          
        # TODO: not checked
        self.unet = UNet(in_channels=latent_dim, out_channels=latent_dim, num_filters=64)
        # TODO: not checked
        self.noise_scheduler = ForwardDiffusion(num_timesteps=num_timesteps)
        self.style_encoder = StyleEncoder(in_channels=1, num_filters=64)
        self.num_timesteps = num_timesteps

    def forward(self, x, style, t):

        x = x.float()
        style = style.float()

        z_0 = self.encoder(x)
        style_embedding = self.style_encoder(style)
        z_t, noise = self.noise_scheduler(z_0, t)
        noise_pred = self.unet(z_t, t, style_embedding)
        reconstructed = self.decoder(z_0)

        return z_t, noise, noise_pred, z_0, reconstructed
    
    def sample(self, z_T, style, num_steps=50, eta=0.0):
        """
        Sample from noise using DDIM with style conditioning
        
        Args:
            z_T: Initial noise
            style: Style spectrogram to condition on
            num_steps: Number of deno ising steps
            eta: Controls the stochasticity (0 = deterministic)
        """
        # Get style embeddings
        style_embedding = self.style_encoder(style)
        
        # Sample using DDIM
        z_0 = ddim_sample(
            z_T, 
            self.unet,
            self.alpha_bar_t,
            self.beta_t,
            style_embedding=style_embedding,
            eta=eta,
            timesteps=num_steps
        )
        
        # Decode to spectrogram
        return self.decoder(z_0)

    