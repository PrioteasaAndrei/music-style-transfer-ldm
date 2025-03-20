import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple
import math
from config import config


class SpectrogramEncoder(nn.Module):
    '''
    Output: [B, latent_dim, H//32, W//32] i.e. [B,32,4,4] for 128x128 spectrograms
    '''
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
    '''
    input: [B, latent_dim, H//32, W//32] i.e. [B,32,4,4] for 128x128 spectrograms
    '''
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

       # Downsampling path to match UNet dimensions
        self.enc1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1)      # 128x128
        self.enc2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, stride=2, padding=1)  # 64x64
        self.enc3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, stride=2, padding=1)  # 32x32
        self.enc4 = nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=3, stride=2, padding=1)  # 16x16
        self.enc5 = nn.Conv2d(num_filters * 8, num_filters * 4, kernel_size=3, stride=2, padding=1)  # 8x8
        self.enc6 = nn.Conv2d(num_filters * 4, num_filters, kernel_size=3, stride=2, padding=1)  # 4x4
        self.enc7 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, stride=2, padding=1)  # 2x2

    def forward(self, style_spectrogram):
        """
        Returns a dictionary of different resolution style embeddings.
        """
        s1 = F.relu(self.enc1(style_spectrogram))  # 128x128
        s2 = F.relu(self.enc2(s1))                # 64x64
        s3 = F.relu(self.enc3(s2))                # 32x32
        s4 = F.relu(self.enc4(s3))                # 16x16
        s5 = F.relu(self.enc5(s4))                # 8x8
        s6 = F.relu(self.enc6(s5))                # 4x4
        s7 = F.relu(self.enc7(s6))                # 2x2

        return {
            "s1": s1,  # 128x128
            "s2": s2,  # 64x64
            "s3": s3,  # 32x32
            "s4": s4,  # 16x16
            "s5": s5,  # 8x8
            "s6": s6,  # 4x4
            "s7": s7   # 2x2
        }

class ForwardDiffusion(nn.Module):
    def __init__(self, num_timesteps=config['forward_diffusion_num_timesteps']):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # Pre-compute values (these are fixed, not learned)
        beta_start = 0.0001
        beta_end = 0.02
        self.register_buffer('beta_t', torch.linspace(beta_start, beta_end, num_timesteps))
        self.register_buffer('alpha_t', 1 - self.beta_t)
        self.register_buffer('alpha_bar_t', torch.cumprod(self.alpha_t, dim=0))
    
    def forward(self, x_0, t):
        # Get alpha_bar_t for the batch
        device = x_0.device
        t = t.to(device)
        self.alpha_bar_t = self.alpha_bar_t.to(device)
        alpha_bar_t = self.alpha_bar_t[t].view(-1, 1, 1, 1)
        
        # Sample noise
        eps = torch.randn_like(x_0, device=device)
        
        # Apply forward process
        z_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * eps
        
        return z_t, eps

    def predict_start_from_noise(self, z_t, t, noise_pred):
        """Predict x_0 from the predicted noise"""
        device = z_t.device
        t = t.to(device)
        self.alpha_bar_t = self.alpha_bar_t.to(device)
        
        alpha_bar_t = self.alpha_bar_t[t].view(-1, 1, 1, 1)
        return (z_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

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
        self.cross_attention1 = CrossAttention(embed_dim=num_filters * 2, num_heads=4)  # For 64x64 feature maps
        self.cross_attention2 = CrossAttention(embed_dim=num_filters * 4, num_heads=4)  # For 32x32 feature maps

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
        print(f"t_embedding: {t_embedding.shape}")
        # Encoder
        z1 = F.relu(self.enc1(z))  # [B, 64, H, W]
        print(f"z1: {z1.shape}")
        print(f"style_embedding['s6']: {style_embedding['s6'].shape}")
        z1_orig = z1
        # z1 = self.cross_attention1(z1, style_embedding["s6"])
        # print(f"z1 after cross attention: {z1.shape}")
        z2 = F.relu(self.enc2(z1)) + t_embedding 
        print(f"z2: {z2.shape}")
        z3 = F.relu(self.enc3(z2)) # 
        print(f"z3: {z3.shape}")
        z3_orig = z3

        z4 = F.relu(self.enc4(z3))
        print(f"z4: {z4.shape}")
        z4_orig = z4




        # z3 = self.cross_attention2(z3, style_embedding["s7"])
        print(f"z3 after cross attention: {z3.shape}")
        bottleneck = F.relu(self.bottleneck(z3))
        print(f"bottleneck: {bottleneck.shape}")
        z3_up = F.relu(self.dec3(bottleneck))
        print(f"z3_up: {z3_up.shape}")
        z3_up = z3_up + z2
        print(f"z3_up + z2: {z3_up.shape}")
        z2_up = F.relu(self.dec2(z3_up))
        print(f"z2_up: {z2_up.shape}")
        z2_up = z2_up + z1_orig
        print(f"z2_up + z1: {z2_up.shape}")
        output = self.dec1(z2_up)   
        print(f"output: {output.shape}")
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
          
        self.unet = UNet(in_channels=latent_dim, out_channels=latent_dim, num_filters=64)
        self.noise_scheduler = ForwardDiffusion(num_timesteps=num_timesteps)
        self.style_encoder = StyleEncoder(in_channels=1, num_filters=64)
        self.num_timesteps = num_timesteps

    def forward(self, x, style, t):

        x = x.float()
        style = style.float()

        z_0 = self.encoder(x)
        style_embedding = self.style_encoder(style)
        z_t, noise = self.noise_scheduler(z_0, t)
        noise_pred = self.unet(z_t, t, style_embedding) # only predicts the noise, not the denoised image
        
        # Get the predicted clean latent from the predicted noise
        alpha_bar_t = self.noise_scheduler.alpha_bar_t[t].view(-1, 1, 1, 1)
        z_0_pred = self.noise_scheduler.predict_start_from_noise(z_t, t, noise_pred)
        
        reconstructed = self.decoder(z_0_pred)

        return {
            'z_t': z_t,                    # Noisy latent
            'noise': noise,                # Ground truth noise
            'noise_pred': noise_pred,      # Predicted noise
            'z_0': z_0,                    # Original clean latent
            'reconstructed': reconstructed  # Reconstructed spectrogram
        }
    

    