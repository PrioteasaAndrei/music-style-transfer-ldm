import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple
import math
from config import config
from models import loss

class SpectrogramEncoder(nn.Module):
    '''
    Output: [B, latent_dim, H//8, W//8] i.e. [B,32,16,16] for 128x128 spectrograms
    '''
    def __init__(self, latent_dim=4):
        super(SpectrogramEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            nn.Conv2d(128, latent_dim, kernel_size=3, stride=2, padding=1),  # 16x16xlatent_dim
            nn.BatchNorm2d(latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)
    
    
class SpectrogramDecoder(nn.Module):
    '''
    input: [B, latent_dim, H//8, W//8] i.e. [B,32,16,16] for 128x128 spectrograms
    '''
    def __init__(self, latent_dim=4):
        super(SpectrogramDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.Tanh()  # Output normalized to [-1, 1]
        )

    def forward(self, z):
        return self.decoder(z)

class StyleEncoder(nn.Module):
    """
    Encodes the style spectrogram into multiple resolution embeddings
    so that it can be injected into different layers of the U-Net.
    Input: [B, 1, 128, 128]
    """
    def __init__(self, in_channels=1, num_filters=64):
        super().__init__()

        # Downsampling path with monotonically increasing channels
        self.enc1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=2, padding=1)      # 64x64
        self.enc2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, stride=2, padding=1)  # 32x32
        self.enc3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, stride=2, padding=1)  # 16x16
        self.enc4 = nn.Conv2d(num_filters * 4, num_filters * 4, kernel_size=3, stride=2, padding=1)  # 8x8
        # Reduce channels to 256 for enc5
        self.enc5 = nn.Conv2d(num_filters * 4, num_filters * 4, kernel_size=3, stride=2, padding=1)  # 4x4
        # Add enc6 with 512 channels and stride 2 to get 2x2
        self.enc6 = nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=3, stride=2, padding=1)  # 2x2

    def forward(self, style_spectrogram):
        """
        Returns a dictionary of different resolution style embeddings.
        """
        s1 = F.relu(self.enc1(style_spectrogram))  # 64x64
        s2 = F.relu(self.enc2(s1))                # 32x32
        s3 = F.relu(self.enc3(s2))                # 16x16
        s4 = F.relu(self.enc4(s3))                # 8x8
        s5 = F.relu(self.enc5(s4))                # 4x4
        s6 = F.relu(self.enc6(s5))                # 2x2

        return {
            "s1": s1,  # [4, 64, 64, 64]
            "s2": s2,  # [4, 128, 32, 32]
            "s3": s3,  # [4, 256, 16, 16]
            "s4": s4,  # [4, 256, 8, 8]
            "s5": s5,  # [4, 256, 4, 4]
            "s6": s6,  # [4, 512, 2, 2]
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
        self.cross_attention1 = CrossAttention(embed_dim=512, num_heads=4)  # For 2x2 feature maps with 512 channels
        self.cross_attention2 = CrossAttention(embed_dim=256, num_heads=4)  # For 4x4 feature maps with 256 channels

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
        z1 = F.relu(self.enc1(z))  # [B, 64, 16, 16]
        z2 = F.relu(self.enc2(z1)) + t_embedding 
        # Apply cross attention to z2 using style embedding s6
        z2_original = z2
        z3 = F.relu(self.enc3(z2))
        z3_original = z3
        z3 = self.cross_attention2(z3, style_embedding["s5"])
        z4 = F.relu(self.enc4(z3))
        z4_original = z4
        z4 = self.cross_attention1(z4, style_embedding["s6"])
        
        # Bottleneck
        z4 = F.relu(self.bottleneck(z4))

        # Decoder with skip connections
        z4 = F.relu(self.dec4(z4))
        z4 = z4 + z3_original  # Skip connection to z3
        
        z3 = F.relu(self.dec3(z4))
        z3 = z3 + z2_original  # Skip connection to z2
        
        z2 = F.relu(self.dec2(z3))
        z2 = z2 + z1  # Skip connection to z1
        
        z1 = self.dec1(z2)
        
        return z1

# TODO: I still doubt the formulas are right here
def ddim_sample(z_T, model, alpha_bar_t, beta_t, style_embedding: dict=None, eta=0.0, timesteps=250, verbose=False):
    """
    DDIM Reverse Sampling Process

    z_T: Initial noisy latent (from Gaussian prior)
    model: Trained denoising U-Net
    alpha_bar_t: Precomputed cumulative noise schedule
    beta_t: Noise variance schedule
    num_steps: Number of denoising steps (less = faster)
    eta: Controls stochasticity (0 = deterministic DDIM)
    verbose: If True, collect and return logs of the sampling process

    Returns: 
        If verbose=False: Denoised latent z_0 (final spectrogram latent)
        If verbose=True: (Denoised latent z_0, sampling_logs)
    """
    device = z_T.device
    batch_size = z_T.shape[0]
    
    # Move alpha_bar_t and beta_t to the same device as z_T
    alpha_bar_t = alpha_bar_t.to(device)
    beta_t = beta_t.to(device)
    
    # For tracking the sampling process when verbose=True
    sampling_logs = {
        'timesteps': [],
        'pred_x0': [],
        'latents': []
    } if verbose else None
    
    z_t = z_T.clone()
    for i in range(timesteps - 1, -1, -1):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        t_float = t.float()
        
        with torch.no_grad():
            noise_pred = model(z_t, t_float, style_embedding)
            
            # Get alpha values and reshape for broadcasting
            alpha_bar_t_prev = alpha_bar_t[t].view(-1, 1, 1, 1)  # [B, 1, 1, 1]
            alpha_bar_t_curr = alpha_bar_t[t+1].view(-1, 1, 1, 1) if i+1 < timesteps else torch.ones(batch_size, 1, 1, 1, device=device)

            # Compute sigma (variance term)
            sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) * (1 - alpha_bar_t_curr) / alpha_bar_t_curr)

            # Compute predicted x_0 (denoised sample)
            pred_x0 = (z_t - torch.sqrt(1 - alpha_bar_t_prev) * noise_pred) / torch.sqrt(alpha_bar_t_prev)

            # Compute direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_bar_t_curr - sigma_t**2) * noise_pred

            # Sample z_{t-1} using the DDIM formula
            z_t = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + sigma_t * torch.randn_like(z_t)

            # Log data if verbose is enabled
            if verbose:
                # some type stuff
                assert sampling_logs is not None
                # Save data every 25 timesteps or at significant points
                if i % 25 == 0 or i in [timesteps-1, timesteps//2, 10, 5, 0]:
                    sampling_logs['timesteps'].append(i)
                    sampling_logs['pred_x0'].append(pred_x0.detach().clone())
                    sampling_logs['latents'].append(z_t.detach().clone())
                    
                    if i % 50 == 0:
                        print(f"Timestep: {i}/{timesteps}, Noise scale: {sigma_t.mean().item():.5f}")

    if verbose:
        return z_t, sampling_logs
    else:
        return z_t

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
    def __init__(self, latent_dim, pretrained_path: str = 'models/pretrained/', num_timesteps=config['forward_diffusion_num_timesteps'], load_full_model=False):
        super(LDM, self).__init__()
        
        # First create all components with correct initialization
        self.encoder = SpectrogramEncoder(latent_dim=latent_dim)
        self.decoder = SpectrogramDecoder(latent_dim=latent_dim)
        self.unet = UNet(in_channels=latent_dim, out_channels=latent_dim, num_filters=64)
        self.noise_scheduler = ForwardDiffusion(num_timesteps=num_timesteps)
        self.style_encoder = StyleEncoder(in_channels=1, num_filters=64)
        self.num_timesteps = num_timesteps
        self.feature_loss_net = loss.VGGishFeatureLoss()

        if pretrained_path:
            if load_full_model:
                # Load the entire LDM model if it exists
                try:
                    # Choose device for loading weights
                    if torch.cuda.is_available():
                        map_location = 'cuda'
                    elif torch.backends.mps.is_available():
                        map_location = 'mps'
                    else: 
                        map_location = 'cpu'
                    
                    # Create a state dict with just encoder and decoder
                    state_dict = torch.load(pretrained_path + 'ldm.pth', map_location=map_location)
                    
                    # Manually load components instead of full model
                    # This avoids issues with module names not matching
                    encoder_prefix = 'encoder.'
                    decoder_prefix = 'decoder.'
                    unet_prefix = 'unet.'
                    style_encoder_prefix = 'style_encoder.'
                    noise_prefix = 'noise_scheduler.'
                    
                    encoder_state_dict = {k[len(encoder_prefix):]: v for k, v in state_dict.items() 
                                         if k.startswith(encoder_prefix)}
                    decoder_state_dict = {k[len(decoder_prefix):]: v for k, v in state_dict.items() 
                                         if k.startswith(decoder_prefix)}
                    unet_state_dict = {k[len(unet_prefix):]: v for k, v in state_dict.items() 
                                      if k.startswith(unet_prefix)}
                    style_encoder_state_dict = {k[len(style_encoder_prefix):]: v for k, v in state_dict.items() 
                                              if k.startswith(style_encoder_prefix)}
                    noise_state_dict = {k[len(noise_prefix):]: v for k, v in state_dict.items() 
                                        if k.startswith(noise_prefix)}
                    
                    # Load state dictionaries into respective components
                    self.encoder.load_state_dict(encoder_state_dict)
                    self.decoder.load_state_dict(decoder_state_dict)
                    self.unet.load_state_dict(unet_state_dict)
                    self.style_encoder.load_state_dict(style_encoder_state_dict)
                    self.noise_scheduler.load_state_dict(noise_state_dict)
                    
                    print(f"Loaded full pretrained LDM components from {pretrained_path + 'ldm.pth'}")
                    
                    # Set appropriate training modes
                    self.encoder.eval()  # Encoder should be frozen
                    self.decoder.train() # Decoder continues to be trained
                    self.unet.train()
                    self.style_encoder.train()
                    
                    # Freeze encoder weights
                    for param in self.encoder.parameters():
                        param.requires_grad = False
                    
                    return
                
                except (FileNotFoundError, RuntimeError) as e:
                    print(f"Could not load full LDM model: {e}")
                    print("Falling back to loading just encoder/decoder weights")
            
            # Default behavior: load just encoder/decoder
            if torch.cuda.is_available():
                # load pretrained weights
                self.encoder.load_state_dict(torch.load(pretrained_path + 'encoder.pth', map_location='cuda'))
                self.decoder.load_state_dict(torch.load(pretrained_path + 'decoder.pth', map_location='cuda'))
            elif torch.backends.mps.is_available():
                # load pretrained weights
                self.encoder.load_state_dict(torch.load(pretrained_path + 'encoder.pth', map_location='mps'))
                self.decoder.load_state_dict(torch.load(pretrained_path + 'decoder.pth', map_location='mps'))
            else: 
                # load pretrained weights
                self.encoder.load_state_dict(torch.load(pretrained_path + 'encoder.pth', map_location='cpu'))
                self.decoder.load_state_dict(torch.load(pretrained_path + 'decoder.pth', map_location='cpu'))
            print("Loaded pretrained weights from", pretrained_path)
          
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
        # normalize to [0, 1] -> Match the original spectrogram range
        reconstructed = (reconstructed + 1) / 2

        return {
            'z_t': z_t,                    # Noisy latent
            'noise': noise,                # Ground truth noise
            'noise_pred': noise_pred,      # Predicted noise
            'z_0': z_0,                    # Original clean latent
            'reconstructed': reconstructed  # Reconstructed spectrogram
        }


