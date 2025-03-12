import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple
import math

### TODO: check all of this

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dims: List[int] = [32, 64, 128, 256],
        latent_dim: int = 64
    ):
        super().__init__()
        
        # Build encoder layers
        modules = []
        for h_dim in hidden_dims:
            modules.extend([
                nn.Conv2d(in_channels, h_dim, 3, stride=2, padding=1),
                nn.GroupNorm(8, h_dim),
                nn.SiLU()
            ])
            in_channels = h_dim
            
        self.encoder = nn.Sequential(*modules)
        self.mu = nn.Conv2d(hidden_dims[-1], latent_dim, 1)
        self.log_var = nn.Conv2d(hidden_dims[-1], latent_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        return self.mu(x), self.log_var(x)

class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dims: List[int] = [256, 128, 64, 32],
        out_channels: int = 1
    ):
        super().__init__()
        
        # Build decoder layers
        modules = []
        in_channels = latent_dim
        for h_dim in hidden_dims:
            modules.extend([
                nn.ConvTranspose2d(in_channels, h_dim, 3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(8, h_dim),
                nn.SiLU()
            ])
            in_channels = h_dim
            
        modules.append(nn.Conv2d(hidden_dims[-1], out_channels, 3, padding=1))
        self.decoder = nn.Sequential(*modules)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

class VAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_dims: List[int] = [32, 64, 128, 256],
        latent_dim: int = 64
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims[::-1], in_channels)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return {"recon": recon, "mu": mu, "log_var": log_var, "z": z}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.encoder(x)
        return self.reparameterize(mu, log_var)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, time_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        time_emb = F.silu(time_emb)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[..., None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + x

class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        
        q = q.view(B, C, -1)
        k = k.view(B, C, -1)
        v = v.view(B, C, -1)

        attn = torch.einsum('bci,bcj->bij', q, k) * (self.channels ** -0.5)
        attn = F.softmax(attn, dim=2)
        
        out = torch.einsum('bij,bcj->bci', attn, v)
        out = out.view(B, C, H, W)
        return self.proj(out) + x

class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,  # Latent dimension
        model_channels: int = 128,
        out_channels: int = 64,  # Latent dimension
        time_emb_dim: int = 256,
        num_res_blocks: int = 2
    ):
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels * 2, model_channels, 3, padding=1)  # *2 for conditioning

        # Downsampling
        self.downs = nn.ModuleList([])
        current_channels = model_channels
        channel_mults = [1, 2, 4]
        
        for mult in channel_mults:
            out_channels = model_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(nn.ModuleList([
                    ResidualBlock(current_channels, time_emb_dim),
                    AttentionBlock(current_channels)
                ]))
                current_channels = out_channels
            
            if mult != channel_mults[-1]:  # Don't downsample at the last block
                self.downs.append(nn.Conv2d(current_channels, current_channels, 4, 2, 1))

        # Middle
        self.mid = nn.ModuleList([
            ResidualBlock(current_channels, time_emb_dim),
            AttentionBlock(current_channels),
            ResidualBlock(current_channels, time_emb_dim)
        ])

        # Upsampling
        self.ups = nn.ModuleList([])
        for mult in reversed(channel_mults[:-1]):
            out_channels = model_channels * mult
            for _ in range(num_res_blocks + 1):
                self.ups.append(nn.ModuleList([
                    ResidualBlock(current_channels, time_emb_dim),
                    AttentionBlock(current_channels)
                ]))
                current_channels = out_channels
            
            self.ups.append(
                nn.ConvTranspose2d(current_channels, current_channels, 4, 2, 1)
            )

        # Final layers
        self.final = nn.Sequential(
            ResidualBlock(model_channels, time_emb_dim),
            nn.Conv2d(model_channels, out_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t = self.time_embed(t)
        
        # Concatenate condition
        x = torch.cat([x, cond], dim=1)
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Store skip connections
        skips = []
        
        # Downsampling
        for layer in self.downs:
            if isinstance(layer, nn.ModuleList):
                for sublayer in layer:
                    h = sublayer(h, t) if isinstance(sublayer, ResidualBlock) else sublayer(h)
            else:
                skips.append(h)
                h = layer(h)
        
        # Middle
        for layer in self.mid:
            h = layer(h, t) if isinstance(layer, ResidualBlock) else layer(h)
        
        # Upsampling
        for layer in self.ups:
            if isinstance(layer, nn.ModuleList):
                for sublayer in layer:
                    h = sublayer(h, t) if isinstance(sublayer, ResidualBlock) else sublayer(h)
            else:
                h = layer(h)
                if skips:
                    h = h + skips.pop()
        
        return self.final(h)

class LatentDiffusion(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int = 64,
        model_channels: int = 128,
        time_emb_dim: int = 256,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        n_timesteps: int = 1000,
        lr: float = 1e-4
    ):
        super().__init__()
        # VAE for encoding/decoding spectrograms
        self.vae = VAE(in_channels=1, latent_dim=latent_dim)
        
        # Denoising UNet
        self.model = UNet(
            in_channels=latent_dim,
            model_channels=model_channels,
            out_channels=latent_dim,
            time_emb_dim=time_emb_dim
        )
        
        self.lr = lr
        
        # Register diffusion parameters
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, n_timesteps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x)
        
    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z)
    
    def add_noise(self, x: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x)
            
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return torch.sqrt(alphas_cumprod_t) * x + torch.sqrt(1 - alphas_cumprod_t) * noise
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode inputs to latent space
        z = self.encode_to_latent(x)
        z_cond = self.encode_to_latent(cond)
        
        # Sample timestep
        batch_size = z.shape[0]
        t = torch.randint(0, len(self.betas), (batch_size,), device=z.device)
        
        # Add noise in latent space
        noise = torch.randn_like(z)
        noisy_z = self.add_noise(z, t, noise)
        
        # Predict noise
        pred = self.model(noisy_z, t, z_cond)
        return pred, noise
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, cond = batch
        
        # Train VAE
        vae_output = self.vae(x)
        recon_loss = F.mse_loss(vae_output["recon"], x)
        kl_loss = -0.5 * torch.mean(1 + vae_output["log_var"] - vae_output["mu"].pow(2) - vae_output["log_var"].exp())
        vae_loss = recon_loss + 0.1 * kl_loss
        
        # Train diffusion
        pred_noise, target_noise = self(x, cond)
        diffusion_loss = F.mse_loss(pred_noise, target_noise)
        
        # Total loss
        total_loss = vae_loss + diffusion_loss
        
        # Logging
        self.log('train_loss', total_loss)
        self.log('vae_loss', vae_loss)
        self.log('diffusion_loss', diffusion_loss)
        
        return total_loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
