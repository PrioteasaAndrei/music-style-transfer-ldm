import torch
import pytest
from model import ddim_sample, ForwardDiffusion, UNet
from model import SpectrogramEncoder, SpectrogramDecoder

def test_ddim_deterministic():
    """Test if DDIM sampling is deterministic when eta=0"""
    # Setup
    batch_size = 2
    channels = 1
    height = width = 256
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # Create test components
    unet = UNet().to(device)
    diffusion = ForwardDiffusion()
    z_T = torch.randn(batch_size, channels, height, width).to(device)
    
    # Run DDIM sampling twice with eta=0
    with torch.no_grad():
        result1 = ddim_sample(z_T, unet, diffusion.alpha_bar_t, diffusion.beta_t, eta=0)
        result2 = ddim_sample(z_T, unet, diffusion.alpha_bar_t, diffusion.beta_t, eta=0)
    
    # Check if results are identical
    assert torch.allclose(result1, result2), "DDIM sampling is not deterministic when eta=0"

    print("Test ddim deterministic passed")

def test_ddim_shape_preservation():
    """Test if DDIM sampling preserves input shape"""
    batch_size = 2
    channels = 1
    height = width = 256
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    unet = UNet().to(device)
    diffusion = ForwardDiffusion()
    z_T = torch.randn(batch_size, channels, height, width).to(device)
    
    with torch.no_grad():
        result = ddim_sample(z_T, unet, diffusion.alpha_bar_t, diffusion.beta_t)
    
    assert result.shape == z_T.shape, f"Shape mismatch: {result.shape} != {z_T.shape}"

    print("Test ddim shape preservation passed")

def test_ddim_value_range():
    """Test if output values are in a reasonable range"""
    batch_size = 2
    channels = 1
    height = width = 256
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    unet = UNet().to(device)
    diffusion = ForwardDiffusion()
    z_T = torch.randn(batch_size, channels, height, width).to(device)
    
    with torch.no_grad():
        result = ddim_sample(z_T, unet, diffusion.alpha_bar_t, diffusion.beta_t)
    
    # Check if values are finite
    assert torch.all(torch.isfinite(result)), "Output contains inf or nan values"
    
    # Check if values are in a reasonable range
    assert torch.max(torch.abs(result)) < 100, "Output values are suspiciously large"

    print("Test ddim value range passed")

def test_forward_reverse_consistency():
    """Test if forward diffusion followed by DDIM sampling is somewhat consistent"""
    batch_size = 2
    channels = 1
    height = width = 256
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    unet = UNet().to(device)
    diffusion = ForwardDiffusion()
    
    # Create a simple test image (e.g., a gradient)
    x_0 = torch.linspace(-1, 1, width).view(1, 1, 1, -1).repeat(batch_size, 1, height, 1).to(device)
    
    # Apply forward diffusion
    t = torch.full((batch_size,), 999, dtype=torch.long).to(device)
    with torch.no_grad():
        z_T, _ = diffusion(x_0, t)
        # Reverse the process
        x_0_recovered = ddim_sample(z_T, unet, diffusion.alpha_bar_t, diffusion.beta_t, eta=0)
    
    # Stack the flattened tensors for correlation calculation
    correlation_input = torch.stack([
        x_0.flatten(),
        x_0_recovered.flatten()
    ])
    
    # Calculate correlation matrix
    correlation = torch.corrcoef(correlation_input)[0, 1]
    assert not torch.isnan(correlation), "Correlation is NaN"

    print("Test forward reverse consistency passed")

def test_sigma_t_behavior():
    """Test if sigma_t behaves correctly with different eta values"""
    batch_size = 2
    channels = 1
    height = width = 256
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    unet = UNet().to(device)
    diffusion = ForwardDiffusion()
    z_T = torch.randn(batch_size, channels, height, width).to(device)
    
    # Test with eta = 0
    with torch.no_grad():
        result_det = ddim_sample(z_T, unet, diffusion.alpha_bar_t, diffusion.beta_t, eta=0)
        result_stoch = ddim_sample(z_T, unet, diffusion.alpha_bar_t, diffusion.beta_t, eta=1)
    
    # The stochastic result should be different from the deterministic one
    assert not torch.allclose(result_det, result_stoch), "Stochastic and deterministic results are identical"

    print("Test sigma_t behavior passed")
def test_encoder_dimensions():
    """Test if encoder preserves expected dimensions"""
    batch_size = 4
    input_channels = 1
    height = width = 256
    latent_dim = 4
    
    # Create encoder
    encoder = SpectrogramEncoder(latent_dim=latent_dim)
    
    # Create dummy input
    x = torch.randn(batch_size, input_channels, height, width)
    
    # Forward pass
    latent = encoder(x)
    
    # Expected output dimensions: [batch_size, latent_dim, height//32, width//32]
    expected_shape = (batch_size, latent_dim, height//32, width//32)
    assert latent.shape == expected_shape, f"Expected shape {expected_shape}, got {latent.shape}"

    print("Test encoder dimensions passed")

def test_decoder_dimensions():
    """Test if decoder restores original dimensions"""
    batch_size = 4
    latent_dim = 4
    height = width = 8  # 256//32 = 8 (matching encoder output)
    
    # Create decoder
    decoder = SpectrogramDecoder(latent_dim=latent_dim)
    
    # Create dummy latent
    z = torch.randn(batch_size, latent_dim, height, width)
    
    # Forward pass
    output = decoder(z)
    
    # Expected output dimensions: [batch_size, 1, height*32, width*32]
    expected_shape = (batch_size, 1, height*32, width*32)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    print("Test decoder dimensions passed")

def test_encoder_decoder_pipeline():
    """Test if encoder-decoder pipeline can reconstruct input"""
    batch_size = 4
    input_channels = 1
    height = width = 256
    latent_dim = 4
    
    # Create models
    encoder = SpectrogramEncoder(latent_dim=latent_dim)
    decoder = SpectrogramDecoder(latent_dim=latent_dim)
    
    # Create dummy input (normalized between -1 and 1 due to tanh in decoder)
    x = torch.rand(batch_size, input_channels, height, width) * 2 - 1
    
    # Forward pass through both models
    latent = encoder(x)
    reconstruction = decoder(latent)
    
    # Check shapes
    assert x.shape == reconstruction.shape, "Input and reconstruction shapes don't match"
    
    # Check if output is in valid range (due to tanh)
    assert torch.all(reconstruction >= -1) and torch.all(reconstruction <= 1), \
        "Decoder output values outside valid range [-1, 1]"

    print("Test encoder-decoder pipeline passed")

def test_decoder_output_range():
    """Test if decoder output respects tanh range"""
    batch_size = 4
    latent_dim = 4
    height = width = 8
    
    decoder = SpectrogramDecoder(latent_dim=latent_dim)
    z = torch.randn(batch_size, latent_dim, height, width)
    
    output = decoder(z)
    
    # Check if output values are finite
    assert torch.all(torch.isfinite(output)), "Decoder output contains inf or nan values"
    
    # Check if output values are in [-1, 1] (due to tanh)
    assert torch.all(output >= -1) and torch.all(output <= 1), \
        "Decoder output values outside tanh range [-1, 1]"

    print("Test decoder output range passed")
if __name__ == "__main__":
    test_ddim_deterministic()
    test_ddim_shape_preservation()
    test_ddim_value_range()
    test_forward_reverse_consistency()
    test_sigma_t_behavior()
    test_encoder_dimensions()
    test_decoder_dimensions()
    test_encoder_decoder_pipeline()
    test_decoder_output_range()
    print("All tests passed!")