import torch
import pytest
from model import ddim_sample, ForwardDiffusion, UNet
from model import SpectrogramEncoder, SpectrogramDecoder
from config import config
from model import StyleEncoder
from dataset import SpectrogramDataset, prepare_dataset
import torch
import matplotlib.pyplot as plt

def test_ddim_deterministic():
    """Test if DDIM sampling is deterministic when eta=0"""
    # Setup
    batch_size = 2
    channels = 1
    height = width = 256
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # Create test components
    unet = UNet(num_filters=config['unet_num_filters']).to(device)
    style_encoder = StyleEncoder(num_filters=config['unet_num_filters']).to(device)
    diffusion = ForwardDiffusion()

    dummy_style_spectrogram = torch.randn(batch_size, channels, height, width).to(device)
    style_embedding = style_encoder(dummy_style_spectrogram)
    
    z_T = torch.randn(batch_size, channels, height, width).to(device)
    
    # Run DDIM sampling twice with eta=0
    with torch.no_grad():
        result1 = ddim_sample(z_T, unet, diffusion.alpha_bar_t, diffusion.beta_t, eta=0, style_embedding=style_embedding)
        result2 = ddim_sample(z_T, unet, diffusion.alpha_bar_t, diffusion.beta_t, eta=0, style_embedding=style_embedding)
    
    # Check if results are identical
    assert torch.allclose(result1, result2), "DDIM sampling is not deterministic when eta=0"

    print("Test ddim deterministic passed")

def test_ddim_shape_preservation():
    """Test if DDIM sampling preserves input shape"""
    batch_size = 2
    channels = 1
    height = width = 256
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    unet = UNet(num_filters=config['unet_num_filters']).to(device)
    style_encoder = StyleEncoder(num_filters=config['unet_num_filters']).to(device)
    diffusion = ForwardDiffusion()
    style_embedding = style_encoder(torch.randn(batch_size, channels, height, width).to(device))
    z_T = torch.randn(batch_size, channels, height, width).to(device)
    
    with torch.no_grad():
        result = ddim_sample(z_T, unet, diffusion.alpha_bar_t, diffusion.beta_t, style_embedding=style_embedding)
    
    assert result.shape == z_T.shape, f"Shape mismatch: {result.shape} != {z_T.shape}"

    print("Test ddim shape preservation passed")

def test_ddim_value_range():
    """Test if output values are in a reasonable range"""
    batch_size = 2
    channels = 1
    height = width = 256
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    unet = UNet(num_filters=config['unet_num_filters']).to(device)
    diffusion = ForwardDiffusion()
    style_encoder = StyleEncoder(num_filters=config['unet_num_filters']).to(device)
    style_embedding = style_encoder(torch.randn(batch_size, channels, height, width).to(device))
    z_T = torch.randn(batch_size, channels, height, width).to(device)
    
    with torch.no_grad():
        result = ddim_sample(z_T, unet, diffusion.alpha_bar_t, diffusion.beta_t, style_embedding=style_embedding)
    
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
    
    unet = UNet(num_filters=config['unet_num_filters']).to(device)
    diffusion = ForwardDiffusion()
    style_encoder = StyleEncoder(num_filters=config['unet_num_filters']).to(device)
    style_embedding = style_encoder(torch.randn(batch_size, channels, height, width).to(device))
    
    # Create a simple test image (e.g., a gradient)
    x_0 = torch.linspace(-1, 1, width).view(1, 1, 1, -1).repeat(batch_size, 1, height, 1).to(device)
    
    # Apply forward diffusion
    t = torch.full((batch_size,), 999, dtype=torch.long).to(device)
    with torch.no_grad():
        z_T, _ = diffusion(x_0, t)
        # Reverse the process
        x_0_recovered = ddim_sample(z_T, unet, diffusion.alpha_bar_t, diffusion.beta_t, eta=0, style_embedding=style_embedding)
    
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
    
    unet = UNet(num_filters=config['unet_num_filters']).to(device)
    style_encoder = StyleEncoder(num_filters=config['unet_num_filters']).to(device)
    style_embedding = style_encoder(torch.randn(batch_size, channels, height, width).to(device))
    diffusion = ForwardDiffusion()
    z_T = torch.randn(batch_size, channels, height, width).to(device)
    
    # Test with eta = 0
    with torch.no_grad():
        result_det = ddim_sample(z_T, unet, diffusion.alpha_bar_t, diffusion.beta_t, eta=0, style_embedding=style_embedding)
        result_stoch = ddim_sample(z_T, unet, diffusion.alpha_bar_t, diffusion.beta_t, eta=1, style_embedding=style_embedding)
    
    # The stochastic result should be different from the deterministic one
    assert not torch.allclose(result_det, result_stoch), "Stochastic and deterministic results are identical"

    print("Test sigma_t behavior passed")
def test_encoder_dimensions():
    """Test if encoder preserves expected dimensions"""
    batch_size = 4
    input_channels = 1
    height = width = 128
    latent_dim = config['latent_dim_encoder']
    
    # Create encoder
    encoder = SpectrogramEncoder(latent_dim=latent_dim)
    
    # Create dummy input
    x = torch.randn(batch_size, input_channels, height, width)
    
    # Forward pass
    latent = encoder(x)
    
    # Expected output dimensions: [batch_size, latent_dim, height//32, width//32]
    expected_shape = (batch_size, latent_dim, 16,16)
    print(f"Expected shape: {expected_shape}, got: {latent.shape}")
    assert latent.shape == expected_shape, f"Expected shape {expected_shape}, got {latent.shape}"

    print("Test encoder dimensions passed")

def test_decoder_dimensions():
    """Test if decoder restores original dimensions"""
    batch_size = 4
    latent_dim = config['latent_dim_encoder']
    height = width = 16  
    
    # Create decoder
    decoder = SpectrogramDecoder(latent_dim=latent_dim)
    
    # Create dummy latent
    z = torch.randn(batch_size, latent_dim, height, width)
    
    # Forward pass
    output = decoder(z)
    
    # Expected output dimensions: [batch_size, 1, height*32, width*32]
    expected_shape = (batch_size, 1, 128,128)
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


def check_dataset_ranges(dataset, num_samples=100):
    """
    Check if images in the dataset are properly normalized between 0 and 1.
    
    Args:
        dataset: SpectrogramDataset instance
        num_samples: Number of samples to check (default: 100)
    """
    print(f"Checking {num_samples} samples from dataset of size {len(dataset)}")
    
    num_samples = min(num_samples, len(dataset))
    all_good = True
    
    for idx in range(num_samples):
        image_tensor, label = dataset[idx]
        
        min_val = image_tensor.min().item()
        max_val = image_tensor.max().item()
        mean_val = image_tensor.mean().item()
        std_val = image_tensor.std().item()
        
        # Check if values are in range [0, 1]
        if min_val < 0 or max_val > 1:
            print(f"WARNING: Image {idx} (label: {label}) has values outside [0, 1] range!")
            print(f"         Min: {min_val:.4f}, Max: {max_val:.4f}")
            all_good = False
        
        # Print statistics for each image
        print(f"Image {idx:3d} - Label: {label:10s} | "
              f"Min: {min_val:.4f}, Max: {max_val:.4f}, "
              f"Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    
    if all_good:
        print("\nAll checked images are properly normalized between 0 and 1!")
    else:
        print("\nWARNING: Some images have values outside the [0, 1] range!")

def check_dataset_dimensions(dataset, expected_size=(256, 256)):
    """
    Check if all images in the dataset have the same dimensions.
    
    Args:
        dataset: SpectrogramDataset instance
        expected_size: Tuple of (height, width) that images should have
    """
    print(f"Checking dimensions for {len(dataset)} images...")
    print(f"Expected size: {expected_size}")
    
    all_correct = True
    incorrect_indices = []
    
    for idx in range(len(dataset)):
        image_tensor, label = dataset[idx]
        
        # Get current image dimensions
        channels, height, width = image_tensor.shape
        current_size = (height, width)
        
        # Check if dimensions match expected
        if current_size != expected_size or channels != 1:
            all_correct = False
            incorrect_indices.append(idx)
            print(f"\nWARNING: Image {idx} (label: {label}) has incorrect dimensions!")
            print(f"Expected: (1, {expected_size[0]}, {expected_size[1]})")
            print(f"Got:      ({channels}, {height}, {width})")
        
        # Print progress every 1000 images
        if (idx + 1) % 1000 == 0:
            print(f"Checked {idx + 1}/{len(dataset)} images...")
    
    # Final report
    if all_correct:
        print(f"\nAll {len(dataset)} images have correct dimensions!")
    else:
        print(f"\nFound {len(incorrect_indices)} images with incorrect dimensions!")
        print(f"Indices of incorrect images: {incorrect_indices}")


def test_autoencoder_reconstruction():
    """
    Load pretrained autoencoder and decoder, reconstruct 5 random images from dataset
    and plot them side by side with originals.
    """

    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load models
    encoder = SpectrogramEncoder(config['latent_dim_encoder']).to(device)
    decoder = SpectrogramDecoder(config['latent_dim_encoder']).to(device)
    
    # Load pretrained weights
    encoder.load_state_dict(torch.load('models/pretrained/encoder.pth'))
    decoder.load_state_dict(torch.load('models/pretrained/decoder.pth'))
    
    # Set to eval mode
    encoder.eval()
    decoder.eval()

    # Load dataset
    train_loader, test_loader = prepare_dataset(config)
    # Get one batch from the test_loader (batch size should be > 5)
    images, labels = next(iter(test_loader))
    num_examples = min(5, images.size(0))
    fig, axes = plt.subplots(2, num_examples, figsize=(15, 6))
    
    with torch.no_grad():
        for idx in range(num_examples):
            # Get image and label from the batch
            image = images[idx].unsqueeze(0).to(device)  # Add batch dimension and send to device
            label = labels[idx]
            
            # Get reconstruction from the encoder and decoder
            latent = encoder(image)
            reconstruction = decoder(latent)
            
            # Plot original image
            axes[0, idx].imshow(image.squeeze().cpu(), cmap='gray')
            axes[0, idx].set_title(f'Original\n{label}')
            axes[0, idx].axis('off')
            
            # Plot reconstructed image
            axes[1, idx].imshow(reconstruction.squeeze().cpu(), cmap='gray')
            axes[1, idx].set_title('Reconstruction')
            axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('models/plots/autoencoder_reconstructions.png')
    plt.close()
    print("Reconstruction test complete. Check models/plots/autoencoder_reconstructions.png")



def test_style_encoder_dimensions():
    """Test if StyleEncoder outputs correct dimensions at each resolution level"""
    batch_size = 4
    channels = 1
    height = width = 128
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Create model
    style_encoder = StyleEncoder(in_channels=channels, num_filters=64).to(device)

    # Print total number of parameters
    total_params = sum(p.numel() for p in style_encoder.parameters())
    print(f"\nTotal number of parameters in StyleEncoder: {total_params:,}")
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in style_encoder.parameters() if p.requires_grad)
    print(f"Trainable parameters in StyleEncoder: {trainable_params:,}\n")

    # Create dummy input
    style_spectrogram = torch.randn(batch_size, channels, height, width).to(device)

    # Get style embeddings
    style_embeddings = style_encoder(style_spectrogram)

    for key, value in style_embeddings.items():
        print(f"{key}: {value.shape}")

    # Check each resolution level
    # Expected shapes for each resolution level
    expected_shapes = {
        "s1": (batch_size, 64, 64, 64),
        "s2": (batch_size, 128, 32, 32),
        "s3": (batch_size, 256, 16, 16),
        "s4": (batch_size, 256, 8, 8),
        "s5": (batch_size, 256, 4, 4),
        "s6": (batch_size, 512, 2, 2)
    }

    # Check each resolution level matches expected shape
    for key, expected_shape in expected_shapes.items():
        assert style_embeddings[key].shape == expected_shape, \
            f"Shape mismatch for {key}. Expected {expected_shape}, got {style_embeddings[key].shape}"

    print("Test style encoder dimensions passed")


def test_unet_dimensions():
    """Test if UNet preserves expected dimensions through the full pipeline"""
    batch_size = 4
    channels = 32
    height = width = 16
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Create model
    unet = UNet(in_channels=channels, out_channels=channels, num_filters=64).to(device)

    # Print total number of parameters
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"\nTotal number of parameters in UNet: {total_params:,}")
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"Trainable parameters in UNet: {trainable_params:,}\n")

    # Create dummy input
    x = torch.randn(batch_size, channels, height, width).to(device)

    # Forward pass
    # Create dummy timesteps and style embeddings
    t = torch.zeros(batch_size).to(device)  # Timestep 0 for all samples
    # Create dummy style embeddings matching expected dimensions
    style_embeddings = {
        "s1": torch.randn(batch_size, 64, 64, 64).to(device),
        "s2": torch.randn(batch_size, 128, 32, 32).to(device),
        "s3": torch.randn(batch_size, 256, 16, 16).to(device),
        "s4": torch.randn(batch_size, 256, 8, 8).to(device),
        "s5": torch.randn(batch_size, 256, 4, 4).to(device),
        "s6": torch.randn(batch_size, 512, 2, 2).to(device)
    }
    z = unet(x, t, style_embeddings)

    # Check if output dimensions match expected
    expected_shape = (batch_size, channels, height, width)
    assert z.shape == expected_shape, f"Output shape mismatch. Expected {expected_shape}, got {z.shape}"

    print("Test UNet dimensions passed")


if __name__ == "__main__":
    # test_ddim_deterministic()
    # test_ddim_shape_preservation()
    # test_ddim_value_range()
    # test_forward_reverse_consistency()
    # test_sigma_t_behavior()
    # test_encoder_dimensions()
    # test_decoder_dimensions()
    # test_encoder_decoder_pipeline()
    # test_decoder_output_range()
    # dataset = SpectrogramDataset(config)
    # check_dataset_ranges(dataset)
    # check_dataset_dimensions(dataset, (128, 128))
    # test_autoencoder_reconstruction()
    test_style_encoder_dimensions()
    test_unet_dimensions()
    print("All tests passed!")