import torch
import pytest
from model import ddim_sample, ForwardDiffusion, UNet, LDM
from model import SpectrogramEncoder, SpectrogramDecoder
from config import config
from model import StyleEncoder
from dataset import SpectrogramDataset, prepare_dataset
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model import StyleEncoder, UNet, LDM
from config import config
import soundfile as sf
import numpy as np
from pathlib import Path
from data.audio_processor import AudioPreprocessor
from dataset import SpectrogramPairDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def test_ddim_deterministic():
    """Test if DDIM sampling is deterministic when eta=0"""
    # Setup
    batch_size = 2
    channels = 1
    height = width = 256
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        raise RuntimeError("CUDA is not available. Please run on a machine with a GPU.")
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


def test_music_style_transfer_pipeline_from_dataset():
    """
    Full music style transfer pipeline test using a sample from the actual dataset.
    """
    
    # Load dataset
    _, test_loader = prepare_dataset(config)
    images, labels = next(iter(test_loader))
    sample_image = images[0]  # Get first image from batch
    print(f"Sample image shape: {sample_image.shape}, Label: {labels[0]}")
    
    # Setup models with correct dimensions
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    latent_dim = config['latent_dim_encoder']  # Get the latent dimension from config
    encoder = SpectrogramEncoder(latent_dim=latent_dim).to(device)
    decoder = SpectrogramDecoder(latent_dim=latent_dim).to(device)
    style_encoder = StyleEncoder(num_filters=config['unet_num_filters']).to(device)
    
    # Initialize UNet with the correct input channels (matching the latent_dim)
    unet = UNet(in_channels=latent_dim, out_channels=latent_dim, num_filters=config['unet_num_filters']).to(device)
    diffusion = ForwardDiffusion()
    
    # Load pretrained weights if available
    try:
        encoder.load_state_dict(torch.load('models/pretrained/encoder.pth'))
        decoder.load_state_dict(torch.load('models/pretrained/decoder.pth'))
        print("Loaded pretrained encoder/decoder weights")
    except:
        print("No pretrained weights found, using random initialization")
    
    # Create output directory
    output_dir = Path('tests/downloads/style_transfer_dataset_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Convert input to latent
    sample_tensor = sample_image.unsqueeze(0).to(device)  # Add batch dimension
    
    # First test: simple reconstruction through encoder-decoder
    with torch.no_grad():
        latent = encoder(sample_tensor)
        print(f"Encoded latent shape: {latent.shape}")
        recon = decoder(latent)
        print(f"Decoded reconstruction shape: {recon.shape}")
    
    # Save reconstructed audio
    proc = AudioPreprocessor()
    recon_audio_path = output_dir / 'reconstructed_audio_through_autoencoder.wav'
    recon_pil = transforms.ToPILImage()(recon.cpu().squeeze(0))
    recon_audio = proc.grayscale_mel_spectogram_image_to_audio(
        recon_pil, sr=22050, im_height=sample_image.shape[1], im_width=sample_image.shape[2])
    sf.write(str(recon_audio_path), np.int16(recon_audio * 32767), 22050)
    print(f'Saved reconstructed audio to {recon_audio_path}')
    
    # Step 2: Style transfer through DDIM
    # Get style embedding from the original image for this test
    style_embedding = style_encoder(sample_tensor)
    
    # Create random noise in latent space (not in image space)
    z_T = torch.randn_like(latent)
    print(f"Random noise latent shape: {z_T.shape}")
    
    # Apply DDIM sampling in latent space
    with torch.no_grad():
        denoised_latent = ddim_sample(
            z_T, unet, diffusion.alpha_bar_t, diffusion.beta_t,
            eta=0, style_embedding=style_embedding, timesteps=100)
        print(f"Denoised latent shape: {denoised_latent.shape}")
        
        # Decode the denoised latent
        generated_spectrogram = decoder(denoised_latent)
        print(f"Generated spectrogram shape: {generated_spectrogram.shape}")
    
    # Save generated audio
    gen_audio_path = output_dir / 'generated_audio_with_style_transfer.wav'
    gen_pil = transforms.ToPILImage()(generated_spectrogram.cpu().squeeze(0))
    gen_audio = proc.grayscale_mel_spectogram_image_to_audio(
        gen_pil, sr=22050, im_height=sample_image.shape[1], im_width=sample_image.shape[2])
    sf.write(str(gen_audio_path), np.int16(gen_audio * 32767), 22050)
    print(f'Saved generated audio with style transfer to {gen_audio_path}')
    
    print("Full music style transfer pipeline test complete.")

def test_music_style_transfer_with_ldm(load_full_model=True):
    """
    Test the full music style transfer pipeline using the integrated LDM class.
    """
    import soundfile as sf
    import numpy as np
    from pathlib import Path
    from data.audio_processor import AudioPreprocessor
    from model import LDM
    
    # Load dataset
    _, test_loader = prepare_dataset(config)
    images, labels = next(iter(test_loader))
    sample_image = images[0]  # Get first image from batch
    print(f"Sample image shape: {sample_image.shape}, Label: {labels[0]}")
    
    # Setup LDM model with correct dimensions
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    latent_dim = config['latent_dim_encoder']
    
    # Initialize LDM model with pretrained path
    ldm = LDM(latent_dim=latent_dim, pretrained_path='models/pretrained/', 
              load_full_model=load_full_model).to(device)
    
    if load_full_model:
        print("Testing with full pretrained LDM model (including UNet and StyleEncoder)")
    else:
        print("Testing with only pretrained encoder/decoder (UNet and StyleEncoder are randomly initialized)")
    
    # Create output directory
    output_dir = Path('tests/downloads/style_transfer_ldm_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process sample image
    sample_tensor = sample_image.unsqueeze(0).to(device)  # Add batch dimension
    
    # Step 1: Use LDM's encoder-decoder for simple reconstruction
    with torch.no_grad():
        latent = ldm.encoder(sample_tensor)
        print(f"Encoded latent shape: {latent.shape}")
        recon = ldm.decoder(latent)
        print(f"Decoded reconstruction shape: {recon.shape}")
    
    # Save reconstructed audio
    proc = AudioPreprocessor()
    recon_audio_path = output_dir / 'reconstructed_audio_through_ldm.wav'
    recon_pil = transforms.ToPILImage()(recon.cpu().squeeze(0))
    recon_audio = proc.grayscale_mel_spectogram_image_to_audio(
        recon_pil, sr=22050, im_height=sample_image.shape[1], im_width=sample_image.shape[2])
    sf.write(str(recon_audio_path), np.int16(recon_audio * 32767), 22050)
    print(f'Saved reconstructed audio to {recon_audio_path}')
    
    # Step 2: Style transfer through DDIM using LDM components
    # Here we use the same image as both content and style for demonstration
    style_embedding = ldm.style_encoder(sample_tensor)
    
    # Create random noise in latent space
    z_T = torch.randn_like(latent)
    print(f"Random noise latent shape: {z_T.shape}")
    
    # Apply DDIM sampling in latent space
    with torch.no_grad():
        denoised_latent = ddim_sample(
            z_T, ldm.unet, ldm.noise_scheduler.alpha_bar_t, ldm.noise_scheduler.beta_t,
            eta=0, style_embedding=style_embedding, timesteps=250)
        print(f"Denoised latent shape: {denoised_latent.shape}")
        
        # Decode the denoised latent using LDM's decoder
        generated_spectrogram = ldm.decoder(denoised_latent)
        print(f"Generated spectrogram shape: {generated_spectrogram.shape}")
    
    # Save generated audio
    gen_audio_path = output_dir / 'generated_audio_with_ldm_style_transfer.wav'
    gen_pil = transforms.ToPILImage()(generated_spectrogram.cpu().squeeze(0))
    gen_audio = proc.grayscale_mel_spectogram_image_to_audio(
        gen_pil, sr=22050, im_height=sample_image.shape[1], im_width=sample_image.shape[2])
    sf.write(str(gen_audio_path), np.int16(gen_audio * 32767), 22050)
    print(f'Saved generated audio with LDM style transfer to {gen_audio_path}')
    
    print("Full LDM music style transfer pipeline test complete.")

def diagnose_ldm_generation(load_full_model=True):
    """
    Diagnose why generated audio lacks structure by visualizing spectrograms
    at different stages of the generation process.
    """
    
    # Load dataset
    _, test_loader = prepare_dataset(config)
    images, labels = next(iter(test_loader))
    sample_image = images[0]  
    
    # Setup 
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    latent_dim = config['latent_dim_encoder']
    ldm = LDM(latent_dim=latent_dim, pretrained_path='models/pretrained/', 
              load_full_model=load_full_model).to(device)
    
    # Create output directory
    output_dir = Path('tests/downloads/ldm_diagnosis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directory for saving individual timestep images
    timestep_images_dir = output_dir / 'timestep_images'
    timestep_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original sample image
    plt.figure(figsize=(5, 5))
    plt.imshow(sample_image.squeeze().cpu(), cmap='gray')
    plt.title(f'Original Sample (Label: {labels[0]})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(timestep_images_dir / 'original_sample.png')
    plt.close()
    
    sample_tensor = sample_image.unsqueeze(0).to(device)  # Add batch dimension
    
    # Step 1: Simple reconstruction
    with torch.no_grad():
        latent = ldm.encoder(sample_tensor)
        recon = ldm.decoder(latent)
    
    # Save reconstruction
    plt.figure(figsize=(5, 5))
    plt.imshow(recon.squeeze().detach().cpu(), cmap='gray')
    plt.title('Autoencoder Reconstruction')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(timestep_images_dir / 'autoencoder_reconstruction.png')
    plt.close()
    
    # Step 2: Sample from noise through diffusion
    z_T = torch.randn_like(latent)
    style_embedding = ldm.style_encoder(sample_tensor)
    
    # Use the verbose mode to get sampling logs
    with torch.no_grad():
        denoised_latent, sampling_logs = ddim_sample(
            z_T, ldm.unet, ldm.noise_scheduler.alpha_bar_t, ldm.noise_scheduler.beta_t,
            eta=0, style_embedding=style_embedding, timesteps=250, verbose=True)
    
    # Typesafety for the logs
    assert isinstance(sampling_logs, dict), "Sampling logs should be a dictionary"

    # Generate final spectrogram
    generated_spectrogram = ldm.decoder(denoised_latent)
    
    # Save each timestep as a separate image
    proc = AudioPreprocessor()
    
    for idx, timestep in enumerate(sampling_logs['timesteps']):
        # Get the predicted clean latent at this timestep
        pred_x0 = sampling_logs['pred_x0'][idx]
        
        # Decode to get the spectrogram
        with torch.no_grad():
            spec_at_t = ldm.decoder(pred_x0)
        
        # Save the spectrogram image
        plt.figure(figsize=(5, 5))
        plt.imshow(spec_at_t.squeeze().detach().cpu(), cmap='gray')
        plt.title(f'Timestep {timestep} / 250')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(timestep_images_dir / f'timestep_{timestep:04d}.png')
        plt.close()
        
        # Optionally, save the audio for selected timesteps
        if timestep % 50 == 0 or timestep in [0, 249]:
            audio_path = timestep_images_dir / f'audio_timestep_{timestep:04d}.wav'
            spec_pil = transforms.ToPILImage()(spec_at_t.detach().cpu().squeeze(0))
            audio = proc.grayscale_mel_spectogram_image_to_audio(
                spec_pil, sr=22050, im_height=sample_image.shape[1], im_width=sample_image.shape[2])
            sf.write(str(audio_path), np.int16(audio * 32767), 22050)
    
    # Save the final generated spectrogram
    plt.figure(figsize=(5, 5))
    plt.imshow(generated_spectrogram.squeeze().detach().cpu(), cmap='gray')
    plt.title('Final Generated Spectrogram')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(timestep_images_dir / 'final_generation.png')
    plt.close()
    
    # Also save the final generated audio
    gen_audio_path = output_dir / 'generated_audio_diagnosis.wav'
    gen_pil = transforms.ToPILImage()(generated_spectrogram.detach().cpu().squeeze(0))
    gen_audio = proc.grayscale_mel_spectogram_image_to_audio(
        gen_pil, sr=22050, im_height=sample_image.shape[1], im_width=sample_image.shape[2])
    sf.write(str(gen_audio_path), np.int16(gen_audio * 32767), 22050)
    
    # Create a summary visualization showing a few key timesteps
    num_timesteps_to_show = min(7, len(sampling_logs['timesteps']))
    step_indices = list(range(0, len(sampling_logs['timesteps']), len(sampling_logs['timesteps'])//num_timesteps_to_show))[:num_timesteps_to_show]
    
    fig, axes = plt.subplots(1, 3 + num_timesteps_to_show, figsize=(20, 4))
    
    # Original spectrogram
    axes[0].imshow(sample_image.squeeze().cpu(), cmap='gray')
    axes[0].set_title(f'Original\nLabel: {labels[0]}')
    axes[0].axis('off')
    
    # Simple reconstruction
    axes[1].imshow(recon.squeeze().detach().cpu(), cmap='gray')
    axes[1].set_title('Autoencoder\nReconstruction')
    axes[1].axis('off')
    
    # Intermediate diffusion outputs
    for idx, log_idx in enumerate(step_indices):
        timestep = sampling_logs['timesteps'][log_idx]
        pred_x0 = sampling_logs['pred_x0'][log_idx]
        
        with torch.no_grad():
            spec_at_t = ldm.decoder(pred_x0)
        
        axes[2 + idx].imshow(spec_at_t.squeeze().detach().cpu(), cmap='gray')
        axes[2 + idx].set_title(f'Timestep {timestep}\nPredicted Output')
        axes[2 + idx].axis('off')
    
    # Final generated spectrogram
    axes[-1].imshow(generated_spectrogram.squeeze().detach().cpu(), cmap='gray')
    axes[-1].set_title('Final Generation')
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'diffusion_process_visualization.png')
    plt.close()
    
    print(f"Saved individual timestep images to {timestep_images_dir}/")
    print(f"Saved diffusion visualization summary to {output_dir / 'diffusion_process_visualization.png'}")
    print(f"Saved generated audio to {gen_audio_path}")
    print("Diagnosis complete. Check the visualizations to understand the generation process.")

def test_model_parameters():
    """Print parameter counts for all model components"""

    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Initialize latent dimension from config
    latent_dim = config['latent_dim_encoder']
    
    # Define all model components to test
    models = {
        "SpectrogramEncoder": SpectrogramEncoder(latent_dim=latent_dim),
        "SpectrogramDecoder": SpectrogramDecoder(latent_dim=latent_dim),
        "StyleEncoder": StyleEncoder(in_channels=1, num_filters=config['unet_num_filters']),
        "UNet": UNet(in_channels=latent_dim, out_channels=latent_dim, num_filters=64),
        "LDM (full)": LDM(latent_dim=latent_dim)
    }
    
    # Print header
    print(f"{'Component':<20} {'Total Parameters':<20} {'Trainable Parameters':<20}")
    print("-" * 60)
    
    # Calculate and print parameters for each model
    for name, model in models.items():
        model = model.to(device)
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Calculate trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Print results
        print(f"{name:<20} {total_params:<20,d} {trainable_params:<20,d}")
    
    print("-" * 60)
    print("Parameter count test completed.\n")


def test_dead_style_encoder():
    """Test if the style encoder is dead"""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    style_dataset = SpectrogramPairDataset(config["processed_spectograms_dataset_folderpath"], config["pairing_file_path"])
    train_dataset, test_dataset = torch.utils.data.random_split(style_dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    
    style_encoder = torch.load('models/pretrained/style_encoder.pth')
    style_encoder.to(device)
    style_encoder.eval()


    with torch.no_grad():
         with tqdm(train_loader) as pbar:
            for batch_idx, element in enumerate(pbar):
                    # Correct unpacking of the batch
                    (content_spec, content_label), (style_spec, style_label) = element
                    # print(content_spec.shape)  # You can keep this if needed
                    
                    # Move data to device
                    content_spec = content_spec.to(device)
                    style_spec = style_spec.to(device)

                    style_embedding = style_encoder(style_spec)

                    weights_deviations_s1 = style_embedding['s1'].std().item()
                    weights_deviations_s2 = style_embedding['s2'].std().item()
                    weights_deviations_s3 = style_embedding['s3'].std().item()
                    weights_deviations_s4 = style_embedding['s4'].std().item()
                    weights_deviations_s5 = style_embedding['s5'].std().item()
                    weights_deviations_s6 = style_embedding['s6'].std().item()

                    print(f"weights_deviations_s1: {weights_deviations_s1}")
                    print(f"weights_deviations_s2: {weights_deviations_s2}")
                    print(f"weights_deviations_s3: {weights_deviations_s3}")
                    print(f"weights_deviations_s4: {weights_deviations_s4}")
                    print(f"weights_deviations_s5: {weights_deviations_s5}")
                    print(f"weights_deviations_s6: {weights_deviations_s6}")

                    # check if they are close to 0

                    if weights_deviations_s1 < 0.0001 and weights_deviations_s2 < 0.0001 and weights_deviations_s3 < 0.0001 and weights_deviations_s4 < 0.0001 and weights_deviations_s5 < 0.0001 and weights_deviations_s6 < 0.0001:
                        print("Style encoder is dead")
                        break

                    break
                    

    print("Style encoder is not dead")

def test_different_images_loader():

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    style_dataset = SpectrogramPairDataset(config["processed_spectograms_dataset_folderpath"], config["pairing_file_path"])
    train_dataset, test_dataset = torch.utils.data.random_split(style_dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)

    content_images = []
    style_images = []

    cont = 0

    with tqdm(train_loader) as pbar:
            for batch_idx, element in enumerate(pbar):
                    # Correct unpacking of the batch
                    (content_spec, content_label), (style_spec, style_label) = element
                    content_images.append(content_spec)
                    style_images.append(style_spec)

                    cont += 1

                    if cont > 8:
                        break



    # Convert lists of tensors to tensors
    content_images = torch.cat(content_images[:8], dim=0)  # Take first 8 images
    style_images = torch.cat(style_images[:8], dim=0)

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    
    # Plot content images on top row
    for i in range(8):
        axes[0,i].imshow(content_images[i,0].cpu().numpy(), cmap='gray')
        axes[0,i].axis('off')
        if i == 0:
            axes[0,i].set_title('Content', pad=10)
    
    # Plot style images on bottom row
    for i in range(8):
        axes[1,i].imshow(style_images[i,0].cpu().numpy(), cmap='gray')
        axes[1,i].axis('off')
        if i == 0:
            axes[1,i].set_title('Style', pad=10)
    
    plt.tight_layout()
    plt.savefig('models/plots/dataset_style_comparison.png')
    plt.close()


    print(content_images.shape)
    print(style_images.shape)


def test_vggish_loss():
    """Test if the vggish loss is working"""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    from models import loss
    feature_loss_net = loss.VGGishFeatureLoss().to(device)

    content_images = torch.randn(64,1 ,128, 128).to(device)
    style_images = torch.randn(64,1 ,128, 128).to(device)

    feature_loss = feature_loss_net(content_images,style_images)

    print(feature_loss)


def test_ddim_wrapper():
    """Test if the DDIM sampling wrapper is working correctly"""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Initialize model
    from models.model import LDM
    model = LDM(latent_dim=config['latent_dim_encoder'], pretraind_filename='ldm.pth, 'load_full_model=True).to(device)
    model.eval()

    # Create dummy style spectrogram
    style_spec = torch.randn(1, 1, 128, 128).to(device)
    
    # Define latent shape matching encoder output dimensions
    z_shape = (1, 32, 16, 16)  # [batch, latent_dim, height/8, width/8]
    
    # Test with different timesteps and eta values
    timesteps_list = [10, 50, 100]
    eta_list = [0.0, 0.5, 1.0]
    
    for timesteps in timesteps_list:
        for eta in eta_list:
            print(f"\nTesting DDIM with timesteps={timesteps}, eta={eta}")
            
            # Generate sample
            with torch.no_grad():
                output = model.style_ddim_sample_wrapper(
                    z_shape=z_shape,
                    style_spec=style_spec,
                    timesteps=timesteps,
                    eta=eta
                )
            
            # Check output properties
            print(f"Output shape: {output.shape}")
            print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
            assert output.shape == (1, 1, 128, 128), "Output shape mismatch"
            assert torch.all(output >= 0) and torch.all(output <= 1), "Output range error"
            
            # Check if different eta values produce different results
            if eta > 0:
                with torch.no_grad():
                    output2 = model.style_ddim_sample_wrapper(
                        z_shape=z_shape,
                        style_spec=style_spec,
                        timesteps=timesteps,
                        eta=eta
                    )
                # With eta > 0, outputs should differ due to stochasticity
                assert not torch.allclose(output, output2), f"Outputs identical with eta={eta}"
            
    print("DDIM wrapper test completed successfully")


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
    # test_style_encoder_dimensions()
    # test_unet_dimensions()
    # test_music_style_transfer_pipeline_from_dataset()
    # test_music_style_transfer_with_ldm()
    # diagnose_ldm_generation(load_full_model=True)
    # test_model_parameters()
    # test_dead_style_encoder()
    # test_different_images_loader()
    # test_vggish_loss()
    test_ddim_wrapper()
    print("All tests passed!")