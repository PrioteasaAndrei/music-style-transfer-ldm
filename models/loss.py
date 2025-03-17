import torch
import torch.nn as nn
from lpips import LPIPS
import torch.nn.functional as F

def perceptual_loss(original, reconstructed):
    '''
    Compute perceptual loss using LPIPS (Learned Perceptual Image Patch Similarity)
    '''
    feature_extractor = LPIPS(net='alex').eval()  # Use AlexNet backbone    
    
    # LPIPS expects values in [-1,1] range
    # Check if inputs are in [-1, 1] range
    assert torch.all((original >= -1) & (original <= 1)), "Original input must be in [-1, 1] range"
    assert torch.all((reconstructed >= -1) & (reconstructed <= 1)), "Reconstructed input must be in [-1, 1] range"
    
    original = 2 * original - 1
    reconstructed = 2 * reconstructed - 1
    
    # Compute perceptual loss using LPIPS
    return feature_extractor(original, reconstructed).mean()

def kl_regularization_loss(latent):
    return torch.mean(0.5 * (latent.pow(2) - 1 - torch.log(latent.pow(2) + 1e-8)))

def compression_loss(original, reconstructed, latent, feature_extractor=None):

    mse_loss = nn.MSELoss()(reconstructed, original)

    # Perceptual loss (Optional)
    perceptual_loss_value = perceptual_loss(original, reconstructed, feature_extractor) if feature_extractor else 0

    # KL Regularization (Applied to latent space directly)
    kl_loss = kl_regularization_loss(latent)

    return mse_loss + 0.1 * perceptual_loss_value + 0.01 * kl_loss


def diffusion_loss(noise_pred, noise_target):
    return F.mse_loss(noise_pred, noise_target)

def style_loss(reconstructed, style_spec):
    # TODO: add a more complicated loss here
    # """Compute style loss using perceptual features"""
    # # Extract features using pretrained network
    # content_features = feature_extractor(content_spec)
    # style_features = feature_extractor(style_spec)
    
    # # Compute Gram matrices
    # content_gram = gram_matrix(content_features)
    # style_gram = gram_matrix(style_features)
    
    # return F.mse_loss(content_gram, style_gram)


    return F.mse_loss(reconstructed, style_spec)

