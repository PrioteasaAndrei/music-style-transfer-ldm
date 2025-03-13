import torch
import torch.nn as nn

def perceptual_loss(original, reconstructed, feature_extractor=None):
    '''
    TODO: Implement perceptual loss
    '''
    if feature_extractor is None:
        raise ValueError("Feature extractor must be provided for perceptual loss")
    
    # Extract features from original and reconstructed images
    original_features = feature_extractor(original)
    reconstructed_features = feature_extractor(reconstructed)
    
    # Compute perceptual loss as the mean absolute difference between features
    loss = torch.mean(torch.abs(original_features - reconstructed_features))
    
    return loss

def kl_regularization_loss(latent):
    return torch.mean(0.5 * (latent.pow(2) - 1 - torch.log(latent.pow(2) + 1e-8)))

def compression_loss(original, reconstructed, latent, feature_extractor=None):
    mse_loss = nn.MSELoss()(reconstructed, original)

    # Perceptual loss (Optional)
    perceptual_loss_value = perceptual_loss(original, reconstructed, feature_extractor) if feature_extractor else 0

    # KL Regularization (Applied to latent space directly)
    kl_loss = kl_regularization_loss(latent)

    return mse_loss + 0.1 * perceptual_loss_value + 0.01 * kl_loss