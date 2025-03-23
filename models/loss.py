import torch
import torch.nn as nn
from lpips import LPIPS
import torch.nn.functional as F

def perceptual_loss(original, reconstructed):
    '''
    Compute perceptual loss using LPIPS (Learned Perceptual Image Patch Similarity)
    '''
    feature_extractor = LPIPS(net='alex', verbose=False).eval()  # Use AlexNet backbone    
    feature_extractor.to(original.device)  # Move to the same device as input tensors
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

def compression_loss(original, reconstructed, latent):

    mse_loss = nn.MSELoss()(reconstructed, original)

    # Perceptual loss (Optional)
    perceptual_loss_value = perceptual_loss(original, reconstructed)

    # KL Regularization (Applied to latent space directly)
    kl_loss = kl_regularization_loss(latent)

    return mse_loss + 0.1 * perceptual_loss_value + 0.01 * kl_loss


def diffusion_loss(noise_pred, noise_target):
    return F.mse_loss(noise_pred, noise_target)


class VGGishFeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Load VGGish but only use the convolutional features part
        vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.features = vggish.features  # Only use the conv layers
        self.features.eval()
        
        # Freeze the weights
        for param in self.features.parameters():
            param.requires_grad = False
            
    def forward(self, predicted, target):
        # Remove channel dimension if needed
        # if predicted.size(1) == 1:
        #     predicted = predicted.squeeze(1)
        #     target = target.squeeze(1)
        
        # # Add channel dimension expected by VGGish
        # predicted = predicted.unsqueeze(1)
        # target = target.unsqueeze(1)
        
        # Get features from multiple layers
        pred_features = []
        target_features = []
        
        with torch.no_grad():
            # Extract features layer by layer
            x_pred = predicted
            x_target = target
            
            for layer in self.features:
                x_pred = layer(x_pred)
                x_target = layer(x_target)
                
                # Collect features after each ReLU layer
                if isinstance(layer, nn.ReLU):
                    pred_features.append(x_pred)
                    target_features.append(x_target)
        
        # Compute loss across batch
        total_loss = 0
        for p_feat, t_feat in zip(pred_features, target_features):
            # Normalize features
            p_feat = p_feat / (torch.std(p_feat, dim=[1,2,3], keepdim=True) + 1e-8)
            t_feat = t_feat / (torch.std(t_feat, dim=[1,2,3], keepdim=True) + 1e-8)
            
            total_loss += torch.nn.functional.mse_loss(p_feat, t_feat)
            
        return total_loss / len(pred_features)


def style_loss(reconstructed, style_spec, feature_loss_net):

    return feature_loss_net(reconstructed, style_spec)

def gram_matrix(features):
    """Compute the Gram matrix from feature maps (B, C, H, W)"""
    B, C, H, W = features.size()
    features = features.view(B, C, H * W)
    return torch.bmm(features, features.transpose(1, 2)) / (C * H * W)