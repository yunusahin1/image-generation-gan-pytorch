import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminator Network for GAN
    Classifies images as real or fake
    """
    def __init__(self, img_channels=3, feature_maps=64):
        super(Discriminator, self).__init__()
        
        # Build the discriminator network using convolutions
        self.main = nn.Sequential(
            # Input: img_channels x 64 x 64
            nn.Conv2d(img_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: feature_maps x 32 x 32
            
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*2) x 16 x 16
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*4) x 8 x 8
            
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (feature_maps*8) x 4 x 4
            
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1 (probability that input is real)
        )
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input images of shape (batch_size, img_channels, 64, 64)
        Returns:
            Probability that input is real, shape (batch_size, 1, 1, 1)
        """
        return self.main(x)