import torch.nn as nn
import torch
from src.vision.resnet import ResBlock

class VQVAE(nn.Module):
    def __init__(
        self,
        n_channels: int,
        latent_dim: int,
        codebook_size: int,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(latent_dim),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(latent_dim),
            ResBlock(latent_dim, latent_dim, nn.ReLU),
            ResBlock(latent_dim, latent_dim, nn.ReLU),
            nn.GroupNorm(8, latent_dim),
            nn.Tanh(),
        )

        self.codebook = nn.Parameter(
            torch.randn(codebook_size, latent_dim) * 0.02, 
            requires_grad=True
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
            ResBlock(latent_dim, latent_dim, nn.ReLU),
            ResBlock(latent_dim, latent_dim, nn.ReLU),
            nn.ConvTranspose2d(latent_dim, latent_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(latent_dim),
            nn.ConvTranspose2d(latent_dim, latent_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(latent_dim),
            nn.ConvTranspose2d(latent_dim, n_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the VAE.

        Args:
            x: input data

        Returns:
            x_hat: reconstructed data
        """
        z_encoder = self.encoder(x)
        z_quantized = self._quantize_encoder_output(z_encoder)
        z_q_st = z_encoder + (z_quantized - z_encoder).detach()
        return self.decoder(z_q_st), z_encoder, z_quantized
    
    def _quantize_encoder_output(self, z_e: torch.Tensor) -> torch.Tensor:
        """
        Quantizing the encoded tensor by snapping its elements to the closest codebook 
        entry.

        Args:
            z_e: encoded representation

        Returns:
            z_q: quantized representation
        """
        batch_size, latent_dim, h, w = z_e.shape
        encoded = z_e.permute(0, 2, 3, 1).reshape(batch_size*h*w, latent_dim)
        quantized = self.codebook[torch.argmin(torch.cdist(encoded, self.codebook), dim=1)]
        z_q = quantized.reshape(batch_size, h, w, latent_dim).permute(0, 3, 1, 2)
        return z_q
