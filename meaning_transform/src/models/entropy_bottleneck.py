from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class EntropyBottleneck(nn.Module):
    """Entropy bottleneck for compressing latent representations."""

    def __init__(self, latent_dim: int, compression_level: float = 1.0):
        """
        Initialize entropy bottleneck.

        Args:
            latent_dim: Dimension of latent space
            compression_level: Level of compression (higher = more compression)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.compression_level = compression_level

        # Learnable parameters for the bottleneck
        self.compress_mu = nn.Parameter(torch.zeros(latent_dim))
        self.compress_log_scale = nn.Parameter(torch.zeros(latent_dim))

        # Projection layers for adaptive compression
        self.proj_compress = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, latent_dim * 2),  # mu and log_scale
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress latent representation using entropy bottleneck.

        Args:
            z: Latent representation [B, D]

        Returns:
            z_compressed: Compressed latent representation
            compression_loss: Loss term for compression entropy
        """
        # Project to get adaptive mu and log_scale
        projection = self.proj_compress(z)
        mu, log_scale = torch.chunk(projection, 2, dim=-1)

        # Apply base parameters with adaptive adjustments
        mu = mu + self.compress_mu
        log_scale = log_scale + self.compress_log_scale

        # Scale compression based on compression_level
        log_scale = log_scale - np.log(self.compression_level)

        # Add noise for quantization
        if self.training:
            # Reparameterization trick during training
            epsilon = torch.randn_like(mu)
            z_compressed = mu + torch.exp(log_scale) * epsilon
        else:
            # Deterministic rounding during inference
            z_compressed = torch.round(mu) / self.compression_level

        # Compute entropy loss (bits per dimension)
        compression_loss = 0.5 * log_scale.exp().pow(2) + 0.5 * torch.log(
            2 * torch.tensor(np.pi, device=z.device)
        )
        compression_loss = compression_loss.mean()

        return z_compressed, compression_loss
