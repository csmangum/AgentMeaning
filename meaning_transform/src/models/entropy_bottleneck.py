from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from meaning_transform.src.models.utils import CompressionBase, set_temp_seed


class EntropyBottleneck(CompressionBase):
    """Entropy bottleneck for compressing latent representations."""

    def __init__(
        self, latent_dim: int, compression_level: float = 1.0, seed: int = None
    ):
        """
        Initialize entropy bottleneck.

        Args:
            latent_dim: Dimension of latent space
            compression_level: Level of compression (higher = more compression)
            seed: Random seed for reproducibility
        """
        super().__init__(latent_dim, compression_level)
        self.seed = seed

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
        # Validate input
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(z)}")
        if z.dim() != 2 or z.size(1) != self.latent_dim:
            raise ValueError(
                f"Expected shape (batch_size, {self.latent_dim}), got {z.shape}"
            )

        # Project to get adaptive mu and log_scale
        projection = self.proj_compress(z)
        mu, log_scale = torch.chunk(projection, 2, dim=-1)

        # Apply base parameters with adaptive adjustments
        mu = mu + self.compress_mu
        log_scale = log_scale + self.compress_log_scale

        # Apply compression level to both mu and log_scale for consistency
        # Scale based on compression_level: higher level = more compression
        mu_scaled = mu / self.compression_level
        log_scale_adjusted = log_scale - torch.log(
            torch.tensor(self.compression_level, device=z.device)
        )

        # Add noise for quantization
        if self.training:
            # Reparameterization trick during training
            with set_temp_seed(self.seed):
                epsilon = torch.randn_like(mu_scaled)
            z_compressed = mu_scaled + torch.exp(log_scale_adjusted) * epsilon
        else:
            # Deterministic rounding during inference
            # Round in scaled space for consistent quantization
            z_compressed = torch.round(mu_scaled)

        # Compute entropy loss (bits per dimension) with improved numerical stability
        # Use log_scale.mul(2).exp() instead of log_scale.exp().pow(2) for stability
        compression_loss = 0.5 * log_scale_adjusted.mul(2).exp() + 0.5 * torch.log(
            2 * torch.tensor(np.pi, device=z.device)
        )
        compression_loss = compression_loss.mean()

        return z_compressed, compression_loss
