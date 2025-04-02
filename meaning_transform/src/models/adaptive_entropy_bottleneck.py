import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from meaning_transform.src.models.utils import CompressionBase, set_temp_seed


class AdaptiveEntropyBottleneck(CompressionBase):
    """
    Adaptive entropy bottleneck that actually changes its structure based on compression level.
    The parameter count scales inversely with compression level.
    """

    def __init__(
        self,
        latent_dim: int,
        compression_level: float = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize adaptive entropy bottleneck.

        Args:
            latent_dim: Dimension of latent space
            compression_level: Level of compression (higher = more compression)
            seed: Random seed for reproducibility
        """
        super().__init__(latent_dim, compression_level)
        self.seed = seed

        # Initialize parameters using temp seed (if provided)
        with set_temp_seed(seed):
            # Learnable parameters sized to effective dimension
            self.compress_mu = nn.Parameter(torch.zeros(self.effective_dim))
            self.compress_log_scale = nn.Parameter(torch.zeros(self.effective_dim))

            # Projection layers with adaptive dimensions
            self.proj_down = nn.Linear(latent_dim, self.effective_dim)
            self.nonlin = nn.LeakyReLU()
            self.proj_up = nn.Linear(
                self.effective_dim, latent_dim * 2
            )  # mu and log_scale for reconstruction

        logging.info(
            f"Created AdaptiveEntropyBottleneck with latent_dim={latent_dim}, "
            f"compression_level={compression_level}, effective_dim={self.effective_dim}"
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress latent representation using adaptive entropy bottleneck.

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

        # Project down to effective dimension
        z_down = self.proj_down(z)
        z_down = self.nonlin(z_down)

        # Apply adaptive compression in the effective space
        mu = z_down + self.compress_mu
        log_scale = self.compress_log_scale.clone()

        # Add noise for quantization in the effective space
        if self.training:
            # Reparameterization trick during training
            with set_temp_seed(self.seed):
                epsilon = torch.randn_like(mu)
            z_compressed_effective = mu + torch.exp(log_scale) * epsilon
        else:
            # Deterministic rounding during inference
            z_compressed_effective = torch.round(mu)

        # Project back up to full latent space
        projected = self.proj_up(z_compressed_effective)
        mu_full, log_scale_full = torch.chunk(projected, 2, dim=-1)

        # Apply quantization in full latent space
        if self.training:
            # During training, use the output from projection with noise
            with set_temp_seed(self.seed):
                epsilon = torch.randn_like(mu_full)
            z_compressed = mu_full + torch.exp(log_scale_full) * epsilon
        else:
            # During inference, apply deterministic rounding for proper quantization
            z_compressed = torch.round(mu_full)

        # Compute entropy loss (bits per dimension) with improved numerical stability
        compression_loss = 0.5 * log_scale.mul(2).exp() + 0.5 * torch.log(
            2 * torch.tensor(np.pi, device=z.device)
        )
        compression_loss = compression_loss.mean()

        return z_compressed, compression_loss

    def get_parameter_count(self) -> int:
        """
        Calculate the total number of parameters in the bottleneck.

        Returns:
            Number of parameters
        """
        total_params = 0
        for param in self.parameters():
            total_params += param.numel()
        return total_params

    def get_effective_compression_rate(self) -> float:
        """
        Calculate the effective compression rate in bits per dimension.

        Returns:
            Effective bits per dimension
        """
        # Avoid division by zero or recursion
        if self.effective_dim <= 0:
            return 0.0

        # Calculate bits per dimension
        return float(self.latent_dim) / float(self.effective_dim)
