from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from meaning_transform.src.models.adaptive_entropy_bottleneck import (
    AdaptiveEntropyBottleneck,
)
from meaning_transform.src.models.decoder import Decoder
from meaning_transform.src.models.encoder import Encoder
from meaning_transform.src.models.utils import BaseModelIO, set_temp_seed


class AdaptiveMeaningVAE(nn.Module, BaseModelIO):
    """
    Adaptive VAE model that changes its bottleneck structure based on compression level.
    The parameter count scales with compression level, making it more efficient at high compression.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        compression_level: float = 1.0,
        use_batch_norm: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize adaptive VAE model.

        Args:
            input_dim: Dimension of input agent state
            latent_dim: Dimension of latent space
            compression_level: Level of compression (higher = more compression)
            use_batch_norm: Whether to use batch normalization
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.compression_level = compression_level
        self.seed = seed
        self.use_batch_norm = use_batch_norm

        # Create encoder and decoder
        self.encoder = Encoder(input_dim, latent_dim, use_batch_norm=use_batch_norm)
        self.decoder = Decoder(latent_dim, input_dim, use_batch_norm=use_batch_norm)

        # Create adaptive compressor
        self.compressor = AdaptiveEntropyBottleneck(
            latent_dim, compression_level, seed=seed
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from latent distribution.

        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution

        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)

        if self.training:
            with set_temp_seed(self.seed):
                eps = torch.randn_like(std)
            z = mu + eps * std
            return z
        else:
            # During evaluation, just use the mean for deterministic results
            return mu

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass through the adaptive VAE model.

        Args:
            x: Input tensor

        Returns:
            dict: Output tensors and loss components
        """
        # Validate input
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"Expected shape (batch_size, {self.input_dim}), got {x.shape}"
            )

        # Encode input to latent space
        mu, log_var = self.encoder(x)

        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)

        # Apply adaptive compression
        z_compressed, compression_loss = self.compressor(z)

        # Decode compressed representation
        x_reconstructed = self.decoder(z_compressed)

        # Compute KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / x.size(0)  # Normalize by batch size

        # Return all tensors and loss components
        return {
            "reconstruction": x_reconstructed,
            "mu": mu,
            "log_var": log_var,
            "z": z,
            "z_compressed": z_compressed,
            "kl_loss": kl_loss,
            "compression_loss": compression_loss,
            "effective_dim": self.compressor.effective_dim,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an input tensor to compressed latent space.

        Args:
            x: Input tensor

        Returns:
            z_compressed: Compressed latent representation
        """
        # Validate input
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"Expected shape (batch_size, {self.input_dim}), got {x.shape}"
            )

        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        z_compressed, _ = self.compressor(z)
        return z_compressed

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent representation to reconstruction.

        Args:
            z: Latent representation

        Returns:
            reconstruction: Reconstructed output
        """
        # Validate input
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(z)}")
        if z.dim() != 2 or z.size(1) != self.latent_dim:
            raise ValueError(
                f"Expected shape (batch_size, {self.latent_dim}), got {z.shape}"
            )

        return self.decoder(z)

    def get_compression_rate(self) -> float:
        """Get the effective compression rate."""
        return self.compressor.get_compression_rate()
