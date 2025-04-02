import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from meaning_transform.src.models.adaptive_entropy_bottleneck import (
    AdaptiveEntropyBottleneck,
)
from meaning_transform.src.models.decoder import Decoder
from meaning_transform.src.models.encoder import Encoder


class FeatureGroupedVAE(nn.Module):
    """
    VAE model that applies different compression rates to different feature groups based on importance.
    This allows for better semantic preservation of high-importance features.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        feature_groups: Optional[Dict[str, Tuple[int, int, float]]] = None,
        base_compression_level: float = 1.0,
        use_batch_norm: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize feature-grouped VAE model.

        Args:
            input_dim: Dimension of input agent state
            latent_dim: Dimension of latent space
            feature_groups: Dictionary mapping group names to (start_idx, end_idx, compression_level)
                            If None, equal groups with equal compression will be used
            base_compression_level: Base compression level to apply to all groups
            use_batch_norm: Whether to use batch normalization
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.base_compression_level = base_compression_level
        self.seed = seed
        self.use_batch_norm = use_batch_norm

        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Default feature groups if not provided
        if feature_groups is None:
            # Equal splits with equal compression
            group_size = input_dim // 3
            feature_groups = {
                "group1": (0, group_size, 1.0),
                "group2": (group_size, 2 * group_size, 1.0),
                "group3": (2 * group_size, input_dim, 1.0),
            }

        self.feature_groups = feature_groups

        # Create encoder and decoder
        self.encoder = Encoder(input_dim, latent_dim, use_batch_norm=use_batch_norm)
        self.decoder = Decoder(latent_dim, input_dim, use_batch_norm=use_batch_norm)

        # Create separate bottlenecks for each feature group
        self.bottlenecks = nn.ModuleDict()
        self.group_latent_dims = {}

        # Determine latent dimension allocation per group
        total_features = sum(end - start for start, end, _ in feature_groups.values())

        latent_start_idx = 0
        group_names = list(feature_groups.keys())

        # First pass: calculate intended dimensions
        intended_dims = {}
        for name, (start_idx, end_idx, compression) in feature_groups.items():
            feature_count = end_idx - start_idx
            intended_dims[name] = max(
                1, int(latent_dim * feature_count / total_features)
            )

        # Adjust to ensure total matches latent_dim exactly
        total_allocated = sum(intended_dims.values())
        remaining = latent_dim - total_allocated

        # Distribute remaining dimensions to groups proportionally
        if remaining != 0:
            sorted_groups = sorted(
                intended_dims.items(),
                key=lambda x: feature_groups[x[0]][
                    2
                ],  # Sort by compression value (lower first)
            )

            for i in range(abs(remaining)):
                group_name = sorted_groups[i % len(sorted_groups)][0]
                intended_dims[group_name] += 1 if remaining > 0 else -1

        # Second pass: create bottlenecks with corrected dimensions
        for i, name in enumerate(group_names):
            start_idx, end_idx, compression = feature_groups[name]
            group_latent_dim = intended_dims[name]

            # Apply group-specific compression
            effective_compression = compression * base_compression_level

            # Store latent dimension range for this group
            self.group_latent_dims[name] = (
                latent_start_idx,
                latent_start_idx + group_latent_dim,
            )

            # Create bottleneck for this group
            self.bottlenecks[name] = AdaptiveEntropyBottleneck(
                latent_dim=group_latent_dim,
                compression_level=effective_compression,
                seed=seed,
            )

            latent_start_idx += group_latent_dim

            logging.info(
                f"Feature group '{name}': features [{start_idx}:{end_idx}], "
                f"latent dim {group_latent_dim}, compression {effective_compression}x"
            )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from latent distribution."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            if self.seed is not None:
                # Use deterministic noise when seed is set
                torch.manual_seed(self.seed)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z
        else:
            # During evaluation, just use the mean for deterministic results
            return mu

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Forward pass through the feature-grouped VAE model."""
        # Encode input to latent space
        mu, log_var = self.encoder(x)

        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)

        # Apply feature-specific compression
        z_compressed = torch.zeros_like(z)
        compression_loss = 0.0

        for name, (start_idx, end_idx) in self.group_latent_dims.items():
            # Get latent vector segment for this group
            z_group = z[:, start_idx:end_idx]

            # Apply group-specific bottleneck
            z_group_compressed, group_loss = self.bottlenecks[name](z_group)

            # Store compressed representation
            z_compressed[:, start_idx:end_idx] = z_group_compressed

            # Accumulate compression loss
            compression_loss += group_loss

        # Average compression loss across groups
        compression_loss = compression_loss / len(self.bottlenecks)

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
            "feature_group_dims": self.group_latent_dims,
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an input tensor to compressed latent space."""
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)

        # Apply feature-specific compression
        z_compressed = torch.zeros_like(z)

        for name, (start_idx, end_idx) in self.group_latent_dims.items():
            # Get latent vector segment for this group
            z_group = z[:, start_idx:end_idx]

            # Apply group-specific bottleneck
            z_group_compressed, _ = self.bottlenecks[name](z_group)

            # Store compressed representation
            z_compressed[:, start_idx:end_idx] = z_group_compressed

        return z_compressed

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent representation to reconstruction.

        Args:
            z: Latent representation

        Returns:
            reconstruction: Reconstructed output
        """
        return self.decoder(z)

    def save(self, filepath: str) -> None:
        """Save model to file."""
        model_data = {
            "state_dict": self.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "feature_groups": self.feature_groups,
                "base_compression_level": self.base_compression_level,
                "seed": self.seed,
            },
        }
        torch.save(model_data, filepath)

    def load(self, filepath: str) -> None:
        """Load model from file."""
        model_data = torch.load(filepath)

        # Load config if available
        if "config" in model_data and "seed" in model_data["config"]:
            self.seed = model_data["config"]["seed"]

        # Load state dict
        if "state_dict" in model_data:
            self.load_state_dict(model_data["state_dict"])
        else:
            # Handle old model format
            self.load_state_dict(model_data)

    def get_compression_rate(self) -> Dict[str, float]:
        """Get the effective compression rate for each feature group."""
        rates = {}

        for name, bottleneck in self.bottlenecks.items():
            start_idx, end_idx = self.group_latent_dims[name]
            group_latent_dim = end_idx - start_idx
            rates[name] = group_latent_dim / bottleneck.effective_dim

        # Also compute overall rate - weighted average based on feature counts
        total_input_features = sum(
            end - start for start, end, _ in self.feature_groups.values()
        )
        weighted_compression = 0.0

        for name, (feature_start, feature_end, _) in self.feature_groups.items():
            feature_count = feature_end - feature_start
            weight = feature_count / total_input_features
            group_rate = rates[name]
            weighted_compression += weight * group_rate

        rates["overall"] = weighted_compression

        return rates

    def get_feature_group_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed analysis of each feature group."""
        analysis = {}

        for name in self.feature_groups:
            start_idx, end_idx, compression = self.feature_groups[name]
            latent_start, latent_end = self.group_latent_dims[name]
            bottleneck = self.bottlenecks[name]

            analysis[name] = {
                "feature_range": (start_idx, end_idx),
                "feature_count": end_idx - start_idx,
                "latent_range": (latent_start, latent_end),
                "latent_dim": latent_end - latent_start,
                "effective_dim": bottleneck.effective_dim,
                "compression": compression,
                "base_compression": self.base_compression_level,
                "overall_compression": compression * self.base_compression_level,
                "importance": 1.0
                / compression,  # Inverse relationship with compression
            }

        return analysis
