#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adaptive model architecture that changes its size based on compression level.

This module implements:
1. AdaptiveEntropyBottleneck - A bottleneck that adapts its dimensions to compression level
2. AdaptiveMeaningVAE - A VAE model that uses the adaptive bottleneck
3. FeatureGroupedVAE - A VAE that applies different compression to different feature groups
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from meaning_transform.src.model import Decoder, Encoder


class AdaptiveEntropyBottleneck(nn.Module):
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
        super().__init__()
        self.latent_dim = latent_dim
        self.compression_level = compression_level
        self.seed = seed

        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Calculate effective dimension based on compression level
        self.effective_dim = max(1, int(latent_dim / compression_level))

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
        # Project down to effective dimension
        z_down = self.proj_down(z)
        z_down = self.nonlin(z_down)

        # Apply adaptive compression in the effective space
        mu = z_down + self.compress_mu
        log_scale = self.compress_log_scale.clone()

        # Add noise for quantization
        if self.training:
            # Reparameterization trick during training
            if self.seed is not None:
                torch.manual_seed(self.seed)
            epsilon = torch.randn_like(mu)
            z_compressed_effective = mu + torch.exp(log_scale) * epsilon
        else:
            # Deterministic rounding during inference
            z_compressed_effective = torch.round(mu)

        # Project back up to full latent space
        projected = self.proj_up(z_compressed_effective)
        mu_full, log_scale_full = torch.chunk(projected, 2, dim=-1)

        # Use the projected values as the compressed representation
        z_compressed = mu_full

        # Compute entropy loss (bits per dimension)
        compression_loss = 0.5 * log_scale.exp().pow(2) + 0.5 * torch.log(
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


class AdaptiveMeaningVAE(nn.Module):
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

        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)

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
        """
        Forward pass through the adaptive VAE model.

        Args:
            x: Input tensor

        Returns:
            dict: Output tensors and loss components
        """
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
            "x_reconstructed": x_reconstructed,
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
            x_reconstructed: Reconstructed output
        """
        return self.decoder(z)

    def save(self, filepath: str) -> None:
        """Save model to file."""
        model_data = {
            "state_dict": self.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "compression_level": self.compression_level,
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

    def get_compression_rate(self) -> float:
        """Get the effective compression rate."""
        return self.latent_dim / self.compressor.effective_dim


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
        for name, (start_idx, end_idx, compression) in feature_groups.items():
            # Allocate latent dimensions proportionally to feature count
            feature_count = end_idx - start_idx
            group_latent_dim = max(1, int(latent_dim * feature_count / total_features))

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
            "x_reconstructed": x_reconstructed,
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
        """Decode a latent representation to reconstruction."""
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

        # Also compute overall rate
        total_latent_dim = self.latent_dim
        total_effective_dim = sum(
            bottleneck.effective_dim for bottleneck in self.bottlenecks.values()
        )
        rates["overall"] = total_latent_dim / total_effective_dim

        return rates

    def get_feature_group_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed analysis of each feature group."""
        analysis = {}

        for name, (start_idx, end_idx, compression) in self.feature_groups.items():
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
