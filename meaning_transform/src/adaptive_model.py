#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adaptive model architecture that changes its size based on compression level.

This module implements:
1. AdaptiveEntropyBottleneck - A bottleneck that adapts its dimensions to compression level
2. AdaptiveMeaningVAE - A VAE model that uses the adaptive bottleneck
3. FeatureGroupedVAE - A VAE that applies different compression to different feature groups
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, List, Optional

from meaning_transform.src.model import Encoder, Decoder


class AdaptiveEntropyBottleneck(nn.Module):
    """Entropy bottleneck that adapts its architecture based on compression level."""
    
    def __init__(self, latent_dim: int, compression_level: float = 1.0):
        """
        Initialize entropy bottleneck with adaptive size.
        
        Args:
            latent_dim: Dimension of latent space
            compression_level: Level of compression (higher = more compression)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.compression_level = compression_level
        
        # Calculate effective dimension based on compression level
        # Higher compression = lower effective dimension
        self.effective_dim = max(1, int(latent_dim / compression_level))
        
        # Learnable parameters for the bottleneck - sized to effective dimension
        self.compress_mu = nn.Parameter(torch.zeros(self.effective_dim))
        self.compress_log_scale = nn.Parameter(torch.zeros(self.effective_dim))
        
        # Projection layers with adaptive dimensions
        self.proj_down = nn.Linear(latent_dim, self.effective_dim)
        self.nonlin = nn.LeakyReLU()
        self.proj_up = nn.Linear(self.effective_dim, latent_dim * 2)  # mu and log_scale
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress latent representation using adaptive bottleneck.
        
        Args:
            z: Latent representation [B, D]
            
        Returns:
            z_compressed: Compressed latent representation
            compression_loss: Loss term for compression entropy
        """
        batch_size = z.size(0)
        device = z.device
        
        # Project down to effective dimension
        z_down = self.proj_down(z)
        z_down = self.nonlin(z_down)
        
        # Project back up with compression parameters
        projection = self.proj_up(z_down)
        # Force the projection to be the right size in case of dimension mismatch
        if projection.size(1) != self.latent_dim * 2:
            # Reshape to match expected dimensionality
            projection = projection.view(batch_size, -1)
            projection = projection[:, :self.latent_dim * 2]
            # If still too small, pad
            if projection.size(1) < self.latent_dim * 2:
                padding = torch.zeros(batch_size, self.latent_dim * 2 - projection.size(1), device=device)
                projection = torch.cat([projection, padding], dim=1)
        
        mu, log_scale = torch.chunk(projection, 2, dim=-1)
        
        # Make sure we're working with the right sizes
        if mu.size(1) != self.latent_dim or log_scale.size(1) != self.latent_dim:
            mu = mu[:, :self.latent_dim]
            log_scale = log_scale[:, :self.latent_dim]
            
            # Pad if necessary
            if mu.size(1) < self.latent_dim:
                padding = torch.zeros(batch_size, self.latent_dim - mu.size(1), device=device)
                mu = torch.cat([mu, padding], dim=1)
            if log_scale.size(1) < self.latent_dim:
                padding = torch.zeros(batch_size, self.latent_dim - log_scale.size(1), device=device)
                log_scale = torch.cat([log_scale, padding], dim=1)
        
        # Apply base parameters with adaptive adjustments - make sure dimensions match
        compress_mu_expanded = self.compress_mu.unsqueeze(0).expand(batch_size, -1)
        # Pad or trim if necessary
        if compress_mu_expanded.size(1) != mu.size(1):
            if compress_mu_expanded.size(1) > mu.size(1):
                compress_mu_expanded = compress_mu_expanded[:, :mu.size(1)]
            else:
                padding = torch.zeros(batch_size, mu.size(1) - compress_mu_expanded.size(1), device=device)
                compress_mu_expanded = torch.cat([compress_mu_expanded, padding], dim=1)
                
        compress_log_scale_expanded = self.compress_log_scale.unsqueeze(0).expand(batch_size, -1)
        # Pad or trim if necessary
        if compress_log_scale_expanded.size(1) != log_scale.size(1):
            if compress_log_scale_expanded.size(1) > log_scale.size(1):
                compress_log_scale_expanded = compress_log_scale_expanded[:, :log_scale.size(1)]
            else:
                padding = torch.zeros(batch_size, log_scale.size(1) - compress_log_scale_expanded.size(1), device=device)
                compress_log_scale_expanded = torch.cat([compress_log_scale_expanded, padding], dim=1)
        
        mu = mu + compress_mu_expanded
        log_scale = log_scale + compress_log_scale_expanded
        
        # Add noise for quantization
        if self.training:
            # Reparameterization trick during training
            epsilon = torch.randn_like(mu)
            z_compressed = mu + torch.exp(log_scale) * epsilon
        else:
            # Deterministic rounding during inference
            z_compressed = torch.round(mu)
        
        # Compute entropy loss (bits per dimension)
        pi_tensor = torch.tensor(np.pi, device=device)
        compression_loss = 0.5 * torch.exp(2 * log_scale) + 0.5 * torch.log(2 * pi_tensor)
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
    """VAE model with adaptive architecture based on compression level."""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        compression_level: float = 1.0,
        hidden_dims: List[int] = None,
        use_batch_norm: bool = True
    ):
        """
        Initialize adaptive VAE model.
        
        Args:
            input_dim: Dimension of input agent state
            latent_dim: Dimension of latent space
            compression_level: Level of compression (higher = more compression)
            hidden_dims: List of hidden layer dimensions
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.compression_level = compression_level
        
        # Create encoder and decoder
        self.encoder = Encoder(input_dim, latent_dim, hidden_dims, use_batch_norm)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dims[::-1] if hidden_dims else None, use_batch_norm)
        
        # Create adaptive compressor
        self.compressor = AdaptiveEntropyBottleneck(latent_dim, compression_level)
        
        # For model tracking
        self.compression_type = "adaptive"
    
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
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass through the VAE model.
        
        Args:
            x: Input tensor of agent state
            
        Returns:
            results: Dictionary containing:
                - x_reconstructed: Reconstructed agent state
                - mu: Mean of latent distribution
                - log_var: Log variance of latent distribution
                - z: Sampled latent vector
                - z_compressed: Compressed latent vector
                - kl_loss: KL divergence loss
                - compression_loss: Loss from compression mechanism
        """
        # Encode input to latent distribution parameters
        mu, log_var = self.encoder(x)
        
        # Sample from latent distribution using reparameterization trick
        z = self.reparameterize(mu, log_var)
        
        # Apply compression
        z_compressed, compression_loss = self.compressor(z)
        
        # Decode compressed latent
        x_reconstructed = self.decoder(z_compressed)
        
        # Compute KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Return all relevant values
        results = {
            "x_reconstructed": x_reconstructed,
            "mu": mu,
            "log_var": log_var,
            "z": z,
            "z_compressed": z_compressed,
            "kl_loss": kl_loss,
            "compression_loss": compression_loss,
            "vq_loss": 0.0,
            "perplexity": 0.0,
            "effective_dim": self.compressor.effective_dim
        }
        
        return results
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input agent state to latent representation.
        
        Args:
            x: Input tensor of agent state
            
        Returns:
            z_compressed: Compressed latent representation
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        z_compressed, _ = self.compressor(z)
        return z_compressed
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to agent state.
        
        Args:
            z: Latent representation
            
        Returns:
            x_reconstructed: Reconstructed agent state
        """
        return self.decoder(z)
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath: str) -> None:
        """Load model from file."""
        self.load_state_dict(torch.load(filepath))
    
    def get_compression_rate(self) -> float:
        """
        Calculate and return the effective compression rate.
        
        Returns:
            compression_rate: Effective bits per dimension
        """
        return self.compressor.get_effective_compression_rate()
    
    def get_parameter_count(self) -> Dict[str, int]:
        """
        Calculate the parameter count for each component of the model.
        
        Returns:
            Dictionary with parameter counts by component
        """
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        compressor_params = self.compressor.get_parameter_count()
        
        return {
            "encoder": encoder_params,
            "decoder": decoder_params,
            "compressor": compressor_params,
            "total": encoder_params + decoder_params + compressor_params
        }


class FeatureGroupedVAE(nn.Module):
    """VAE model that applies different compression to different feature groups based on importance."""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        feature_groups: Dict[str, Tuple[int, int, float]] = None,
        hidden_dims: List[int] = None,
        use_batch_norm: bool = True,
        base_compression_level: float = 1.0
    ):
        """
        Initialize feature-grouped VAE model.
        
        Args:
            input_dim: Dimension of input agent state
            latent_dim: Total dimension of latent space
            feature_groups: Dictionary mapping group names to (start_idx, end_idx, compression_level)
            hidden_dims: List of hidden layer dimensions
            use_batch_norm: Whether to use batch normalization
            base_compression_level: Base compression level to scale group-specific levels
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.base_compression_level = base_compression_level
        
        # Default feature groups if none provided (evenly split)
        if feature_groups is None:
            third = input_dim // 3
            feature_groups = {
                "spatial": (0, third, 0.5),                        # Low compression
                "resources": (third, 2*third, 2.0),                # Medium compression
                "other": (2*third, input_dim, 5.0)                 # High compression
            }
        
        self.feature_groups = feature_groups
        
        # Create encoder and decoder for the full input
        self.encoder = Encoder(input_dim, latent_dim, hidden_dims, use_batch_norm)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dims[::-1] if hidden_dims else None, use_batch_norm)
        
        # Calculate feature group sizes (number of features in each group)
        group_sizes = {}
        for name, (start_idx, end_idx, _) in feature_groups.items():
            group_sizes[name] = end_idx - start_idx
        
        # Calculate importance-weighted sizes
        total_size = sum(group_sizes.values())
        group_importance = {}
        for name, (start_idx, end_idx, compression) in feature_groups.items():
            # Inverse of compression = importance
            group_importance[name] = 1.0 / max(0.01, compression)  # Avoid division by zero
        
        # Normalize importance weights
        total_importance = sum(group_importance.values())
        for name in group_importance:
            group_importance[name] /= total_importance
        
        # Allocate latent dimensions based on importance and group size
        group_latent_dims = {}
        remaining_dims = latent_dim
        
        for name, (start_idx, end_idx, _) in sorted(
            feature_groups.items(), 
            key=lambda x: group_importance[x[0]], 
            reverse=True
        ):
            # Higher importance gets more dimensions relative to size
            group_size = end_idx - start_idx
            size_proportion = group_size / total_size
            
            # Blend of raw size proportion and importance
            blended_proportion = 0.3 * size_proportion + 0.7 * group_importance[name]
            
            # Calculate dimensions for this group, ensuring at least 1 dimension
            if name == list(sorted(feature_groups.items(), key=lambda x: group_importance[x[0]], reverse=True))[-1]:
                # Last group gets all remaining dimensions
                group_latent_dims[name] = remaining_dims
            else:
                group_dim = max(1, int(blended_proportion * latent_dim))
                group_latent_dims[name] = min(group_dim, remaining_dims - 1)  # Leave at least 1 for remaining groups
                remaining_dims -= group_latent_dims[name]
        
        # Create separate bottlenecks for each feature group
        self.bottlenecks = nn.ModuleDict()
        self.group_dims = {}
        
        latent_start_idx = 0
        for name, (start_idx, end_idx, compression) in feature_groups.items():
            group_dim = group_latent_dims[name]
            
            # Scale compression by base compression level
            effective_compression = compression * self.base_compression_level
            
            # Create bottleneck
            self.bottlenecks[name] = AdaptiveEntropyBottleneck(group_dim, effective_compression)
            
            # Store group dimensions for reference
            self.group_dims[name] = {
                "latent_indices": (latent_start_idx, latent_start_idx + group_dim),
                "feature_indices": (start_idx, end_idx),
                "compression": effective_compression,
                "importance": group_importance[name]
            }
            
            latent_start_idx += group_dim
        
        # Store feature-to-group mapping for easier lookup
        self.feature_to_group = torch.zeros(input_dim, dtype=torch.long)
        self.group_names = list(feature_groups.keys())
        
        for group_idx, (name, (start_idx, end_idx, _)) in enumerate(feature_groups.items()):
            self.feature_to_group[start_idx:end_idx] = group_idx
        
        # For model tracking
        self.compression_type = "feature_grouped"
    
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
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass through the grouped VAE model.
        
        Args:
            x: Input tensor of agent state
            
        Returns:
            results: Dictionary containing:
                - x_reconstructed: Reconstructed agent state
                - mu: Mean of latent distribution
                - log_var: Log variance of latent distribution
                - z: Sampled latent vector
                - z_compressed: Compressed latent vector
                - kl_loss: KL divergence loss
                - compression_loss: Loss from compression mechanism
                - feature_group_info: Information about feature groups
        """
        # Encode input to latent distribution parameters
        mu, log_var = self.encoder(x)
        
        # Sample from latent distribution using reparameterization trick
        z = self.reparameterize(mu, log_var)
        
        # Apply compression for each group
        z_compressed = torch.zeros_like(z)
        compression_loss = 0.0
        group_info = {}
        
        for name, info in self.group_dims.items():
            latent_start, latent_end = info["latent_indices"]
            z_group = z[:, latent_start:latent_end]
            
            # Apply bottleneck compression to this group
            z_group_compressed, group_loss = self.bottlenecks[name](z_group)
            
            # Store compressed representation
            z_compressed[:, latent_start:latent_end] = z_group_compressed
            
            # Add weighted compression loss (proportional to importance)
            weighted_loss = group_loss * info["importance"]
            compression_loss += weighted_loss
            
            # Store information about this group for analysis
            group_info[name] = {
                "latent_dim": latent_end - latent_start,
                "feature_dim": info["feature_indices"][1] - info["feature_indices"][0],
                "compression": info["compression"],
                "importance": info["importance"],
                "effective_dim": self.bottlenecks[name].effective_dim,
                "compression_loss": group_loss.item()
            }
        
        # Decode compressed latent
        x_reconstructed = self.decoder(z_compressed)
        
        # Compute KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Return all relevant values
        results = {
            "x_reconstructed": x_reconstructed,
            "mu": mu,
            "log_var": log_var,
            "z": z,
            "z_compressed": z_compressed,
            "kl_loss": kl_loss,
            "compression_loss": compression_loss,
            "vq_loss": 0.0,
            "perplexity": 0.0,
            "feature_group_info": group_info,
            "base_compression_level": self.base_compression_level
        }
        
        return results
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input agent state to latent representation.
        
        Args:
            x: Input tensor of agent state
            
        Returns:
            z_compressed: Compressed latent representation
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        
        # Apply compression for each group
        z_compressed = torch.zeros_like(z)
        
        for name, info in self.group_dims.items():
            latent_start, latent_end = info["latent_indices"]
            z_group = z[:, latent_start:latent_end]
            z_group_compressed, _ = self.bottlenecks[name](z_group)
            z_compressed[:, latent_start:latent_end] = z_group_compressed
        
        return z_compressed
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to agent state.
        
        Args:
            z: Latent representation
            
        Returns:
            x_reconstructed: Reconstructed agent state
        """
        return self.decoder(z)
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath: str) -> None:
        """Load model from file."""
        self.load_state_dict(torch.load(filepath))
    
    def get_feature_group_analysis(self) -> Dict[str, Dict[str, float]]:
        """
        Get detailed analysis of compression applied to each feature group.
        
        Returns:
            Dictionary containing metrics for each feature group
        """
        analysis = {}
        
        for name, info in self.group_dims.items():
            bottleneck = self.bottlenecks[name]
            feature_start, feature_end = info["feature_indices"]
            latent_start, latent_end = info["latent_indices"]
            
            # Calculate effective compression metrics
            feature_count = feature_end - feature_start
            latent_count = latent_end - latent_start
            effective_dim = bottleneck.effective_dim
            
            # Compression statistics
            analysis[name] = {
                "feature_count": feature_count,
                "latent_dim": latent_count,
                "effective_dim": effective_dim,
                "input_to_latent_ratio": feature_count / max(1, latent_count),
                "latent_to_effective_ratio": latent_count / max(1, effective_dim),
                "overall_compression": feature_count / max(1, effective_dim),
                "importance": info["importance"],
                "target_compression": info["compression"]
            }
        
        return analysis
    
    def get_compression_rate(self) -> Dict[str, float]:
        """
        Calculate and return the effective compression rate for each group.
        
        Returns:
            compression_rates: Dictionary of effective bits per dimension by group
        """
        rates = {}
        try:
            total_input_dim = 0
            total_effective_dim = 0
            
            for name, info in self.group_dims.items():
                feature_start, feature_end = info["feature_indices"]
                feature_count = feature_end - feature_start
                total_input_dim += feature_count
                
                try:
                    if hasattr(self.bottlenecks[name], 'get_effective_compression_rate'):
                        # Get rate from bottleneck
                        bottleneck_rate = self.bottlenecks[name].get_effective_compression_rate()
                        effective_dim = self.bottlenecks[name].effective_dim
                    else:
                        # Fallback calculation
                        latent_start, latent_end = info["latent_indices"]
                        latent_dim = latent_end - latent_start
                        effective_dim = getattr(self.bottlenecks[name], 'effective_dim', 1)
                        bottleneck_rate = float(latent_dim) / float(max(1, effective_dim))
                    
                    # Store rate and add to total effective dimension
                    rates[name] = bottleneck_rate
                    total_effective_dim += effective_dim
                    
                except Exception as e:
                    print(f"Error getting compression rate for {name}: {e}")
                    rates[name] = 1.0
            
            # Calculate overall rate as input dimension divided by total effective dimension
            if total_effective_dim > 0:
                rates["overall"] = float(self.input_dim) / float(total_effective_dim)
            else:
                rates["overall"] = 1.0
            
            return rates
        except Exception as e:
            print(f"Error calculating compression rates: {e}")
            return {"overall": 1.0}
    
    def get_parameter_count(self) -> Dict[str, int]:
        """
        Calculate the parameter count for each component of the model.
        
        Returns:
            Dictionary with parameter counts by component
        """
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        # Count bottleneck parameters by group
        bottleneck_params = {}
        for name, bottleneck in self.bottlenecks.items():
            bottleneck_params[name] = sum(p.numel() for p in bottleneck.parameters())
        
        total_bottleneck_params = sum(bottleneck_params.values())
        
        return {
            "encoder": encoder_params,
            "decoder": decoder_params,
            "compressor": total_bottleneck_params,
            "bottlenecks": bottleneck_params,
            "total": encoder_params + decoder_params + total_bottleneck_params
        } 