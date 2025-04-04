#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Loss functions for the meaning-preserving transformation system.

This module defines:
1. Reconstruction loss (MSE, BCE)
2. Latent loss (KL divergence)
3. Semantic loss (feature-based similarity)
4. Combined loss function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
import math


def beta_annealing(epoch: int, max_epochs: int, min_beta: float = 0.0, 
                  max_beta: float = 1.0, schedule_type: str = "linear") -> float:
    """
    Calculate annealed beta value for KL divergence weight.
    
    This implements different annealing schedules for the KL weight (beta)
    to stabilize training and prevent posterior collapse.
    
    Args:
        epoch: Current training epoch
        max_epochs: Maximum number of epochs
        min_beta: Minimum beta value (starting value)
        max_beta: Maximum beta value (ending value)
        schedule_type: Type of annealing schedule. Options:
                      "linear" - linear increase from min_beta to max_beta
                      "sigmoid" - sigmoid-shaped curve for smooth transition
                      "cyclical" - cyclical annealing (repeating pattern)
                      "exponential" - exponential increase
    
    Returns:
        beta: Annealed beta value for current epoch
    """
    # Normalize epoch to [0, 1]
    t = epoch / max_epochs
    
    if schedule_type == "linear":
        # Linear annealing
        beta = min_beta + (max_beta - min_beta) * t
    
    elif schedule_type == "sigmoid":
        # Sigmoid annealing for smooth transition
        # Adjusted to ensure we reach close to min_beta and max_beta at endpoints
        steepness = 10  # Controls sigmoid steepness
        shift = 0.5     # Center of sigmoid
        sigmoid_val = 1.0 / (1.0 + math.exp(-steepness * (t - shift)))
        beta = min_beta + (max_beta - min_beta) * sigmoid_val
    
    elif schedule_type == "cyclical":
        # Cyclical annealing (as in "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing")
        # Number of cycles
        n_cycles = 4
        # Calculate position within cycle
        cycle_ratio = n_cycles * t
        cycle_t = cycle_ratio - math.floor(cycle_ratio)
        
        # Annealing within each cycle
        if cycle_t < 0.5:
            # Linear annealing for first half of cycle
            beta_cycle = min_beta + (max_beta - min_beta) * (cycle_t * 2)
        else:
            # Hold at max_beta for second half of cycle
            beta_cycle = max_beta
            
        beta = beta_cycle
    
    elif schedule_type == "exponential":
        # Exponential annealing
        exponent = 3  # Controls curve steepness
        beta = min_beta + (max_beta - min_beta) * (t ** exponent)
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    return beta


class ReconstructionLoss(nn.Module):
    """Loss for measuring reconstruction quality."""
    
    def __init__(self, loss_type: str = "mse"):
        """
        Initialize reconstruction loss.
        
        Args:
            loss_type: Type of reconstruction loss ("mse" or "bce")
        """
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction="sum")
        elif loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, 
                x_reconstructed: torch.Tensor, 
                x_original: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            x_reconstructed: Reconstructed tensor
            x_original: Original tensor
            
        Returns:
            loss: Reconstruction loss
        """
        return self.loss_fn(x_reconstructed, x_original)


class KLDivergenceLoss(nn.Module):
    """Kullback-Leibler divergence loss for latent space regularization."""
    
    def __init__(self):
        """Initialize KL divergence loss."""
        super().__init__()
    
    def forward(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss.
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            kl_loss: KL divergence loss
        """
        # KL divergence between N(mu, var) and N(0, 1)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return kl_loss


class SemanticLoss(nn.Module):
    """Loss for measuring semantic preservation between original and reconstructed states."""
    
    def __init__(self, feature_extractors: List[str] = None, similarity_type: str = "cosine"):
        """
        Initialize semantic loss.
        
        Args:
            feature_extractors: List of feature extractors to use
            similarity_type: Type of similarity metric ("cosine", "euclidean", or "manhattan")
        """
        super().__init__()
        self.feature_extractors = feature_extractors or ["position", "health", "has_target", 
                                                         "energy", "is_alive", "role", "threatened"]
        self.similarity_type = similarity_type
    
    def extract_semantic_features(self, state_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extracts higher-level semantic info from serialized state tensor.
        Assumes a specific input vector structure.
        
        Args:
            state_tensor: The serialized agent state tensor
            
        Returns:
            features: Dictionary of extracted semantic features
        """
        # Extract core features based on expected tensor structure
        # This should match the serialization format in data.py
        x = state_tensor[:, 0]
        y = state_tensor[:, 1]
        health = state_tensor[:, 2] * 100.0  # denormalized
        is_alive = (health > 10).float()
        has_target = state_tensor[:, 3]
        energy = state_tensor[:, 4] * 100.0  # denormalized
        role_idx = torch.argmax(state_tensor[:, 5:10], dim=1)  # Assuming roles are one-hot encoded in dims 5-9
        
        # Derived semantic condition: "threatened"
        # Agent is alive, has a target, and low health
        threatened = ((has_target == 1.0) & (health < 30)).float()
        
        # Create dictionary of semantic features
        features = {
            "position": torch.stack([x, y], dim=1),
            "health": health.unsqueeze(1) / 100.0,  # normalize back
            "has_target": has_target.unsqueeze(1),
            "energy": energy.unsqueeze(1) / 100.0,  # normalize back
            "is_alive": is_alive.unsqueeze(1),
            "role": role_idx.float().unsqueeze(1) / 5.0,  # normalized for loss
            "threatened": threatened.unsqueeze(1)
        }
        
        return features
    
    def compute_similarity(self, 
                          x1: torch.Tensor, 
                          x2: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between two feature vectors.
        
        Args:
            x1: First feature vector
            x2: Second feature vector
            
        Returns:
            similarity: Similarity score
        """
        if self.similarity_type == "cosine":
            # Cosine similarity (higher is better, so we negate for loss)
            similarity = 1.0 - F.cosine_similarity(x1, x2, dim=1).mean()
        elif self.similarity_type == "euclidean":
            # Euclidean distance (lower is better)
            similarity = F.pairwise_distance(x1, x2, p=2).mean()
        elif self.similarity_type == "manhattan":
            # Manhattan distance (lower is better)
            similarity = F.pairwise_distance(x1, x2, p=1).mean()
        else:
            raise ValueError(f"Unsupported similarity type: {self.similarity_type}")
        
        return similarity
    
    def forward(self, 
                x_reconstructed: torch.Tensor, 
                x_original: torch.Tensor) -> torch.Tensor:
        """
        Compute semantic loss.
        
        Args:
            x_reconstructed: Reconstructed tensor
            x_original: Original tensor
            
        Returns:
            loss: Semantic loss
        """
        # Extract semantic features
        original_features = self.extract_semantic_features(x_original)
        reconstructed_features = self.extract_semantic_features(x_reconstructed)
        
        # Compute similarity for each feature type
        losses = {}
        total_loss = 0.0
        
        for feature_name in self.feature_extractors:
            if feature_name not in original_features:
                continue
                
            # Select appropriate loss function based on feature type
            if feature_name in ["has_target", "is_alive", "threatened"]:
                # Binary features use BCE - clamp values to [0, 1] for safety
                reconstructed_value = torch.clamp(reconstructed_features[feature_name], 0.0, 1.0)
                original_value = torch.clamp(original_features[feature_name], 0.0, 1.0)
                feature_loss = F.binary_cross_entropy(
                    reconstructed_value, 
                    original_value
                )
            elif feature_name in ["position", "health", "energy", "role"]:
                # Continuous features use MSE
                feature_loss = F.mse_loss(
                    original_features[feature_name], 
                    reconstructed_features[feature_name]
                )
            else:
                # Default to similarity metric for other features
                feature_loss = self.compute_similarity(
                    original_features[feature_name], 
                    reconstructed_features[feature_name]
                )
                
            losses[feature_name] = feature_loss
            total_loss += feature_loss
        
        # Average across all feature types
        return total_loss / len(losses)
    
    def detailed_breakdown(self, 
                          x_reconstructed: torch.Tensor, 
                          x_original: torch.Tensor) -> Dict[str, float]:
        """
        Compute detailed breakdown of semantic loss components.
        
        Args:
            x_reconstructed: Reconstructed tensor
            x_original: Original tensor
            
        Returns:
            losses: Dictionary of loss components
        """
        # Extract semantic features
        original_features = self.extract_semantic_features(x_original)
        reconstructed_features = self.extract_semantic_features(x_reconstructed)
        
        # Compute detailed loss breakdown
        losses = {}
        
        for feature_name in self.feature_extractors:
            if feature_name not in original_features:
                continue
                
            # Select appropriate loss function based on feature type
            if feature_name in ["has_target", "is_alive", "threatened"]:
                # Binary features use BCE - clamp values to [0, 1] for safety
                reconstructed_value = torch.clamp(reconstructed_features[feature_name], 0.0, 1.0)
                original_value = torch.clamp(original_features[feature_name], 0.0, 1.0)
                feature_loss = F.binary_cross_entropy(
                    reconstructed_value,
                    original_value
                ).item()
            elif feature_name in ["position", "health", "energy", "role"]:
                # Continuous features use MSE
                feature_loss = F.mse_loss(
                    original_features[feature_name], 
                    reconstructed_features[feature_name]
                ).item()
            else:
                # Default to similarity metric for other features
                feature_loss = self.compute_similarity(
                    original_features[feature_name], 
                    reconstructed_features[feature_name]
                ).item()
                
            losses[feature_name] = feature_loss
        
        return losses


class CombinedLoss(nn.Module):
    """Combined loss function for the VAE model."""
    
    def __init__(self,
                 recon_loss_weight: float = 1.0,
                 kl_loss_weight: float = 0.1,
                 semantic_loss_weight: float = 0.5,
                 recon_loss_type: str = "mse",
                 semantic_feature_extractors: List[str] = None,
                 semantic_similarity_type: str = "cosine"):
        """
        Initialize combined loss.
        
        Args:
            recon_loss_weight: Weight for reconstruction loss
            kl_loss_weight: Weight for KL divergence loss
            semantic_loss_weight: Weight for semantic loss
            recon_loss_type: Type of reconstruction loss
            semantic_feature_extractors: List of semantic feature extractors
            semantic_similarity_type: Type of semantic similarity metric
        """
        super().__init__()
        
        self.recon_loss_weight = recon_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.semantic_loss_weight = semantic_loss_weight
        
        # Initialize component losses
        self.recon_loss = ReconstructionLoss(recon_loss_type)
        self.kl_loss = KLDivergenceLoss()
        self.semantic_loss = SemanticLoss(semantic_feature_extractors, semantic_similarity_type)
    
    def forward(self, 
                model_output: Dict[str, Any], 
                x_original: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            model_output: Output from VAE model
            x_original: Original input tensor
            
        Returns:
            loss_dict: Dictionary of loss components and total loss
        """
        # Extract values from model output
        x_reconstructed = model_output["reconstruction"]
        mu = model_output["mu"]
        log_var = model_output["log_var"]
        compression_loss = model_output.get("compression_loss", torch.tensor(0.0))  # Ensure tensor type
        
        # Compute individual loss components
        recon_loss = self.recon_loss(x_reconstructed, x_original)
        kld_loss = self.kl_loss(mu, log_var)
        sem_loss = self.semantic_loss(x_reconstructed, x_original)
        
        # Make sure all losses are detached for logging purposes
        recon_loss_detached = recon_loss.detach().clone()
        kld_loss_detached = kld_loss.detach().clone()
        sem_loss_detached = sem_loss.detach().clone()
        comp_loss_detached = compression_loss.detach().clone() if isinstance(compression_loss, torch.Tensor) else torch.tensor(compression_loss)
        
        # Compute weighted total loss
        total_loss = (
            self.recon_loss_weight * recon_loss +
            self.kl_loss_weight * kld_loss +
            self.semantic_loss_weight * sem_loss +
            compression_loss  # Compression loss is applied directly
        )
        
        # Optional: Calculate semantic breakdown for logging/analysis
        if self.training and torch.rand(1).item() < 0.1:  # Only sometimes to save computation
            semantic_breakdown = self.semantic_loss.detailed_breakdown(x_reconstructed, x_original)
        else:
            semantic_breakdown = {}
        
        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss_detached,
            "kl_loss": kld_loss_detached,
            "semantic_loss": sem_loss_detached,
            "compression_loss": comp_loss_detached,
            "semantic_breakdown": semantic_breakdown
        }


class FeatureWeightedLoss(CombinedLoss):
    """
    Feature-weighted loss function that prioritizes critical semantic properties.
    
    This loss extends the CombinedLoss by:
    1. Applying feature-specific weights based on importance scores
    2. Supporting dynamic/progressive weight adjustment during training
    3. Providing more granular control over semantic preservation
    """
    
    # Canonical feature importance weights based on analysis
    CANONICAL_WEIGHTS = {
        "position": 0.554,      # Spatial features - highest importance
        "health": 0.150,        # Resource features
        "energy": 0.101,        # Resource features
        "is_alive": 0.050,      # Performance features
        "has_target": 0.035,    # Performance features
        "threatened": 0.020,    # Performance features
        "role": 0.050           # Role features - lowest importance
    }
    
    # Feature groups for convenient adjustment
    FEATURE_GROUPS = {
        "spatial": ["position"],
        "resources": ["health", "energy"],
        "performance": ["is_alive", "has_target", "threatened"],
        "role": ["role"]
    }
    
    def __init__(self,
                 recon_loss_weight: float = 1.0,
                 kl_loss_weight: float = 0.1,
                 semantic_loss_weight: float = 0.5,
                 recon_loss_type: str = "mse",
                 semantic_feature_extractors: List[str] = None,
                 semantic_similarity_type: str = "cosine",
                 feature_weights: Dict[str, float] = None,
                 use_canonical_weights: bool = True,
                 progressive_weight_schedule: str = None,
                 progressive_weight_epochs: int = 50,
                 feature_stability_adjustment: bool = False):
        """
        Initialize feature-weighted loss.
        
        Args:
            recon_loss_weight: Weight for reconstruction loss
            kl_loss_weight: Weight for KL divergence loss
            semantic_loss_weight: Weight for semantic loss
            recon_loss_type: Type of reconstruction loss
            semantic_feature_extractors: List of semantic feature extractors
            semantic_similarity_type: Type of semantic similarity metric
            feature_weights: Dictionary of feature weights (overrides canonical weights if provided)
            use_canonical_weights: Whether to use canonical weights from feature importance analysis
            progressive_weight_schedule: Type of progressive weight schedule ("linear", "exp", or None)
            progressive_weight_epochs: Number of epochs for progressive weight adjustment
            feature_stability_adjustment: Whether to adjust weights based on feature stability
        """
        super().__init__(
            recon_loss_weight, 
            kl_loss_weight,
            semantic_loss_weight,
            recon_loss_type,
            semantic_feature_extractors,
            semantic_similarity_type
        )
        
        # Initialize feature weights
        self.feature_weights = {}
        
        if feature_weights is not None:
            # Use provided weights
            self.feature_weights = feature_weights
        elif use_canonical_weights:
            # Use canonical weights from feature importance analysis
            self.feature_weights = self.CANONICAL_WEIGHTS.copy()
        else:
            # Equal weights if no weights provided and not using canonical weights
            feature_extractors = semantic_feature_extractors or ["position", "health", "has_target", 
                                                               "energy", "is_alive", "role", "threatened"]
            weight = 1.0 / len(feature_extractors)
            self.feature_weights = {feature: weight for feature in feature_extractors}
        
        # Ensure all features have weights
        for feature in self.semantic_loss.feature_extractors:
            if feature not in self.feature_weights:
                self.feature_weights[feature] = 0.1  # Default weight for unknown features
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.feature_weights.values())
        if total_weight > 0:
            for feature in self.feature_weights:
                self.feature_weights[feature] /= total_weight
        
        # Progressive weight adjustment
        self.progressive_weight_schedule = progressive_weight_schedule
        self.progressive_weight_epochs = progressive_weight_epochs
        self.current_epoch = 0
        
        # Store initial weights for progressive scheduling
        self.initial_weights = self.feature_weights.copy()
        self.target_weights = self.feature_weights.copy()
        
        # Feature stability tracking
        self.feature_stability_adjustment = feature_stability_adjustment
        self.feature_stability_scores = {feature: 1.0 for feature in self.feature_weights}
        self.stability_history = {feature: [] for feature in self.feature_weights}
    
    def update_epoch(self, epoch: int):
        """
        Update current epoch for progressive weight adjustment.
        
        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch
        
        # Apply progressive weight adjustment if enabled
        if self.progressive_weight_schedule is not None:
            self._update_progressive_weights()
    
    def update_feature_weights(self, feature_weights: Dict[str, float]):
        """
        Update feature weights directly.
        
        Args:
            feature_weights: New feature weights
        """
        # Update target weights
        for feature, weight in feature_weights.items():
            if feature in self.feature_weights:
                self.target_weights[feature] = weight
        
        # Normalize target weights
        total_weight = sum(self.target_weights.values())
        if total_weight > 0:
            for feature in self.target_weights:
                self.target_weights[feature] /= total_weight
        
        # If not using progressive scheduling, apply immediately
        if self.progressive_weight_schedule is None:
            self.feature_weights = self.target_weights.copy()
    
    def update_stability_scores(self, feature_errors: Dict[str, float]):
        """
        Update feature stability scores based on recent errors.
        
        Args:
            feature_errors: Dictionary of recent feature errors
        """
        if not self.feature_stability_adjustment:
            return
        
        # Update stability history
        for feature, error in feature_errors.items():
            if feature in self.stability_history:
                self.stability_history[feature].append(error)
                # Keep only recent history (last 10 epochs)
                if len(self.stability_history[feature]) > 10:
                    self.stability_history[feature].pop(0)
        
        # Calculate stability scores (inverse of error variance)
        for feature in self.feature_stability_scores:
            if feature in self.stability_history and len(self.stability_history[feature]) > 1:
                errors = self.stability_history[feature]
                # Use coefficient of variation (std/mean) as stability metric
                mean_error = sum(errors) / len(errors)
                if mean_error > 0:
                    variance = sum((e - mean_error)**2 for e in errors) / len(errors)
                    std_dev = variance**0.5
                    coef_var = std_dev / mean_error
                    # Inverse of coefficient of variation, capped for stability
                    stability = min(5.0, 1.0 / max(0.01, coef_var))
                    self.feature_stability_scores[feature] = stability
        
        # Adjust weights based on stability
        if self.feature_stability_adjustment:
            self._adjust_weights_for_stability()
    
    def _update_progressive_weights(self):
        """Update weights according to progressive schedule."""
        if self.progressive_weight_schedule is None:
            return
            
        # Calculate progress factor (0.0 to 1.0)
        progress = min(1.0, self.current_epoch / self.progressive_weight_epochs)
        
        if self.progressive_weight_schedule == "linear":
            # Linear interpolation between initial and target weights
            for feature in self.feature_weights:
                initial = self.initial_weights.get(feature, 0.0)
                target = self.target_weights.get(feature, 0.0)
                self.feature_weights[feature] = initial + progress * (target - initial)
        
        elif self.progressive_weight_schedule == "exp":
            # Exponential approach to target weights
            for feature in self.feature_weights:
                initial = self.initial_weights.get(feature, 0.0)
                target = self.target_weights.get(feature, 0.0)
                self.feature_weights[feature] = target - (target - initial) * (1 - progress)**2
    
    def _adjust_weights_for_stability(self):
        """Adjust weights based on feature stability."""
        if not self.feature_stability_adjustment:
            return
            
        # Calculate total stability-weighted importance
        total = 0.0
        for feature in self.feature_weights:
            importance = self.target_weights.get(feature, 0.0)
            stability = self.feature_stability_scores.get(feature, 1.0)
            total += importance * stability
        
        # Adjust weights based on stability
        if total > 0:
            for feature in self.feature_weights:
                importance = self.target_weights.get(feature, 0.0)
                stability = self.feature_stability_scores.get(feature, 1.0)
                self.feature_weights[feature] = (importance * stability) / total
    
    def forward(self, 
                model_output: Dict[str, Any], 
                x_original: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute feature-weighted loss.
        
        Args:
            model_output: Output from VAE model
            x_original: Original input tensor
            
        Returns:
            loss_dict: Dictionary of loss components and total loss
        """
        # Extract values from model output
        x_reconstructed = model_output["reconstruction"]
        mu = model_output["mu"]
        log_var = model_output["log_var"]
        compression_loss = model_output.get("compression_loss", torch.tensor(0.0))
        
        # Compute reconstruction and KL loss normally
        recon_loss = self.recon_loss(x_reconstructed, x_original)
        kld_loss = self.kl_loss(mu, log_var)
        
        # Extract semantic features
        original_features = self.semantic_loss.extract_semantic_features(x_original)
        reconstructed_features = self.semantic_loss.extract_semantic_features(x_reconstructed)
        
        # Compute weighted semantic loss for each feature type
        semantic_losses = {}
        weighted_semantic_loss = 0.0
        
        for feature_name in self.semantic_loss.feature_extractors:
            if feature_name not in original_features:
                continue
                
            # Select appropriate loss function based on feature type
            if feature_name in ["has_target", "is_alive", "threatened"]:
                # Binary features use BCE
                reconstructed_value = torch.clamp(reconstructed_features[feature_name], 0.0, 1.0)
                original_value = torch.clamp(original_features[feature_name], 0.0, 1.0)
                feature_loss = F.binary_cross_entropy(
                    reconstructed_value, 
                    original_value
                )
            elif feature_name in ["position", "health", "energy", "role"]:
                # Continuous features use MSE
                feature_loss = F.mse_loss(
                    original_features[feature_name], 
                    reconstructed_features[feature_name]
                )
            else:
                # Default to similarity metric for other features
                feature_loss = self.semantic_loss.compute_similarity(
                    original_features[feature_name], 
                    reconstructed_features[feature_name]
                )
                
            # Apply feature-specific weight
            feature_weight = self.feature_weights.get(feature_name, 1.0)
            weighted_feature_loss = feature_weight * feature_loss
            
            semantic_losses[feature_name] = feature_loss.detach().clone()
            weighted_semantic_loss += weighted_feature_loss
        
        # Make sure all losses are detached for logging purposes
        recon_loss_detached = recon_loss.detach().clone()
        kld_loss_detached = kld_loss.detach().clone()
        sem_loss_detached = weighted_semantic_loss.detach().clone()
        comp_loss_detached = compression_loss.detach().clone() if isinstance(compression_loss, torch.Tensor) else torch.tensor(compression_loss)
        
        # Compute weighted total loss
        total_loss = (
            self.recon_loss_weight * recon_loss +
            self.kl_loss_weight * kld_loss +
            self.semantic_loss_weight * weighted_semantic_loss +
            compression_loss  # Compression loss is applied directly
        )
        
        # Update stability scores if enabled
        if self.feature_stability_adjustment and self.training:
            self.update_stability_scores({k: v.item() for k, v in semantic_losses.items()})
        
        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss_detached,
            "kl_loss": kld_loss_detached,
            "semantic_loss": sem_loss_detached,
            "compression_loss": comp_loss_detached,
            "semantic_breakdown": {k: v.item() for k, v in semantic_losses.items()},
            "feature_weights": self.feature_weights.copy()
        }


# Import the feature importance analyzer for convenience
try:
    from .feature_importance import FeatureImportanceAnalyzer
except ImportError:
    FeatureImportanceAnalyzer = None 