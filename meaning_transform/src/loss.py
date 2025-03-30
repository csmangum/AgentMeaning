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
        
        # Return all loss components
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss_detached,
            "kl_loss": kld_loss_detached,
            "semantic_loss": sem_loss_detached,
            "compression_loss": comp_loss_detached,
            "semantic_breakdown": semantic_breakdown
        } 