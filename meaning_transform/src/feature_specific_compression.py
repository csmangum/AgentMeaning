#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature-specific compression strategy implementation.

This module implements:
1. Feature importance mapping functions
2. Adaptive compression configuration based on importance scores
3. Specialized FeatureSpecificVAE model implementing these strategies
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

from meaning_transform.src.adaptive_model import FeatureGroupedVAE, AdaptiveEntropyBottleneck
from meaning_transform.src.metrics import SemanticMetrics


# Feature importance scores from step 12 analysis
FEATURE_IMPORTANCE = {
    "spatial": 55.4,  # Position coordinates
    "resource": 25.1,  # Health, energy, resource level
    "performance": 10.5,  # Age, total reward
    "status": 4.7,  # Current health, is_defending
    "role": 4.3  # Role
}

# Feature group definitions (start_idx, end_idx)
# Based on AgentState.to_tensor implementation
FEATURE_GROUPS = {
    "spatial": (0, 3),      # Position x, y, z (3 features)
    "resource": (3, 6),     # Health, energy, resource_level (3 features)
    "status": (6, 8),       # Current health, is_defending (2 features)
    "performance": (8, 10), # Age, total_reward (2 features)
    "role": (10, 15)        # One-hot encoded role (5 features)
}


def calculate_compression_levels(
    importance_scores: Dict[str, float],
    base_compression: float = 1.0,
    min_compression: float = 0.5,
    max_compression: float = 5.0
) -> Dict[str, float]:
    """
    Calculate feature-specific compression levels inversely proportional to importance.
    
    Args:
        importance_scores: Dictionary of feature importance scores (higher = more important)
        base_compression: Base compression level to scale from
        min_compression: Minimum compression level (for highest importance features)
        max_compression: Maximum compression level (for lowest importance features)
    
    Returns:
        Dictionary mapping feature groups to compression levels
    """
    # Normalize importance scores to sum to 100
    total_importance = sum(importance_scores.values())
    normalized_importance = {k: v / total_importance * 100 for k, v in importance_scores.items()}
    
    # Calculate inverse importance (higher importance -> lower compression)
    inverse_importance = {k: 100 / max(v, 0.1) for k, v in normalized_importance.items()}
    
    # Normalize inverse importance to derive compression multipliers
    total_inverse = sum(inverse_importance.values())
    compression_multipliers = {k: v / total_inverse * len(importance_scores) 
                             for k, v in inverse_importance.items()}
    
    # Apply base compression and clamp to min/max
    compression_levels = {}
    for feature, multiplier in compression_multipliers.items():
        compression = base_compression * multiplier
        compression = max(min_compression, min(max_compression, compression))
        compression_levels[feature] = compression
    
    return compression_levels


class FeatureSpecificCompressionVAE(FeatureGroupedVAE):
    """
    Enhanced VAE model that applies compression based on feature importance.
    Builds on the FeatureGroupedVAE but with more sophisticated importance-based allocation.
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        importance_scores: Optional[Dict[str, float]] = None,
        feature_groups: Optional[Dict[str, Tuple[int, int]]] = None,
        base_compression_level: float = 1.0,
        min_compression: float = 0.5,
        max_compression: float = 5.0,
        use_batch_norm: bool = True
    ):
        """
        Initialize feature-specific compression VAE.
        
        Args:
            input_dim: Dimension of input agent state
            latent_dim: Dimension of latent space
            importance_scores: Dictionary of feature importance scores (higher = more important)
            feature_groups: Dictionary mapping group names to (start_idx, end_idx)
            base_compression_level: Base compression level
            min_compression: Minimum compression level for any feature group
            max_compression: Maximum compression level for any feature group
            use_batch_norm: Whether to use batch normalization
        """
        # Use default importance scores if not provided
        if importance_scores is None:
            importance_scores = FEATURE_IMPORTANCE
        
        # Use default feature groups if not provided
        if feature_groups is None:
            feature_groups = FEATURE_GROUPS
        
        # Calculate feature-specific compression levels
        compression_levels = calculate_compression_levels(
            importance_scores,
            base_compression=base_compression_level,
            min_compression=min_compression,
            max_compression=max_compression
        )
        
        # Convert feature groups format from (start, end) to (start, end, compression)
        feature_groups_with_compression = {
            name: (start, end, compression_levels[name])
            for name, (start, end) in feature_groups.items()
        }
        
        # Initialize parent class with feature-specific compression levels
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            feature_groups=feature_groups_with_compression,
            base_compression_level=1.0,  # Base is already applied in compression_levels
            use_batch_norm=use_batch_norm
        )
        
        # Store metadata for analysis
        self.importance_scores = importance_scores
        self.min_compression = min_compression
        self.max_compression = max_compression
        self.base_compression_level = base_compression_level
    
    def get_importance_based_analysis(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed analysis of feature groups with importance metrics.
        
        Returns:
            Dictionary mapping feature groups to detailed metrics
        """
        analysis = self.get_feature_group_analysis()
        
        # Add importance scores and normalized metrics
        for group_name, metrics in analysis.items():
            metrics["importance_score"] = self.importance_scores.get(group_name, 0.0)
            metrics["normalized_importance"] = (metrics["importance_score"] / 
                                               sum(self.importance_scores.values()) * 100)
            
            # Calculate efficiency metrics
            metrics["importance_to_space_ratio"] = (
                metrics["normalized_importance"] / (metrics["latent_dim"] / self.latent_dim * 100)
            )
        
        return analysis


def calculate_optimal_latent_allocation(
    input_dim: int,
    latent_dim: int,
    importance_scores: Dict[str, float],
    feature_groups: Dict[str, Tuple[int, int]]
) -> Dict[str, int]:
    """
    Calculate optimal latent dimension allocation based on feature importance.
    
    Args:
        input_dim: Total input dimension
        latent_dim: Total latent dimension
        importance_scores: Dictionary of feature importance scores
        feature_groups: Dictionary mapping groups to (start_idx, end_idx)
    
    Returns:
        Dictionary mapping feature groups to latent dimensions
    """
    # Calculate normalized importance
    total_importance = sum(importance_scores.values())
    normalized_importance = {k: v / total_importance for k, v in importance_scores.items()}
    
    # Calculate feature counts per group
    feature_counts = {name: end - start for name, (start, end) in feature_groups.items()}
    total_features = sum(feature_counts.values())
    
    # Calculate dimension allocation
    # Formula: latent_dim_i = latent_dim * (importance_i * feature_count_i / total_features)
    # This balances both importance and the number of features in each group
    latent_allocations = {}
    remaining_dim = latent_dim
    
    for name in importance_scores:
        if name not in feature_groups:
            continue
            
        feature_count = feature_counts[name]
        importance = normalized_importance[name]
        
        # Allocate latent dimensions proportionally to importance and feature count
        allocation = max(1, int(latent_dim * importance * feature_count / total_features * 2))
        
        # Ensure we don't exceed remaining dimensions
        allocation = min(allocation, remaining_dim - len(importance_scores) + 1)
        
        latent_allocations[name] = allocation
        remaining_dim -= allocation
    
    # Distribute any remaining dimensions
    if remaining_dim > 0:
        # Sort groups by importance
        sorted_groups = sorted(
            importance_scores.keys(), 
            key=lambda k: importance_scores.get(k, 0),
            reverse=True
        )
        
        # Distribute remaining dimensions to the most important groups
        for name in sorted_groups:
            if name in latent_allocations and remaining_dim > 0:
                latent_allocations[name] += 1
                remaining_dim -= 1
                
            if remaining_dim == 0:
                break
    
    return latent_allocations


def create_feature_specific_model(
    input_dim: int,
    latent_dim: int,
    base_compression_level: float = 1.0,
    custom_importance: Optional[Dict[str, float]] = None,
    custom_feature_groups: Optional[Dict[str, Tuple[int, int]]] = None
) -> FeatureSpecificCompressionVAE:
    """
    Create a feature-specific compression VAE with optimal settings.
    
    Args:
        input_dim: Input dimension
        latent_dim: Latent dimension
        base_compression_level: Base compression level
        custom_importance: Optional custom importance scores
        custom_feature_groups: Optional custom feature group definitions
    
    Returns:
        Configured FeatureSpecificCompressionVAE
    """
    importance = custom_importance or FEATURE_IMPORTANCE
    feature_groups = custom_feature_groups or FEATURE_GROUPS
    
    model = FeatureSpecificCompressionVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        importance_scores=importance,
        feature_groups=feature_groups,
        base_compression_level=base_compression_level,
        min_compression=0.5,  # Minimum compression for most important features
        max_compression=5.0,  # Maximum compression for least important features
        use_batch_norm=True
    )
    
    return model


def print_compression_strategy_analysis(model: FeatureSpecificCompressionVAE) -> None:
    """
    Print detailed analysis of the feature-specific compression strategy.
    
    Args:
        model: FeatureSpecificCompressionVAE model
    """
    analysis = model.get_importance_based_analysis()
    
    print("\n=== FEATURE-SPECIFIC COMPRESSION STRATEGY ANALYSIS ===")
    print(f"Base compression level: {model.base_compression_level}x")
    print(f"Total latent dimension: {model.latent_dim}")
    print(f"Total input dimension: {model.input_dim}")
    print("\nFeature Group Analysis:")
    
    # Sort groups by importance
    sorted_groups = sorted(
        analysis.keys(), 
        key=lambda k: analysis[k].get("importance_score", 0),
        reverse=True
    )
    
    for group in sorted_groups:
        metrics = analysis[group]
        if "feature_range" not in metrics:
            continue
            
        print(f"\n{group.upper()} - {metrics.get('importance_score', 0):.1f}% importance")
        print(f"  Features: {metrics['feature_range'][0]}-{metrics['feature_range'][1]} "
              f"({metrics['feature_count']} features)")
        print(f"  Latent allocation: {metrics['latent_range'][0]}-{metrics['latent_range'][1]} "
              f"({metrics['latent_dim']} dimensions, {metrics['latent_dim']/model.latent_dim*100:.1f}% of total)")
        print(f"  Compression: {metrics['overall_compression']:.2f}x "
              f"(Effective dim: {metrics['effective_dim']})")
        print(f"  Efficiency: {metrics.get('importance_to_space_ratio', 0):.2f} "
              f"(Importance-to-space ratio)")
    
    # Overall compression
    compression_rates = model.get_compression_rate()
    print(f"\nOverall compression rate: {compression_rates.get('overall', 0):.2f}x") 