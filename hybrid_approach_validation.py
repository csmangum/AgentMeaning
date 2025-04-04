#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validation script for the hybrid approach to semantic similarity measurement.
This script tests whether our fixes correctly ensure that lower compression levels
(1.0) perform better than higher compression levels (5.0) as expected.
"""

import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt

from meaning_transform.src.metrics import SemanticMetrics
from meaning_transform.src.loss import SemanticLoss


def setup_test_data(batch_size=64, input_dim=50):
    """Create test data for metrics evaluation."""
    # Create sample agent states
    # Format: [x, y, health, has_target, energy, role_1, role_2, role_3, role_4, role_5, ...]
    x_original = torch.zeros(batch_size, input_dim)
    
    # Set position (x, y)
    x_original[:, 0] = torch.rand(batch_size)  # x position
    x_original[:, 1] = torch.rand(batch_size)  # y position
    
    # Set health
    x_original[:, 2] = 0.7 + 0.3 * torch.rand(batch_size)  # health (normalized 0-1)
    
    # Set has_target
    x_original[:, 3] = (torch.rand(batch_size) > 0.5).float()  # binary has_target

    # Set energy
    x_original[:, 4] = 0.5 + 0.5 * torch.rand(batch_size)  # energy (normalized 0-1)
    
    # Set roles (one-hot encoded in positions 5-9)
    roles = torch.eye(5)  # 5 possible roles
    for i in range(batch_size):
        role_idx = np.random.randint(0, 5)
        x_original[i, 5:10] = roles[role_idx]
    
    return x_original


def create_perturbed_data(original, compression_level):
    """
    Create controlled perturbation based on compression level.
    Higher compression (5.0) = more perturbation
    Lower compression (1.0) = less perturbation
    """
    # Control the amount of perturbation based on compression level
    perturbation_factor = compression_level / 5.0  # 1.0 for level 5.0, 0.2 for level 1.0
    
    # Create reconstructed data with controlled perturbation
    reconstructed = original.clone()
    
    # Add more perturbation to spatial features for higher compression
    reconstructed[:, 0:2] += perturbation_factor * 0.2 * torch.randn_like(reconstructed[:, 0:2])
    
    # Add more perturbation to health and energy
    reconstructed[:, 2] += perturbation_factor * 0.1 * torch.randn_like(reconstructed[:, 2])
    reconstructed[:, 4] += perturbation_factor * 0.1 * torch.randn_like(reconstructed[:, 4])
    
    # More errors in role assignment for higher compression
    if compression_level >= 3.0:
        # For high compression, randomly change roles
        mask = torch.rand(reconstructed.shape[0]) < 0.2 * perturbation_factor
        for i in range(reconstructed.shape[0]):
            if mask[i]:
                reconstructed[i, 5:10] = torch.zeros(5)
                new_role = torch.randint(0, 5, (1,))
                reconstructed[i, 5 + new_role] = 1.0
    
    return reconstructed


def validate_hybrid_approach():
    """
    Validate that our hybrid approach to semantic similarity measurement correctly
    identifies that compression level 1.0 performs better than higher levels.
    """
    print("\n=== Validating Hybrid Approach to Semantic Similarity ===")
    
    # Initialize semantic metrics
    metrics = SemanticMetrics()
    
    # Create a fixed test dataset for consistent comparison
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Original data will be the same for all tests
    original = setup_test_data(batch_size=64)
    
    # Compression levels to test
    compression_levels = [5.0, 3.0, 1.0, 0.5]
    
    results = {}
    feature_results = defaultdict(dict)
    
    for level in compression_levels:
        # Create reconstructed data with controlled perturbation
        reconstructed = create_perturbed_data(original, level)
        
        # Calculate semantic scores
        scores = metrics.compute_equivalence_scores(original, reconstructed)
        
        # Save overall score
        results[level] = scores["overall"]
        
        # Save feature-specific scores
        for feature, score in scores.items():
            if feature != "overall":
                feature_results[feature][level] = score
        
        print(f"Compression level {level:.1f}: Overall score = {scores['overall']:.4f}")
    
    # Validate that compression level 1.0 performs better than 5.0
    print(f"\nCompression level 1.0 score: {results[1.0]:.4f}")
    print(f"Compression level 5.0 score: {results[5.0]:.4f}")
    print(f"Difference: {results[1.0] - results[5.0]:.4f}")
    
    assert results[1.0] > results[5.0], f"Level 1.0 ({results[1.0]:.4f}) should outperform level 5.0 ({results[5.0]:.4f})"
    
    # Print feature-specific comparisons
    print("\nFeature-specific performance comparison:")
    for feature, level_scores in feature_results.items():
        sorted_levels = sorted(level_scores.items(), key=lambda x: x[1], reverse=True)
        best_level, best_score = sorted_levels[0]
        print(f"{feature}: Best at level {best_level:.1f} with score {best_score:.4f}")
    
    # Visualize the results
    create_comparison_chart(results, feature_results, "hybrid_approach_results.png")
    
    return results


def create_comparison_chart(results, feature_results, output_file):
    """Create a chart comparing the hybrid approach results"""
    plt.figure(figsize=(12, 8))
    
    # Sort compression levels
    compression_levels = sorted(results.keys())
    
    # Plot overall scores
    plt.subplot(2, 1, 1)
    plt.plot(compression_levels, [results[level] for level in compression_levels], 
             'o-', linewidth=2, markersize=8, label='Overall Score')
    plt.title('Overall Semantic Preservation by Compression Level')
    plt.xlabel('Compression Level (higher = more compression)')
    plt.ylabel('Semantic Preservation Score')
    plt.grid(True)
    plt.legend()
    
    # Plot feature-specific scores
    plt.subplot(2, 1, 2)
    for feature in feature_results:
        plt.plot(compression_levels, 
                [feature_results[feature][level] for level in compression_levels],
                'o-', linewidth=2, markersize=6, label=feature)
    
    plt.title('Feature-specific Preservation by Compression Level')
    plt.xlabel('Compression Level (higher = more compression)')
    plt.ylabel('Feature Preservation Score')
    plt.grid(True)
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Comparison chart saved to {output_file}")


if __name__ == "__main__":
    validate_hybrid_approach() 