#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for metrics.py module.

This script tests:
1. Semantic feature extraction
2. Semantic equivalence evaluation
3. Drift tracking functionality
4. Latent space metrics
"""

import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import unittest
from collections import defaultdict
import matplotlib.pyplot as plt

# Add the project root to the path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from meaning_transform.src.metrics import (
    SemanticMetrics, 
    DriftTracker, 
    compute_latent_space_metrics,
    generate_t_sne_visualization,
    CompressionThresholdFinder,
    compute_reconstruction_error,
    compute_latent_statistics,
    compute_meaning_metrics,
    normalized_hamming_distance,
    binary_accuracy,
    compute_drift_metrics
)


def setup_test_data(batch_size=32, input_dim=50):
    """Create test data for metrics evaluation."""
    # Create sample agent states
    # Format: [x, y, health, has_target, energy, role_1, role_2, role_3, role_4, role_5, ...]
    x_original = torch.zeros(batch_size, input_dim)
    x_reconstructed = torch.zeros(batch_size, input_dim)
    
    # Set position (x, y)
    x_original[:, 0] = torch.rand(batch_size)  # x position
    x_original[:, 1] = torch.rand(batch_size)  # y position
    x_reconstructed[:, 0] = x_original[:, 0] + 0.1 * torch.randn(batch_size)  # slightly perturbed x
    x_reconstructed[:, 1] = x_original[:, 1] + 0.1 * torch.randn(batch_size)  # slightly perturbed y
    
    # Set health
    x_original[:, 2] = 0.7 + 0.3 * torch.rand(batch_size)  # health (normalized 0-1)
    x_reconstructed[:, 2] = x_original[:, 2] + 0.05 * torch.randn(batch_size)
    x_reconstructed[:, 2] = torch.clamp(x_reconstructed[:, 2], 0, 1)  # ensure in [0, 1]
    
    # Set has_target
    x_original[:, 3] = (torch.rand(batch_size) > 0.5).float()  # binary has_target
    x_reconstructed[:, 3] = (torch.rand(batch_size) > 0.3).float()  # different has_target

    # Set energy
    x_original[:, 4] = 0.5 + 0.5 * torch.rand(batch_size)  # energy (normalized 0-1)
    x_reconstructed[:, 4] = x_original[:, 4] + 0.1 * torch.randn(batch_size)
    x_reconstructed[:, 4] = torch.clamp(x_reconstructed[:, 4], 0, 1)
    
    # Set roles (one-hot encoded in positions 5-9)
    roles = torch.eye(5)  # 5 possible roles
    for i in range(batch_size):
        role_idx = np.random.randint(0, 5)
        x_original[i, 5:10] = roles[role_idx]
        
        # 80% of time keep same role, 20% different
        if np.random.random() < 0.8:
            x_reconstructed[i, 5:10] = roles[role_idx]
        else:
            new_role_idx = np.random.randint(0, 5)
            x_reconstructed[i, 5:10] = roles[new_role_idx]
    
    return x_original, x_reconstructed


def test_semantic_metrics():
    """Test the semantic metrics functionality."""
    print("\n=== Testing Semantic Metrics ===")
    
    # Create test data
    original, reconstructed = setup_test_data()
    
    # Initialize semantic metrics
    metrics = SemanticMetrics()
    
    # Test feature extraction
    features = metrics.extract_features(original)
    print(f"Extracted features: {list(features.keys())}")
    
    # Test equivalence scores
    scores = metrics.compute_equivalence_scores(original, reconstructed)
    print(f"Semantic equivalence scores: {scores}")
    
    # Test binary feature accuracy
    binary_acc = metrics.binary_feature_accuracy(original, reconstructed)
    print(f"Binary feature accuracy: {binary_acc}")
    
    # Test role accuracy
    role_acc = metrics.role_accuracy(original, reconstructed)
    print(f"Role accuracy: {role_acc['role_accuracy']}")
    
    # Test numeric feature errors
    numeric_errors = metrics.numeric_feature_errors(original, reconstructed)
    print(f"Numeric feature errors: {numeric_errors}")
    
    # Test complete evaluation
    evaluation = metrics.evaluate(original, reconstructed)
    print(f"Overall semantic preservation score: {evaluation['overall']}")
    
    return evaluation


def test_drift_tracker():
    """Test the drift tracking functionality."""
    print("\n=== Testing Drift Tracker ===")
    
    # Create temporary results directory
    log_dir = "test_results/drift_tracking"
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize drift tracker
    tracker = DriftTracker(log_dir=log_dir)
    
    # Simulate compression levels
    compression_levels = [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
    
    # Log metrics for each compression level
    for i, compression_level in enumerate(compression_levels):
        # Create data with increasing perturbation as compression increases
        original, reconstructed = setup_test_data()
        
        # Add more perturbation as compression increases (lower bpp)
        perturbation = (4.0 - compression_level) / 4.0  # 0.0 to 0.875
        reconstructed = original + perturbation * torch.randn_like(original)
        reconstructed = torch.clamp(reconstructed, 0, 1)
        
        # Log metrics
        metrics = tracker.log_iteration(
            iteration=i,
            compression_level=compression_level,
            original=original,
            reconstructed=reconstructed
        )
        print(f"Iteration {i}, Compression: {compression_level} bpp, Overall Score: {metrics['overall']:.4f}")
    
    # Test visualization
    output_file = os.path.join(log_dir, "drift_visualization.png")
    tracker.visualize_drift(output_file)
    print(f"Drift visualization saved to: {output_file}")
    
    # Test report generation
    report_file = os.path.join(log_dir, "drift_report.md")
    report = tracker.generate_report(report_file)
    print(f"Drift report saved to: {report_file}")
    
    return tracker


def test_latent_space_metrics():
    """Test the latent space metrics functionality."""
    print("\n=== Testing Latent Space Metrics ===")
    
    # Create synthetic latent vectors
    batch_size = 100
    latent_dim = 32
    
    # Create two clusters in latent space
    cluster1 = torch.randn(batch_size // 2, latent_dim) + torch.tensor([2.0, 1.0] + [0.0] * (latent_dim - 2))
    cluster2 = torch.randn(batch_size // 2, latent_dim) + torch.tensor([-2.0, -1.0] + [0.0] * (latent_dim - 2))
    latent_vectors = torch.cat([cluster1, cluster2], dim=0)
    
    # Create labels
    labels = torch.cat([
        torch.zeros(batch_size // 2),
        torch.ones(batch_size // 2)
    ], dim=0)
    
    # Compute metrics
    metrics = compute_latent_space_metrics(latent_vectors, labels)
    print(f"Latent space metrics: {metrics}")
    
    # Test t-SNE visualization
    output_dir = "test_results/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "tsne_visualization.png")
    generate_t_sne_visualization(latent_vectors, labels, output_file)
    print(f"t-SNE visualization saved to: {output_file}")
    
    return metrics


def test_compression_threshold_finder():
    """Test the compression threshold finder functionality."""
    print("\n=== Testing Compression Threshold Finder ===")
    
    # Initialize threshold finder
    threshold_finder = CompressionThresholdFinder(semantic_threshold=0.9)
    
    # Simulate compression levels
    compression_levels = [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
    
    # Evaluate each compression level
    for compression_level in compression_levels:
        # Create data with increasing perturbation as compression increases
        original, reconstructed = setup_test_data()
        
        # Add more perturbation as compression increases (lower bpp)
        perturbation = (4.0 - compression_level) / 4.0  # 0.0 to 0.875
        reconstructed = original + perturbation * torch.randn_like(original)
        reconstructed = torch.clamp(reconstructed, 0, 1)
        
        # Evaluate compression level
        result = threshold_finder.evaluate_compression_level(
            compression_level=compression_level,
            original=original,
            reconstructed=reconstructed
        )
        print(f"Compression: {compression_level} bpp, Score: {result['overall_score']:.4f}, " +
              f"Meets threshold: {result['meets_threshold']}")
    
    # Find optimal threshold
    optimal = threshold_finder.find_optimal_threshold()
    print(f"Optimal compression threshold: {optimal}")
    
    return optimal


def main():
    """Run all tests."""
    print("=== Running Tests for Metrics Module ===")
    
    # Create results directory
    os.makedirs("test_results", exist_ok=True)
    
    # Run tests
    test_semantic_metrics()
    test_drift_tracker()
    test_latent_space_metrics()
    test_compression_threshold_finder()
    
    print("\n=== All Tests Completed ===")


if __name__ == "__main__":
    main() 