#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Examples of using the visualization utilities.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import sys
from typing import Dict, List, Tuple

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import visualize


def example_latent_space_visualization():
    """Example of latent space visualization."""
    print("Running latent space visualization example...")
    
    # Create some synthetic latent vectors
    latent_dim = 32
    num_samples = 200
    
    # Create random latent vectors
    latent_vectors = torch.randn(num_samples, latent_dim)
    
    # Create synthetic labels (5 classes)
    labels = torch.randint(0, 5, (num_samples,))
    
    # Visualize using both t-SNE and PCA
    output_paths = visualize.visualize_latent_space(
        latent_vectors=latent_vectors,
        labels=labels,
        metadata={"model": "VAE", "latent_dim": latent_dim}
    )
    
    print(f"Generated latent space visualizations at: {output_paths}")


def example_loss_curves_visualization():
    """Example of loss curves visualization."""
    print("Running loss curves visualization example...")
    
    # Create synthetic loss history
    epochs = 100
    loss_history = {
        "total_loss": [(i, 1.0 * np.exp(-i / 50) + 0.1 + 0.05 * np.sin(i / 10)) for i in range(epochs)],
        "reconstruction_loss": [(i, 0.7 * np.exp(-i / 60) + 0.05 + 0.03 * np.sin(i / 8)) for i in range(epochs)],
        "kl_loss": [(i, 0.3 * np.exp(-i / 40) + 0.05 + 0.02 * np.sin(i / 12)) for i in range(epochs)],
        "semantic_loss": [(i, 0.4 * np.exp(-i / 55) + 0.02 + 0.01 * np.sin(i / 15)) for i in range(epochs)],
        "val_total_loss": [(i, 1.1 * np.exp(-i / 50) + 0.12 + 0.07 * np.sin(i / 10)) for i in range(epochs)],
    }
    
    # Visualize loss curves
    output_path = visualize.visualize_loss_curves(
        loss_history=loss_history,
        loss_names=["total_loss", "reconstruction_loss", "kl_loss", "semantic_loss"]
    )
    
    print(f"Generated loss curves visualization at: {output_path}")
    
    # Create synthetic compression vs. reconstruction data
    compression_levels = np.linspace(1, 32, 15)
    reconstruction_errors = [0.8 * np.exp(-x / 10) + 0.1 for x in compression_levels]
    semantic_losses = [0.7 * np.exp(-x / 8) + 0.05 for x in compression_levels]
    
    # Visualize compression vs. reconstruction
    output_path = visualize.visualize_compression_vs_reconstruction(
        compression_levels=compression_levels,
        reconstruction_errors=reconstruction_errors,
        semantic_losses=semantic_losses
    )
    
    print(f"Generated compression vs. reconstruction visualization at: {output_path}")


def example_state_comparison_visualization():
    """Example of state comparison visualization."""
    print("Running state comparison visualization example...")
    
    # Create synthetic feature data
    num_samples = 10
    features = ["position", "health", "energy", "is_alive", "has_target", "role"]
    
    original_features = {
        "position": np.random.rand(num_samples, 2),
        "health": np.random.rand(num_samples) * 100,
        "energy": np.random.rand(num_samples) * 50,
        "is_alive": np.random.randint(0, 2, num_samples),
        "has_target": np.random.randint(0, 2, num_samples),
        "role": np.random.randint(0, 5, num_samples)
    }
    
    # Create reconstructed features with some noise
    reconstructed_features = {}
    for feature, values in original_features.items():
        if feature in ["is_alive", "has_target", "role"]:
            # For categorical features, flip some values
            reconstructed = values.copy()
            flip_indices = np.random.choice(num_samples, size=num_samples // 10, replace=False)
            if feature == "role":
                reconstructed[flip_indices] = np.random.randint(0, 5, len(flip_indices))
            else:
                reconstructed[flip_indices] = 1 - reconstructed[flip_indices]
            reconstructed_features[feature] = reconstructed
        else:
            # For continuous features, add noise
            noise = np.random.normal(0, 0.05 * np.std(values), values.shape)
            reconstructed_features[feature] = values + noise
    
    # Visualize feature comparison
    output_path = visualize.visualize_state_comparison(
        original_features=original_features,
        reconstructed_features=reconstructed_features,
        example_indices=[0, 1, 2]  # Show first 3 examples
    )
    
    print(f"Generated state comparison visualization at: {output_path}")
    
    # Create synthetic state trajectories
    num_steps = 20
    original_states = [torch.tensor([float(i)/num_steps, np.sin(i * np.pi / 10)]) for i in range(num_steps)]
    
    # Add some noise to reconstructed states
    reconstructed_states = [state + torch.randn_like(state) * 0.05 for state in original_states]
    
    # Visualize state trajectories
    output_path = visualize.visualize_state_trajectories(
        original_states=original_states,
        reconstructed_states=reconstructed_states
    )
    
    print(f"Generated state trajectory visualization at: {output_path}")
    
    # Create synthetic confusion matrices
    confusion_matrices = {
        "role": np.array([
            [18, 2, 0, 0, 0],
            [1, 16, 3, 0, 0],
            [0, 2, 19, 1, 0],
            [0, 0, 2, 17, 1],
            [0, 0, 0, 2, 18]
        ]),
        "is_alive": np.array([
            [45, 5],
            [3, 47]
        ]),
        "has_target": np.array([
            [42, 8],
            [6, 44]
        ])
    }
    
    # Visualize confusion matrices
    output_path = visualize.visualize_confusion_matrices(
        confusion_matrices=confusion_matrices
    )
    
    print(f"Generated confusion matrices visualization at: {output_path}")


def example_drift_tracking_visualization():
    """Example of drift tracking visualization."""
    print("Running drift tracking visualization example...")
    
    # Create synthetic semantic drift data
    iterations = list(range(0, 100, 5))
    features = ["position", "health", "energy", "is_alive", "has_target", "role", "overall"]
    
    semantic_scores = {}
    for feature in features:
        if feature == "overall":
            # Overall score is a weighted average
            base = 0.95
            decay = 0.001
            scores = [base - decay * i for i in iterations]
        elif feature in ["is_alive", "has_target"]:
            # Binary features have high importance
            base = 0.97
            decay = 0.0008
            scores = [base - decay * i for i in iterations]
        else:
            # Other features have more variability
            base = 0.92
            decay = 0.002
            noise = 0.05
            scores = [max(0, min(1, base - decay * i + np.random.normal(0, noise))) for i in iterations]
        
        semantic_scores[feature] = scores
    
    # Visualize semantic drift
    output_path = visualize.visualize_semantic_drift(
        iterations=iterations,
        semantic_scores=semantic_scores
    )
    
    print(f"Generated semantic drift visualization at: {output_path}")
    
    # Create synthetic compression threshold data
    compression_levels = np.linspace(1, 32, 15)
    semantic_scores = [min(1.0, 0.7 + 0.3 * (1 - np.exp(-x / 8))) for x in compression_levels]
    reconstruction_errors = [0.8 * np.exp(-x / 10) + 0.1 for x in compression_levels]
    
    # Visualize threshold finder
    output_path, optimal_compression = visualize.visualize_threshold_finder(
        compression_levels=compression_levels,
        semantic_scores=semantic_scores,
        reconstruction_errors=reconstruction_errors,
        threshold=0.9
    )
    
    print(f"Generated threshold finder visualization at: {output_path}")
    print(f"Optimal compression level: {optimal_compression:.2f} bits")


def example_latent_interpolation():
    """Example of latent interpolation visualization."""
    print("Running latent interpolation example...")
    
    # Create a simple dummy VAE for demonstration
    latent_dim = 8
    input_dim = 10
    
    class DummyVAE:
        def __init__(self):
            self.encoder = lambda x: torch.randn(latent_dim)
            
            # Create a decoder that deterministically maps from latent to feature space
            # Just for demonstration purposes
            self.decoder_matrix = torch.randn(latent_dim, input_dim)
            self.decoder = lambda z: torch.matmul(z, self.decoder_matrix)
    
    model = DummyVAE()
    
    # Create two synthetic states
    state_a = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0])
    state_b = torch.tensor([0.8, 0.7, 0.6, 0.5, 0.4, 0.0, 1.0, 0.0, 0.0, 0.0])
    
    # Visualize latent interpolation
    output_path = visualize.visualize_latent_interpolation(
        decode_fn=model.decoder,
        state_a=state_a,
        state_b=state_b,
        encoder=model.encoder,
        steps=10
    )
    
    print(f"Generated latent interpolation visualization at: {output_path}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Make sure the visualization directories exist
    visualize.setup_visualization_dirs()
    
    # Run examples
    example_latent_space_visualization()
    example_loss_curves_visualization()
    example_state_comparison_visualization()
    example_drift_tracking_visualization()
    example_latent_interpolation()
    
    print("\nAll visualization examples completed. Check the results/visualizations directory for outputs.") 