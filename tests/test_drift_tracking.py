#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script specifically for the drift tracking functionality in train.py.
"""

import os
import torch
import numpy as np
from meaning_transform.src.metrics import (
    DriftTracker,
    CompressionThresholdFinder
)
from meaning_transform.src.standardized_metrics import StandardizedMetrics

# Create test data with expected format
def create_test_batch(batch_size=10, input_dim=50):
    """Create a test batch that has the expected structure."""
    batch = torch.zeros(batch_size, input_dim)
    
    # Set position (x, y)
    batch[:, 0] = torch.rand(batch_size)  # x position
    batch[:, 1] = torch.rand(batch_size)  # y position
    
    # Set health
    batch[:, 2] = 0.7 + 0.3 * torch.rand(batch_size)  # health (normalized 0-1)
    
    # Set has_target
    batch[:, 3] = (torch.rand(batch_size) > 0.5).float()  # binary has_target

    # Set energy
    batch[:, 4] = 0.5 + 0.5 * torch.rand(batch_size)  # energy (normalized 0-1)
    
    # Set roles (one-hot encoded in positions 5-9)
    roles = torch.eye(5)  # 5 possible roles
    for i in range(batch_size):
        role_idx = np.random.randint(0, 5)
        batch[i, 5:10] = roles[role_idx]
    
    return batch

# Create output directory
output_dir = "test_results/drift_test"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)

# Create synthetic data
print("Creating test data...")
original_batch = create_test_batch(batch_size=10, input_dim=50)

# Simulate model output dictionary
def simulate_model_output(original_batch, perturbation=0.1):
    """Simulate a model's output including reconstructed data."""
    reconstructed = original_batch + perturbation * torch.randn_like(original_batch)
    reconstructed = torch.clamp(reconstructed, 0, 1)
    
    # In our real model, we would also have z (latent vector)
    # For testing, we'll create a random z
    z = torch.randn(original_batch.size(0), 32)
    
    return {
        "x_reconstructed": reconstructed,
        "z": z
    }

# Custom t-SNE visualization for small sample sizes
def custom_tsne_visualization(latent_vectors, labels=None, output_file=None):
    """
    Generate t-SNE visualization with adjusted parameters for small sample sizes.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    # Convert to numpy for t-SNE
    latent_np = latent_vectors.detach().cpu().numpy()
    
    # Adjust perplexity for small sample sizes (perplexity must be < n_samples)
    n_samples = latent_np.shape[0]
    perplexity = min(n_samples - 1, 5)  # Use a small perplexity for small sample sizes
    
    # Apply t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    latent_2d = tsne.fit_transform(latent_np)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        labels_np = labels.detach().cpu().numpy()
        unique_labels = np.unique(labels_np)
        for label in unique_labels:
            mask = labels_np == label
            plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                      label=f"Class {label}", alpha=0.7)
        plt.legend()
    else:
        plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.7)
    
    plt.title("t-SNE Visualization of Latent Space (Small Sample)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, alpha=0.3)
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

# Simulate the track_semantic_drift method from train.py
def test_track_semantic_drift(iterations=3):
    """Test the semantic drift tracking functionality."""
    # Initialize metrics tools
    drift_tracker = DriftTracker(log_dir=os.path.join(output_dir, "drift_tracking"))
    semantic_metrics = StandardizedMetrics()
    
    # Store history
    drift_history = []
    
    # Simulate training loop
    for i in range(iterations):
        print(f"\nIteration {i+1}/{iterations}")
        
        # Create test batch with increasing perturbation to simulate drift
        perturbation = 0.05 * (i + 1)
        batch = create_test_batch(batch_size=10)
        results = simulate_model_output(batch, perturbation)
        
        # Set compression level
        compression_level = 4.0 - i * 0.5  # Decreasing from 4.0 to 2.5
        
        # Track metrics using drift tracker
        print(f"Running drift tracking (compression level: {compression_level} bpp)")
        drift_metrics = drift_tracker.log_iteration(
            iteration=i,
            compression_level=compression_level,
            original=batch,
            reconstructed=results["x_reconstructed"]
        )
        
        # Calculate metrics using standardized metrics
        standard_metrics = semantic_metrics.evaluate(
            batch, results["x_reconstructed"]
        )
        
        # Create metrics for history tracking
        history_metrics = {
            "epoch": i,
            "total_semantic_loss": 1.0 - standard_metrics["overall_preservation"],
            "feature_losses": {
                k: 1.0 - v for k, v in standard_metrics.items() 
                if k.endswith("_preservation") and k != "overall_preservation"
            }
        }
        
        # Save to history
        drift_history.append(history_metrics)
        
        # Generate latent space visualization periodically
        if i % 2 == 0:
            # Extract latent vectors
            latent_vectors = results["z"]
            
            # Create role labels for visualization
            role_indices = torch.argmax(batch[:, 5:10], dim=1)
            
            # Generate custom t-SNE visualization
            vis_path = os.path.join(output_dir, "visualizations", f"latent_tsne_epoch_{i}.png")
            print(f"Generating t-SNE visualization: {vis_path}")
            custom_tsne_visualization(
                latent_vectors, 
                labels=role_indices,
                output_file=vis_path
            )
        
        print(f"Overall preservation: {standard_metrics['overall_preservation']:.4f}")
        print(f"Overall fidelity: {standard_metrics['overall_fidelity']:.4f}")
        print(f"Semantic loss: {history_metrics['total_semantic_loss']:.4f}")
    
    # Generate visualization and report
    print("\nGenerating visualization and report...")
    drift_tracker.visualize_drift(os.path.join(output_dir, "drift_visualization.png"))
    report = drift_tracker.generate_report(os.path.join(output_dir, "drift_report.md"))
    
    return drift_history

# Run the test
print("Testing drift tracking functionality...")
drift_history = test_track_semantic_drift(iterations=4)

print("\nTest completed. Check the test_results/drift_test directory for outputs.") 