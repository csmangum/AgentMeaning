#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test evaluation methods for the meaning preservation model.
"""

import os
import sys
from pathlib import Path

import torch
import numpy as np

# Add the project root to the path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from meaning_transform.src.model import MeaningVAE
from meaning_transform.src.metrics import (
    SemanticMetrics,
    DriftTracker,
    CompressionThresholdFinder,
    generate_t_sne_visualization
)

# Create output directory
output_dir = "test_results/semantic_eval"
os.makedirs(output_dir, exist_ok=True)

# Create synthetic data directly - bypassing the data module's complexity
def create_test_batch(batch_size=5, input_dim=50):
    """Create a test batch that has the expected structure for our metrics."""
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

# Generate synthetic data
print("Creating synthetic data...")
original_batch = create_test_batch(batch_size=5, input_dim=50)
print("Original batch shape:", original_batch.shape)

# Create a slightly altered reconstruction batch for testing
reconstructed_batch = original_batch.clone()

# Add some noise to the reconstruction to simulate compression artifacts
print("Creating simulated reconstructed data...")
perturbation = 0.1
reconstructed_batch = original_batch + perturbation * torch.randn_like(original_batch)
reconstructed_batch = torch.clamp(reconstructed_batch, 0, 1)

# Initialize metrics
print("Initializing metrics...")
semantic_metrics = SemanticMetrics()
drift_tracker = DriftTracker(log_dir=os.path.join(output_dir, "drift_tracking"))
threshold_finder = CompressionThresholdFinder(semantic_threshold=0.9)

# Evaluate semantic preservation
metrics = semantic_metrics.evaluate(original_batch, reconstructed_batch)
print(f"Overall semantic score: {metrics['overall']:.4f}")

# Detailed metric breakdown
print("\nDetailed metrics:")
for key, value in metrics.items():
    if isinstance(value, float):  # Only print the scalar metrics
        print(f"  {key}: {value:.4f}")

# Log drift metrics
drift_metrics = drift_tracker.log_iteration(
    iteration=0, 
    compression_level=2.0, 
    original=original_batch, 
    reconstructed=reconstructed_batch
)

# Generate visualization
print("\nGenerating visualization...")
drift_tracker.visualize_drift(os.path.join(output_dir, "drift_visualization.png"))

# Generate report
print("Generating report...")
report = drift_tracker.generate_report(os.path.join(output_dir, "drift_report.md"))

print("\nTest completed. Check the test_results/semantic_eval directory for outputs.") 