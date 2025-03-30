#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for loss functions in the meaning-preserving transformation system.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Add the project root to the path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from meaning_transform.src.loss import ReconstructionLoss, KLDivergenceLoss, SemanticLoss, CombinedLoss


def test_reconstruction_loss():
    """Test the reconstruction loss component."""
    batch_size = 32
    input_dim = 50
    
    # Create random data
    x_original = torch.rand(batch_size, input_dim)
    x_reconstructed = torch.rand(batch_size, input_dim)
    
    # Test MSE loss
    mse_loss = ReconstructionLoss(loss_type="mse")
    mse_loss_value = mse_loss(x_reconstructed, x_original)
    print(f"MSE Reconstruction Loss: {mse_loss_value.item():.4f}")
    
    # Test BCE loss
    bce_loss = ReconstructionLoss(loss_type="bce")
    bce_loss_value = bce_loss(x_reconstructed, x_original)
    print(f"BCE Reconstruction Loss: {bce_loss_value.item():.4f}")
    
    assert mse_loss_value >= 0, "MSE loss should be non-negative"
    assert bce_loss_value >= 0, "BCE loss should be non-negative"


def test_kl_divergence_loss():
    """Test the KL divergence loss component."""
    batch_size = 32
    latent_dim = 16
    
    # Create random latent variables
    mu = torch.randn(batch_size, latent_dim)
    log_var = torch.randn(batch_size, latent_dim)
    
    # Test KL divergence loss
    kl_loss = KLDivergenceLoss()
    kl_loss_value = kl_loss(mu, log_var)
    print(f"KL Divergence Loss: {kl_loss_value.item():.4f}")
    
    assert kl_loss_value >= 0, "KL divergence loss should be non-negative"


def test_semantic_loss():
    """Test the semantic loss component."""
    batch_size = 32
    input_dim = 50
    
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
    
    # Now create some low health agents to trigger "threatened" state
    low_health_indices = np.random.choice(batch_size, size=int(batch_size/4), replace=False)
    x_original[low_health_indices, 2] = 0.1 + 0.15 * torch.rand(len(low_health_indices))  # low health
    x_original[low_health_indices, 3] = 1.0  # has_target
    
    x_reconstructed[low_health_indices, 2] = x_original[low_health_indices, 2] + 0.05 * torch.randn(len(low_health_indices))
    x_reconstructed[low_health_indices, 2] = torch.clamp(x_reconstructed[low_health_indices, 2], 0, 1)
    x_reconstructed[low_health_indices, 3] = 1.0  # has_target
    
    # Test semantic loss
    sem_loss = SemanticLoss()
    sem_loss_value = sem_loss(x_reconstructed, x_original)
    print(f"Semantic Loss: {sem_loss_value.item():.4f}")
    
    # Test detailed breakdown
    breakdown = sem_loss.detailed_breakdown(x_reconstructed, x_original)
    print("Semantic Loss Breakdown:")
    for feature, loss in breakdown.items():
        print(f"  - {feature}: {loss:.4f}")
    
    assert sem_loss_value >= 0, "Semantic loss should be non-negative"
    assert len(breakdown) > 0, "Semantic breakdown should contain items"
    expected_features = ["position", "health", "has_target", "energy", "is_alive", "role", "threatened"]
    for feature in expected_features:
        assert feature in breakdown, f"Expected feature '{feature}' not in breakdown"


def test_combined_loss():
    """Test the combined loss function."""
    batch_size = 32
    input_dim = 50
    latent_dim = 16
    
    # Create sample tensors
    x_original = torch.zeros(batch_size, input_dim)
    x_reconstructed = torch.zeros(batch_size, input_dim)
    
    try:
        print("Creating test data...")
        # Fill with test data (similar to test_semantic_loss)
        # Position (x, y)
        x_original[:, 0] = torch.rand(batch_size)
        x_original[:, 1] = torch.rand(batch_size)
        x_reconstructed[:, 0] = x_original[:, 0] + 0.1 * torch.randn(batch_size)
        x_reconstructed[:, 1] = x_original[:, 1] + 0.1 * torch.randn(batch_size)
        
        # Health
        x_original[:, 2] = 0.7 + 0.3 * torch.rand(batch_size)
        x_reconstructed[:, 2] = x_original[:, 2] + 0.05 * torch.randn(batch_size)
        x_reconstructed[:, 2] = torch.clamp(x_reconstructed[:, 2], 0, 1)
        
        # Has target
        x_original[:, 3] = (torch.rand(batch_size) > 0.5).float()
        x_reconstructed[:, 3] = (torch.rand(batch_size) > 0.3).float()
        
        # Energy
        x_original[:, 4] = 0.5 + 0.5 * torch.rand(batch_size)
        x_reconstructed[:, 4] = x_original[:, 4] + 0.1 * torch.randn(batch_size)
        x_reconstructed[:, 4] = torch.clamp(x_reconstructed[:, 4], 0, 1)
        
        # Roles (one-hot encoded in positions 5-9)
        roles = torch.eye(5)
        for i in range(batch_size):
            role_idx = np.random.randint(0, 5)
            x_original[i, 5:10] = roles[role_idx]
            
            if np.random.random() < 0.8:
                x_reconstructed[i, 5:10] = roles[role_idx]
            else:
                new_role_idx = np.random.randint(0, 5)
                x_reconstructed[i, 5:10] = roles[new_role_idx]
        
        # Create some low health agents to trigger "threatened" state
        low_health_indices = np.random.choice(batch_size, size=int(batch_size/4), replace=False)
        x_original[low_health_indices, 2] = 0.1 + 0.15 * torch.rand(len(low_health_indices))  # low health
        x_original[low_health_indices, 3] = 1.0  # has_target
        
        x_reconstructed[low_health_indices, 2] = x_original[low_health_indices, 2] + 0.05 * torch.randn(len(low_health_indices))
        x_reconstructed[low_health_indices, 2] = torch.clamp(x_reconstructed[low_health_indices, 2], 0, 1)
        x_reconstructed[low_health_indices, 3] = 1.0  # has_target
        
        print("Creating latent variables...")
        # Create latent variables
        mu = torch.randn(batch_size, latent_dim)
        log_var = torch.randn(batch_size, latent_dim)
        
        print("Initializing combined loss...")
        # Create model output dictionary
        model_output = {
            "x_reconstructed": x_reconstructed,
            "mu": mu,
            "log_var": log_var,
            "compression_loss": torch.tensor(0.1)
        }
        
        print("Computing loss...")
        # Test combined loss
        combined_loss = CombinedLoss(
            recon_loss_weight=1.0,
            kl_loss_weight=0.1,
            semantic_loss_weight=0.5
        )
        
        loss_dict = combined_loss(model_output, x_original)
        print("Loss computation complete.")
        
        print("\nCombined Loss Components:")
        for key, value in loss_dict.items():
            if key != "semantic_breakdown":
                value_display = value.item() if isinstance(value, torch.Tensor) else value
                print(f"  - {key}: {value_display:.4f}")
        
        # Check for semantic breakdown details
        if "semantic_breakdown" in loss_dict and loss_dict["semantic_breakdown"]:
            print("\nSemantic Loss Breakdown:")
            for feature, value in loss_dict["semantic_breakdown"].items():
                print(f"  - {feature}: {value:.4f}")
        
        assert loss_dict["total_loss"] >= 0, "Total loss should be non-negative"
        assert "recon_loss" in loss_dict, "Result should contain reconstruction loss"
        assert "kl_loss" in loss_dict, "Result should contain KL divergence loss"
        assert "semantic_loss" in loss_dict, "Result should contain semantic loss"
        
    except Exception as e:
        print(f"Error in combined loss test: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("\n=== Testing Reconstruction Loss ===")
    test_reconstruction_loss()
    
    print("\n=== Testing KL Divergence Loss ===")
    test_kl_divergence_loss()
    
    print("\n=== Testing Semantic Loss ===")
    test_semantic_loss()
    
    print("\n=== Testing Combined Loss ===")
    test_combined_loss()
    
    print("\nAll tests completed successfully!") 