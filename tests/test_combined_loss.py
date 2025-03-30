#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone test for the combined loss function.
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

from meaning_transform.src.loss import CombinedLoss


def test_combined_loss():
    """Test the combined loss function."""
    print("Starting combined loss test...")
    batch_size = 32
    input_dim = 50
    latent_dim = 16
    
    # Create sample tensors
    x_original = torch.zeros(batch_size, input_dim)
    x_reconstructed = torch.zeros(batch_size, input_dim)
    
    try:
        print("Creating test data...")
        # Fill with test data
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
        
        print("Setting up roles...")
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
        
        print("Creating threatened state...")
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
        
        # Create model output dictionary
        model_output = {
            "reconstruction": x_reconstructed,
            "mu": mu,
            "log_var": log_var,
            "compression_loss": torch.tensor(0.1)
        }
        
        print("Initializing combined loss...")
        # Test combined loss
        combined_loss = CombinedLoss(
            recon_loss_weight=1.0,
            kl_loss_weight=0.1,
            semantic_loss_weight=0.5
        )
        
        print("Computing loss...")
        loss_dict = combined_loss(model_output, x_original)
        print("Loss computation complete.")
        
        # Get all the values safely
        total_loss = loss_dict.get("total_loss", None)
        recon_loss = loss_dict.get("recon_loss", None)
        kl_loss = loss_dict.get("kl_loss", None)
        semantic_loss = loss_dict.get("semantic_loss", None)
        compression_loss = loss_dict.get("compression_loss", None)
        semantic_breakdown = loss_dict.get("semantic_breakdown", {})
        
        print("\nCombined Loss Components:")
        
        # Print each component explicitly
        if total_loss is not None:
            print(f"  - total_loss: {total_loss.item():.4f}")
            
        if recon_loss is not None:
            print(f"  - recon_loss: {recon_loss.item():.4f}")
            
        if kl_loss is not None:
            print(f"  - kl_loss: {kl_loss.item():.4f}")
            
        if semantic_loss is not None:
            print(f"  - semantic_loss: {semantic_loss.item():.4f}")
            
        if compression_loss is not None:
            value = compression_loss.item() if isinstance(compression_loss, torch.Tensor) else compression_loss
            print(f"  - compression_loss: {value:.4f}")
        
        # Check for semantic breakdown details
        if semantic_breakdown:
            print("\nSemantic Loss Breakdown:")
            for feature, value in semantic_breakdown.items():
                print(f"  - {feature}: {value:.4f}")
        
        # Assertions to validate the loss components
        assert "total_loss" in loss_dict, "Result should contain total loss"
        assert "recon_loss" in loss_dict, "Result should contain reconstruction loss"
        assert "kl_loss" in loss_dict, "Result should contain KL divergence loss"
        assert "semantic_loss" in loss_dict, "Result should contain semantic loss"
        
        print("Combined loss test completed successfully!")
        
    except Exception as e:
        print(f"Error in combined loss test: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_combined_loss() 