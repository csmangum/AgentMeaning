#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Beta Annealing Debug Experiment for Meaning-Preserving Transformations

This script demonstrates how to use beta annealing to avoid zero-valued loss issues
by gradually increasing the KL weight during training.
"""

import os
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from meaning_transform.src.models.encoder import Encoder
from meaning_transform.src.models.decoder import Decoder
from meaning_transform.src.loss import ReconstructionLoss, KLDivergenceLoss, CombinedLoss, beta_annealing
from meaning_transform.src.models.utils import set_temp_seed


class SimpleVAE(torch.nn.Module):
    """Simple VAE for demonstration purposes."""
    
    def __init__(self, input_dim=10, latent_dim=4, seed=42):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seed = seed
        
        # Create encoder and decoder
        self.encoder = Encoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=[32, 16],
            use_batch_norm=True,
        )
        
        self.decoder = Decoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=[16, 32],
            use_batch_norm=True,
        )
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        
        if self.training:
            with set_temp_seed(self.seed):
                eps = torch.randn_like(std)
            z = mu + eps * std
            return z
        else:
            return mu
    
    def forward(self, x):
        """Forward pass through the VAE."""
        # Encode
        mu, log_var = self.encoder(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x_reconstructed = self.decoder(z)
        
        # Return results
        return {
            "reconstruction": x_reconstructed,
            "mu": mu,
            "log_var": log_var,
            "z": z,
        }


def generate_sample_data(batch_size=32, input_dim=10, seed=42):
    """Generate random sample data."""
    torch.manual_seed(seed)
    # Create random data with reasonable values (not too extreme)
    return torch.randn(batch_size, input_dim) * 0.1 + 0.5


def run_beta_annealing_experiment():
    """Run the beta annealing experiment."""
    print("Starting beta annealing experiment for loss functions...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set dimensions
    input_dim = 10
    latent_dim = 4
    batch_size = 32
    
    # Generate data
    data = generate_sample_data(batch_size, input_dim)
    data = data.to(device)
    
    # Create model
    model = SimpleVAE(input_dim, latent_dim)
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Number of epochs
    epochs = 30
    
    # Create result storage
    results = {
        "epoch": [],
        "beta": [],
        "recon_loss": [],
        "kl_loss": [],
        "total_loss": [],
    }
    
    # Run training with beta annealing
    print("\nTraining with beta annealing:")
    
    for epoch in range(epochs):
        model.train()
        
        # Calculate beta using sigmoid annealing
        beta = beta_annealing(
            epoch=epoch,
            max_epochs=epochs,
            min_beta=0.0001,
            max_beta=1.0,
            schedule_type="sigmoid"
        )
        
        # Create loss function with current beta
        loss_fn = CombinedLoss(
            recon_loss_weight=1.0,
            kl_loss_weight=beta,
            semantic_loss_weight=0.0,  # Disable semantic loss for simplicity
        )
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Calculate loss
        loss_dict = loss_fn(output, data)
        loss = loss_dict["loss"]
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent instability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, Beta={beta:.4f}, "
              f"Recon Loss={loss_dict['reconstruction_loss'].item():.6f}, "
              f"KL Loss={loss_dict['kl_loss'].item():.6f}, "
              f"Total Loss={loss.item():.6f}")
        
        # Store results
        results["epoch"].append(epoch)
        results["beta"].append(beta)
        results["recon_loss"].append(loss_dict["reconstruction_loss"].item())
        results["kl_loss"].append(loss_dict["kl_loss"].item())
        results["total_loss"].append(loss.item())
    
    # Create results directory
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot beta values
    plt.subplot(2, 2, 1)
    plt.plot(results["epoch"], results["beta"], 'r-')
    plt.title("Beta (KL Weight) Annealing")
    plt.xlabel("Epoch")
    plt.ylabel("Beta Value")
    
    # Plot reconstruction loss
    plt.subplot(2, 2, 2)
    plt.plot(results["epoch"], results["recon_loss"], 'b-')
    plt.title("Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    
    # Plot KL loss
    plt.subplot(2, 2, 3)
    plt.plot(results["epoch"], results["kl_loss"], 'g-')
    plt.title("KL Divergence Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    
    # Plot total loss
    plt.subplot(2, 2, 4)
    plt.plot(results["epoch"], results["total_loss"], 'm-')
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    
    plt.tight_layout()
    plt.savefig(results_dir / "beta_annealing_results.png")
    print(f"Results saved to {results_dir / 'beta_annealing_results.png'}")
    plt.close()
    
    return results


if __name__ == "__main__":
    run_beta_annealing_experiment() 