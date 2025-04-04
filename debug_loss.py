#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug experiment for loss function issues in MeaningVAE model

This script provides a minimal experiment to diagnose why the reconstruction 
and KL losses are zero-valued during training.
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
from meaning_transform.src.loss import ReconstructionLoss, KLDivergenceLoss, CombinedLoss
from meaning_transform.src.models.utils import set_temp_seed


class MinimalVAE(torch.nn.Module):
    """Minimal VAE for debugging loss functions."""
    
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


def train_one_epoch(model, data, optimizer, loss_fn, verbose=True):
    """Train the model for one epoch."""
    model.train()
    
    # Process entire batch
    optimizer.zero_grad()
    
    # Forward pass
    output = model(data)
    
    # Calculate loss
    loss_dict = loss_fn(output, data)
    loss = loss_dict["loss"]
    
    if verbose:
        print(f"Loss components:")
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.8f}")
            else:
                print(f"  {key}: {value}")
    
    # Check for zero-valued losses
    if loss_dict["reconstruction_loss"] < 1e-8 or loss_dict["kl_loss"] < 1e-8:
        print("\nWARNING: Detected potentially zero-valued loss components!")
        print(f"  Reconstruction: {loss_dict['reconstruction_loss'].item():.10f}")
        print(f"  KL Divergence: {loss_dict['kl_loss'].item():.10f}")
        print("\nDiagnostic information:")
        
        # Check mu and log_var
        mu = output["mu"]
        log_var = output["log_var"]
        print(f"  mu stats: mean={mu.mean().item():.6f}, std={mu.std().item():.6f}")
        print(f"  log_var stats: mean={log_var.mean().item():.6f}, std={log_var.std().item():.6f}")
        
        # Check reconstructions vs original
        recon = output["reconstruction"]
        print(f"  Reconstruction stats: mean={recon.mean().item():.6f}, std={recon.std().item():.6f}")
        print(f"  Original stats: mean={data.mean().item():.6f}, std={data.std().item():.6f}")
        
        # Check if reconstruction is identical to input
        if torch.allclose(recon, data, rtol=1e-3, atol=1e-3):
            print("  ISSUE: Reconstruction is suspiciously close to original input!")
            
        # Check if latent representation is collapsed
        if torch.allclose(mu, torch.zeros_like(mu), rtol=1e-3, atol=1e-3):
            print("  ISSUE: mu is suspiciously close to zero (collapsed encoding)!")
            
        # Check if loss function weights are set properly
        print(f"  Loss weights: recon={loss_fn.recon_loss_weight}, kl={loss_fn.kl_loss_weight}")
    
    # Backward pass
    loss.backward()
    
    # Optional: Print gradient norms
    total_grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_grad_norm += param_norm.item() ** 2
            if verbose and param_norm > 1.0:
                print(f"  Gradient norm for {name}: {param_norm.item():.4f}")
    
    total_grad_norm = total_grad_norm ** 0.5
    if verbose:
        print(f"  Total gradient norm: {total_grad_norm:.4f}")
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss_dict


def run_debug_experiment():
    """Run the debug experiment."""
    print("Starting debug experiment for loss functions...")
    
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
    model = MinimalVAE(input_dim, latent_dim)
    model = model.to(device)
    
    # Create loss function with different weight variations
    loss_configs = [
        {"name": "Default", "recon": 1.0, "kl": 0.1, "semantic": 0.5},
        {"name": "Higher KL", "recon": 1.0, "kl": 1.0, "semantic": 0.5},
        {"name": "No Semantic", "recon": 1.0, "kl": 0.1, "semantic": 0.0},
        {"name": "Balanced", "recon": 1.0, "kl": 1.0, "semantic": 1.0},
    ]
    
    # History for plotting
    history = {config["name"]: {"recon": [], "kl": [], "total": []} for config in loss_configs}
    
    # Run experiment with each loss configuration
    for config in loss_configs:
        print(f"\n{'='*80}")
        print(f"Testing loss configuration: {config['name']}")
        print(f"  Weights: recon={config['recon']}, kl={config['kl']}, semantic={config['semantic']}")
        print(f"{'='*80}")
        
        # Create loss function
        loss_fn = CombinedLoss(
            recon_loss_weight=config["recon"],
            kl_loss_weight=config["kl"],
            semantic_loss_weight=config["semantic"],
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train for a few epochs
        for epoch in range(10):
            print(f"\nEpoch {epoch+1}/10")
            loss_dict = train_one_epoch(model, data, optimizer, loss_fn)
            
            # Store history
            history[config["name"]]["recon"].append(loss_dict["reconstruction_loss"].item())
            history[config["name"]]["kl"].append(loss_dict["kl_loss"].item())
            history[config["name"]]["total"].append(loss_dict["loss"].item())
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot reconstruction loss
    plt.subplot(1, 3, 1)
    for config_name, values in history.items():
        plt.plot(values["recon"], label=config_name)
    plt.title("Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.legend()
    
    # Plot KL loss
    plt.subplot(1, 3, 2)
    for config_name, values in history.items():
        plt.plot(values["kl"], label=config_name)
    plt.title("KL Divergence Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.legend()
    
    # Plot total loss
    plt.subplot(1, 3, 3)
    for config_name, values in history.items():
        plt.plot(values["total"], label=config_name)
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.legend()
    
    # Save the figure
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / "loss_debug_results.png")
    print(f"Results saved to {results_dir / 'loss_debug_results.png'}")
    plt.close()


if __name__ == "__main__":
    run_debug_experiment() 