#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extended Debug Experiment for Loss Function Issues in MeaningVAE

This script provides a more thorough investigation of the zero-valued loss issue:
1. Direct inspection of loss computation and intermediate tensors 
2. Gradient flow analysis
3. Modifications to test various hypotheses
4. Isolated testing of individual loss components
"""

import os
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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


class CustomVAE(torch.nn.Module):
    """Custom VAE with hooks for inspecting internal values."""
    
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
        
        # Store intermediate values
        self._intermediate_values = {}
        
        # Register hooks to track gradients
        self._hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to track gradients during backpropagation."""
        self._grad_values = {}
        
        def hook_fn(name):
            def fn(grad):
                self._grad_values[name] = {
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'min': grad.min().item(),
                    'max': grad.max().item(),
                    'has_nan': torch.isnan(grad).any().item(),
                    'has_inf': torch.isinf(grad).any().item(),
                }
                return grad
            return fn
        
        # Register hooks for encoder parameters
        for name, param in self.encoder.named_parameters():
            hook = param.register_hook(hook_fn(f"encoder.{name}"))
            self._hooks.append(hook)
        
        # Register hooks for decoder parameters
        for name, param in self.decoder.named_parameters():
            hook = param.register_hook(hook_fn(f"decoder.{name}"))
            self._hooks.append(hook)
    
    def remove_hooks(self):
        """Remove hooks to avoid memory leaks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick with debugging information."""
        std = torch.exp(0.5 * log_var)
        
        # Store for debugging
        self._intermediate_values["std"] = std.detach().clone()
        
        if self.training:
            with set_temp_seed(self.seed):
                eps = torch.randn_like(std)
            
            # Store for debugging
            self._intermediate_values["eps"] = eps.detach().clone()
            
            z = mu + eps * std
            return z
        else:
            return mu
    
    def forward(self, x):
        """Forward pass with extra debugging information."""
        # Store input
        self._intermediate_values["input"] = x.detach().clone()
        
        # Encode
        mu, log_var = self.encoder(x)
        
        # Store for debugging
        self._intermediate_values["mu"] = mu.detach().clone()
        self._intermediate_values["log_var"] = log_var.detach().clone()
        
        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)
        
        # Store for debugging
        self._intermediate_values["z"] = z.detach().clone()
        
        # Decode
        x_reconstructed = self.decoder(z)
        
        # Store for debugging
        self._intermediate_values["reconstruction"] = x_reconstructed.detach().clone()
        
        # Return results
        return {
            "reconstruction": x_reconstructed,
            "mu": mu,
            "log_var": log_var,
            "z": z,
        }
    
    def get_intermediate_values(self):
        """Get stored intermediate values."""
        return self._intermediate_values
    
    def get_gradient_info(self):
        """Get gradient information."""
        return self._grad_values


def generate_sample_data(batch_size=32, input_dim=10, seed=42, data_type="normal"):
    """Generate different types of sample data."""
    torch.manual_seed(seed)
    
    if data_type == "normal":
        # Standard normal with mild offset
        return torch.randn(batch_size, input_dim) * 0.1 + 0.5
    elif data_type == "uniform":
        # Uniform between 0 and 1
        return torch.rand(batch_size, input_dim)
    elif data_type == "binary":
        # Random binary data
        return torch.randint(0, 2, (batch_size, input_dim)).float()
    elif data_type == "one_hot":
        # One-hot encoded data
        data = torch.zeros(batch_size, input_dim)
        indices = torch.randint(0, input_dim, (batch_size,))
        for i, idx in enumerate(indices):
            data[i, idx] = 1.0
        return data
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def debug_kl_loss(mu, log_var):
    """Manually debug KL loss computation."""
    print("\nDebugging KL loss computation...")
    
    # Original KL loss formula
    kl_orig = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Step-by-step calculation
    term1 = 1
    term2 = log_var
    term3 = -mu.pow(2)
    term4 = -log_var.exp()
    
    sum_terms = term1 + term2 + term3 + term4
    kl_step = -0.5 * torch.sum(sum_terms)
    
    # Per-dimension values for inspection
    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    
    print(f"KL loss (original): {kl_orig.item():.10f}")
    print(f"KL loss (step by step): {kl_step.item():.10f}")
    print(f"Per-dimension KL (first 5 dimensions of first sample):")
    for i in range(min(5, kl_per_dim.size(1))):
        print(f"  Dim {i}: {kl_per_dim[0, i].item():.10f}")
    
    print(f"Term stats:")
    print(f"  Term 1 (constant): {term1}")
    print(f"  Term 2 (log_var): mean={term2.mean().item():.6f}, std={term2.std().item():.6f}")
    print(f"  Term 3 (-mu²): mean={term3.mean().item():.6f}, std={term3.std().item():.6f}")
    print(f"  Term 4 (-exp(log_var)): mean={term4.mean().item():.6f}, std={term4.std().item():.6f}")
    print(f"  Sum: mean={sum_terms.mean().item():.6f}, std={sum_terms.std().item():.6f}")
    
    return kl_orig.item()


def debug_recon_loss(x_reconstructed, x_original):
    """Manually debug reconstruction loss computation."""
    print("\nDebugging reconstruction loss computation...")
    
    # Original MSE calculation
    mse_loss = F.mse_loss(x_reconstructed, x_original, reduction="sum")
    
    # Manual calculation
    sq_diff = (x_reconstructed - x_original).pow(2)
    manual_mse = sq_diff.sum()
    
    # Per-feature MSE for inspection
    mse_per_feature = sq_diff.mean(dim=0)
    
    print(f"MSE loss (original): {mse_loss.item():.10f}")
    print(f"MSE loss (manual): {manual_mse.item():.10f}")
    print(f"Per-feature MSE (first 5 features):")
    for i in range(min(5, mse_per_feature.size(0))):
        print(f"  Feature {i}: {mse_per_feature[i].item():.10f}")
    
    print(f"Squared difference stats:")
    print(f"  Mean: {sq_diff.mean().item():.10f}")
    print(f"  Std: {sq_diff.std().item():.10f}")
    print(f"  Min: {sq_diff.min().item():.10f}")
    print(f"  Max: {sq_diff.max().item():.10f}")
    
    # Check if reconstructions are identical to inputs
    if torch.allclose(x_reconstructed, x_original, rtol=1e-5, atol=1e-7):
        print("WARNING: Reconstruction is effectively identical to original input!")
        
        # Detailed comparison
        print("Detailed comparison (first sample, first 5 features):")
        for i in range(min(5, x_original.size(1))):
            orig = x_original[0, i].item()
            recon = x_reconstructed[0, i].item()
            diff = orig - recon
            print(f"  Feature {i}: original={orig:.10f}, recon={recon:.10f}, diff={diff:.10f}")
    
    return mse_loss.item()


def inspect_model_parameters(model):
    """Inspect model parameters for problems."""
    print("\nInspecting model parameters...")
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            stats = {
                "mean": param.data.mean().item(),
                "std": param.data.std().item(),
                "min": param.data.min().item(),
                "max": param.data.max().item(),
                "has_nan": torch.isnan(param.data).any().item(),
                "has_inf": torch.isinf(param.data).any().item(),
            }
            
            # Flag suspicious values
            flags = []
            if stats["has_nan"]:
                flags.append("HAS_NAN")
            if stats["has_inf"]:
                flags.append("HAS_INF")
            if abs(stats["mean"]) < 1e-8 and stats["std"] < 1e-8:
                flags.append("COLLAPSED")
            if stats["std"] > 100:
                flags.append("HIGH_VAR")
            
            print(f"Parameter {name}:")
            print(f"  Shape: {param.data.shape}")
            print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
            print(f"  Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
            
            if flags:
                print(f"  FLAGS: {', '.join(flags)}")
            print()


def run_extended_debug():
    """Run the extended debug experiment."""
    print("Starting extended debug experiment for loss functions...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set dimensions
    input_dim = 10
    latent_dim = 4
    batch_size = 32
    
    # Create results directory
    results_dir = project_root / "results" / "debug_loss"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Test with different data types
    data_types = ["normal", "uniform", "binary", "one_hot"]
    
    for data_type in data_types:
        print(f"\n{'='*80}")
        print(f"Testing with {data_type} data")
        print(f"{'='*80}")
        
        # Generate data
        data = generate_sample_data(batch_size, input_dim, data_type=data_type)
        data = data.to(device)
        
        # Create model
        model = CustomVAE(input_dim, latent_dim)
        model = model.to(device)
        
        # Initial parameter inspection
        print("\nInitial model parameters:")
        inspect_model_parameters(model)
        
        # Create loss functions for testing
        recon_loss_fn = ReconstructionLoss(loss_type="mse")
        kl_loss_fn = KLDivergenceLoss()
        combined_loss_fn = CombinedLoss(
            recon_loss_weight=1.0,
            kl_loss_weight=1.0,
            semantic_loss_weight=0.0,  # Disable semantic loss for debugging
        )
        
        # Test untrained model once
        print("\nTesting untrained model...")
        model.eval()
        with torch.no_grad():
            output = model(data)
            
            # Get intermediate values
            values = model.get_intermediate_values()
            
            # Print key statistics
            print("\nUntrained model stats:")
            print(f"  mu: mean={values['mu'].mean().item():.6f}, std={values['mu'].std().item():.6f}")
            print(f"  log_var: mean={values['log_var'].mean().item():.6f}, std={values['log_var'].std().item():.6f}")
            print(f"  std: mean={values['std'].mean().item():.6f}, std={values['std'].std().item():.6f}")
            print(f"  z: mean={values['z'].mean().item():.6f}, std={values['z'].std().item():.6f}")
            print(f"  reconstruction: mean={values['reconstruction'].mean().item():.6f}, std={values['reconstruction'].std().item():.6f}")
            print(f"  input: mean={values['input'].mean().item():.6f}, std={values['input'].std().item():.6f}")
            
            # Calculate initial losses directly
            recon_loss = recon_loss_fn(output["reconstruction"], data)
            kl_loss = kl_loss_fn(output["mu"], output["log_var"])
            
            print(f"\nInitial losses:")
            print(f"  Reconstruction loss: {recon_loss.item():.10f}")
            print(f"  KL divergence loss: {kl_loss.item():.10f}")
            
            # Manually debug each loss
            debug_kl_loss(output["mu"], output["log_var"])
            debug_recon_loss(output["reconstruction"], data)
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop with detailed diagnostics
        print("\nStarting training with detailed diagnostics...")
        
        losses = {"reconstruction": [], "kl": [], "total": []}
        
        for epoch in range(10):
            model.train()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss_dict = combined_loss_fn(output, data)
            loss = loss_dict["loss"]
            
            # Print loss components
            print(f"\nEpoch {epoch+1}/10:")
            print(f"  Total loss: {loss.item():.10f}")
            print(f"  Reconstruction loss: {loss_dict['reconstruction_loss'].item():.10f}")
            print(f"  KL loss: {loss_dict['kl_loss'].item():.10f}")
            
            # Check for zero losses
            if loss_dict["reconstruction_loss"] < 1e-8 or loss_dict["kl_loss"] < 1e-8:
                print("\nWARNING: Detected potentially zero-valued loss components!")
                
                # Get intermediate values for inspection
                values = model.get_intermediate_values()
                
                # Analyze specific tensors that might be causing issues
                if loss_dict["reconstruction_loss"] < 1e-8:
                    print("\nAnalyzing reconstruction loss issue:")
                    debug_recon_loss(output["reconstruction"], data)
                
                if loss_dict["kl_loss"] < 1e-8:
                    print("\nAnalyzing KL loss issue:")
                    debug_kl_loss(output["mu"], output["log_var"])
                    
                    # Try fixing the KL loss calculation
                    print("\nAttempting to manually calculate and backpropagate KL loss...")
                    corrected_kl = 0.5 * torch.mean(
                        torch.sum(mu.pow(2) + log_var.exp() - log_var - 1, dim=1)
                    )
                    print(f"Corrected KL loss: {corrected_kl.item():.10f}")
            
            # Backward pass
            loss.backward()
            
            # Get gradient information
            grad_info = model.get_gradient_info()
            
            # Check for gradient issues
            print("\nGradient information:")
            for param_name, grad_stats in grad_info.items():
                # Only print suspicious gradients
                if grad_stats["has_nan"] or grad_stats["has_inf"] or abs(grad_stats["mean"]) < 1e-10:
                    print(f"  {param_name}:")
                    print(f"    Mean: {grad_stats['mean']:.10f}, Std: {grad_stats['std']:.10f}")
                    print(f"    Min: {grad_stats['min']:.10f}, Max: {grad_stats['max']:.10f}")
                    if grad_stats["has_nan"]:
                        print("    WARNING: Contains NaN gradients!")
                    if grad_stats["has_inf"]:
                        print("    WARNING: Contains Inf gradients!")
                    if abs(grad_stats["mean"]) < 1e-10 and grad_stats["std"] < 1e-10:
                        print("    WARNING: Gradient effectively zero!")
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Step optimizer
            optimizer.step()
            
            # Store losses for plotting
            losses["reconstruction"].append(loss_dict["reconstruction_loss"].item())
            losses["kl"].append(loss_dict["kl_loss"].item())
            losses["total"].append(loss.item())
            
            # Every few epochs, check parameters
            if epoch % 3 == 0:
                inspect_model_parameters(model)
        
        # Remove hooks to prevent memory leaks
        model.remove_hooks()
        
        # Visualize loss trends
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(losses["reconstruction"], 'b-')
        plt.title(f"Reconstruction Loss ({data_type} data)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log')
        
        plt.subplot(1, 3, 2)
        plt.plot(losses["kl"], 'r-')
        plt.title(f"KL Divergence Loss ({data_type} data)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log')
        
        plt.subplot(1, 3, 3)
        plt.plot(losses["total"], 'g-')
        plt.title(f"Total Loss ({data_type} data)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(results_dir / f"loss_trends_{data_type}.png")
        plt.close()
        
        # Test with beta-annealing (gradual KL term increase)
        print("\nTesting with beta-annealing (KL weight scheduled increase)...")
        
        # Reset model
        model = CustomVAE(input_dim, latent_dim)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        annealing_losses = {"reconstruction": [], "kl": [], "total": [], "kl_weight": []}
        
        # Schedule KL weight from 0.0001 to 1.0 over epochs
        epochs = 30
        for epoch in range(epochs):
            model.train()
            
            # Calculate beta (KL weight) using a sigmoid schedule
            beta = sigmoid_schedule(epoch, epochs, k=5)  # k controls steepness
            annealing_losses["kl_weight"].append(beta)
            
            # Create loss function with current beta
            loss_fn = CombinedLoss(
                recon_loss_weight=1.0,
                kl_loss_weight=beta,
                semantic_loss_weight=0.0,
            )
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss_dict = loss_fn(output, data)
            loss = loss_dict["loss"]
            
            # Store losses
            annealing_losses["reconstruction"].append(loss_dict["reconstruction_loss"].item())
            annealing_losses["kl"].append(loss_dict["kl_loss"].item())
            annealing_losses["total"].append(loss.item())
            
            # Print progress
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Beta={beta:.6f}")
                print(f"  Recon loss: {loss_dict['reconstruction_loss'].item():.6f}")
                print(f"  KL loss: {loss_dict['kl_loss'].item():.6f}")
                print(f"  Total loss: {loss.item():.6f}")
            
            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()
        
        # Remove hooks
        model.remove_hooks()
        
        # Visualize annealing results
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(annealing_losses["reconstruction"], 'b-')
        plt.title(f"Reconstruction Loss ({data_type} data)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log')
        
        plt.subplot(2, 2, 2)
        plt.plot(annealing_losses["kl"], 'r-')
        plt.title(f"KL Divergence Loss ({data_type} data)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log')
        
        plt.subplot(2, 2, 3)
        plt.plot(annealing_losses["total"], 'g-')
        plt.title(f"Total Loss ({data_type} data)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log')
        
        plt.subplot(2, 2, 4)
        plt.plot(annealing_losses["kl_weight"], 'm-')
        plt.title(f"KL Weight (Beta)")
        plt.xlabel("Epoch")
        plt.ylabel("Weight")
        
        plt.tight_layout()
        plt.savefig(results_dir / f"annealing_results_{data_type}.png")
        plt.close()
    
    print(f"\nDebug experiment complete. Results saved to {results_dir}")


def sigmoid_schedule(epoch, total_epochs, k=5, min_value=0.0001, max_value=1.0):
    """Create a sigmoid schedule for beta annealing."""
    x = epoch / total_epochs
    scaled_x = k * (2 * x - 1)  # Scale to be centered at 0.5
    sigmoid = 1 / (1 + np.exp(-scaled_x))
    return min_value + (max_value - min_value) * sigmoid


if __name__ == "__main__":
    # For importing F.mse_loss in debug_recon_loss
    import torch.nn.functional as F
    
    run_extended_debug() 