#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the Adaptive Bottleneck implementation.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os

# Add the project root to the path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from meaning_transform.src.adaptive_model import AdaptiveEntropyBottleneck, AdaptiveMeaningVAE, FeatureGroupedVAE
from meaning_transform.src.model import EntropyBottleneck, MeaningVAE

def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())

def test_parameter_count():
    """Test that the parameter count changes with compression level for adaptive models."""
    print("Testing parameter count vs compression level...")
    
    input_dim = 128
    latent_dim = 32
    compression_levels = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    # Compare standard vs adaptive bottleneck
    standard_params = []
    adaptive_params = []
    effective_dims = []
    
    for level in compression_levels:
        # Standard bottleneck
        standard = EntropyBottleneck(latent_dim, compression_level=level)
        standard_count = count_parameters(standard)
        standard_params.append(standard_count)
        
        # Adaptive bottleneck
        adaptive = AdaptiveEntropyBottleneck(latent_dim, compression_level=level)
        adaptive_count = count_parameters(adaptive)
        adaptive_params.append(adaptive_count)
        effective_dims.append(adaptive.effective_dim)
        
        print(f"Compression level: {level:.1f}")
        print(f"  Standard bottleneck parameters: {standard_count}")
        print(f"  Adaptive bottleneck parameters: {adaptive_count}")
        print(f"  Effective dimension: {adaptive.effective_dim}")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.plot(compression_levels, standard_params, 'o-', label='Standard Bottleneck')
    plt.plot(compression_levels, adaptive_params, 's-', label='Adaptive Bottleneck')
    plt.xlabel('Compression Level')
    plt.ylabel('Parameter Count')
    plt.title('Parameter Count vs Compression Level')
    plt.grid(True)
    plt.legend()
    
    # Create output directory if it doesn't exist
    output_dir = Path('meaning_transform/tests/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the plot
    plt.savefig(output_dir / 'bottleneck_parameter_count.png')
    
    # Also plot effective dimension
    plt.figure(figsize=(10, 6))
    plt.plot(compression_levels, effective_dims, 'o-')
    plt.xlabel('Compression Level')
    plt.ylabel('Effective Dimension')
    plt.title('Effective Dimension vs Compression Level')
    plt.grid(True)
    plt.savefig(output_dir / 'effective_dimension.png')
    
    # Plot log scale for better visibility
    plt.figure(figsize=(10, 6))
    plt.plot(compression_levels, effective_dims, 'o-')
    plt.xscale('log')
    plt.xlabel('Compression Level (log scale)')
    plt.ylabel('Effective Dimension')
    plt.title('Effective Dimension vs Compression Level (Log Scale)')
    plt.grid(True)
    plt.savefig(output_dir / 'effective_dimension_log.png')
    
    return standard_params, adaptive_params, effective_dims

def test_full_vae_models():
    """Test full VAE models with standard vs adaptive compression."""
    print("\nTesting full VAE models...")
    
    input_dim = 128
    latent_dim = 32
    compression_levels = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    # Compare standard vs adaptive VAE
    standard_params = []
    adaptive_params = []
    
    for level in compression_levels:
        # Standard VAE
        standard = MeaningVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            compression_type="entropy",
            compression_level=level
        )
        standard_count = count_parameters(standard)
        standard_params.append(standard_count)
        
        # Adaptive VAE
        adaptive = AdaptiveMeaningVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            compression_level=level
        )
        adaptive_count = count_parameters(adaptive)
        adaptive_params.append(adaptive_count)
        
        print(f"Compression level: {level:.1f}")
        print(f"  Standard VAE parameters: {standard_count}")
        print(f"  Adaptive VAE parameters: {adaptive_count}")
        print(f"  Parameter reduction: {100 * (standard_count - adaptive_count) / standard_count:.2f}%")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.plot(compression_levels, standard_params, 'o-', label='Standard VAE')
    plt.plot(compression_levels, adaptive_params, 's-', label='Adaptive VAE')
    plt.xlabel('Compression Level')
    plt.ylabel('Parameter Count')
    plt.title('VAE Parameter Count vs Compression Level')
    plt.grid(True)
    plt.legend()
    
    output_dir = Path('meaning_transform/tests/results')
    plt.savefig(output_dir / 'vae_parameter_count.png')
    
    return standard_params, adaptive_params

def test_feature_grouped_vae():
    """Test the feature-grouped VAE model with different compression rates for different features."""
    print("\nTesting Feature-Grouped VAE...")
    
    input_dim = 128
    latent_dim = 32
    base_compression = 1.0
    
    # Define feature groups based on importance findings
    feature_groups = {
        "spatial": (0, 42, 0.5),             # High importance - low compression
        "resources": (42, 85, 2.0),          # Medium importance - medium compression
        "other": (85, input_dim, 5.0)        # Low importance - high compression
    }
    
    # Create model
    model = FeatureGroupedVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        feature_groups=feature_groups,
        base_compression_level=base_compression
    )
    
    # Get model analysis
    analysis = model.get_feature_group_analysis()
    
    # Print model summary
    print(f"Feature-Grouped VAE with base compression {base_compression}:")
    print(f"  Total parameters: {count_parameters(model)}")
    
    # Print group details
    print("\nFeature Group Analysis:")
    print(f"{'Group':<10} {'Features':<10} {'Latent Dim':<12} {'Effective Dim':<14} {'Compression':<12}")
    print("-" * 65)
    
    for name, metrics in analysis.items():
        print(f"{name:<10} {metrics['feature_count']:<10d} {metrics['latent_dim']:<12d} "
              f"{metrics['effective_dim']:<14d} {metrics['overall_compression']:<12.2f}")
    
    # Get compression rates
    compression_rates = model.get_compression_rate()
    print("\nCompression Rates:")
    for group, rate in compression_rates.items():
        print(f"  {group}: {rate:.2f}x")
    
    # Visualize the group dimensions
    plt.figure(figsize=(10, 6))
    groups = list(analysis.keys())
    
    # Create data for plotting
    feature_counts = [analysis[g]["feature_count"] for g in groups]
    latent_dims = [analysis[g]["latent_dim"] for g in groups]
    effective_dims = [analysis[g]["effective_dim"] for g in groups]
    
    # Set up bar positions
    x = np.arange(len(groups))
    width = 0.25
    
    # Plot
    plt.bar(x - width, feature_counts, width=width, label='Input Features')
    plt.bar(x, latent_dims, width=width, label='Latent Dimensions')
    plt.bar(x + width, effective_dims, width=width, label='Effective Dimensions')
    
    plt.xlabel('Feature Groups')
    plt.ylabel('Dimensions')
    plt.title('Feature-Grouped VAE Dimension Distribution')
    plt.xticks(x, groups)
    plt.legend()
    
    output_dir = Path('meaning_transform/tests/results')
    plt.savefig(output_dir / 'feature_grouped_dimensions.png')
    
    return model, analysis

def test_forward_pass():
    """Test the forward pass of the adaptive models with a random input tensor."""
    print("\nTesting forward pass...")
    
    input_dim = 128
    latent_dim = 32
    batch_size = 16
    
    # Create random input
    x = torch.randn(batch_size, input_dim)
    
    # Create models with different compression levels
    standard_vae = MeaningVAE(
        input_dim=input_dim, 
        latent_dim=latent_dim,
        compression_level=1.0
    )
    
    adaptive_vae = AdaptiveMeaningVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        compression_level=1.0
    )
    
    # Define feature groups
    feature_groups = {
        "spatial": (0, 42, 0.5), 
        "resources": (42, 85, 2.0),
        "other": (85, input_dim, 5.0)
    }
    
    grouped_vae = FeatureGroupedVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        feature_groups=feature_groups
    )
    
    # Forward pass
    with torch.no_grad():
        standard_output = standard_vae(x)
        adaptive_output = adaptive_vae(x)
        grouped_output = grouped_vae(x)
    
    # Check outputs
    print(f"Standard VAE output shapes:")
    print(f"  x_reconstructed: {standard_output['x_reconstructed'].shape}")
    print(f"  z_compressed: {standard_output['z_compressed'].shape}")
    
    print(f"\nAdaptive VAE output shapes:")
    print(f"  x_reconstructed: {adaptive_output['x_reconstructed'].shape}")
    print(f"  z_compressed: {adaptive_output['z_compressed'].shape}")
    print(f"  Effective dimension: {adaptive_output['effective_dim']}")
    
    print(f"\nFeature-Grouped VAE output shapes:")
    print(f"  x_reconstructed: {grouped_output['x_reconstructed'].shape}")
    print(f"  z_compressed: {grouped_output['z_compressed'].shape}")
    
    # Check that all models produce outputs of the expected shape
    assert standard_output['x_reconstructed'].shape == (batch_size, input_dim)
    assert adaptive_output['x_reconstructed'].shape == (batch_size, input_dim)
    assert grouped_output['x_reconstructed'].shape == (batch_size, input_dim)
    
    print("All forward passes successful!")
    
    return standard_output, adaptive_output, grouped_output

def main():
    """Run all tests."""
    print("Testing Adaptive Bottleneck Implementation")
    print("=" * 40)
    
    # Parameter count test
    std_params, adaptive_params, effective_dims = test_parameter_count()
    
    # Full VAE model test
    vae_std_params, vae_adaptive_params = test_full_vae_models()
    
    # Feature-grouped VAE test
    grouped_model, analysis = test_feature_grouped_vae()
    
    # Forward pass test
    std_output, adaptive_output, grouped_output = test_forward_pass()
    
    print("\nAll tests completed successfully!")
    print("=" * 40)
    print(f"Check results in meaning_transform/tests/results/")

if __name__ == "__main__":
    main() 