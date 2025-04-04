#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug script focused specifically on spatial feature preservation in the model.

This script will:
1. Generate test data with varying spatial locations
2. Encode and decode through the model
3. Compare original and reconstructed spatial features
4. Visualize spatial reconstructions
5. Log detailed metrics about spatial feature preservation
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Create results directory and configure logging
results_dir = Path("results/spatial_debug")
results_dir.mkdir(exist_ok=True, parents=True)
log_file = results_dir / f"spatial_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(log_file))
    ]
)

# Import model and metrics
from meaning_transform.src.models.meaning_vae import MeaningVAE
from meaning_transform.src.metrics import SemanticMetrics
from meaning_transform.src.loss import SemanticLoss, FeatureWeightedLoss

def generate_spatial_test_data(num_samples=100, input_dim=15):
    """Generate test data with varying spatial locations."""
    # Create a tensor of zeros
    data = torch.zeros(num_samples, input_dim)
    
    # Set spatial coordinates (first 3 dimensions) to cover a grid of values
    # Create a grid in 3D space
    grid_size = int(np.ceil(num_samples**(1/3)))
    
    # Generate grid coordinates in [0.1, 0.9] range (normalized)
    x_coords = torch.linspace(0.1, 0.9, grid_size)
    y_coords = torch.linspace(0.1, 0.9, grid_size)
    z_coords = torch.linspace(0.1, 0.9, grid_size)
    
    # Assign coordinates to samples
    sample_idx = 0
    for i in range(min(grid_size, num_samples)):
        for j in range(min(grid_size, num_samples)):
            for k in range(min(grid_size, num_samples)):
                if sample_idx >= num_samples:
                    break
                data[sample_idx, 0] = x_coords[i]
                data[sample_idx, 1] = y_coords[j]
                data[sample_idx, 2] = z_coords[k]
                sample_idx += 1
                if sample_idx >= num_samples:
                    break
            if sample_idx >= num_samples:
                break
    
    # Set other features to reasonable values
    # Health and energy (indices 3-4)
    data[:, 3] = 0.8  # health at 80%
    data[:, 4] = 0.7  # energy at 70%
    
    # Resource level (index 5)
    data[:, 5] = 0.5  # resource at 50%
    
    # Current health (index 6)
    data[:, 6] = 0.75  # current health at 75%
    
    # Is defending (index 7)
    data[:, 7] = 0.0  # not defending
    
    # Other flags (indices 8-9)
    data[:, 8] = 0.0
    data[:, 9] = 0.0
    
    # Role (one-hot encoded in indices 10-14)
    # Set each sample to a random role
    roles = torch.zeros(num_samples, 5)
    random_roles = torch.randint(0, 5, (num_samples,))
    roles[torch.arange(num_samples), random_roles] = 1.0
    data[:, 10:15] = roles
    
    return data

def visualize_spatial_reconstruction(original, reconstructed, title, output_file=None):
    """Visualize original vs. reconstructed spatial coordinates."""
    # Extract spatial coordinates
    original_spatial = original[:, :3].cpu().numpy()
    reconstructed_spatial = reconstructed[:, :3].cpu().numpy()
    
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original points
    ax.scatter(
        original_spatial[:, 0],
        original_spatial[:, 1],
        original_spatial[:, 2],
        c='blue',
        marker='o',
        label='Original'
    )
    
    # Plot reconstructed points
    ax.scatter(
        reconstructed_spatial[:, 0],
        reconstructed_spatial[:, 1],
        reconstructed_spatial[:, 2],
        c='red',
        marker='x',
        label='Reconstructed'
    )
    
    # Add connecting lines
    for i in range(len(original_spatial)):
        ax.plot(
            [original_spatial[i, 0], reconstructed_spatial[i, 0]],
            [original_spatial[i, 1], reconstructed_spatial[i, 1]],
            [original_spatial[i, 2], reconstructed_spatial[i, 2]],
            'gray',
            alpha=0.3
        )
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # Save or show the plot
    if output_file:
        plt.savefig(output_file)
        logging.info(f"Saved visualization to {output_file}")
    else:
        plt.show()
    
    plt.close()

def calculate_spatial_metrics(original, reconstructed):
    """Calculate detailed metrics for spatial feature preservation."""
    # Extract spatial coordinates
    original_spatial = original[:, :3]
    reconstructed_spatial = reconstructed[:, :3]
    
    # Calculate error metrics
    mse = torch.mean((original_spatial - reconstructed_spatial) ** 2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(original_spatial - reconstructed_spatial)).item()
    
    # Calculate per-dimension metrics
    dim_mse = torch.mean((original_spatial - reconstructed_spatial) ** 2, dim=0).tolist()
    dim_mae = torch.mean(torch.abs(original_spatial - reconstructed_spatial), dim=0).tolist()
    
    # Calculate Euclidean distances
    distances = torch.sqrt(torch.sum((original_spatial - reconstructed_spatial) ** 2, dim=1))
    mean_distance = torch.mean(distances).item()
    max_distance = torch.max(distances).item()
    
    # Calculate relative error
    # Avoid division by zero
    epsilon = 1e-8
    relative_error = torch.abs(original_spatial - reconstructed_spatial) / (torch.abs(original_spatial) + epsilon)
    mean_relative_error = torch.mean(relative_error).item()
    
    # Calculate correlation
    def correlation(x, y):
        # Flatten tensors
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        
        # Calculate means
        x_mean = torch.mean(x_flat)
        y_mean = torch.mean(y_flat)
        
        # Calculate covariance and variances
        cov = torch.mean((x_flat - x_mean) * (y_flat - y_mean))
        x_var = torch.mean((x_flat - x_mean) ** 2)
        y_var = torch.mean((y_flat - y_mean) ** 2)
        
        # Calculate correlation
        if x_var == 0 or y_var == 0:
            return 0.0
        return cov / (torch.sqrt(x_var) * torch.sqrt(y_var))
    
    corr = correlation(original_spatial, reconstructed_spatial).item()
    
    # Return all metrics
    return {
        "spatial_mse": mse,
        "spatial_rmse": rmse,
        "spatial_mae": mae,
        "spatial_x_mse": dim_mse[0],
        "spatial_y_mse": dim_mse[1],
        "spatial_z_mse": dim_mse[2],
        "spatial_x_mae": dim_mae[0],
        "spatial_y_mae": dim_mae[1],
        "spatial_z_mae": dim_mae[2],
        "spatial_mean_distance": mean_distance,
        "spatial_max_distance": max_distance,
        "spatial_mean_relative_error": mean_relative_error,
        "spatial_correlation": corr
    }

def debug_spatial_preservation():
    """Main function to debug spatial feature preservation."""
    logging.info("Starting spatial feature preservation debug")
    
    # Configuration
    input_dim = 15
    latent_dim = 32
    compression_level = 1.0
    
    # Generate test data
    logging.info("Generating test data")
    data = generate_spatial_test_data(num_samples=100, input_dim=input_dim)
    logging.info(f"Generated data shape: {data.shape}")
    
    # Log data statistics
    logging.info(f"Data stats: mean={data.mean().item():.6f}, std={data.std().item():.6f}")
    logging.info(f"Spatial stats: mean={data[:, :3].mean().item():.6f}, std={data[:, :3].std().item():.6f}")
    
    # Create a model
    logging.info(f"Creating model with latent_dim={latent_dim}, compression_level={compression_level}")
    model = MeaningVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        compression_level=compression_level,
        spatial_priority=False,  # Disable spatial priority as recommended
    )
    
    # Set to evaluation mode
    model.eval()
    
    # Forward pass through the model
    logging.info("Running forward pass")
    with torch.no_grad():
        output = model(data)
        reconstructed = output["reconstruction"]
    
    # Check if reconstruction has NaN values
    if torch.isnan(reconstructed).any():
        logging.error("NaN values detected in reconstruction!")
    
    # Log reconstruction statistics
    logging.info(f"Reconstruction shape: {reconstructed.shape}")
    logging.info(f"Reconstruction stats: mean={reconstructed.mean().item():.6f}, std={reconstructed.std().item():.6f}")
    logging.info(f"Spatial recon stats: mean={reconstructed[:, :3].mean().item():.6f}, std={reconstructed[:, :3].std().item():.6f}")
    
    # Calculate spatial metrics
    logging.info("Calculating spatial metrics")
    metrics = calculate_spatial_metrics(data, reconstructed)
    
    # Log metrics
    logging.info("Spatial metrics:")
    for key, value in metrics.items():
        logging.info(f"  {key}: {value:.10f}")
    
    # Calculate semantic metrics
    logging.info("Calculating semantic metrics")
    semantic_metrics = SemanticMetrics()
    equivalence_scores = semantic_metrics.compute_equivalence_scores(data, reconstructed)
    
    # Log semantic scores
    logging.info("Semantic equivalence scores:")
    for key, value in equivalence_scores.items():
        logging.info(f"  {key}: {value:.10f}")
    
    # Visualize spatial reconstruction
    logging.info("Creating visualization")
    visualize_spatial_reconstruction(
        data,
        reconstructed,
        f"Spatial Reconstruction (Compression Level {compression_level})",
        output_file=results_dir / f"spatial_reconstruction_{int(compression_level*10)}.png"
    )
    
    # Test with different compression levels
    compression_levels = [0.5, 1.0, 2.0, 5.0]
    results = {}
    
    for level in compression_levels:
        if level == compression_level:
            # Skip if we already calculated this
            results[level] = metrics
            continue
            
        logging.info(f"Testing compression level {level}")
        test_model = MeaningVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            compression_level=level,
            spatial_priority=False,
        )
        test_model.eval()
        
        with torch.no_grad():
            test_output = test_model(data)
            test_recon = test_output["reconstruction"]
            
        test_metrics = calculate_spatial_metrics(data, test_recon)
        results[level] = test_metrics
        
        # Log metrics for this compression level
        logging.info(f"Metrics for compression level {level}:")
        for key, value in test_metrics.items():
            logging.info(f"  {key}: {value:.10f}")
            
        # Visualize
        visualize_spatial_reconstruction(
            data,
            test_recon,
            f"Spatial Reconstruction (Compression Level {level})",
            output_file=results_dir / f"spatial_reconstruction_{int(level*10)}.png"
        )
    
    # Create comparison chart for compression levels
    plt.figure(figsize=(12, 8))
    
    # Extract MSE values for each compression level
    mse_values = [results[level]["spatial_mse"] for level in compression_levels]
    mae_values = [results[level]["spatial_mae"] for level in compression_levels]
    corr_values = [results[level]["spatial_correlation"] for level in compression_levels]
    
    # Create a plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot MSE and MAE on left y-axis
    ax1.plot(compression_levels, mse_values, 'b-o', label='MSE')
    ax1.plot(compression_levels, mae_values, 'g-s', label='MAE')
    ax1.set_xlabel('Compression Level')
    ax1.set_ylabel('Error (MSE/MAE)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Create second y-axis for correlation
    ax2 = ax1.twinx()
    ax2.plot(compression_levels, corr_values, 'r-^', label='Correlation')
    ax2.set_ylabel('Correlation', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.title('Spatial Metrics vs. Compression Level')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save chart
    plt.savefig(results_dir / "compression_comparison.png")
    logging.info(f"Saved comparison chart to {results_dir / 'compression_comparison.png'}")
    plt.close()
    
    logging.info("Spatial debug completed")
    return metrics, equivalence_scores

if __name__ == "__main__":
    try:
        metrics, scores = debug_spatial_preservation()
        print("\nSpatial Metrics Summary:")
        for key, value in metrics.items():
            if key.startswith("spatial_"):
                print(f"  {key}: {value:.10f}")
                
        print("\nSemantic Equivalence Scores:")
        for key, value in scores.items():
            print(f"  {key}: {value:.10f}")
            
        print(f"\nResults and log file saved to: {results_dir}")
    except Exception as e:
        logging.exception("Error in spatial debug")
        print(f"Error: {e}") 