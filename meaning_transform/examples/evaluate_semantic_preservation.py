#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script for evaluating semantic preservation in a trained model.

This script demonstrates:
1. Loading a trained model
2. Evaluating semantic preservation metrics
3. Visualizing semantic drift
4. Finding optimal compression threshold
"""

import os
import sys
import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MeaningVAE
from src.data import AgentStateDataset, generate_agent_states
from src.metrics import (
    SemanticMetrics, 
    DriftTracker, 
    CompressionThresholdFinder,
    generate_t_sne_visualization
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate semantic preservation in a trained model")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/semantic_evaluation",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=100,
        help="Number of agent states to evaluate"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--compression_levels",
        type=str,
        default="4.0,3.5,3.0,2.5,2.0,1.5,1.0,0.5",
        help="Comma-separated list of compression levels to evaluate"
    )
    
    return parser.parse_args()


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from {model_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model parameters
    model_config = checkpoint.get("model_config", {})
    input_dim = model_config.get("input_dim", 50)
    latent_dim = model_config.get("latent_dim", 32)
    compression_type = model_config.get("compression_type", "entropy_bottleneck")
    compression_level = model_config.get("compression_level", 2.0)
    vq_num_embeddings = model_config.get("vq_num_embeddings", 512)
    
    # Create model
    model = MeaningVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        compression_type=compression_type,
        compression_level=compression_level,
        vq_num_embeddings=vq_num_embeddings
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"Model loaded successfully.")
    return model, model_config


def evaluate_model(model, dataset, device, output_dir, compression_levels):
    """Evaluate semantic preservation in model."""
    print("Evaluating semantic preservation...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics
    semantic_metrics = SemanticMetrics()
    drift_tracker = DriftTracker(log_dir=os.path.join(output_dir, "drift_tracking"))
    threshold_finder = CompressionThresholdFinder(semantic_threshold=0.9)
    
    # Save original compression level
    original_compression_level = model.compression_level if hasattr(model, "compression_level") else None
    
    # Evaluate at different compression levels
    for i, compression_level in enumerate(compression_levels):
        print(f"Evaluating at compression level {compression_level} bits per dimension...")
        
        # Set compression level if model supports it
        if hasattr(model, "set_compression_level"):
            model.set_compression_level(compression_level)
        
        # Reset dataset index
        dataset._current_idx = 0
        
        # Collect reconstructions for all batches
        all_originals = []
        all_reconstructed = []
        
        with torch.no_grad():
            while dataset._current_idx < len(dataset.states):
                # Get batch
                batch = dataset.get_batch()
                batch = batch.to(device)
                
                # Forward pass
                results = model(batch)
                
                # Store original and reconstructed
                all_originals.append(batch)
                all_reconstructed.append(results["x_reconstructed"])
                
        # Concatenate all batches
        all_originals = torch.cat(all_originals, dim=0)
        all_reconstructed = torch.cat(all_reconstructed, dim=0)
        
        # Evaluate semantic preservation
        metrics = semantic_metrics.evaluate(all_originals, all_reconstructed)
        print(f"Overall semantic score: {metrics['overall']:.4f}")
        
        # Track drift
        drift_tracker.log_iteration(i, compression_level, all_originals, all_reconstructed)
        
        # Evaluate compression threshold
        threshold_finder.evaluate_compression_level(
            compression_level, all_originals, all_reconstructed
        )
        
        # Generate latent space visualization
        if "z" in results:
            # Extract latent vectors
            latent_vectors = results["z"]
            
            # Create role labels for visualization
            role_indices = torch.argmax(all_originals[:, 5:10], dim=1)
            
            # Generate t-SNE visualization
            vis_path = os.path.join(output_dir, f"latent_tsne_level_{compression_level}.png")
            generate_t_sne_visualization(latent_vectors, labels=role_indices, output_file=vis_path)
    
    # Restore original compression level
    if original_compression_level is not None and hasattr(model, "set_compression_level"):
        model.set_compression_level(original_compression_level)
    
    # Generate drift visualization
    print("Generating drift visualization...")
    drift_tracker.visualize_drift(os.path.join(output_dir, "drift_visualization.png"))
    
    # Generate drift report
    print("Generating drift report...")
    report = drift_tracker.generate_report(os.path.join(output_dir, "drift_report.md"))
    
    # Find optimal compression threshold
    print("Finding optimal compression threshold...")
    optimal = threshold_finder.find_optimal_threshold()
    print(f"Optimal compression threshold: {optimal['optimal_level']} bits per dimension")
    print(f"Optimal semantic score: {optimal['optimal_score']:.4f}")
    
    # Save optimal threshold report
    with open(os.path.join(output_dir, "optimal_threshold.json"), "w") as f:
        import json
        json.dump(optimal, f, indent=2)
    
    return {
        "drift_tracker": drift_tracker,
        "optimal_threshold": optimal
    }


def main():
    """Run semantic preservation evaluation."""
    args = parse_args()
    
    # Parse compression levels
    compression_levels = [float(level) for level in args.compression_levels.split(",")]
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, model_config = load_model(args.model_path, device)
    
    # Generate test data
    print(f"Generating {args.num_samples} synthetic agent states...")
    states = generate_agent_states(args.num_samples)
    dataset = AgentStateDataset(states, batch_size=args.batch_size)
    
    # Evaluate model
    results = evaluate_model(
        model=model,
        dataset=dataset,
        device=device,
        output_dir=args.output_dir,
        compression_levels=compression_levels
    )
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Optimal compression level: {results['optimal_threshold']['optimal_level']} bits per dimension")


if __name__ == "__main__":
    main() 