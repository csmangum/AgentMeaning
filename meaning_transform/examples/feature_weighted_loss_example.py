#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example demonstrating the feature-weighted loss implementation.

This script shows:
1. How to configure feature weights based on importance scores
2. How to use progressive weight adjustment during training
3. How to test if feature-weighted loss preserves critical features better than standard loss
4. How to visualize the impact of feature weighting on semantic preservation
"""

# Add parent directory to Python path so we can import the module without installation
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import argparse
import os
from datetime import datetime
from typing import Dict, List, Tuple
import random

# Import project components
from meaning_transform.src.loss import CombinedLoss, FeatureWeightedLoss
from meaning_transform.src.model import MeaningVAE
from meaning_transform.src.data import AgentStateDataset
from meaning_transform.src.metrics import calculate_semantic_similarity
from meaning_transform.src.feature_importance import FeatureImportanceAnalyzer, analyze_feature_importance


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_epoch(
    model: MeaningVAE,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_semantic_loss = 0.0
    
    # Update feature weights if using feature-weighted loss with progressive schedule
    if hasattr(loss_fn, 'update_epoch'):
        loss_fn.update_epoch(epoch)
    
    for batch_idx, (data,) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Calculate loss
        loss_dict = loss_fn(output, data)
        loss = loss_dict["loss"]
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate statistics
        total_loss += loss.item()
        total_recon_loss += loss_dict["reconstruction_loss"].item()
        total_kl_loss += loss_dict["kl_loss"].item()
        total_semantic_loss += loss_dict["semantic_loss"].item()
    
    # Calculate average loss
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    avg_semantic_loss = total_semantic_loss / num_batches
    
    return {
        "loss": avg_loss,
        "reconstruction_loss": avg_recon_loss,
        "kl_loss": avg_kl_loss,
        "semantic_loss": avg_semantic_loss
    }


def evaluate(
    model: MeaningVAE,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    feature_similarities = {}
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, (data,) in enumerate(dataloader):
            data = data.to(device)
            
            # Forward pass
            output = model(data)
            reconstruction = output["reconstruction"]
            
            # Calculate loss
            loss_dict = loss_fn(output, data)
            total_loss += loss_dict["loss"].item()
            
            # Calculate feature-specific semantic similarities
            if batch_idx == 0 or batch_idx % 5 == 0:  # Only sample some batches for similarity
                feature_sim = calculate_semantic_similarity(data, reconstruction)
                for feature, sim in feature_sim.items():
                    if feature not in feature_similarities:
                        feature_similarities[feature] = 0.0
                    feature_similarities[feature] += sim
                batch_count += 1
    
    # Calculate average metrics
    avg_loss = total_loss / len(dataloader)
    
    # Average feature similarities
    if batch_count > 0:
        for feature in feature_similarities:
            feature_similarities[feature] /= batch_count
    
    return {
        "loss": avg_loss,
        "feature_similarities": feature_similarities
    }


def visualize_feature_similarities(
    standard_similarities: Dict[str, float],
    weighted_similarities: Dict[str, float],
    output_dir: str,
):
    """Visualize feature-specific semantic similarities."""
    features = list(standard_similarities.keys())
    standard_values = [standard_similarities[f] for f in features]
    weighted_values = [weighted_similarities[f] for f in features]
    
    x = np.arange(len(features))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, standard_values, width, label='Standard Loss')
    ax.bar(x + width/2, weighted_values, width, label='Feature-Weighted Loss')
    
    ax.set_ylabel('Semantic Similarity')
    ax.set_title('Feature-Specific Semantic Preservation')
    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_similarities.png'), dpi=300)
    plt.close()


def plot_training_curves(
    standard_history: Dict[str, List[float]],
    weighted_history: Dict[str, List[float]],
    output_dir: str,
):
    """Plot training curves comparing standard and feature-weighted loss."""
    epochs = range(1, len(standard_history["loss"]) + 1)
    
    # Plot total loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, standard_history["loss"], 'b-', label='Standard Loss')
    ax.plot(epochs, weighted_history["loss"], 'r-', label='Feature-Weighted Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_loss_comparison.png'), dpi=300)
    plt.close()
    
    # Plot semantic loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, standard_history["semantic_loss"], 'b-', label='Standard Loss')
    ax.plot(epochs, weighted_history["semantic_loss"], 'r-', label='Feature-Weighted Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Semantic Loss')
    ax.set_title('Semantic Loss Comparison')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'semantic_loss_comparison.png'), dpi=300)
    plt.close()


def run_comparison(
    train_dataset: AgentStateDataset,
    val_dataset: AgentStateDataset,
    test_dataset: AgentStateDataset,
    compression_level: float = 1.0,
    epochs: int = 20,
    batch_size: int = 32,
    output_dir: str = "results",
):
    """
    Run comparison between standard loss and feature-weighted loss.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        compression_level: Compression level (0.5-2.0)
        epochs: Number of training epochs
        batch_size: Batch size for training
        output_dir: Output directory for results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get input dimensions from dataset
    sample_tensor = train_dataset[0][0]  # First item, first element of tuple
    input_dim = sample_tensor.shape[0]
    print(f"Input dimension: {input_dim}")
    
    # Run feature importance analysis on validation data
    print("Running feature importance analysis...")
    val_tensors = torch.stack([x[0] for x in val_dataset])
    importance_results = analyze_feature_importance(val_tensors, create_visualizations=True)
    feature_importance = importance_results["feature_importance"]
    
    print("Feature importance scores:")
    for feature, score in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {score:.4f}")
    
    # Create standard and feature-weighted loss functions
    standard_loss = CombinedLoss(
        recon_loss_weight=1.0,
        kl_loss_weight=0.1,
        semantic_loss_weight=0.5
    )
    
    weighted_loss = FeatureWeightedLoss(
        recon_loss_weight=1.0,
        kl_loss_weight=0.1,
        semantic_loss_weight=0.5,
        feature_weights=feature_importance,
        progressive_weight_schedule="linear",
        progressive_weight_epochs=epochs // 2,
        feature_stability_adjustment=True
    )
    
    # Create separate models for standard and weighted loss
    standard_model = MeaningVAE(
        input_dim=input_dim,
        latent_dim=int(input_dim / compression_level),
        compression_level=compression_level
    ).to(device)
    
    weighted_model = MeaningVAE(
        input_dim=input_dim,
        latent_dim=int(input_dim / compression_level),
        compression_level=compression_level
    ).to(device)
    
    # Ensure both models start with the same weights
    weighted_model.load_state_dict(standard_model.state_dict())
    
    # Create optimizers
    standard_optimizer = torch.optim.Adam(standard_model.parameters(), lr=1e-3)
    weighted_optimizer = torch.optim.Adam(weighted_model.parameters(), lr=1e-3)
    
    # Training histories
    standard_history = {
        "loss": [],
        "reconstruction_loss": [],
        "kl_loss": [],
        "semantic_loss": []
    }
    
    weighted_history = {
        "loss": [],
        "reconstruction_loss": [],
        "kl_loss": [],
        "semantic_loss": []
    }
    
    # Training loop
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        
        # Train both models
        standard_metrics = train_epoch(
            standard_model, train_loader, standard_loss, standard_optimizer, device, epoch
        )
        
        weighted_metrics = train_epoch(
            weighted_model, train_loader, weighted_loss, weighted_optimizer, device, epoch
        )
        
        # Evaluate on validation set
        standard_val = evaluate(standard_model, val_loader, standard_loss, device)
        weighted_val = evaluate(weighted_model, val_loader, weighted_loss, device)
        
        # Print progress
        print(f"  Standard - Loss: {standard_metrics['loss']:.4f}, Val Loss: {standard_val['loss']:.4f}")
        print(f"  Weighted - Loss: {weighted_metrics['loss']:.4f}, Val Loss: {weighted_val['loss']:.4f}")
        
        # Record history
        for key in standard_history:
            if key in standard_metrics:
                standard_history[key].append(standard_metrics[key])
                weighted_history[key].append(weighted_metrics[key])
    
    # Final evaluation on test set
    standard_test = evaluate(standard_model, test_loader, standard_loss, device)
    weighted_test = evaluate(weighted_model, test_loader, weighted_loss, device)
    
    print("\nTest Results:")
    print(f"  Standard Loss - Test Loss: {standard_test['loss']:.4f}")
    print(f"  Feature-Weighted Loss - Test Loss: {weighted_test['loss']:.4f}")
    
    # Visualize feature-specific semantic similarities
    print("\nFeature-specific semantic similarities:")
    print("  Standard Loss:")
    for feature, sim in sorted(standard_test["feature_similarities"].items()):
        print(f"    {feature}: {sim:.4f}")
    
    print("  Feature-Weighted Loss:")
    for feature, sim in sorted(weighted_test["feature_similarities"].items()):
        print(f"    {feature}: {sim:.4f}")
    
    # Create visualizations
    visualize_feature_similarities(
        standard_test["feature_similarities"],
        weighted_test["feature_similarities"],
        output_dir
    )
    
    plot_training_curves(standard_history, weighted_history, output_dir)
    
    # Save models
    torch.save(standard_model.state_dict(), os.path.join(output_dir, "standard_model.pt"))
    torch.save(weighted_model.state_dict(), os.path.join(output_dir, "weighted_model.pt"))
    
    print(f"Results saved to {output_dir}")
    
    return {
        "standard_model": standard_model,
        "weighted_model": weighted_model,
        "standard_test": standard_test,
        "weighted_test": weighted_test,
        "standard_history": standard_history,
        "weighted_history": weighted_history
    }


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="Feature-weighted loss example")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="results/feature_weighted", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--compression", type=float, default=1.0, help="Compression level (0.5-2.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--db_limit", type=int, default=2000, help="Maximum number of states to load from database")
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"compression_{args.compression}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load from simulation.db
    print(f"Loading data from {os.path.join(args.data_dir, 'simulation.db')}")
    
    # Create dataset
    all_dataset = AgentStateDataset()
    db_path = os.path.join(args.data_dir, "simulation.db")
    all_dataset.load_from_db(db_path, limit=args.db_limit)
    
    if not all_dataset.states:
        raise ValueError(f"Failed to load any states from the database: {db_path}")
        
    print(f"Loaded {len(all_dataset.states)} agent states from database")
    
    # Split into train/val/test sets
    states = all_dataset.states
    total_size = len(states)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    
    # Shuffle the states
    random.shuffle(states)
    
    train_states = states[:train_size]
    val_states = states[train_size:train_size+val_size]
    test_states = states[train_size+val_size:]
    
    train_dataset = AgentStateDataset(train_states)
    val_dataset = AgentStateDataset(val_states)
    test_dataset = AgentStateDataset(test_states)
    
    # Print dataset sizes
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset.states)}")
    print(f"  Val: {len(val_dataset.states)}")
    print(f"  Test: {len(test_dataset.states)}")
    
    # Run comparison
    results = run_comparison(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        compression_level=args.compression,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=output_dir
    )
    
    # Calculate improvements for high-importance features
    std_similarities = results["standard_test"]["feature_similarities"]
    weighted_similarities = results["weighted_test"]["feature_similarities"]
    
    # Calculate overall improvement
    improvements = {}
    for feature in std_similarities:
        if feature in weighted_similarities:
            rel_improvement = (weighted_similarities[feature] - std_similarities[feature]) / max(0.001, std_similarities[feature]) * 100
            improvements[feature] = rel_improvement
    
    print("\nRelative improvements in semantic preservation:")
    for feature, improvement in sorted(improvements.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {improvement:+.2f}%")
    
    # Print overall summary
    print("\nSummary:")
    spatial_improvement = improvements.get("position", 0)
    resource_improvement = (improvements.get("health", 0) + improvements.get("energy", 0)) / 2
    
    print(f"  Spatial feature improvement: {spatial_improvement:+.2f}%")
    print(f"  Resource feature improvement: {resource_improvement:+.2f}%")
    print(f"  Overall test loss reduction: {(results['standard_test']['loss'] - results['weighted_test']['loss']) / results['standard_test']['loss'] * 100:.2f}%")
    
    # Save configuration and results
    import json
    
    config = {
        "compression_level": args.compression,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "feature_importance": results["standard_test"]["feature_similarities"],
        "improvements": {k: float(v) for k, v in improvements.items()},
        "standard_final_loss": float(results["standard_test"]["loss"]),
        "weighted_final_loss": float(results["weighted_test"]["loss"])
    }
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Experiment completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main() 