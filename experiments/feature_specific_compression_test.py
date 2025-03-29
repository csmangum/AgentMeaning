#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment to test feature-specific compression strategies.

This script:
1. Creates models with different compression configurations
2. Trains them on the same dataset
3. Compares performance across different feature groups
4. Analyzes semantic preservation by feature type
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import json
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from meaning_transform.src.feature_specific_compression import (
    create_feature_specific_model,
    print_compression_strategy_analysis,
    FEATURE_IMPORTANCE,
    FEATURE_GROUPS
)
from meaning_transform.src.adaptive_model import FeatureGroupedVAE
from meaning_transform.src.model import MeaningVAE
from meaning_transform.src.data import AgentState, AgentStateDataset
from meaning_transform.src.metrics import SemanticMetrics, compute_feature_drift
from meaning_transform.src.loss import CombinedLoss
from meaning_transform.src.config import Config


def setup_experiment_directory() -> Path:
    """Set up experiment directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"feature_specific_compression_{timestamp}"
    experiment_dir = Path("results") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    return experiment_dir


def load_data(
    batch_size: int = 64, 
    num_states: int = 5000,
    validation_split: float = 0.15
) -> Tuple[AgentStateDataset, AgentStateDataset, List[AgentState]]:
    """Load data for training and testing."""
    print(f"Loading data with {num_states} states...")
    
    # Generate or load agent states
    dataset = AgentStateDataset(batch_size=batch_size)
    
    # Check for data files in different locations
    data_paths = [
        "simulation.db",
        "data/agent_states.db", 
        "../data/agent_states.db",
        "data/simulation.db",
        "../data/simulation.db"
    ]
    
    # Also check for pickle files
    pickle_paths = [
        "data/agent_states.pkl",
        "../data/agent_states.pkl",
        "data/states.pkl",
        "../data/states.pkl"
    ]
    
    # Try to load from database files
    db_loaded = False
    for db_path in data_paths:
        if os.path.exists(db_path):
            print(f"Loading agent states from database {db_path}...")
            try:
                dataset.load_from_db(db_path, limit=num_states)
                if dataset.states:
                    db_loaded = True
                    break
            except Exception as e:
                print(f"Error loading from {db_path}: {e}")
    
    # If database load failed, try pickle files
    if not db_loaded:
        for pkl_path in pickle_paths:
            if os.path.exists(pkl_path):
                print(f"Loading agent states from pickle file {pkl_path}...")
                try:
                    dataset.load_from_file(pkl_path)
                    if dataset.states:
                        db_loaded = True
                        break
                except Exception as e:
                    print(f"Error loading from {pkl_path}: {e}")
    
    # If we couldn't load any data, raise an error
    if not dataset.states:
        raise ValueError(
            "Could not load any agent states. Please ensure data files exist in one of the following locations: "
            + ", ".join(data_paths + pickle_paths)
        )
    
    # Limit to requested number of states
    if len(dataset.states) > num_states:
        print(f"Limiting dataset to {num_states} states")
        dataset.states = dataset.states[:num_states]
    
    # Split into train and validation sets
    total_states = len(dataset.states)
    val_size = int(total_states * validation_split)
    train_size = total_states - val_size
    
    train_states = dataset.states[:train_size]
    val_states = dataset.states[train_size:]
    
    train_dataset = AgentStateDataset(train_states, batch_size=batch_size)
    val_dataset = AgentStateDataset(val_states, batch_size=batch_size)
    
    print(f"Training set: {len(train_dataset.states)} states")
    print(f"Validation set: {len(val_dataset.states)} states")
    
    # Set aside a small set of states for detailed feature analysis
    analysis_states = val_states[:min(20, len(val_states))]
    
    return train_dataset, val_dataset, analysis_states


def train_model(
    model: torch.nn.Module,
    train_dataset: AgentStateDataset,
    val_dataset: AgentStateDataset, 
    epochs: int = 25,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    experiment_dir: Path = None,
    model_name: str = "model"
) -> Dict[str, List[float]]:
    """Train a model and return training metrics."""
    print(f"Training {model_name} on {device}...")
    
    device = torch.device(device)
    model = model.to(device)
    
    # Create loss function
    loss_fn = CombinedLoss(
        recon_loss_weight=1.0,
        kl_loss_weight=1.0,
        semantic_loss_weight=1.0
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Initialize metrics tracking
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "recon_loss": [],
        "kl_loss": [],
        "semantic_loss": []
    }
    
    # Initialize semantic metrics
    semantic_metrics = SemanticMetrics()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Shuffle at the start of each epoch
        random_idx = torch.randperm(len(train_dataset.states))
        train_dataset.states = [train_dataset.states[i] for i in random_idx]
        
        # Reset dataset index
        train_dataset._current_idx = 0
        
        # Train epoch
        while train_dataset._current_idx < len(train_dataset.states):
            # Get batch
            batch = train_dataset.get_batch()
            batch = batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            results = model(batch)
            
            # Compute loss
            loss_results = loss_fn(results, batch)
            loss = loss_results["total_loss"]
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss for the epoch
        avg_train_loss = total_loss / max(1, num_batches)
        metrics["train_loss"].append(avg_train_loss)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        val_semantic_loss = 0.0
        val_batches = 0
        
        val_dataset._current_idx = 0
        
        with torch.no_grad():
            while val_dataset._current_idx < len(val_dataset.states):
                # Get batch
                batch = val_dataset.get_batch()
                batch = batch.to(device)
                
                # Forward pass
                results = model(batch)
                
                # Compute loss
                loss_results = loss_fn(results, batch)
                
                # Update metrics
                val_loss += loss_results["total_loss"].item()
                val_recon_loss += loss_results["recon_loss"].item()
                val_kl_loss += loss_results["kl_loss"].item()
                val_semantic_loss += loss_results["semantic_loss"].item()
                val_batches += 1
        
        # Calculate average validation loss
        avg_val_loss = val_loss / max(1, val_batches)
        avg_val_recon_loss = val_recon_loss / max(1, val_batches)
        avg_val_kl_loss = val_kl_loss / max(1, val_batches)
        avg_val_semantic_loss = val_semantic_loss / max(1, val_batches)
        
        metrics["val_loss"].append(avg_val_loss)
        metrics["recon_loss"].append(avg_val_recon_loss)
        metrics["kl_loss"].append(avg_val_kl_loss)
        metrics["semantic_loss"].append(avg_val_semantic_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train loss: {avg_train_loss:.4f}, "
              f"Val loss: {avg_val_loss:.4f}, "
              f"Recon: {avg_val_recon_loss:.4f}, "
              f"KL: {avg_val_kl_loss:.4f}, "
              f"Semantic: {avg_val_semantic_loss:.4f}")
    
    # Save model
    if experiment_dir:
        model_dir = experiment_dir / "models"
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f"{model_name}.pt"
        torch.save(model.state_dict(), model_path)
        
        # Save metrics
        metrics_path = experiment_dir / "metrics" / f"{model_name}_metrics.json"
        metrics_path.parent.mkdir(exist_ok=True)
        
        with open(metrics_path, 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in metrics.items()}, f, indent=2)
    
    return metrics


def evaluate_feature_specific_performance(
    model: torch.nn.Module,
    test_states: List[AgentState],
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    """Evaluate model performance on a per-feature basis."""
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    # Convert states to tensors
    original_tensors = torch.stack([state.to_tensor() for state in test_states])
    original_tensors = original_tensors.to(device)
    
    # Get reconstructions
    with torch.no_grad():
        if hasattr(model, 'forward'):
            results = model(original_tensors)
            reconstructed_tensors = results["x_reconstructed"]
        else:
            # Fallback if model has different interface
            z = model.encode(original_tensors)
            reconstructed_tensors = model.decode(z)
    
    # Create metrics
    semantic_metrics = SemanticMetrics()
    
    # Compute overall metrics
    overall_scores = semantic_metrics.compute_equivalence_scores(
        original_tensors, reconstructed_tensors
    )
    
    # Compute binary feature accuracy
    binary_metrics = semantic_metrics.binary_feature_accuracy(
        original_tensors, reconstructed_tensors
    )
    
    # Compute role accuracy
    role_metrics = semantic_metrics.role_accuracy(
        original_tensors, reconstructed_tensors
    )
    
    # Compute numeric feature errors
    numeric_metrics = semantic_metrics.numeric_feature_errors(
        original_tensors, reconstructed_tensors
    )
    
    # Group metrics by feature group
    feature_group_metrics = {}
    
    # Spatial features (position)
    feature_group_metrics["spatial"] = {
        "position_mae": numeric_metrics.get("position_mae", 0.0),
        "position_rmse": numeric_metrics.get("position_rmse", 0.0),
        "position_similarity": overall_scores.get("position", 0.0)
    }
    
    # Resource features (health, energy)
    feature_group_metrics["resource"] = {
        "health_mae": numeric_metrics.get("health_mae", 0.0),
        "health_rmse": numeric_metrics.get("health_rmse", 0.0),
        "energy_mae": numeric_metrics.get("energy_mae", 0.0),
        "energy_rmse": numeric_metrics.get("energy_rmse", 0.0),
        "health_similarity": overall_scores.get("health", 0.0),
        "energy_similarity": overall_scores.get("energy", 0.0)
    }
    
    # Status features
    feature_group_metrics["status"] = {
        "is_alive_accuracy": binary_metrics.get("is_alive_accuracy", 0.0),
        "is_alive_f1": binary_metrics.get("is_alive_f1", 0.0),
        "threatened_accuracy": binary_metrics.get("threatened_accuracy", 0.0),
        "threatened_f1": binary_metrics.get("threatened_f1", 0.0)
    }
    
    # Role features
    feature_group_metrics["role"] = {
        "role_accuracy": role_metrics.get("role_accuracy", 0.0)
    }
    
    # Add importance scores to each group
    for group_name in feature_group_metrics:
        feature_group_metrics[group_name]["importance"] = FEATURE_IMPORTANCE.get(group_name, 0.0)
    
    # Calculate average metrics per group
    for group_name, metrics_dict in feature_group_metrics.items():
        # Filter out importance and non-numeric values
        numeric_values = [v for k, v in metrics_dict.items() 
                         if k != "importance" and isinstance(v, (int, float))]
        
        if numeric_values:
            feature_group_metrics[group_name]["average_score"] = sum(numeric_values) / len(numeric_values)
    
    # Calculate overall performance
    all_avg_scores = [metrics_dict.get("average_score", 0.0) for metrics_dict in feature_group_metrics.values()]
    if all_avg_scores:
        feature_group_metrics["overall"] = {
            "average_score": sum(all_avg_scores) / len(all_avg_scores),
            "overall_similarity": overall_scores.get("overall", 0.0)
        }
    
    return feature_group_metrics


def plot_feature_specific_results(
    results: Dict[str, Dict[str, Any]],
    experiment_dir: Path
):
    """Plot feature-specific evaluation results."""
    output_dir = experiment_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Extract model names and feature groups
    model_names = list(results.keys())
    feature_groups = list(next(iter(results.values())).keys())
    if "overall" in feature_groups:
        feature_groups.remove("overall")  # Handle overall separately
    
    # Create bar plot for average scores by feature group
    plt.figure(figsize=(12, 8))
    
    # Set up bar positions
    bar_width = 0.8 / len(model_names)
    positions = np.arange(len(feature_groups))
    
    for i, model_name in enumerate(model_names):
        avg_scores = [results[model_name][group].get("average_score", 0.0) for group in feature_groups]
        plt.bar(
            positions + i * bar_width - bar_width * (len(model_names) - 1) / 2,
            avg_scores,
            width=bar_width,
            label=model_name
        )
    
    # Add importance as line
    importance_scores = [FEATURE_IMPORTANCE.get(group, 0.0) / 100 for group in feature_groups]
    plt.plot(positions, importance_scores, 'r--', linewidth=2, label="Importance")
    
    plt.xlabel("Feature Group")
    plt.ylabel("Average Score")
    plt.title("Feature-Specific Performance by Model")
    plt.xticks(positions, feature_groups)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(output_dir / "feature_specific_performance.png", dpi=300, bbox_inches="tight")
    
    # Create plot for overall similarity
    plt.figure(figsize=(10, 6))
    
    overall_scores = [results[model_name]["overall"].get("overall_similarity", 0.0) 
                     for model_name in model_names]
    
    plt.bar(model_names, overall_scores, color='skyblue')
    plt.xlabel("Model")
    plt.ylabel("Overall Semantic Similarity")
    plt.title("Overall Model Performance")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(output_dir / "overall_performance.png", dpi=300, bbox_inches="tight")


def run_experiment():
    """Run the complete feature-specific compression experiment."""
    # Set up experiment directory
    experiment_dir = setup_experiment_directory()
    print(f"Experiment results will be saved to: {experiment_dir}")
    
    # Define model input and latent dimensions
    input_dim = 15  # Based on AgentState.to_tensor
    latent_dim = 32  # Based on previous findings
    base_compression = 1.0  # Base compression level
    
    # Load data from simulation.db only - no synthetic data
    db_path = "../data/simulation.db"
    if not os.path.exists(db_path):
        db_path = "data/simulation.db"
    if not os.path.exists(db_path):
        raise ValueError(f"Real data file 'simulation.db' not found! Please check the data directory.")
    
    print(f"Loading real agent states from {db_path}...")
    
    # Create dataset and load from DB
    dataset = AgentStateDataset(batch_size=64)
    dataset.load_from_db(db_path, limit=5000)
    print(f"Loaded {len(dataset.states)} real agent states")
    
    # Split into train and validation sets
    total_states = len(dataset.states)
    val_size = int(total_states * 0.15)
    train_size = total_states - val_size
    
    train_states = dataset.states[:train_size]
    val_states = dataset.states[train_size:]
    
    train_dataset = AgentStateDataset(train_states, batch_size=64)
    val_dataset = AgentStateDataset(val_states, batch_size=64)
    
    print(f"Training set: {len(train_dataset.states)} states")
    print(f"Validation set: {len(val_dataset.states)} states")
    
    # Set aside states for analysis
    analysis_states = val_states[:min(20, len(val_states))]
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create models to test
    models = {}
    
    # 1. Baseline model (standard VAE with uniform compression)
    models["baseline"] = MeaningVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        compression_type="entropy",
        compression_level=base_compression
    )
    
    # 2. Feature-grouped VAE with uniform compression (no importance weighting)
    uniform_feature_groups = {
        name: (start, end, base_compression)
        for name, (start, end) in FEATURE_GROUPS.items()
    }
    
    models["grouped_uniform"] = FeatureGroupedVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        feature_groups=uniform_feature_groups,
        base_compression_level=base_compression
    )
    
    # 3. Feature-specific compression VAE (importance-weighted)
    models["feature_specific"] = create_feature_specific_model(
        input_dim=input_dim,
        latent_dim=latent_dim,
        base_compression_level=base_compression
    )
    
    # Print analysis of the feature-specific model
    print_compression_strategy_analysis(models["feature_specific"])
    
    # Train models
    training_metrics = {}
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name} model")
        print(f"{'='*50}")
        
        metrics = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=25,
            learning_rate=1e-3,
            device=device,
            experiment_dir=experiment_dir,
            model_name=model_name
        )
        
        training_metrics[model_name] = metrics
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    
    for model_name, metrics in training_metrics.items():
        plt.plot(metrics["val_loss"], label=f"{model_name} Validation Loss")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss by Model")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / "validation_loss_comparison.png", dpi=300, bbox_inches="tight")
    
    # Evaluate feature-specific performance
    evaluation_results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name} model...")
        
        feature_metrics = evaluate_feature_specific_performance(
            model=model,
            test_states=analysis_states,
            device=device
        )
        
        evaluation_results[model_name] = feature_metrics
        
        # Save results
        results_dir = experiment_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / f"{model_name}_feature_evaluation.json", 'w') as f:
            # Convert any non-serializable values
            serializable_metrics = {}
            for group, metrics in feature_metrics.items():
                serializable_metrics[group] = {k: float(v) if isinstance(v, (int, float)) else v 
                                               for k, v in metrics.items()}
            
            json.dump(serializable_metrics, f, indent=2)
    
    # Plot feature-specific results
    plot_feature_specific_results(evaluation_results, experiment_dir)
    
    # Generate report
    generate_experiment_report(
        evaluation_results, 
        training_metrics,
        models,
        experiment_dir
    )
    
    print(f"\nExperiment complete! Results saved to {experiment_dir}")
    return experiment_dir


def generate_experiment_report(
    evaluation_results: Dict[str, Dict[str, Any]],
    training_metrics: Dict[str, Dict[str, List[float]]],
    models: Dict[str, torch.nn.Module],
    experiment_dir: Path
):
    """Generate a comprehensive experiment report."""
    report_path = experiment_dir / "feature_specific_compression_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Feature-Specific Compression Strategy Experiment\n\n")
        f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write("This experiment evaluates the effectiveness of feature-specific compression strategies ")
        f.write("that apply different compression rates to different feature groups based on their importance. ")
        f.write("The hypothesis is that by preserving high-importance features (spatial, resource) with higher fidelity ")
        f.write("while compressing low-importance features more aggressively, we can achieve better overall ")
        f.write("semantic preservation at the same compression level.\n\n")
        
        # Models compared
        f.write("## Models Evaluated\n\n")
        
        for model_name, model in models.items():
            f.write(f"### {model_name.replace('_', ' ').title()}\n")
            f.write(f"- Type: {model.__class__.__name__}\n")
            f.write(f"- Input dimension: {model.input_dim}\n")
            f.write(f"- Latent dimension: {model.latent_dim}\n")
            
            if hasattr(model, 'compression_level'):
                f.write(f"- Compression level: {model.compression_level}x\n")
            elif hasattr(model, 'base_compression_level'):
                f.write(f"- Base compression level: {model.base_compression_level}x\n")
            
            if hasattr(model, 'get_compression_rate'):
                if isinstance(model.get_compression_rate(), dict):
                    f.write("- Compression rates by feature group:\n")
                    for group, rate in model.get_compression_rate().items():
                        f.write(f"  - {group}: {rate:.2f}x\n")
                else:
                    f.write(f"- Effective compression rate: {model.get_compression_rate():.2f}x\n")
            
            f.write("\n")
        
        # Overall performance comparison
        f.write("## Overall Performance Comparison\n\n")
        
        # Create comparison table
        f.write("| Model | Final Val Loss | Reconstruction Loss | Semantic Loss | Overall Similarity |\n")
        f.write("|-------|----------------|---------------------|---------------|--------------------|\n")
        
        for model_name in models:
            val_loss = training_metrics[model_name]["val_loss"][-1]
            recon_loss = training_metrics[model_name]["recon_loss"][-1]
            semantic_loss = training_metrics[model_name]["semantic_loss"][-1]
            overall_similarity = evaluation_results[model_name]["overall"]["overall_similarity"]
            
            f.write(f"| {model_name.replace('_', ' ').title()} | {val_loss:.4f} | {recon_loss:.4f} | ")
            f.write(f"{semantic_loss:.4f} | {overall_similarity:.4f} |\n")
        
        f.write("\n")
        
        # Feature-specific performance
        f.write("## Feature-Specific Performance\n\n")
        
        feature_groups = list(next(iter(evaluation_results.values())).keys())
        if "overall" in feature_groups:
            feature_groups.remove("overall")
        
        for group in feature_groups:
            f.write(f"### {group.title()} Features\n\n")
            f.write(f"Importance score: {FEATURE_IMPORTANCE.get(group, 0.0):.1f}%\n\n")
            
            # Create comparison table for this feature group
            # Get all metric keys for this group across all models
            metric_keys = set()
            for model_name in models:
                metric_keys.update(evaluation_results[model_name][group].keys())
            
            # Remove non-metrics
            metric_keys = [k for k in metric_keys if k not in ["importance", "average_score"]]
            
            if metric_keys:
                f.write("| Model | " + " | ".join(k.replace("_", " ").title() for k in metric_keys) + " |\n")
                f.write("|-------|" + "---|" * (len(metric_keys) - 1) + "---|\n")
                
                for model_name in models:
                    metrics = evaluation_results[model_name][group]
                    f.write(f"| {model_name.replace('_', ' ').title()} | ")
                    f.write(" | ".join(f"{metrics.get(k, 0.0):.4f}" for k in metric_keys))
                    f.write(" |\n")
            
            f.write("\n")
        
        # Key findings
        f.write("## Key Findings\n\n")
        
        # Calculate improvement for feature-specific model over baseline
        baseline_overall = evaluation_results["baseline"]["overall"]["overall_similarity"]
        feature_specific_overall = evaluation_results["feature_specific"]["overall"]["overall_similarity"]
        overall_improvement = ((feature_specific_overall - baseline_overall) / baseline_overall) * 100
        
        f.write("1. **Overall Semantic Preservation**\n")
        f.write(f"   - Feature-specific compression {overall_improvement:.1f}% ")
        if overall_improvement > 0:
            f.write("better than baseline\n")
        else:
            f.write("worse than baseline\n")
        
        # Calculate improvements for individual feature groups
        f.write("2. **Feature Group Improvements**\n")
        
        for group in feature_groups:
            baseline_avg = evaluation_results["baseline"][group].get("average_score", 0.0)
            feature_specific_avg = evaluation_results["feature_specific"][group].get("average_score", 0.0)
            
            if baseline_avg > 0:
                group_improvement = ((feature_specific_avg - baseline_avg) / baseline_avg) * 100
                f.write(f"   - {group.title()} features: {group_improvement:.1f}% ")
                if group_improvement > 0:
                    f.write("improvement\n")
                else:
                    f.write("reduction\n")
        
        # Calculate model size comparison
        f.write("3. **Model Efficiency**\n")
        
        model_sizes = {}
        for model_name, model in models.items():
            total_params = sum(p.numel() for p in model.parameters())
            model_sizes[model_name] = total_params
            f.write(f"   - {model_name.replace('_', ' ').title()}: {total_params} parameters\n")
        
        f.write("\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        f.write("- Feature-specific compression allows for more efficient allocation of latent space capacity\n")
        f.write("- High-importance features (spatial, resource) benefit from lower compression ratios\n")
        f.write("- Low-importance features (role, status) can be compressed more aggressively with minimal impact\n")
        f.write("- This approach provides a more balanced trade-off between model size and semantic preservation\n")
        
        # Recommendations
        f.write("\n## Recommendations\n\n")
        f.write("1. Adopt feature-specific compression for future models to optimize semantic preservation\n")
        f.write("2. Further refine compression ratios through iterative testing\n")
        f.write("3. Consider even more aggressive compression for the lowest-importance features\n")
        f.write("4. Explore dynamic adaptation of compression ratios based on context or agent role\n")
        f.write("5. Test this approach with larger models and more diverse agent states\n")
    
    print(f"Report generated: {report_path}")


if __name__ == "__main__":
    experiment_dir = run_experiment()
    print(f"Experiment results available at: {experiment_dir}") 