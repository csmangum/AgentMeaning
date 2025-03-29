#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature-Grouped VAE Experiment

This script implements and tests the Feature-Grouped VAE architecture that applies
different compression rates to different feature groups based on their importance.

The compression rates are based on the findings from Step 12 (Feature Importance Analysis):
- Spatial features (55.4%): Low compression (0.5x)
- Resource features (25.1%): Medium compression (2.0x)
- Performance features (10.5%): Higher compression (3.0x)
- Status features (4.5%): High compression (5.0x)
- Role features (4.5%): High compression (5.0x)
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging
import json
from datetime import datetime

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from meaning_transform.src.adaptive_model import FeatureGroupedVAE
from meaning_transform.src.data import AgentStateDataset
from meaning_transform.src.train import Trainer
from meaning_transform.src.config import Config
from meaning_transform.src.metrics import SemanticMetrics, compute_feature_drift

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("feature_grouped_vae.log")
    ]
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Feature-Grouped VAE Experiment")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--latent-dim", type=int, default=32,
                        help="Dimension of latent space")
    parser.add_argument("--base-compression", type=float, default=1.0,
                        help="Base compression level")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for training if available")
    parser.add_argument("--output-dir", type=str, default="results/feature_grouped_vae",
                        help="Directory to save results")
    parser.add_argument("--quick", action="store_true",
                        help="Run a quick version with fewer epochs")
    return parser.parse_args()

def load_data(args):
    """Load agent state data from file or database."""
    dataset = AgentStateDataset(batch_size=args.batch_size)
    
    # Use the real simulation.db in the data folder
    db_path = Path("data/simulation.db")
    if db_path.exists():
        logging.info(f"Loading data from database: {db_path}")
        try:
            dataset.load_from_db(str(db_path))
            logging.info(f"Loaded {len(dataset.states)} agent states")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise RuntimeError(f"Failed to load data: {e}")
    else:
        logging.error(f"Database file not found: {db_path}")
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    if len(dataset.states) == 0:
        logging.error("No data loaded from database.")
        raise ValueError("No agent states were loaded from the database.")
    
    return dataset

def split_data(dataset, train_ratio=0.8):
    """Split dataset into training and validation sets."""
    train_size = int(train_ratio * len(dataset.states))
    val_size = len(dataset.states) - train_size
    
    train_states = dataset.states[:train_size]
    val_states = dataset.states[train_size:]
    
    train_dataset = AgentStateDataset(train_states, batch_size=dataset.batch_size)
    val_dataset = AgentStateDataset(val_states, batch_size=dataset.batch_size)
    
    logging.info(f"Split data into {len(train_states)} training and {len(val_states)} validation states")
    return train_dataset, val_dataset

def create_feature_grouped_vae(input_dim, latent_dim, base_compression):
    """Create a Feature-Grouped VAE model with importance-based compression rates."""
    # Define feature groups based on importance findings from Step 12
    # Format: {name: (start_index, end_index, compression_level)}
    
    # The features in the tensor representation are in this order:
    # - Position (x, y, z): 3 features - highest importance (spatial)
    # - Health, resource_level: 2 features - high importance (resources)
    # - is_defending, total_reward: 2 features - medium importance (performance)
    # - Other state flags and role encoding: remaining features - low importance
    
    # Distribute features according to their importance
    position_end = 3  # position_x, position_y, position_z - highest importance
    resources_end = position_end + 2  # health, resource_level - high importance
    performance_end = resources_end + 2  # is_defending, total_reward - medium importance
    
    feature_groups = {
        "spatial": (0, position_end, 0.5),                   # 55.4% importance - low compression
        "resources": (position_end, resources_end, 2.0),     # 25.1% importance - medium compression
        "performance": (resources_end, performance_end, 3.0),  # 10.5% importance - higher compression
        "status": (performance_end, input_dim, 5.0)         # 9.0% importance - high compression
    }
    
    logging.info("Creating Feature-Grouped VAE with the following feature groups:")
    for name, (start, end, compression) in feature_groups.items():
        logging.info(f"  {name}: features [{start}:{end}], compression {compression}x")
    
    # Create the model
    model = FeatureGroupedVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        feature_groups=feature_groups,
        base_compression_level=base_compression,
        use_batch_norm=True
    )
    
    return model, feature_groups

def train_model(model, train_dataset, val_dataset, args):
    """Train the Feature-Grouped VAE model."""
    # Set up configuration
    config = Config()
    config.training.num_epochs = 5 if args.quick else args.epochs
    config.training.learning_rate = args.learning_rate
    config.training.batch_size = args.batch_size
    config.model.latent_dim = args.latent_dim
    config.model.compression_level = args.base_compression
    config.use_gpu = args.gpu and torch.cuda.is_available()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    config.training.checkpoint_dir = str(output_dir / "checkpoints")
    
    # Set up trainer
    trainer = Trainer(config)
    trainer.model = model
    trainer.train_dataset = train_dataset
    trainer.val_dataset = val_dataset
    
    # Sample some states for drift tracking
    drift_states = val_dataset.states[:min(20, len(val_dataset.states))]
    trainer.drift_tracking_states = drift_states
    
    # Train the model
    logging.info(f"Starting training for {config.training.num_epochs} epochs")
    results = trainer.train()
    
    # Save the final model
    model_path = output_dir / "feature_grouped_vae_model.pt"
    model.save(str(model_path))
    logging.info(f"Model saved to {model_path}")
    
    return results, model_path

def evaluate_model(model, val_dataset, feature_groups, output_dir, use_gpu=False):
    """Evaluate the trained model and analyze feature-specific compression."""
    # Set device
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Get feature group analysis
    analysis = model.get_feature_group_analysis()
    
    # Print and log analysis
    logging.info("\nFeature Group Analysis:")
    logging.info(f"{'Group':<10} {'Features':<10} {'Latent Dim':<12} {'Effective Dim':<14} {'Compression':<12}")
    logging.info("-" * 65)
    
    for name, metrics in analysis.items():
        logging.info(f"{name:<10} {metrics['feature_count']:<10d} {metrics['latent_dim']:<12d} "
                    f"{metrics['effective_dim']:<14d} {metrics['overall_compression']:<12.2f}")
    
    # Get compression rates
    compression_rates = model.get_compression_rate()
    logging.info("\nCompression Rates:")
    for group, rate in compression_rates.items():
        logging.info(f"  {group}: {rate:.2f}x")
    
    # Save analysis to JSON
    analysis_path = output_dir / "feature_group_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump({
            "feature_groups": {
                name: {"start": start, "end": end, "compression": comp}
                for name, (start, end, comp) in feature_groups.items()
            },
            "analysis": {
                name: {k: v for k, v in metrics.items() if not isinstance(v, torch.Tensor)}
                for name, metrics in analysis.items()
            },
            "compression_rates": compression_rates
        }, f, indent=4)
    
    # Evaluate reconstruction quality for each feature group
    feature_group_mse = {}
    total_mse = 0.0
    count = 0
    
    # Sample states for evaluation
    eval_states = val_dataset.states[:min(100, len(val_dataset.states))]
    
    with torch.no_grad():
        for state in eval_states:
            # Convert state to tensor and move to device
            tensor = state.to_tensor().unsqueeze(0).to(device)
            
            # Forward pass
            output = model(tensor)
            reconstructed = output["x_reconstructed"][0].cpu()
            
            # Calculate MSE for the whole state
            mse = torch.mean((tensor.cpu() - reconstructed) ** 2).item()
            total_mse += mse
            count += 1
            
            # Calculate MSE for each feature group
            for name, (start, end, _) in feature_groups.items():
                group_mse = torch.mean((tensor.cpu()[0, start:end] - reconstructed[start:end]) ** 2).item()
                if name not in feature_group_mse:
                    feature_group_mse[name] = []
                feature_group_mse[name].append(group_mse)
    
    # Average MSE values
    avg_mse = total_mse / count
    avg_group_mse = {name: sum(values) / len(values) for name, values in feature_group_mse.items()}
    
    logging.info(f"\nOverall reconstruction MSE: {avg_mse:.6f}")
    logging.info("Feature group reconstruction MSE:")
    for name, mse in avg_group_mse.items():
        logging.info(f"  {name}: {mse:.6f}")
    
    # Return evaluation results
    return {
        "analysis": analysis,
        "compression_rates": compression_rates,
        "overall_mse": avg_mse,
        "group_mse": avg_group_mse
    }

def visualize_results(eval_results, feature_groups, output_dir):
    """Create visualizations of the feature group compression and reconstruction quality."""
    # Create output directory for visualizations
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract data for plotting
    analysis = eval_results["analysis"]
    groups = list(analysis.keys())
    
    # 1. Feature Group Dimensions Bar Chart
    feature_counts = [analysis[g]["feature_count"] for g in groups]
    latent_dims = [analysis[g]["latent_dim"] for g in groups]
    effective_dims = [analysis[g]["effective_dim"] for g in groups]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(groups))
    width = 0.25
    
    plt.bar(x - width, feature_counts, width=width, label='Input Features')
    plt.bar(x, latent_dims, width=width, label='Latent Dimensions')
    plt.bar(x + width, effective_dims, width=width, label='Effective Dimensions')
    
    plt.xlabel('Feature Groups')
    plt.ylabel('Dimensions')
    plt.title('Feature-Grouped VAE Dimension Distribution')
    plt.xticks(x, groups)
    plt.legend()
    plt.tight_layout()
    plt.savefig(viz_dir / "feature_group_dimensions.png")
    
    # 2. Compression Rates Bar Chart
    compression_rates = eval_results["compression_rates"]
    # Filter out 'overall' from bar chart
    group_rates = {k: v for k, v in compression_rates.items() if k != 'overall'}
    
    plt.figure(figsize=(10, 6))
    plt.bar(group_rates.keys(), group_rates.values())
    plt.axhline(y=compression_rates.get('overall', 1.0), color='r', linestyle='--', 
                label=f'Overall ({compression_rates.get("overall", 1.0):.2f}x)')
    plt.xlabel('Feature Groups')
    plt.ylabel('Compression Rate')
    plt.title('Compression Rates by Feature Group')
    plt.legend()
    plt.tight_layout()
    plt.savefig(viz_dir / "compression_rates.png")
    
    # 3. Reconstruction MSE by Feature Group
    group_mse = eval_results["group_mse"]
    
    plt.figure(figsize=(10, 6))
    plt.bar(group_mse.keys(), group_mse.values())
    plt.axhline(y=eval_results["overall_mse"], color='r', linestyle='--', 
                label=f'Overall MSE ({eval_results["overall_mse"]:.6f})')
    plt.xlabel('Feature Groups')
    plt.ylabel('Mean Squared Error')
    plt.title('Reconstruction Quality by Feature Group')
    plt.legend()
    plt.tight_layout()
    plt.savefig(viz_dir / "reconstruction_mse.png")
    
    # 4. MSE vs Importance Chart (scatter plot)
    # Estimate importance from compression level (inverse relationship)
    importances = {name: 1.0 / comp for name, (_, _, comp) in feature_groups.items()}
    importance_values = [importances[name] for name in group_mse.keys()]
    mse_values = list(group_mse.values())
    
    plt.figure(figsize=(10, 6))
    plt.scatter(importance_values, mse_values)
    
    # Add group names as labels
    for i, name in enumerate(group_mse.keys()):
        plt.annotate(name, (importance_values[i], mse_values[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel('Feature Importance (1/compression)')
    plt.ylabel('Mean Squared Error')
    plt.title('Reconstruction Quality vs Feature Importance')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(viz_dir / "importance_vs_mse.png")
    
    logging.info(f"Visualizations saved to {viz_dir}")

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"run_{timestamp}"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Configure log file in output directory
    file_handler = logging.FileHandler(output_dir / "experiment.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Save arguments
    with open(output_dir / "args.json", 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Log start
    logging.info(f"Starting Feature-Grouped VAE experiment (output dir: {output_dir})")
    logging.info(f"Arguments: {args}")
    
    # Load data
    dataset = load_data(args)
    
    if len(dataset.states) == 0:
        logging.error("No data available. Exiting.")
        return
    
    # Determine input dimension from data
    input_dim = dataset.states[0].to_tensor().shape[0]
    logging.info(f"Input dimension: {input_dim}")
    
    # Split data
    train_dataset, val_dataset = split_data(dataset)
    
    # Create model
    model, feature_groups = create_feature_grouped_vae(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        base_compression=args.base_compression
    )
    
    # Train model
    try:
        training_results, model_path = train_model(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            args=args
        )
        
        # Save training results
        with open(output_dir / "training_results.json", 'w') as f:
            # Convert any non-serializable values
            serializable_results = {}
            for key, value in training_results.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    serializable_results[key] = value
                elif isinstance(value, list):
                    serializable_results[key] = [float(v) if isinstance(v, torch.Tensor) else v for v in value]
                else:
                    serializable_results[key] = str(value)
            
            json.dump(serializable_results, f, indent=4)
        
        # Evaluate model
        eval_results = evaluate_model(
            model=model,
            val_dataset=val_dataset,
            feature_groups=feature_groups,
            output_dir=output_dir,
            use_gpu=args.gpu
        )
        
        # Visualize results
        visualize_results(
            eval_results=eval_results,
            feature_groups=feature_groups,
            output_dir=output_dir
        )
        
        # Save evaluation results
        with open(output_dir / "evaluation_results.json", 'w') as f:
            # Convert any non-serializable values
            serializable_eval = {}
            for key, value in eval_results.items():
                if key == "analysis":
                    serializable_eval[key] = {
                        name: {k: float(v) if isinstance(v, (torch.Tensor, np.float32, np.float64)) else v 
                               for k, v in metrics.items() if not isinstance(v, torch.Tensor)}
                        for name, metrics in value.items()
                    }
                elif isinstance(value, dict):
                    serializable_eval[key] = {k: float(v) if isinstance(v, (torch.Tensor, np.float32, np.float64)) else v 
                                            for k, v in value.items()}
                else:
                    serializable_eval[key] = float(value) if isinstance(value, (torch.Tensor, np.float32, np.float64)) else value
            
            json.dump(serializable_eval, f, indent=4)
        
        logging.info("Experiment completed successfully!")
        logging.info(f"Results saved to {output_dir}")
        
    except Exception as e:
        logging.error(f"Error during experiment: {e}", exc_info=True)
        
if __name__ == "__main__":
    main() 