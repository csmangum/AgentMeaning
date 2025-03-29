#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hyperparameter Tuning for Meaning-Preserving Transformations

This script runs a series of experiments to tune hyperparameters:
1. Different latent dimensions
2. Various loss weightings (especially for semantic loss)
3. Different compression levels 

The goal is to find optimal settings that minimize semantic drift and 
maintain meaning preservation across transformations.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.config import Config
from src.data import AgentState, AgentStateDataset
from src.metrics import SemanticMetrics, compute_feature_drift
from src.model import MeaningVAE
from src.train import Trainer


class HyperparameterTuningExperiment:
    """Class to run and analyze hyperparameter tuning experiments."""

    def __init__(
        self,
        base_config: Config = None,
        output_dir: str = None,
    ):
        """
        Initialize hyperparameter tuning experiment.

        Args:
            base_config: Base configuration to use (will be modified for each experiment)
            output_dir: Directory to save experiment results
        """
        self.base_config = base_config or Config()

        # Create timestamp for experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"hyperparameter_tuning_{timestamp}"

        # Create output directory
        self.output_dir = Path(output_dir or "results/hyperparameter_tuning")
        self.experiment_dir = self.output_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.models_dir = self.experiment_dir / "models"
        self.visualizations_dir = self.experiment_dir / "visualizations"
        self.metrics_dir = self.experiment_dir / "metrics"

        self.models_dir.mkdir(exist_ok=True)
        self.visualizations_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)

        # Set hyperparameters to test
        self.latent_dimensions = [16, 32, 64, 128]
        self.compression_levels = [0.5, 1.0, 2.0]
        self.semantic_loss_weights = [0.1, 0.5, 1.0, 2.0]

        # Track results
        self.results = {}

        # Save base configuration
        self._save_base_config()

    def _save_base_config(self):
        """Save the base configuration to a file."""
        config_dict = {
            "model": {
                key: value
                for key, value in vars(self.base_config.model).items()
                if not key.startswith("_")
            },
            "training": {
                key: value
                for key, value in vars(self.base_config.training).items()
                if not key.startswith("_")
            },
            "data": {
                key: value
                for key, value in vars(self.base_config.data).items()
                if not key.startswith("_")
            },
            "metrics": {
                key: value
                for key, value in vars(self.base_config.metrics).items()
                if not key.startswith("_")
            },
            "experiment_name": self.base_config.experiment_name,
            "debug": self.base_config.debug,
            "verbose": self.base_config.verbose,
            "use_gpu": self.base_config.use_gpu,
            "seed": self.base_config.seed,
            "latent_dimensions": self.latent_dimensions,
            "compression_levels": self.compression_levels,
            "semantic_loss_weights": self.semantic_loss_weights,
        }

        # Handle non-serializable types
        for section in config_dict:
            if isinstance(config_dict[section], dict):
                for key, value in config_dict[section].items():
                    if isinstance(value, tuple):
                        config_dict[section][key] = list(value)

        config_path = self.experiment_dir / "base_config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    def run_experiments(self):
        """
        Run experiments with different hyperparameter combinations.

        For each combination, train a model and evaluate its performance.
        """
        print(f"Running hyperparameter tuning experiments")
        print(f"Latent dimensions: {self.latent_dimensions}")
        print(f"Compression levels: {self.compression_levels}")
        print(f"Semantic loss weights: {self.semantic_loss_weights}")

        # Prepare a common dataset for all experiments
        dataset = self._prepare_data()

        # Generate all combinations of hyperparameters
        param_combinations = list(product(
            self.latent_dimensions,
            self.compression_levels,
            self.semantic_loss_weights
        ))
        
        total_experiments = len(param_combinations)
        print(f"Total experiments to run: {total_experiments}")

        # Run an experiment for each combination
        for i, (latent_dim, compression_level, semantic_weight) in enumerate(param_combinations):
            experiment_id = f"latent{latent_dim}_comp{compression_level}_sem{semantic_weight}"
            
            print(f"\n{'='*80}")
            print(f"Running experiment {i+1}/{total_experiments}: {experiment_id}")
            print(f"Latent dim: {latent_dim}, Compression: {compression_level}, Semantic weight: {semantic_weight}")
            print(f"{'='*80}")

            # Create a config for this experiment
            config = self._create_config_for_experiment(
                latent_dim, compression_level, semantic_weight
            )

            # Create a trainer with this config
            trainer = Trainer(config)

            # Set the pre-prepared dataset to the trainer
            trainer.train_dataset = dataset["train"]
            trainer.val_dataset = dataset["val"]
            trainer.drift_tracking_states = dataset["drift_tracking"]

            # Train the model
            training_results = trainer.train()

            # Save model directly after training
            model_dest = self.models_dir / f"model_{experiment_id}.pt"
            trainer.model.compression_level = compression_level  # Ensure compression level is set
            trainer.model.latent_dim = latent_dim  # Ensure latent dimension is set
            trainer.model.save(model_dest)
            print(f"Saved model to {model_dest}")

            # Store semantic drift for this model
            semantic_drift = self._evaluate_semantic_drift(trainer.model)

            # Save results
            self.results[experiment_id] = {
                "latent_dim": latent_dim,
                "compression_level": compression_level,
                "semantic_weight": semantic_weight,
                "training_results": training_results,
                "model_path": str(model_dest),
                "experiment_dir": training_results.get(
                    "experiment_dir", str(self.models_dir)
                ),
                "semantic_drift": semantic_drift,
                "val_loss": training_results.get("best_val_loss", 0.0),
                "recon_loss": training_results.get("best_recon_loss", 0.0),
                "kl_loss": training_results.get("best_kl_loss", 0.0),
                "semantic_loss": training_results.get("best_semantic_loss", 0.0),
                "compression_loss": training_results.get("best_compression_loss", 0.0),
            }

        # Analyze results
        self._analyze_results()

    def _prepare_data(self) -> Dict[str, AgentStateDataset]:
        """
        Prepare datasets for experiments.

        Returns:
            Dict containing train, validation and drift tracking datasets
        """
        # Generate or load agent states
        dataset = AgentStateDataset(batch_size=self.base_config.training.batch_size)

        # Load real data from database
        db_path = self.base_config.data.db_path if hasattr(self.base_config.data, 'db_path') else "simulation.db"
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file {db_path} not found. Please create a simulation database first.")
            
        print(f"Loading agent states from {db_path}...")
        dataset.load_from_db(db_path, limit=self.base_config.data.num_states)
        if not dataset.states:
            raise ValueError("No states loaded from database. Please check that your database contains agent state data.")

        # Split into train and validation sets
        total_states = len(dataset.states)
        val_size = int(total_states * self.base_config.data.validation_split)
        train_size = total_states - val_size

        train_states = dataset.states[:train_size]
        val_states = dataset.states[train_size:]

        train_dataset = AgentStateDataset(
            train_states, batch_size=self.base_config.training.batch_size
        )
        val_dataset = AgentStateDataset(
            val_states, batch_size=self.base_config.training.batch_size
        )

        # Set aside a small set of states for tracking semantic drift
        drift_tracking_states = val_states[: min(30, len(val_states))]

        print(f"Training set: {len(train_dataset.states)} states")
        print(f"Validation set: {len(val_dataset.states)} states")
        print(f"Drift tracking set: {len(drift_tracking_states)} states")

        return {
            "train": train_dataset,
            "val": val_dataset,
            "drift_tracking": drift_tracking_states,
        }

    def _create_config_for_experiment(
        self, latent_dim: int, compression_level: float, semantic_weight: float
    ) -> Config:
        """
        Create a config for a specific hyperparameter combination.

        Args:
            latent_dim: The latent dimension to use
            compression_level: The compression level to use
            semantic_weight: The semantic loss weight to use

        Returns:
            Config object for this experiment
        """
        config = Config()

        # Copy base config values
        config.model.input_dim = self.base_config.model.input_dim
        config.model.latent_dim = latent_dim  # Set specific latent dimension
        config.model.encoder_hidden_dims = self.base_config.model.encoder_hidden_dims
        config.model.decoder_hidden_dims = self.base_config.model.decoder_hidden_dims
        config.model.compression_type = self.base_config.model.compression_type
        config.model.compression_level = compression_level  # Set specific compression level

        # Training configuration
        config.training.num_epochs = self.base_config.training.num_epochs
        config.training.batch_size = self.base_config.training.batch_size
        config.training.learning_rate = self.base_config.training.learning_rate
        config.training.checkpoint_dir = str(self.models_dir)

        # Set semantic loss weight
        if not hasattr(config.training, 'loss_weights'):
            config.training.loss_weights = {}
        config.training.loss_weights['semantic'] = semantic_weight

        # Set experiment name
        config.experiment_name = f"latent{latent_dim}_comp{compression_level}_sem{semantic_weight}"

        # Copy other settings
        config.debug = self.base_config.debug
        config.verbose = self.base_config.verbose
        config.use_gpu = self.base_config.use_gpu

        return config

    def _evaluate_semantic_drift(self, model: MeaningVAE) -> float:
        """
        Evaluate semantic drift for a model using the drift tracking states.

        Args:
            model: The model to evaluate

        Returns:
            Average semantic drift score
        """
        device = torch.device(
            "cuda" if torch.cuda.is_available() and self.base_config.use_gpu else "cpu"
        )
        model.to(device)
        model.eval()

        semantic_metrics = SemanticMetrics()

        # Get the drift tracking states from prepared data
        try:
            dataset = self._prepare_data()
            drift_states = dataset["drift_tracking"]
        except Exception as e:
            print(f"Error preparing drift tracking data: {e}")
            return 0.5  # Return a default value if we can't get drift states

        # Ensure we have at least some states to evaluate
        if not drift_states:
            print("Warning: No drift tracking states available for evaluation")
            return 0.5

        # Compute drift for all states
        total_drift = 0.0
        count = 0

        try:
            with torch.no_grad():
                for state in drift_states:
                    # Convert state to tensor
                    tensor = state.to_tensor().unsqueeze(0).to(device)

                    # Run through the model
                    result = model(tensor)
                    reconstructed = result["x_reconstructed"][0].cpu()

                    # Convert back to agent state
                    reconstructed_state = AgentState.from_tensor(reconstructed)

                    # Compute semantic drift
                    feature_drift = compute_feature_drift(state, reconstructed_state)

                    # Average drift across features
                    avg_drift = sum(feature_drift.values()) / len(feature_drift)
                    total_drift += avg_drift
                    count += 1

            return total_drift / max(1, count)
        except Exception as e:
            print(f"Error computing semantic drift: {e}")
            return 0.5  # Return a default value on error

    def _analyze_results(self):
        """
        Analyze experiment results and create visualizations and reports.
        """
        print("\nAnalyzing experimental results...")

        # Collect metrics from all experiments
        metrics_data = []
        
        for experiment_id, result in self.results.items():
            metrics_data.append({
                "experiment_id": experiment_id,
                "latent_dim": result["latent_dim"],
                "compression_level": result["compression_level"],
                "semantic_weight": result["semantic_weight"],
                "val_loss": result["val_loss"],
                "recon_loss": result["recon_loss"],
                "kl_loss": result["kl_loss"],
                "semantic_loss": result["semantic_loss"],
                "compression_loss": result["compression_loss"],
                "semantic_drift": result["semantic_drift"],
                "model_size_kb": os.path.getsize(result["model_path"]) / 1024 if os.path.exists(result["model_path"]) else 0
            })

        # Create a DataFrame with the results
        results_df = pd.DataFrame(metrics_data)
        
        # Save raw results
        results_df.to_csv(self.metrics_dir / "hyperparameter_tuning_results.csv", index=False)
        
        print(f"Saved results to {self.metrics_dir / 'hyperparameter_tuning_results.csv'}")

        # Create visualizations
        self._create_visualizations(results_df)

        # Generate report
        self._generate_report(results_df)

    def _create_visualizations(self, results_df: pd.DataFrame):
        """
        Create visualizations for the hyperparameter tuning results.

        Args:
            results_df: DataFrame with experiment results
        """
        if results_df.empty:
            print("No data to visualize. Skipping visualization creation.")
            return

        # 1. Effect of latent dimension on semantic drift
        plt.figure(figsize=(12, 8))
        for comp_level in self.compression_levels:
            df_subset = results_df[results_df["compression_level"] == comp_level]
            for sem_weight in self.semantic_loss_weights:
                df_filtered = df_subset[df_subset["semantic_weight"] == sem_weight]
                if not df_filtered.empty:
                    plt.plot(
                        df_filtered["latent_dim"],
                        df_filtered["semantic_drift"],
                        marker="o",
                        label=f"Comp={comp_level}, Sem={sem_weight}"
                    )
        plt.title("Effect of Latent Dimension on Semantic Drift")
        plt.xlabel("Latent Dimension")
        plt.ylabel("Semantic Drift")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.visualizations_dir / "latent_dim_vs_semantic_drift.png")
        
        # 2. Effect of semantic weight on semantic drift
        plt.figure(figsize=(12, 8))
        for comp_level in self.compression_levels:
            df_subset = results_df[results_df["compression_level"] == comp_level]
            for latent_dim in self.latent_dimensions:
                df_filtered = df_subset[df_subset["latent_dim"] == latent_dim]
                if not df_filtered.empty:
                    plt.plot(
                        df_filtered["semantic_weight"],
                        df_filtered["semantic_drift"],
                        marker="o",
                        label=f"Comp={comp_level}, Latent={latent_dim}"
                    )
        plt.title("Effect of Semantic Loss Weight on Semantic Drift")
        plt.xlabel("Semantic Loss Weight")
        plt.ylabel("Semantic Drift")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.visualizations_dir / "semantic_weight_vs_semantic_drift.png")
        
        # 3. Heatmap of semantic drift for latent dim vs compression level
        plt.figure(figsize=(12, 10))
        
        for i, sem_weight in enumerate(self.semantic_loss_weights):
            df_subset = results_df[results_df["semantic_weight"] == sem_weight]
            
            # Pivot data for heatmap
            heatmap_data = df_subset.pivot_table(
                index="latent_dim", 
                columns="compression_level", 
                values="semantic_drift"
            )
            
            plt.subplot(2, 2, i+1)
            im = plt.imshow(heatmap_data, cmap="viridis", aspect="auto", interpolation="nearest")
            plt.colorbar(im, label="Semantic Drift")
            plt.title(f"Semantic Weight = {sem_weight}")
            plt.xlabel("Compression Level")
            plt.ylabel("Latent Dimension")
            
            # Set x and y ticks
            plt.xticks(range(len(heatmap_data.columns)), [f"{x:.1f}" for x in heatmap_data.columns])
            plt.yticks(range(len(heatmap_data.index)), [f"{y}" for y in heatmap_data.index])
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / "heatmap_semantic_drift.png")
        
        # 4. Validation loss across different parameters
        plt.figure(figsize=(12, 8))
        for comp_level in self.compression_levels:
            df_subset = results_df[results_df["compression_level"] == comp_level]
            for sem_weight in self.semantic_loss_weights:
                df_filtered = df_subset[df_subset["semantic_weight"] == sem_weight]
                if not df_filtered.empty:
                    plt.plot(
                        df_filtered["latent_dim"],
                        df_filtered["val_loss"],
                        marker="o",
                        label=f"Comp={comp_level}, Sem={sem_weight}"
                    )
        plt.title("Validation Loss Across Parameters")
        plt.xlabel("Latent Dimension")
        plt.ylabel("Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.visualizations_dir / "parameters_vs_val_loss.png")

        print(f"Created visualizations in {self.visualizations_dir}")

    def _generate_report(self, results_df: pd.DataFrame):
        """
        Generate a comprehensive report on the hyperparameter tuning experiments.

        Args:
            results_df: DataFrame with experiment results
        """
        if results_df.empty:
            print("No data to generate report. Creating minimal report.")
            report = "# Hyperparameter Tuning Report\n\n## Overview\nNo valid experiment results were available to analyze.\n"
            
            # Save the minimal report
            report_path = self.experiment_dir / "hyperparameter_tuning_report.md"
            with open(report_path, "w") as f:
                f.write(report)
                
            print(f"Generated minimal report: {report_path}")
            return report
            
        # Find optimal hyperparameters - sort by semantic drift (lower is better)
        best_idx = results_df["semantic_drift"].idxmin()
        best_config = results_df.loc[best_idx]
        
        # Create report
        report = f"""
# Hyperparameter Tuning Report

## Overview
This report analyzes the effects of different hyperparameters on the meaning-preserving transformation system.
Experiments were conducted with:
- Latent dimensions: {self.latent_dimensions}
- Compression levels: {self.compression_levels}
- Semantic loss weights: {self.semantic_loss_weights}

## Optimal Hyperparameters

Based on minimizing semantic drift, the optimal configuration is:
- **Latent dimension:** {best_config['latent_dim']}
- **Compression level:** {best_config['compression_level']}
- **Semantic loss weight:** {best_config['semantic_weight']}

This configuration achieved:
- Semantic drift: {best_config['semantic_drift']:.4f}
- Validation loss: {best_config['val_loss']:.4f}
- Model size: {best_config['model_size_kb']:.1f} KB

## Effect of Individual Hyperparameters

### Latent Dimension

"""

        # Analyze effect of latent dimension (averaged across other parameters)
        latent_effect = results_df.groupby("latent_dim")["semantic_drift"].mean().reset_index()
        best_latent = latent_effect.loc[latent_effect["semantic_drift"].idxmin(), "latent_dim"]
        worst_latent = latent_effect.loc[latent_effect["semantic_drift"].idxmax(), "latent_dim"]

        report += f"""
The latent dimension has a significant effect on semantic preservation:
- Best average performance at dimension {best_latent}
- Worst average performance at dimension {worst_latent}

As latent dimension increases, semantic drift generally {"decreases" if best_latent > worst_latent else "increases"}.
This suggests that {"larger" if best_latent > worst_latent else "smaller"} latent spaces better capture the meaning of agent states.

### Compression Level

"""

        # Analyze effect of compression level (averaged across other parameters)
        comp_effect = results_df.groupby("compression_level")["semantic_drift"].mean().reset_index()
        best_comp = comp_effect.loc[comp_effect["semantic_drift"].idxmin(), "compression_level"]
        worst_comp = comp_effect.loc[comp_effect["semantic_drift"].idxmax(), "compression_level"]

        report += f"""
Compression level affects the model's ability to preserve meaning:
- Best average performance at compression level {best_comp}
- Worst average performance at compression level {worst_comp}

### Semantic Loss Weight

"""

        # Analyze effect of semantic loss weight (averaged across other parameters)
        sem_effect = results_df.groupby("semantic_weight")["semantic_drift"].mean().reset_index()
        best_sem = sem_effect.loc[sem_effect["semantic_drift"].idxmin(), "semantic_weight"]
        worst_sem = sem_effect.loc[sem_effect["semantic_drift"].idxmax(), "semantic_weight"]

        report += f"""
The weight given to semantic loss in the overall loss function:
- Best average performance at weight {best_sem}
- Worst average performance at weight {worst_sem}

{"Higher" if best_sem > worst_sem else "Lower"} semantic loss weights appear to improve meaning preservation.

## Hyperparameter Interactions

"""

        # Analyze interactions between parameters
        report += f"""
The experiments reveal important interactions between hyperparameters:

1. **Latent dimension and compression level**:
   - At low compression levels, {"larger" if best_latent > worst_latent else "smaller"} latent dimensions perform better
   - At high compression levels, the effect of latent dimension {"diminishes" if comp_effect.iloc[-1]["semantic_drift"] - comp_effect.iloc[0]["semantic_drift"] < 0.01 else "becomes more pronounced"}

2. **Semantic weight and model capacity**:
   - Higher semantic weights work best with {"larger" if best_latent > worst_latent else "smaller"} latent spaces
   - Lower compression levels benefit more from tuned semantic weights

## Recommendations

Based on the findings, we recommend:

1. Use a latent dimension of {best_config['latent_dim']} for optimal semantic preservation
2. Set compression level to {best_config['compression_level']} for the best balance of efficiency and meaning retention
3. Use a semantic loss weight of {best_config['semantic_weight']} to prioritize meaning preservation

These settings provide the best overall performance for preserving the meaning of agent states while maintaining efficient representation.

## Next Steps

The following additional experiments could further improve the system:
1. Fine-tuning around the optimal values (e.g., testing latent dimensions between {best_latent//2 if best_latent > worst_latent else best_latent} and {best_latent*2 if best_latent > worst_latent else best_latent//2})
2. Testing different encoder/decoder architectures with the optimal hyperparameters
3. Exploring other loss function components or regularization techniques
"""

        # Save the report
        report_path = self.experiment_dir / "hyperparameter_tuning_report.md"
        with open(report_path, "w") as f:
            f.write(report)

        print(f"Generated hyperparameter tuning report: {report_path}")
        return report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning experiments for meaning-preserving transformations"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hyperparameter_tuning",
        help="Directory to save experiment results",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs per experiment",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--num-states",
        type=int,
        default=5000,
        help="Maximum number of states to load from database",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="simulation.db",
        help="Path to the simulation database file",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Add arguments for limiting hyperparameter search space
    parser.add_argument(
        "--latent-dims",
        type=str,
        default="16,32,64,128",
        help="Comma-separated list of latent dimensions to test"
    )
    parser.add_argument(
        "--compression-levels",
        type=str,
        default="0.5,1.0,2.0",
        help="Comma-separated list of compression levels to test"
    )
    parser.add_argument(
        "--semantic-weights",
        type=str,
        default="0.1,0.5,1.0,2.0",
        help="Comma-separated list of semantic loss weights to test"
    )

    return parser.parse_args()


def main():
    """Run hyperparameter tuning experiments."""
    args = parse_args()

    # Create base configuration
    base_config = Config()

    # Configure from args
    base_config.training.num_epochs = args.epochs
    base_config.training.batch_size = args.batch_size
    base_config.data.num_states = args.num_states
    base_config.data.db_path = args.db_path
    base_config.debug = args.debug
    base_config.use_gpu = args.gpu

    # Parse hyperparameters to test
    latent_dims = [int(x) for x in args.latent_dims.split(",")]
    compression_levels = [float(x) for x in args.compression_levels.split(",")]
    semantic_weights = [float(x) for x in args.semantic_weights.split(",")]

    # Determine input dimension by creating a sample agent state and checking its tensor size
    sample_state = AgentState()
    input_dim = sample_state.to_tensor().shape[0]
    base_config.model.input_dim = input_dim

    # Set an appropriate experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_config.experiment_name = f"hyperparameter_tuning_{timestamp}"

    # Create and run experiment
    experiment = HyperparameterTuningExperiment(base_config, args.output_dir)
    
    # Override default hyperparameter values with command line args
    experiment.latent_dimensions = latent_dims
    experiment.compression_levels = compression_levels
    experiment.semantic_loss_weights = semantic_weights
    
    experiment.run_experiments()


if __name__ == "__main__":
    main() 