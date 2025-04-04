#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compression Experiments for Meaning-Preserving Transformations

This script runs a series of compression experiments to analyze how different
compression levels affect semantic preservation in the meaning-preserving transformation system.

Experiments:
1. Run the model with varying compression levels (0.5, 1.0, 2.0, 5.0)
2. Analyze how different compression rates affect semantic preservation
3. Create visualization comparisons between compression levels
4. Identify optimal compression setting for balancing information density with meaning retention
5. Document findings in a compression analysis report
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from meaning_transform.src.config import Config
from meaning_transform.src.data import AgentState, AgentStateDataset
from meaning_transform.src.metrics import compute_feature_drift
from meaning_transform.src.model import MeaningVAE
from meaning_transform.src.standardized_metrics import StandardizedMetrics
from meaning_transform.src.train import Trainer


class CompressionExperiment:
    """Experiment runner for compression studies."""

    def __init__(self, base_config: Config, output_dir: str = "results/compression_experiments"):
        """
        Initialize compression experiment.

        Args:
            base_config: Base configuration for all experiments
            output_dir: Directory to store results
        """
        self.base_config = base_config
        self.compression_levels = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0]
        
        # Initialize drift tracking states
        self.drift_tracking_states = []

        # Create results storage
        self.results = {}

        # Create output directories
        self.experiment_dir = Path(output_dir) / f"{base_config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = self.experiment_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.metrics_dir = self.experiment_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)

        self.visualizations_dir = self.experiment_dir / "visualizations"
        self.visualizations_dir.mkdir(exist_ok=True)

        # Save the base configuration
        self.save_base_config()

    def save_base_config(self):
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
            "compression_levels": self.compression_levels,
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
        Run experiments with different compression levels.

        For each compression level, train a model and evaluate its performance.
        """
        print(f"Running compression experiments with levels: {self.compression_levels}")

        # Prepare a common dataset for all experiments
        dataset = self._prepare_data()
        
        # Store drift tracking states at the experiment level for evaluation
        self.drift_tracking_states = dataset["drift_tracking"]

        # Run an experiment for each compression level
        for level in self.compression_levels:
            print(f"\n{'='*80}")
            print(f"Running experiment with compression level: {level}")
            print(f"{'='*80}")

            # Create a config for this experiment
            config = self._create_config_for_level(level)

            # Create a trainer with this config
            trainer = Trainer(config)

            # Set the pre-prepared dataset to the trainer
            trainer.train_dataset = dataset["train"]
            trainer.val_dataset = dataset["val"]
            trainer.drift_tracking_states = self.drift_tracking_states

            # Train the model
            training_results = trainer.train()

            # Save model directly after training
            model_dest = self.models_dir / f"model_compression_{level}.pt"
            trainer.model.compression_level = level  # Ensure compression level is set
            trainer.model.save(model_dest)
            print(f"Saved model to {model_dest}")

            # Store semantic drift for this model
            semantic_drift = self._evaluate_semantic_drift(trainer.model)

            # Save results
            self.results[level] = {
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
        db_path = (
            self.base_config.data.db_path
            if hasattr(self.base_config.data, "db_path")
            else "data/simulation.db"
        )
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Database file {db_path} not found. Please create a simulation database first."
            )

        print(f"Loading agent states from {db_path}...")
        dataset.load_from_db(db_path, limit=self.base_config.data.num_states)
        if not dataset.states:
            raise ValueError(
                "No states loaded from database. Please check that your database contains agent state data."
            )

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

    def _create_config_for_level(self, compression_level: float) -> Config:
        """
        Create a config for a specific compression level.

        Args:
            compression_level: The compression level to use

        Returns:
            Config object for this experiment
        """
        config = Config()

        # Copy base config values
        config.model.input_dim = self.base_config.model.input_dim
        config.model.latent_dim = self.base_config.model.latent_dim
        config.model.encoder_hidden_dims = self.base_config.model.encoder_hidden_dims
        config.model.decoder_hidden_dims = self.base_config.model.decoder_hidden_dims
        config.model.compression_type = self.base_config.model.compression_type

        # Set compression level for this experiment
        config.model.compression_level = compression_level

        # Training configuration
        config.training.num_epochs = self.base_config.training.num_epochs
        config.training.batch_size = self.base_config.training.batch_size
        config.training.learning_rate = self.base_config.training.learning_rate
        config.training.checkpoint_dir = str(self.models_dir)

        # Set experiment name
        config.experiment_name = f"compression_{compression_level}"

        # Copy other settings
        config.debug = self.base_config.debug
        config.verbose = self.base_config.verbose
        config.use_gpu = self.base_config.use_gpu

        return config

    def _analyze_results(self):
        """
        Analyze results of all compression experiments.
        
        This includes:
        1. Creating comparative visualizations
        2. Generating summary metrics
        3. Creating a comprehensive report
        """
        print("\nAnalyzing compression experiment results...")
        
        # Create a DataFrame from results
        results_data = []
        for level, metrics in self.results.items():
            row = {
                "compression_level": level,
                "val_loss": metrics["val_loss"],
                "recon_loss": metrics["recon_loss"],
                "kl_loss": metrics["kl_loss"],
                "semantic_loss": metrics["semantic_loss"],
                "compression_loss": metrics.get("compression_loss", 0.0),
            }
            
            # Add the standardized metrics
            semantic_drift = metrics["semantic_drift"]
            row.update({
                "overall_drift": semantic_drift["overall_drift"],
                "preservation_score": semantic_drift["preservation"],
                "fidelity_score": semantic_drift["fidelity"],
                "drift_category": semantic_drift["drift_category"],
                "spatial_drift": semantic_drift["spatial_drift"],
                "resources_drift": semantic_drift["resources_drift"],
                "performance_drift": semantic_drift["performance_drift"],
                "role_drift": semantic_drift["role_drift"],
            })
            
            results_data.append(row)
        
        # Create DataFrame and sort by compression level
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values("compression_level")
        
        # Save results to CSV
        results_csv_path = self.metrics_dir / "compression_results.csv"
        results_df.to_csv(results_csv_path, index=False)
        print(f"Saved results to {results_csv_path}")
        
        # Create visualizations
        self._create_visualizations(results_df)
        
        # Generate report
        self._generate_report(results_df)

    def _evaluate_semantic_drift(self, model: MeaningVAE) -> Dict[str, Any]:
        """
        Evaluate semantic drift for a given model.

        Args:
            model: Trained model to evaluate

        Returns:
            Dictionary of semantic drift metrics
        """
        # Create standardized metrics instance
        metrics = StandardizedMetrics()
        
        # Get device from model parameters
        device = next(model.parameters()).device

        # Apply model to drift tracking states
        original_tensors = []
        reconstructed_tensors = []

        # Process states
        for state in self.drift_tracking_states:
            # Convert to tensor
            x = state.to_tensor().unsqueeze(0)
            
            # Move to device
            x = x.to(device)
            
            # Run through model
            with torch.no_grad():
                # Model returns a dictionary with reconstructed data
                model_output = model(x)
                # Get the reconstruction from the output
                recon_x = model_output["reconstruction"]
            
            # Add to lists for batch evaluation
            original_tensors.append(x)
            reconstructed_tensors.append(recon_x)
            
        # Concatenate tensors
        originals = torch.cat(original_tensors, dim=0)
        reconstructions = torch.cat(reconstructed_tensors, dim=0)
        
        # Use standardized metrics to comprehensively evaluate the model
        evaluation_results = metrics.evaluate(originals, reconstructions)
        
        # Extract the key metrics we're interested in for experiments
        semantic_drift = {
            "overall_drift": evaluation_results["drift"]["overall_drift"],
            "preservation": evaluation_results["preservation"]["overall_preservation"],
            "fidelity": evaluation_results["fidelity"]["overall_fidelity"],
            "drift_category": evaluation_results["drift"]["drift_category"],
            "spatial_drift": evaluation_results["drift"].get("spatial_drift", 0.0),
            "resources_drift": evaluation_results["drift"].get("resources_drift", 0.0),
            "performance_drift": evaluation_results["drift"].get("performance_drift", 0.0),
            "role_drift": evaluation_results["drift"].get("role_drift", 0.0)
        }
        
        return semantic_drift

    def _create_visualizations(self, results_df: pd.DataFrame):
        """
        Create visualizations for comparison across compression levels.
        
        Args:
            results_df: DataFrame containing experiment results
        """
        # Create the main comparison plot
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Overall drift vs Compression Level
        plt.subplot(2, 2, 1)
        plt.plot(results_df["compression_level"], results_df["overall_drift"], "r-o", linewidth=2)
        plt.xlabel("Compression Level")
        plt.ylabel("Overall Semantic Drift")
        plt.title("Semantic Drift vs. Compression Level")
        plt.grid(True)
        
        # Plot 2: Preservation Score vs Compression Level
        plt.subplot(2, 2, 2)
        plt.plot(results_df["compression_level"], results_df["preservation_score"], "g-o", linewidth=2)
        plt.xlabel("Compression Level")
        plt.ylabel("Meaning Preservation Score")
        plt.title("Meaning Preservation vs. Compression Level")
        plt.grid(True)
        
        # Plot 3: Feature-specific drift scores
        plt.subplot(2, 2, 3)
        plt.plot(results_df["compression_level"], results_df["spatial_drift"], "r-o", label="Spatial")
        plt.plot(results_df["compression_level"], results_df["resources_drift"], "g-o", label="Resources")
        plt.plot(results_df["compression_level"], results_df["performance_drift"], "b-o", label="Performance")
        plt.plot(results_df["compression_level"], results_df["role_drift"], "m-o", label="Role")
        plt.xlabel("Compression Level")
        plt.ylabel("Feature Group Drift")
        plt.title("Feature-Specific Drift vs. Compression Level")
        plt.legend()
        plt.grid(True)
        
        # Plot 4: Loss metrics
        plt.subplot(2, 2, 4)
        plt.plot(results_df["compression_level"], results_df["val_loss"], "k-o", label="Validation Loss")
        plt.plot(results_df["compression_level"], results_df["recon_loss"], "b-o", label="Reconstruction Loss")
        plt.plot(results_df["compression_level"], results_df["kl_loss"], "g-o", label="KL Loss")
        plt.xlabel("Compression Level")
        plt.ylabel("Loss Value")
        plt.title("Training Losses vs. Compression Level")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / "compression_comparison.png")
        plt.close()
        
        # Create a radar chart to compare all metrics at different compression levels
        self._create_radar_chart(results_df)
        
        # Create a drift category visualization
        self._create_drift_category_chart(results_df)

    def _create_radar_chart(self, results_df: pd.DataFrame):
        """
        Create a radar chart to compare all metrics at different compression levels.
        
        Args:
            results_df: DataFrame containing experiment results
        """
        # Create a radar chart
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Define the metrics to be plotted
        metrics = ["overall_drift", "preservation_score", "fidelity_score", "spatial_drift", "resources_drift", "performance_drift", "role_drift"]
        
        # Define the colors for each metric
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(metrics)))
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            values = results_df[metric]
            ax.plot(np.linspace(0, 2 * np.pi, len(values)), values, color=colors[i], label=metric)
            ax.fill(np.linspace(0, 2 * np.pi, len(values)), values, color=colors[i], alpha=0.2)
        
        # Set the title and labels
        ax.set_thetagrids(np.linspace(0, 2 * np.pi, len(metrics)), metrics)
        ax.set_title("Comparison of Metrics Across Compression Levels")
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Save the chart
        plt.savefig(self.visualizations_dir / "radar_chart.png")
        plt.close()

    def _create_drift_category_chart(self, results_df: pd.DataFrame):
        """
        Create a drift category visualization for the experiment results.
        
        Args:
            results_df: DataFrame containing experiment results
        """
        # Create a bar chart for drift categories
        drift_categories = results_df["drift_category"].value_counts().index
        counts = results_df["drift_category"].value_counts().values
        
        plt.figure(figsize=(10, 6))
        plt.bar(drift_categories, counts)
        plt.xlabel("Drift Category")
        plt.ylabel("Count")
        plt.title("Drift Category Distribution")
        plt.savefig(self.visualizations_dir / "drift_category_distribution.png")
        plt.close()

    def _generate_report(self, results_df: pd.DataFrame):
        """
        Generate a comprehensive report for the compression experiments.
        
        Args:
            results_df: DataFrame containing experiment results
        """
        report_file = self.experiment_dir / "compression_report.md"
        
        with open(report_file, "w") as f:
            # Write header
            f.write("# Compression Experiment Report\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write summary
            f.write("## Summary\n\n")
            f.write(f"Tested compression levels: {', '.join(map(str, self.compression_levels))}\n\n")
            
            # Find the best compression level based on combined metrics
            # Lower drift, higher preservation, higher fidelity
            results_df['combined_score'] = (
                (1 - results_df['overall_drift']) * 0.4 + 
                results_df['preservation_score'] * 0.3 + 
                results_df['fidelity_score'] * 0.3
            )
            
            best_level = results_df.loc[results_df['combined_score'].idxmax()]['compression_level']
            f.write(f"**Best overall compression level: {best_level}**\n\n")
            
            # Create a summary table
            f.write("## Results Summary\n\n")
            f.write("| Compression Level | Overall Drift | Preservation | Fidelity | Drift Category |\n")
            f.write("|-------------------|--------------|--------------|----------|----------------|\n")
            
            for _, row in results_df.sort_values('compression_level').iterrows():
                f.write(f"| {row['compression_level']} | {row['overall_drift']:.4f} | {row['preservation_score']:.4f} | {row['fidelity_score']:.4f} | {row['drift_category']} |\n")
            
            f.write("\n")
            
            # Feature-specific analysis
            f.write("## Feature-Specific Analysis\n\n")
            f.write("### Spatial Features (55.4% importance)\n\n")
            f.write("| Compression Level | Spatial Drift |\n")
            f.write("|-------------------|---------------|\n")
            for _, row in results_df.sort_values('compression_level').iterrows():
                f.write(f"| {row['compression_level']} | {row['spatial_drift']:.4f} |\n")
            
            f.write("\n### Resource Features (25.1% importance)\n\n")
            f.write("| Compression Level | Resources Drift |\n")
            f.write("|-------------------|------------------|\n")
            for _, row in results_df.sort_values('compression_level').iterrows():
                f.write(f"| {row['compression_level']} | {row['resources_drift']:.4f} |\n")
            
            f.write("\n### Performance Features (10.5% importance)\n\n")
            f.write("| Compression Level | Performance Drift |\n")
            f.write("|-------------------|-------------------|\n")
            for _, row in results_df.sort_values('compression_level').iterrows():
                f.write(f"| {row['compression_level']} | {row['performance_drift']:.4f} |\n")
            
            f.write("\n### Role Features (<5% importance)\n\n")
            f.write("| Compression Level | Role Drift |\n")
            f.write("|-------------------|------------|\n")
            for _, row in results_df.sort_values('compression_level').iterrows():
                f.write(f"| {row['compression_level']} | {row['role_drift']:.4f} |\n")
            
            f.write("\n## Recommendations\n\n")
            
            # Find best compression level for each feature group
            best_spatial = results_df.loc[results_df['spatial_drift'].idxmin()]['compression_level']
            best_resources = results_df.loc[results_df['resources_drift'].idxmin()]['compression_level']
            best_performance = results_df.loc[results_df['performance_drift'].idxmin()]['compression_level']
            best_role = results_df.loc[results_df['role_drift'].idxmin()]['compression_level']
            
            f.write(f"- **Best for Spatial Features**: Compression Level {best_spatial}\n")
            f.write(f"- **Best for Resource Features**: Compression Level {best_resources}\n")
            f.write(f"- **Best for Performance Features**: Compression Level {best_performance}\n")
            f.write(f"- **Best for Role Features**: Compression Level {best_role}\n\n")
            
            # Add detailed analysis for each compression level
            f.write("## Detailed Analysis\n\n")
            
            for level in sorted(self.compression_levels):
                level_data = results_df[results_df['compression_level'] == level]
                
                if level_data.empty:
                    continue
                    
                row = level_data.iloc[0]
                f.write(f"### Compression Level {level}\n\n")
                
                f.write("#### Metrics\n\n")
                f.write(f"- **Overall Drift**: {row['overall_drift']:.4f}\n")
                f.write(f"- **Preservation Score**: {row['preservation_score']:.4f}\n")
                f.write(f"- **Fidelity Score**: {row['fidelity_score']:.4f}\n")
                f.write(f"- **Drift Category**: {row['drift_category']}\n")
                f.write(f"- **Validation Loss**: {row['val_loss']:.4f}\n")
                f.write(f"- **Reconstruction Loss**: {row['recon_loss']:.4f}\n")
                f.write(f"- **KL Loss**: {row['kl_loss']:.4f}\n\n")
                
                f.write("#### Feature-Specific Drift\n\n")
                f.write(f"- **Spatial Drift**: {row['spatial_drift']:.4f}\n")
                f.write(f"- **Resources Drift**: {row['resources_drift']:.4f}\n")
                f.write(f"- **Performance Drift**: {row['performance_drift']:.4f}\n")
                f.write(f"- **Role Drift**: {row['role_drift']:.4f}\n\n")
                
                # Analysis
                f.write("#### Analysis\n\n")
                
                if row['overall_drift'] < 0.1:
                    f.write("- Excellent semantic retention\n")
                elif row['overall_drift'] < 0.2:
                    f.write("- Good semantic retention\n")
                elif row['overall_drift'] < 0.3:
                    f.write("- Acceptable semantic retention\n")
                else:
                    f.write("- Poor semantic retention\n")
                
                if row['spatial_drift'] > row['resources_drift'] and row['spatial_drift'] > row['performance_drift']:
                    f.write("- Spatial features show the highest drift\n")
                elif row['resources_drift'] > row['spatial_drift'] and row['resources_drift'] > row['performance_drift']:
                    f.write("- Resource features show the highest drift\n")
                elif row['performance_drift'] > row['spatial_drift'] and row['performance_drift'] > row['resources_drift']:
                    f.write("- Performance features show the highest drift\n")
                
                f.write("\n")
            
            # Conclusion
            f.write("## Conclusion\n\n")
            f.write("Based on the standardized metrics analysis, we can conclude that:\n\n")
            
            # Find the level with the best balance
            balanced_level = results_df.loc[results_df['combined_score'].idxmax()]['compression_level']
            
            f.write(f"1. Compression level **{balanced_level}** provides the best balance between drift, preservation, and fidelity.\n")
            f.write(f"2. Spatial features (55.4% importance) are most sensitive to compression, with best results at level {best_spatial}.\n")
            f.write(f"3. Resource features (25.1% importance) are best preserved at compression level {best_resources}.\n")
            f.write("4. The chosen compression level should prioritize maintaining spatial and resource feature integrity due to their higher importance weights.\n")
            
        print(f"Generated comprehensive report at {report_file}")
        
        return report_file


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run compression experiments for meaning-preserving transformations"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/compression_experiments",
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
        "--latent-dim", type=int, default=32, help="Dimension of latent space"
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
        default="data/simulation.db",
        help="Path to the simulation database file",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    return parser.parse_args()


def main():
    """Run compression experiments."""
    args = parse_args()

    # Create base configuration
    base_config = Config()

    # Configure from args
    base_config.training.num_epochs = args.epochs
    base_config.training.batch_size = args.batch_size
    base_config.model.latent_dim = args.latent_dim
    base_config.data.num_states = args.num_states
    base_config.data.db_path = args.db_path  # Set the database path
    base_config.debug = args.debug
    base_config.use_gpu = args.gpu

    # Determine input dimension by creating a sample agent state and checking its tensor size
    sample_state = AgentState()
    input_dim = sample_state.to_tensor().shape[0]
    base_config.model.input_dim = input_dim

    # Set an appropriate experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_config.experiment_name = f"compression_study_{timestamp}"

    # Create and run experiment
    experiment = CompressionExperiment(base_config, args.output_dir)
    experiment.run_experiments()


if __name__ == "__main__":
    main()
