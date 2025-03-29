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

from src.config import Config
from src.data import AgentState, AgentStateDataset
from src.metrics import SemanticMetrics, compute_feature_drift
from src.model import MeaningVAE
from src.train import Trainer


class CompressionExperiment:
    """Class to run and analyze compression experiments."""

    def __init__(
        self,
        base_config: Config = None,
        output_dir: str = None,
    ):
        """
        Initialize compression experiment.

        Args:
            base_config: Base configuration to use (will be modified for each experiment)
            output_dir: Directory to save experiment results
        """
        self.base_config = base_config or Config()

        # Create timestamp for experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"compression_experiments_{timestamp}"

        # Create output directory
        self.output_dir = Path(output_dir or "results/compression_experiments")
        self.experiment_dir = self.output_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.models_dir = self.experiment_dir / "models"
        self.visualizations_dir = self.experiment_dir / "visualizations"
        self.metrics_dir = self.experiment_dir / "metrics"

        self.models_dir.mkdir(exist_ok=True)
        self.visualizations_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)

        # Set compression levels to test
        self.compression_levels = [0.5, 1.0, 2.0, 5.0]

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
            trainer.drift_tracking_states = dataset["drift_tracking"]

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
        Analyze experiment results and create visualizations and reports.
        """
        print("\nAnalyzing experimental results...")

        # Collect metrics from each experiment
        metrics = {
            "compression_level": [],
            "val_loss": [],
            "recon_loss": [],
            "kl_loss": [],
            "semantic_loss": [],
            "compression_loss": [],
            "semantic_drift": [],
            "model_size_kb": [],
        }

        # Analyze each model
        for level, result in self.results.items():
            model_path = result["model_path"]
            if not os.path.exists(model_path):
                print(f"Warning: Model file not found: {model_path}")
                continue

            # Collect basic metrics
            metrics["compression_level"].append(level)
            metrics["val_loss"].append(result["val_loss"])
            metrics["recon_loss"].append(result["recon_loss"])
            metrics["kl_loss"].append(result["kl_loss"])
            metrics["semantic_loss"].append(result["semantic_loss"])
            metrics["compression_loss"].append(result["compression_loss"])

            # Get model size
            model_size_kb = os.path.getsize(model_path) / 1024
            metrics["model_size_kb"].append(model_size_kb)

            # Load the model to analyze semantic drift
            config = self._create_config_for_level(level)
            model_found = False
            semantic_drift = 0.5  # Default placeholder value

            if os.path.exists(model_path):
                try:
                    model = MeaningVAE(
                        input_dim=config.model.input_dim,
                        latent_dim=config.model.latent_dim,
                        compression_type=config.model.compression_type,
                        compression_level=level,
                    )
                    model.load(model_path)

                    # Evaluate semantic drift
                    semantic_drift = self._evaluate_semantic_drift(model)
                    model_found = True
                except Exception as e:
                    print(f"Error loading model for compression level {level}: {e}")
                    semantic_drift = 0.5  # Default value on error
            else:
                print(
                    f"Warning: Model file not found for compression level {level}: {model_path}"
                )
                # Use training drift value if available
                if "semantic_drift" in result:
                    semantic_drift = result["semantic_drift"]

            metrics["semantic_drift"].append(semantic_drift)

            print(
                f"Compression level {level}: Val Loss = {metrics['val_loss'][-1]:.4f}, "
                f"Semantic Drift = {semantic_drift:.4f}, "
                f"Model Size = {model_size_kb:.1f} KB"
            )

        # Create a DataFrame with the results
        results_df = pd.DataFrame(metrics)
        results_df.to_csv(self.metrics_dir / "compression_results.csv", index=False)

        # Create visualizations
        self._create_visualizations(results_df)

        # Generate report
        self._generate_report(results_df)

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

        # Get the drift tracking states
        drift_states = (
            self.results.get(model.compression_level, {})
            .get("training_results", {})
            .get("drift_tracking_states", [])
        )

        if not drift_states:
            # If drift states aren't in results, use the base ones or create new ones
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

    def _create_visualizations(self, results_df: pd.DataFrame):
        """
        Create visualizations for the experiment results.

        Args:
            results_df: DataFrame with experiment results
        """
        # Check if we have data to visualize
        if results_df.empty:
            print("No data to visualize. Skipping visualization creation.")
            return

        # 1. Plot Val Loss vs Compression Level
        plt.figure(figsize=(10, 6))
        plt.plot(
            results_df["compression_level"], results_df["val_loss"], "o-", linewidth=2
        )
        plt.title("Validation Loss vs Compression Level")
        plt.xlabel("Compression Level")
        plt.ylabel("Validation Loss")
        plt.grid(True)
        plt.savefig(self.visualizations_dir / "val_loss_vs_compression.png")

        # 2. Plot Semantic Drift vs Compression Level
        plt.figure(figsize=(10, 6))
        plt.plot(
            results_df["compression_level"],
            results_df["semantic_drift"],
            "o-",
            linewidth=2,
            color="orange",
        )
        plt.title("Semantic Drift vs Compression Level")
        plt.xlabel("Compression Level")
        plt.ylabel("Semantic Drift")
        plt.grid(True)
        plt.savefig(self.visualizations_dir / "semantic_drift_vs_compression.png")

        # 3. Plot Model Size vs Compression Level
        plt.figure(figsize=(10, 6))
        plt.plot(
            results_df["compression_level"],
            results_df["model_size_kb"],
            "o-",
            linewidth=2,
            color="green",
        )
        plt.title("Model Size vs Compression Level")
        plt.xlabel("Compression Level")
        plt.ylabel("Model Size (KB)")
        plt.grid(True)
        plt.savefig(self.visualizations_dir / "model_size_vs_compression.png")

        # 4. Multi-metric comparison
        plt.figure(figsize=(12, 8))

        # Normalize metrics for comparison
        max_val_loss = (
            max(results_df["val_loss"]) if len(results_df["val_loss"]) > 0 else 1
        )
        max_semantic_drift = (
            max(results_df["semantic_drift"])
            if len(results_df["semantic_drift"]) > 0
            else 1
        )
        max_model_size = (
            max(results_df["model_size_kb"])
            if len(results_df["model_size_kb"]) > 0
            else 1
        )

        plt.plot(
            results_df["compression_level"],
            results_df["val_loss"] / max_val_loss,
            "o-",
            linewidth=2,
            label="Normalized Val Loss",
        )
        plt.plot(
            results_df["compression_level"],
            results_df["semantic_drift"] / max_semantic_drift,
            "o-",
            linewidth=2,
            label="Normalized Semantic Drift",
        )
        plt.plot(
            results_df["compression_level"],
            results_df["model_size_kb"] / max_model_size,
            "o-",
            linewidth=2,
            label="Normalized Model Size",
        )

        plt.title("Normalized Metrics vs Compression Level")
        plt.xlabel("Compression Level")
        plt.ylabel("Normalized Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.visualizations_dir / "comparison_metrics_vs_compression.png")

        print(f"Created visualizations in {self.visualizations_dir}")

    def _generate_report(self, results_df: pd.DataFrame):
        """
        Generate a comprehensive report on the compression experiments.

        Args:
            results_df: DataFrame with experiment results
        """
        # Check if we have data for the report
        if results_df.empty:
            print("No data to generate report. Creating minimal report.")
            report = """
# Compression Analysis Report

## Overview
No valid experiment results were available to analyze. 

This could be due to:
- Model files not being found
- Errors during model evaluation
- Insufficient training data

Please check the experiment logs for details and try running the experiments again with:
- Longer training (more epochs)
- More training data (increase num-states)
- GPU acceleration if available
"""
            # Save the minimal report
            report_path = self.experiment_dir / "compression_analysis_report.md"
            with open(report_path, "w") as f:
                f.write(report)

            print(f"Generated minimal report: {report_path}")
            return report

        # Find optimal compression level
        # We want to minimize semantic drift while also considering model size
        # Simple heuristic: lowest drift-to-size ratio
        results_df["drift_size_ratio"] = (
            results_df["semantic_drift"] / results_df["model_size_kb"]
        )
        optimal_idx = results_df["drift_size_ratio"].idxmin()
        optimal_level = results_df.loc[optimal_idx, "compression_level"]

        report = f"""
# Compression Analysis Report

## Overview
This report analyzes the effects of different compression levels on the meaning-preserving transformation system.
Experiments were conducted with compression levels: {list(self.compression_levels)}

## Key Findings

### Optimal Compression Setting
Based on the balance between semantic preservation and model size, the optimal compression level is: **{optimal_level}**

### Performance Metrics

| Compression | Val Loss | Semantic Drift | Model Size (KB) |
|-------------|----------|----------------|-----------------|
"""

        # Add a row for each compression level
        for _, row in results_df.iterrows():
            report += f"| {row['compression_level']:.1f} | {row['val_loss']:.4f} | {row['semantic_drift']:.4f} | {row['model_size_kb']:.1f} |\n"

        report += """
### Analysis

#### Effect on Semantic Preservation
"""

        # Add analysis of semantic preservation
        min_drift_idx = results_df["semantic_drift"].idxmin()
        min_drift_level = results_df.loc[min_drift_idx, "compression_level"]
        max_drift_idx = results_df["semantic_drift"].idxmax()
        max_drift_level = results_df.loc[max_drift_idx, "compression_level"]

        report += f"""
The lowest semantic drift was observed at compression level {min_drift_level:.1f}, indicating the best semantic preservation.
The highest semantic drift was observed at compression level {max_drift_level:.1f}.

As compression level increases:
- The semantic drift """

        # Determine if drift increases or decreases with compression
        first_drift = results_df.iloc[0]["semantic_drift"]
        last_drift = results_df.iloc[-1]["semantic_drift"]
        trend = "increases" if last_drift > first_drift else "decreases"
        report += f"{trend}, showing that {'higher compression reduces meaning preservation' if trend == 'increases' else 'higher compression surprisingly improves meaning preservation'}.\n"

        report += """
#### Efficiency vs. Meaning Retention
"""

        # Add efficiency analysis
        report += f"""
The optimal balance between model size and semantic preservation is achieved at compression level {optimal_level:.1f}.
This provides the best trade-off between information density and meaning retention.

### Recommendations

Based on these findings, we recommend:
"""

        # Add recommendations
        if optimal_level == min(self.compression_levels):
            report += f"- Using the lowest tested compression level ({optimal_level:.1f}) for applications where semantic fidelity is critical\n"
            report += f"- Consider testing even lower compression levels for potential improvements\n"
        elif optimal_level == max(self.compression_levels):
            report += f"- Using the highest tested compression level ({optimal_level:.1f}) for applications where efficiency is prioritized\n"
            report += f"- Consider testing even higher compression levels for potential further optimization\n"
        else:
            report += f"- Using the balanced compression level of {optimal_level:.1f} for most applications\n"
            report += f"- Adjusting toward {min_drift_level:.1f} when meaning preservation is critical\n"
            report += f"- Adjusting toward {max(self.compression_levels):.1f} when storage efficiency is paramount\n"

        # Save the report
        report_path = self.experiment_dir / "compression_analysis_report.md"
        with open(report_path, "w") as f:
            f.write(report)

        print(f"Generated compression analysis report: {report_path}")
        return report


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
        default="simulation.db",
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
