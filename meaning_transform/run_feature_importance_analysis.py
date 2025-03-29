#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature Importance Analysis for Meaning-Preserving Transformations

This script analyzes the importance of different features in agent state representations:
1. Uses trained models to evaluate feature importance
2. Implements permutation importance method
3. Groups features for meaningful analysis
4. Generates visualizations and reports on feature importance findings
"""

import argparse
import json
import os
import sys
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.config import Config
from src.data import AgentState, AgentStateDataset
from src.metrics import SemanticMetrics, compute_feature_drift
from src.model import MeaningVAE
from src.train import Trainer


class FeatureImportanceAnalysis:
    """Class to run feature importance analysis on agent state representations."""

    def __init__(
        self,
        base_config: Config = None,
        output_dir: str = None,
        feature_groups: List[str] = None,
        permutation_iterations: int = 10,
    ):
        """
        Initialize feature importance analysis.

        Args:
            base_config: Base configuration to use
            output_dir: Directory to save analysis results
            feature_groups: List of feature groups to analyze
            permutation_iterations: Number of permutation iterations for importance calculation
        """
        self.base_config = base_config or Config()
        self.permutation_iterations = permutation_iterations

        # Create timestamp for experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"feature_importance_{timestamp}"

        # Create output directory
        self.output_dir = Path(output_dir or "results/feature_importance")
        self.experiment_dir = self.output_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.models_dir = self.experiment_dir / "models"
        self.visualizations_dir = self.experiment_dir / "visualizations"
        self.metrics_dir = self.experiment_dir / "metrics"

        self.models_dir.mkdir(exist_ok=True)
        self.visualizations_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)

        # Define feature groups for analysis
        self.feature_groups = feature_groups or ["spatial", "resource", "status", "performance", "role"]
        
        # Define feature mappings (customize based on your agent state structure)
        self.feature_mappings = self._define_feature_mappings()
        
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
            "feature_groups": self.feature_groups,
            "permutation_iterations": self.permutation_iterations,
        }

        # Handle non-serializable types
        for section in config_dict:
            if isinstance(config_dict[section], dict):
                for key, value in config_dict[section].items():
                    if isinstance(value, tuple):
                        config_dict[section][key] = list(value)

        config_path = self.experiment_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    def _define_feature_mappings(self) -> Dict[str, List[int]]:
        """
        Define mappings from feature groups to feature indices.
        
        This should be customized based on your agent state structure.
        Returns:
            Dictionary mapping feature group names to list of feature indices
        """
        # Create a sample agent state to get feature names
        sample_state = AgentState()
        feature_names = sample_state.get_feature_names()
        
        # Define mappings based on the exact AgentState.to_tensor structure
        # Looking at the implementation of to_tensor() and from_tensor() methods
        mappings = {
            # Position x, y, z (first 3 features)
            "spatial": [0, 1, 2],
            
            # Health, energy, resource_level (next 3 features)
            "resource": [3, 4, 5],
            
            # Current health, is_defending, age (next 3 features)
            "status": [6, 7, 8],
            
            # Total reward (next 1 feature)
            "performance": [9],
            
            # Role one-hot encoding (next 5 features)
            "role": [10, 11, 12, 13, 14]
        }
        
        # Fill in any missing mappings with some reasonable defaults
        total_features = len(feature_names)
        chunk_size = total_features // len(self.feature_groups)
        
        for i, group in enumerate(self.feature_groups):
            if group not in mappings or not mappings[group]:
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < len(self.feature_groups) - 1 else total_features
                mappings[group] = list(range(start_idx, end_idx))
        
        return mappings

    def run_analysis(self):
        """
        Run feature importance analysis.
        
        This includes:
        1. Training a model with optimal hyperparameters
        2. Analyzing feature importance using permutation method
        3. Generating visualizations and reports
        """
        print(f"Running feature importance analysis")
        print(f"Feature groups: {self.feature_groups}")
        print(f"Permutation iterations: {self.permutation_iterations}")
        
        # Prepare dataset
        dataset = self._prepare_data()
        
        # Train a model with optimal hyperparameters
        model = self._train_model(dataset)
        
        # Calculate feature importance
        importance_results = self._calculate_feature_importance(model, dataset)
        
        # Analyze and visualize results
        self._analyze_results(importance_results)
        
        print(f"Feature importance analysis completed!")

    def _prepare_data(self) -> Dict[str, Any]:
        """
        Prepare datasets for analysis.

        Returns:
            Dict containing train, validation and test datasets
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

        # Split into train, validation and test sets
        total_states = len(dataset.states)
        val_size = int(total_states * 0.15)
        test_size = int(total_states * 0.15)
        train_size = total_states - val_size - test_size

        train_states = dataset.states[:train_size]
        val_states = dataset.states[train_size:train_size+val_size]
        test_states = dataset.states[train_size+val_size:]

        train_dataset = AgentStateDataset(
            train_states, batch_size=self.base_config.training.batch_size
        )
        val_dataset = AgentStateDataset(
            val_states, batch_size=self.base_config.training.batch_size
        )
        test_dataset = AgentStateDataset(
            test_states, batch_size=self.base_config.training.batch_size
        )

        print(f"Training set: {len(train_dataset.states)} states")
        print(f"Validation set: {len(val_dataset.states)} states")
        print(f"Test set: {len(test_dataset.states)} states")

        return {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }

    def _train_model(self, dataset: Dict[str, AgentStateDataset]) -> MeaningVAE:
        """
        Train a model with optimal hyperparameters.
        
        Args:
            dataset: Dictionary containing train/val/test datasets
            
        Returns:
            Trained MeaningVAE model
        """
        print("Training model with optimal hyperparameters...")
        
        # Create config for optimal parameters
        config = Config()
        
        # Copy base config values
        config.model.input_dim = self.base_config.model.input_dim
        config.model.latent_dim = getattr(self.base_config.model, 'latent_dim', 64)  # Default to 64 if not specified
        config.model.encoder_hidden_dims = self.base_config.model.encoder_hidden_dims
        config.model.decoder_hidden_dims = self.base_config.model.decoder_hidden_dims
        config.model.compression_type = self.base_config.model.compression_type
        config.model.compression_level = getattr(self.base_config.model, 'compression_level', 1.0)  # Default to 1.0

        # Training configuration
        config.training.num_epochs = self.base_config.training.num_epochs
        config.training.batch_size = self.base_config.training.batch_size
        config.training.learning_rate = self.base_config.training.learning_rate
        config.training.checkpoint_dir = str(self.models_dir)

        # Set semantic loss weight
        if not hasattr(config.training, 'loss_weights'):
            config.training.loss_weights = {}
        config.training.loss_weights['semantic'] = getattr(self.base_config.training, 'semantic_weight', 1.0)

        # Set experiment name
        config.experiment_name = f"feature_importance_model"
        
        # Copy other settings
        config.debug = self.base_config.debug
        config.verbose = self.base_config.verbose
        config.use_gpu = self.base_config.use_gpu
        
        # Create a trainer
        trainer = Trainer(config)
        
        # Set datasets
        trainer.train_dataset = dataset["train"]
        trainer.val_dataset = dataset["val"]
        
        # Initialize drift tracking states to prevent error
        trainer.drift_tracking_states = dataset["val"].states[:min(10, len(dataset["val"].states))]
        
        # Train the model
        training_results = trainer.train()
        
        # Save model
        model_path = self.models_dir / "feature_importance_model.pt"
        trainer.model.save(model_path)
        print(f"Model saved to {model_path}")
        
        return trainer.model

    def _calculate_feature_importance(
        self, model: MeaningVAE, dataset: Dict[str, AgentStateDataset]
    ) -> Dict[str, List[float]]:
        """
        Calculate feature importance using permutation importance method.
        
        Args:
            model: Trained model
            dataset: Dictionary containing train/val/test datasets
            
        Returns:
            Dictionary mapping feature groups to importance scores
            
        Note:
            Future improvement: Add feature interaction detection by analyzing pairs of
            feature groups together to identify potential interaction effects that
            aren't captured by individual permutation importance.
        """
        print("Calculating feature importance using permutation method...")
        
        device = torch.device(
            "cuda" if torch.cuda.is_available() and self.base_config.use_gpu else "cpu"
        )
        model.to(device)
        model.eval()
        
        # Use test dataset for feature importance
        test_dataset = dataset["test"]
        
        # Calculate baseline performance
        baseline_score = self._evaluate_model_performance(model, test_dataset)
        print(f"Baseline performance score: {baseline_score:.4f}")
        
        # Calculate importance for each feature group
        importance_scores = {}
        
        for group_name in self.feature_groups:
            if group_name not in self.feature_mappings:
                print(f"Warning: Feature group '{group_name}' not defined in mappings. Skipping.")
                continue
                
            feature_indices = self.feature_mappings[group_name]
            if not feature_indices:
                print(f"Warning: No features mapped for group '{group_name}'. Skipping.")
                continue
                
            print(f"Calculating importance for feature group: {group_name} ({len(feature_indices)} features)")
            
            # Run multiple permutation iterations and average the results
            group_scores = []
            
            for iteration in tqdm(range(self.permutation_iterations), desc=f"Permuting {group_name}"):
                # Create permuted dataset
                permuted_dataset = self._create_permuted_dataset(test_dataset, feature_indices)
                
                # Evaluate performance on permuted dataset
                permuted_score = self._evaluate_model_performance(model, permuted_dataset)
                
                # Importance is the drop in performance
                importance = baseline_score - permuted_score
                group_scores.append(importance)
            
            # Average importance across iterations
            avg_importance = sum(group_scores) / len(group_scores)
            importance_scores[group_name] = {
                'mean': avg_importance,
                'raw_scores': group_scores
            }
            
            print(f"  Importance score: {avg_importance:.4f}")
        
        # Save raw importance scores
        with open(self.metrics_dir / "feature_importance_raw.json", "w") as f:
            json.dump(importance_scores, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        
        return importance_scores

    def _create_permuted_dataset(
        self, dataset: AgentStateDataset, feature_indices: List[int]
    ) -> AgentStateDataset:
        """
        Create a copy of the dataset with specified features permuted.
        
        Args:
            dataset: Original dataset
            feature_indices: Indices of features to permute
            
        Returns:
            New dataset with permuted features
        """
        # Make a deep copy of the states
        permuted_states = []
        
        for state in dataset.states:
            # Convert to tensor
            tensor = state.to_tensor().clone()
            
            # Get all values for the specified features
            all_values = torch.stack([s.to_tensor()[feature_indices] for s in dataset.states])
            
            # Shuffle values
            shuffled_indices = torch.randperm(len(dataset.states))
            shuffled_values = all_values[shuffled_indices]
            
            # Replace feature values with shuffled ones
            tensor[feature_indices] = shuffled_values[0]  # Take the first shuffled value
            
            # Convert back to agent state
            permuted_state = AgentState.from_tensor(tensor)
            permuted_states.append(permuted_state)
        
        # Create new dataset with permuted states
        permuted_dataset = AgentStateDataset(
            permuted_states, batch_size=dataset.batch_size
        )
        
        return permuted_dataset

    def _evaluate_model_performance(self, model: MeaningVAE, dataset: AgentStateDataset) -> float:
        """
        Evaluate model performance on a dataset.
        
        For feature importance, we use a combination of reconstruction error and semantic drift.
        
        Args:
            model: Model to evaluate
            dataset: Dataset to evaluate on
            
        Returns:
            Performance score (higher is better)
        """
        device = torch.device(
            "cuda" if torch.cuda.is_available() and self.base_config.use_gpu else "cpu"
        )
        
        # Initialize metrics
        total_recon_loss = 0.0
        total_semantic_drift = 0.0
        count = 0
        
        # Only process a subset of the dataset to speed up evaluation
        eval_states = dataset.states[:min(200, len(dataset.states))]
        
        with torch.no_grad():
            for state in eval_states:
                # Convert to tensor
                tensor = state.to_tensor().unsqueeze(0).to(device)
                
                # Get reconstruction
                output = model(tensor)
                reconstructed = output["x_reconstructed"][0].cpu()
                
                # Compute reconstruction loss
                recon_loss = torch.nn.functional.mse_loss(
                    reconstructed, state.to_tensor()
                ).item()
                
                # Convert reconstructed tensor to agent state
                reconstructed_state = AgentState.from_tensor(reconstructed)
                
                # Compute semantic drift
                feature_drift = compute_feature_drift(state, reconstructed_state)
                avg_drift = sum(feature_drift.values()) / max(1, len(feature_drift))
                
                # Add to totals
                total_recon_loss += recon_loss
                total_semantic_drift += avg_drift
                count += 1
        
        # Compute averages
        avg_recon_loss = total_recon_loss / max(1, count)
        avg_semantic_drift = total_semantic_drift / max(1, count)
        
        # Combine into a single score (invert so higher is better)
        # We use 1 - normalized_score so that importance = baseline - permuted will be positive for important features
        combined_score = 1.0 - (0.5 * avg_recon_loss + 0.5 * avg_semantic_drift)
        
        return combined_score

    def _analyze_results(self, importance_results: Dict[str, Dict[str, Any]]):
        """
        Analyze and visualize feature importance results.
        
        Args:
            importance_results: Dictionary mapping feature groups to importance scores
        """
        print("Analyzing feature importance results...")
        
        # Extract mean importance scores
        mean_scores = {group: data['mean'] for group, data in importance_results.items()}
        
        # Handle negative importance scores
        has_negative = any(score < 0 for score in mean_scores.values())
        if has_negative:
            print("Warning: Negative importance scores detected. This may indicate that:")
            print("  1. The model is overfitting to certain feature groups")
            print("  2. There are complex feature interactions not captured by permutation importance")
            print("  3. The performance metric may need adjustment")
            
            # For normalized scores, use absolute values with a note
            abs_scores = {group: abs(score) for group, score in mean_scores.items()}
            total_abs_importance = sum(abs_scores.values())
            
            if total_abs_importance > 0:
                normalized_scores = {
                    group: (abs_score / total_abs_importance) * 100
                    for group, abs_score in abs_scores.items()
                }
            else:
                normalized_scores = {group: 0 for group in mean_scores}
                
            # Add direction (positive/negative effect)
            directions = {group: "negative" if score < 0 else "positive" for group, score in mean_scores.items()}
        else:
            # No negative scores, proceed as before
            total_importance = sum(mean_scores.values())
            if total_importance > 0:
                normalized_scores = {
                    group: (score / total_importance) * 100
                    for group, score in mean_scores.items()
                }
            else:
                normalized_scores = {group: 0 for group in mean_scores}
            directions = {group: "positive" for group in mean_scores}
        
        # Create DataFrame for results
        if has_negative:
            results_df = pd.DataFrame({
                'Feature Group': list(normalized_scores.keys()),
                'Relative Importance (%)': list(normalized_scores.values()),
                'Raw Importance': list(mean_scores.values()),
                'Direction': list(directions.values())
            })
        else:
            results_df = pd.DataFrame({
                'Feature Group': list(normalized_scores.keys()),
                'Relative Importance (%)': list(normalized_scores.values()),
                'Raw Importance': list(mean_scores.values())
            })
        
        # Sort by absolute importance
        results_df = results_df.sort_values('Relative Importance (%)', ascending=False)
        
        # Save results to CSV
        results_df.to_csv(self.metrics_dir / "feature_importance_results.csv", index=False)
        
        # Generate visualizations
        self._create_visualizations(results_df, importance_results)
        
        # Generate report
        self._generate_report(results_df, importance_results, has_negative)

    def _create_visualizations(
        self, 
        results_df: pd.DataFrame, 
        importance_results: Dict[str, Dict[str, Any]]
    ):
        """
        Create visualizations for feature importance results.
        
        Args:
            results_df: DataFrame with feature importance results
            importance_results: Raw importance results with individual scores
        """
        # Set style
        plt.style.use('ggplot')
        sns.set(style="whitegrid")
        
        # 1. Bar chart of feature group importance
        plt.figure(figsize=(12, 8))
        bar_plot = sns.barplot(
            x='Feature Group',
            y='Relative Importance (%)',
            data=results_df,
            palette='viridis'
        )
        
        # Add value labels on top of bars
        for i, p in enumerate(bar_plot.patches):
            bar_plot.annotate(
                f'{p.get_height():.1f}%',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center',
                va='bottom',
                fontsize=12
            )
        
        plt.title('Relative Importance of Feature Groups', fontsize=16)
        plt.xlabel('Feature Group', fontsize=14)
        plt.ylabel('Relative Importance (%)', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / "feature_group_importance.png")
        
        # 2. Boxplot of importance distribution across permutations
        plt.figure(figsize=(12, 8))
        
        # Prepare data for boxplot
        boxplot_data = []
        group_names = []
        
        for group, data in importance_results.items():
            if 'raw_scores' in data:
                boxplot_data.append(data['raw_scores'])
                group_names.append(group)
        
        # Create boxplot
        boxplot = plt.boxplot(
            boxplot_data,
            labels=group_names,
            patch_artist=True
        )
        
        # Customize boxplot colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(boxplot_data)))
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.title('Distribution of Feature Importance Across Permutations', fontsize=16)
        plt.xlabel('Feature Group', fontsize=14)
        plt.ylabel('Importance Score (Performance Drop)', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / "feature_importance_distribution.png")
        
        # 3. Pie chart of relative importance
        plt.figure(figsize=(10, 10))
        
        # Ensure all values are positive for the pie chart
        pie_values = results_df['Relative Importance (%)'].copy()
        
        # Handle negative values by taking absolute values
        if (pie_values < 0).any():
            print("Warning: Negative importance values detected. Using absolute values for pie chart.")
            pie_values = pie_values.abs()
            
        # If sum is close to zero, set to equal values
        if abs(pie_values.sum()) < 0.001:
            pie_values = pd.Series([1.0] * len(results_df), index=results_df.index)
            
        plt.pie(
            pie_values,
            labels=results_df['Feature Group'],
            autopct='%1.1f%%',
            startangle=90,
            shadow=True,
            colors=plt.cm.viridis(np.linspace(0, 1, len(results_df)))
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Relative Importance of Feature Groups', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / "feature_importance_pie.png")
        
        print(f"Created visualizations in {self.visualizations_dir}")

    def _generate_report(
        self, 
        results_df: pd.DataFrame, 
        importance_results: Dict[str, Dict[str, Any]],
        has_negative: bool = False
    ):
        """
        Generate a comprehensive report on feature importance findings.
        
        Args:
            results_df: DataFrame with feature importance results
            importance_results: Raw importance results with individual scores
            has_negative: Flag indicating whether negative importance values were found
        """
        # Extract most and least important features
        most_important = results_df.iloc[0]['Feature Group'] if not results_df.empty else "None"
        least_important = results_df.iloc[-1]['Feature Group'] if len(results_df) > 1 else "None"
        
        # Calculate stability of importance scores
        stability_scores = {}
        for group, data in importance_results.items():
            if 'raw_scores' in data and len(data['raw_scores']) > 1:
                # Coefficient of variation (standard deviation / mean)
                mean = np.mean(data['raw_scores'])
                std = np.std(data['raw_scores'])
                if mean != 0:
                    cv = std / abs(mean)
                    stability_scores[group] = 1.0 - min(1.0, cv)  # Convert to stability score (higher is more stable)
                else:
                    stability_scores[group] = 0.0
            else:
                stability_scores[group] = 0.0
        
        # Note about interpretation if negative values
        negative_note = ""
        if has_negative:
            negative_note = """
> **Note on Negative Importance Values**: Negative importance values were detected, indicating that permuting these features actually improved model performance in some cases. This could suggest:
> 1. The model may be overfitting to these features
> 2. There are complex interactions between features that permutation importance doesn't capture well
> 3. The feature groups contain noisy or redundant information
>
> For the relative importance calculations, absolute values were used, with the direction of effect noted separately.
"""
        
        report = f"""
# Feature Importance Analysis Report

## Overview
This report analyzes the importance of different feature groups in agent state representations.
The analysis was conducted using a permutation importance method with {self.permutation_iterations} iterations.
{negative_note}
## Key Findings

### Relative Importance
The analysis reveals the following importance ranking of feature groups:

{results_df.to_markdown(index=False)}

### Most Important Feature Group
The **{most_important}** features are most critical for meaning preservation, accounting for {results_df.iloc[0]['Relative Importance (%)']:.1f}% of overall importance.
"""

        # Add information about the direction of effect if there are negative values
        if has_negative and 'Direction' in results_df.columns:
            most_important_direction = results_df.iloc[0]['Direction']
            report += f"\nThe effect is **{most_important_direction}**, meaning that permuting these features {'decreased' if most_important_direction == 'positive' else 'increased'} model performance.\n"

        report += f"""
### Least Important Feature Group
The **{least_important}** features contribute least to meaning preservation, accounting for only {results_df.iloc[-1]['Relative Importance (%)']:.1f}% of overall importance.
"""

        # Add information about the direction of effect if there are negative values
        if has_negative and 'Direction' in results_df.columns:
            least_important_direction = results_df.iloc[-1]['Direction']
            report += f"\nThe effect is **{least_important_direction}**, meaning that permuting these features {'decreased' if least_important_direction == 'positive' else 'increased'} model performance.\n"

        report += """
### Stability Analysis
The stability of importance scores across permutation iterations:
"""

        # Add stability analysis to report
        stability_df = pd.DataFrame({
            'Feature Group': list(stability_scores.keys()),
            'Stability Score (0-1)': list(stability_scores.values())
        }).sort_values('Stability Score (0-1)', ascending=False)
        
        report += f"\n{stability_df.to_markdown(index=False)}\n"
        
        report += f"""
A higher stability score indicates more consistent importance across permutations.

## Implications

### Model Optimization
Based on these findings, the following optimizations can be considered:

1. **Prioritize {most_important} Features**: Since these features contribute most to meaning preservation, ensure they receive special attention in the model architecture.

2. **Potential Dimensionality Reduction**: The {least_important} features could potentially be reduced or compressed more aggressively without significant loss of meaning.

3. **Targeted Regularization**: Apply different regularization strengths to different feature groups based on their importance.
"""

        if has_negative:
            report += """
4. **Investigate Negative Importance**: For feature groups with negative importance, consider whether the model is overfitting or if there are redundant features that could be simplified.
"""

        report += f"""
### Simulation Insights
The importance ranking provides insights into what aspects of agent states are most crucial for maintaining their essential meaning:

1. The high importance of **{most_important}** suggests these aspects are fundamental to agent identity and behavior.

2. The lower importance of **{least_important}** suggests these aspects may be more contextual or situational rather than core to agent meaning.

## Visualization

Feature importance visualizations are available in the visualizations directory:
- Bar chart of relative importance
- Distribution of importance scores across permutations
- Pie chart of importance distribution

## Next Steps

1. **Fine-grained Analysis**: Break down each feature group into individual features to identify specific high-importance features.

2. **Cross-context Validation**: Test if feature importance rankings remain consistent across different simulation contexts.

3. **Adaptive Compression**: Develop a compression strategy that adjusts based on feature importance.

4. **Interactive Visualization**: Create an interactive dashboard for exploring feature importance in relation to specific agent behaviors.
"""

        # Save report
        report_path = self.experiment_dir / "feature_importance_report.md"
        with open(report_path, "w") as f:
            f.write(report)
            
        print(f"Generated feature importance report: {report_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run feature importance analysis for meaning-preserving transformations"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/feature_importance",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
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
    parser.add_argument(
        "--latent-dims",
        type=str,
        default="64",
        help="Comma-separated list of latent dimensions to use"
    )
    parser.add_argument(
        "--compression-levels",
        type=str,
        default="1.0",
        help="Comma-separated list of compression levels to use"
    )
    parser.add_argument(
        "--semantic-weights",
        type=str,
        default="1.0",
        help="Comma-separated list of semantic loss weights to use"
    )
    parser.add_argument(
        "--permutation-iterations",
        type=int,
        default=10,
        help="Number of permutation iterations for importance calculation"
    )
    parser.add_argument(
        "--feature-groups",
        type=str,
        default="spatial,resource,status,performance,role",
        help="Comma-separated list of feature groups to analyze"
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    return parser.parse_args()


def main():
    """Run feature importance analysis."""
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

    # Parse hyperparameters
    latent_dims = [int(x) for x in args.latent_dims.split(",")]
    compression_levels = [float(x) for x in args.compression_levels.split(",")]
    semantic_weights = [float(x) for x in args.semantic_weights.split(",")]
    
    # Use first values for model training
    base_config.model.latent_dim = latent_dims[0]
    base_config.model.compression_level = compression_levels[0]
    
    if not hasattr(base_config.training, 'loss_weights'):
        base_config.training.loss_weights = {}
    base_config.training.loss_weights['semantic'] = semantic_weights[0]

    # Parse feature groups
    feature_groups = args.feature_groups.split(",")

    # Determine input dimension by creating a sample agent state and checking its tensor size
    sample_state = AgentState()
    input_dim = sample_state.to_tensor().shape[0]
    base_config.model.input_dim = input_dim

    # Set an appropriate experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_config.experiment_name = f"feature_importance_{timestamp}"

    # Create and run analysis
    analysis = FeatureImportanceAnalysis(
        base_config,
        args.output_dir,
        feature_groups,
        args.permutation_iterations
    )
    
    analysis.run_analysis()


if __name__ == "__main__":
    main() 