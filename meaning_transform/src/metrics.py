#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metrics and evaluation module for the meaning-preserving transformation system.

This module provides:
1. Semantic feature extraction functions
2. Metrics for evaluating semantic equivalence between original and reconstructed states
3. Drift tracking tools to measure semantic degradation over time or compression levels
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

from .loss import SemanticLoss


class SemanticMetrics:
    """Class for computing semantic metrics between original and reconstructed states."""
    
    def __init__(self, feature_extractors: List[str] = None):
        """
        Initialize semantic metrics.
        
        Args:
            feature_extractors: List of semantic features to extract and compare
        """
        self.feature_extractors = feature_extractors or [
            "position", "health", "has_target", "energy", "is_alive", 
            "role", "threatened"
        ]
        self.semantic_loss = SemanticLoss(self.feature_extractors)
    
    def extract_features(self, state_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract semantic features from agent state tensor.
        
        Args:
            state_tensor: Agent state tensor
            
        Returns:
            features: Dictionary of extracted semantic features
        """
        return self.semantic_loss.extract_semantic_features(state_tensor)
    
    def compute_equivalence_scores(self, 
                                  original: torch.Tensor, 
                                  reconstructed: torch.Tensor) -> Dict[str, float]:
        """
        Compute semantic equivalence scores between original and reconstructed states.
        
        Args:
            original: Original agent states
            reconstructed: Reconstructed agent states
            
        Returns:
            scores: Dictionary of semantic equivalence scores
        """
        # Get detailed breakdown of semantic loss components
        loss_breakdown = self.semantic_loss.detailed_breakdown(reconstructed, original)
        
        # Convert losses to similarity scores (1.0 means identical, 0.0 means completely different)
        similarity_scores = {}
        for feature, loss in loss_breakdown.items():
            # Apply exponential decay transformation: score = exp(-loss)
            # This maps loss of 0 -> score of 1.0, and larger losses -> scores closer to 0
            similarity_scores[feature] = float(np.exp(-loss))
        
        # Calculate overall score as weighted average
        feature_weights = {
            "position": 1.0,
            "health": 1.0,
            "has_target": 1.0,
            "energy": 1.0,
            "is_alive": 2.0,  # Higher weight for critical properties
            "role": 1.5,
            "threatened": 1.0
        }
        
        total_weight = sum(feature_weights.get(f, 1.0) for f in similarity_scores.keys())
        weighted_score = sum(similarity_scores[f] * feature_weights.get(f, 1.0) 
                           for f in similarity_scores.keys()) / total_weight
        
        similarity_scores["overall"] = weighted_score
        return similarity_scores
    
    def binary_feature_accuracy(self, 
                               original: torch.Tensor, 
                               reconstructed: torch.Tensor) -> Dict[str, float]:
        """
        Compute accuracy for binary semantic features.
        
        Args:
            original: Original agent states
            reconstructed: Reconstructed agent states
            
        Returns:
            accuracies: Dictionary of accuracy scores for binary features
        """
        original_features = self.extract_features(original)
        reconstructed_features = self.extract_features(reconstructed)
        
        binary_features = ["has_target", "is_alive", "threatened"]
        accuracies = {}
        
        for feature in binary_features:
            if feature not in original_features:
                continue
                
            # Convert to binary predictions
            orig_binary = (original_features[feature] > 0.5).cpu().numpy().flatten()
            recon_binary = (reconstructed_features[feature] > 0.5).cpu().numpy().flatten()
            
            # Compute metrics
            acc = accuracy_score(orig_binary, recon_binary)
            precision = precision_score(orig_binary, recon_binary, zero_division=1.0)
            recall = recall_score(orig_binary, recon_binary, zero_division=1.0)
            f1 = f1_score(orig_binary, recon_binary, zero_division=1.0)
            
            accuracies[f"{feature}_accuracy"] = acc
            accuracies[f"{feature}_precision"] = precision
            accuracies[f"{feature}_recall"] = recall
            accuracies[f"{feature}_f1"] = f1
            
            # Compute confusion matrix
            cm = confusion_matrix(orig_binary, recon_binary)
            # Store as list for JSON serialization
            accuracies[f"{feature}_confusion_matrix"] = cm.tolist()
            
        return accuracies
    
    def role_accuracy(self, 
                     original: torch.Tensor, 
                     reconstructed: torch.Tensor) -> Dict[str, float]:
        """
        Compute accuracy for agent role classification.
        
        Args:
            original: Original agent states
            reconstructed: Reconstructed agent states
            
        Returns:
            metrics: Dictionary of role accuracy metrics
        """
        # Extract role indices (assumes one-hot encoding in positions 5-9)
        original_roles = torch.argmax(original[:, 5:10], dim=1).cpu().numpy()
        reconstructed_roles = torch.argmax(reconstructed[:, 5:10], dim=1).cpu().numpy()
        
        # Compute metrics
        accuracy = accuracy_score(original_roles, reconstructed_roles)
        
        # Compute confusion matrix
        conf_matrix = confusion_matrix(original_roles, reconstructed_roles)
        
        return {
            "role_accuracy": accuracy,
            "role_confusion_matrix": conf_matrix.tolist()
        }
    
    def numeric_feature_errors(self, 
                              original: torch.Tensor, 
                              reconstructed: torch.Tensor) -> Dict[str, float]:
        """
        Compute errors for numeric semantic features.
        
        Args:
            original: Original agent states
            reconstructed: Reconstructed agent states
            
        Returns:
            errors: Dictionary of error metrics for numeric features
        """
        original_features = self.extract_features(original)
        reconstructed_features = self.extract_features(reconstructed)
        
        numeric_features = ["position", "health", "energy"]
        errors = {}
        
        for feature in numeric_features:
            if feature not in original_features:
                continue
                
            orig = original_features[feature].cpu().numpy()
            recon = reconstructed_features[feature].cpu().numpy()
            
            # Mean Absolute Error
            mae = np.mean(np.abs(orig - recon))
            # Root Mean Squared Error
            rmse = np.sqrt(np.mean(np.square(orig - recon)))
            # Mean Absolute Percentage Error (with epsilon to avoid division by zero)
            epsilon = 1e-6
            mape = np.mean(np.abs((orig - recon) / (orig + epsilon))) * 100
            
            errors[f"{feature}_mae"] = float(mae)
            errors[f"{feature}_rmse"] = float(rmse)
            errors[f"{feature}_mape"] = float(mape)
            
        return errors
    
    def evaluate(self, 
                original: torch.Tensor, 
                reconstructed: torch.Tensor) -> Dict[str, Any]:
        """
        Comprehensive evaluation of semantic preservation.
        
        Args:
            original: Original agent states
            reconstructed: Reconstructed agent states
            
        Returns:
            evaluation: Dictionary of all evaluation metrics
        """
        # Combine all metrics
        evaluation = {}
        
        # Overall equivalence scores
        evaluation.update(self.compute_equivalence_scores(original, reconstructed))
        
        # Binary feature accuracy
        evaluation.update(self.binary_feature_accuracy(original, reconstructed))
        
        # Role accuracy
        evaluation.update(self.role_accuracy(original, reconstructed))
        
        # Numeric feature errors
        evaluation.update(self.numeric_feature_errors(original, reconstructed))
        
        return evaluation


class DriftTracker:
    """Tool for tracking semantic drift over time or compression levels."""
    
    def __init__(self, log_dir: str = "results/drift_tracking"):
        """
        Initialize drift tracker.
        
        Args:
            log_dir: Directory to store drift tracking logs
        """
        self.log_dir = log_dir
        self.metrics = SemanticMetrics()
        self.history = defaultdict(list)
        self.compression_levels = []
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def log_iteration(self, 
                     iteration: int,
                     compression_level: float,
                     original: torch.Tensor, 
                     reconstructed: torch.Tensor) -> Dict[str, Any]:
        """
        Log metrics for current iteration.
        
        Args:
            iteration: Current iteration number
            compression_level: Current compression level (bits per dimension)
            original: Original agent states
            reconstructed: Reconstructed agent states
            
        Returns:
            metrics: Dictionary of current metrics
        """
        # Compute metrics
        metrics = self.metrics.evaluate(original, reconstructed)
        
        # Add metadata
        metrics["iteration"] = iteration
        metrics["compression_level"] = compression_level
        
        # Update history
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.history[key].append(value)
        
        self.compression_levels.append(compression_level)
        
        # Save current metrics
        self._save_iteration(iteration, metrics)
        
        return metrics
    
    def _save_iteration(self, iteration: int, metrics: Dict[str, Any]) -> None:
        """
        Save metrics for current iteration to disk.
        
        Args:
            iteration: Current iteration number
            metrics: Dictionary of current metrics
        """
        # Remove confusion matrices for JSON serialization
        metrics_to_save = {k: v for k, v in metrics.items() 
                         if not k.endswith("confusion_matrix")}
        
        # Save as JSON
        with open(f"{self.log_dir}/iteration_{iteration:06d}.json", "w") as f:
            json.dump(metrics_to_save, f, indent=2)
    
    def visualize_drift(self, output_file: str = None) -> None:
        """
        Visualize semantic drift over iterations.
        
        Args:
            output_file: Path to save visualization
        """
        if not self.history:
            print("No drift history to visualize.")
            return
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Semantic Drift Analysis", fontsize=16)
        
        # Plot overall semantic score vs compression level
        ax = axes[0, 0]
        ax.plot(self.compression_levels, self.history["overall"], marker="o")
        ax.set_title("Overall Semantic Preservation")
        ax.set_xlabel("Compression Level (bits per dimension)")
        ax.set_ylabel("Semantic Similarity Score")
        ax.grid(True)
        
        # Plot individual feature scores
        ax = axes[0, 1]
        features = ["position", "health", "energy", "is_alive", "role", "threatened"]
        for feature in features:
            if feature in self.history:
                ax.plot(self.compression_levels, self.history[feature], 
                      marker="o", label=feature)
        ax.set_title("Feature-level Semantic Preservation")
        ax.set_xlabel("Compression Level (bits per dimension)")
        ax.set_ylabel("Feature Similarity Score")
        ax.legend()
        ax.grid(True)
        
        # Plot binary feature accuracy
        ax = axes[1, 0]
        binary_metrics = ["has_target_accuracy", "is_alive_accuracy", "threatened_accuracy"]
        for metric in binary_metrics:
            if metric in self.history:
                ax.plot(self.compression_levels, self.history[metric],
                      marker="o", label=metric.replace("_accuracy", ""))
        ax.set_title("Binary Feature Accuracy")
        ax.set_xlabel("Compression Level (bits per dimension)")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True)
        
        # Plot role accuracy
        ax = axes[1, 1]
        if "role_accuracy" in self.history:
            ax.plot(self.compression_levels, self.history["role_accuracy"],
                  marker="o", color="purple")
        ax.set_title("Role Classification Accuracy")
        ax.set_xlabel("Compression Level (bits per dimension)")
        ax.set_ylabel("Accuracy")
        ax.grid(True)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
        else:
            plt.show()
    
    def generate_report(self, output_file: str = None) -> str:
        """
        Generate comprehensive drift analysis report.
        
        Args:
            output_file: Path to save report
            
        Returns:
            report: Report as string
        """
        if not self.history:
            return "No drift history to report."
        
        # Convert history to DataFrame
        df = pd.DataFrame(self.history)
        
        # Generate summary statistics
        report = "# Semantic Drift Analysis Report\n\n"
        
        report += "## Summary Statistics\n\n"
        report += "| Metric | Mean | Min | Max | Std Dev |\n"
        report += "|--------|------|-----|-----|--------|\n"
        
        metrics_to_report = [
            "overall", "position", "health", "energy", "is_alive", "role", 
            "has_target_accuracy", "is_alive_accuracy", "role_accuracy"
        ]
        
        for metric in metrics_to_report:
            if metric in df.columns:
                mean = df[metric].mean()
                min_val = df[metric].min()
                max_val = df[metric].max()
                std = df[metric].std()
                report += f"| {metric} | {mean:.4f} | {min_val:.4f} | {max_val:.4f} | {std:.4f} |\n"
        
        report += "\n## Compression Analysis\n\n"
        report += "Relationship between compression level and semantic preservation:\n\n"
        
        # Find lowest compression level with acceptable semantic preservation
        threshold = 0.9  # 90% semantic preservation
        acceptable_df = df[df["overall"] >= threshold]
        if not acceptable_df.empty:
            min_acceptable = acceptable_df["compression_level"].min()
            report += f"- **Minimum acceptable compression level**: {min_acceptable:.2f} bits per dimension\n"
            report += f"  (maintains at least {threshold*100:.0f}% semantic preservation)\n\n"
        
        # Identify which features degrade first
        feature_scores = {f: df[f].values for f in ["position", "health", "energy", 
                                                   "is_alive", "role", "threatened"] 
                         if f in df.columns}
        
        if feature_scores:
            # Calculate average degradation rate for each feature
            degradation_rates = {}
            for feature, values in feature_scores.items():
                if len(values) >= 2:
                    degradation_rates[feature] = (values[0] - values[-1]) / len(values)
            
            # Sort by degradation rate (highest first)
            sorted_features = sorted(degradation_rates.items(), key=lambda x: x[1], reverse=True)
            
            report += "### Feature Degradation Order (fastest to slowest):\n\n"
            for feature, rate in sorted_features:
                report += f"- **{feature}**: degrades at {rate:.4f} per step\n"
        
        # Save report if output_file is provided
        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
        
        return report


def compute_latent_space_metrics(latent_vectors: torch.Tensor, 
                               labels: torch.Tensor = None) -> Dict[str, float]:
    """
    Compute metrics on latent space structure.
    
    Args:
        latent_vectors: Encoded latent vectors
        labels: Optional semantic labels for analysis
        
    Returns:
        metrics: Dictionary of latent space metrics
    """
    # Convert to numpy for compatibility with sklearn
    latent_np = latent_vectors.detach().cpu().numpy()
    
    metrics = {}
    
    # Compute basic statistics
    metrics["latent_mean"] = float(np.mean(latent_np))
    metrics["latent_std"] = float(np.std(latent_np))
    metrics["latent_min"] = float(np.min(latent_np))
    metrics["latent_max"] = float(np.max(latent_np))
    
    # Compute percentage of "dead" latent dimensions (near zero variance)
    variances = np.var(latent_np, axis=0)
    dead_dims = np.sum(variances < 1e-6)
    metrics["dead_dimensions_percent"] = float(dead_dims / latent_np.shape[1] * 100)
    
    # Compute average distance between latent points
    from scipy.spatial.distance import pdist, squareform
    if latent_np.shape[0] > 1:
        distances = pdist(latent_np, 'euclidean')
        metrics["avg_latent_distance"] = float(np.mean(distances))
        metrics["max_latent_distance"] = float(np.max(distances))
    
    # If labels are provided, compute cluster quality metrics
    if labels is not None:
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        labels_np = labels.detach().cpu().numpy()
        
        # Compute cluster separation metrics if we have multiple classes
        unique_labels = np.unique(labels_np)
        if len(unique_labels) > 1 and len(unique_labels) < len(labels_np):
            try:
                # Silhouette score (higher is better)
                metrics["silhouette_score"] = float(silhouette_score(latent_np, labels_np))
                # Davies-Bouldin score (lower is better)
                metrics["davies_bouldin_score"] = float(davies_bouldin_score(latent_np, labels_np))
            except:
                # Skip if there's an error computing cluster metrics
                pass
    
    return metrics


def generate_t_sne_visualization(latent_vectors: torch.Tensor, 
                               labels: Optional[torch.Tensor] = None,
                               output_file: Optional[str] = None) -> None:
    """
    Generate t-SNE visualization of latent space.
    
    Args:
        latent_vectors: Encoded latent vectors
        labels: Optional semantic labels for coloring points
        output_file: Path to save visualization
    """
    # Convert to numpy for t-SNE
    latent_np = latent_vectors.detach().cpu().numpy()
    
    # Apply t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_np)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        labels_np = labels.detach().cpu().numpy()
        unique_labels = np.unique(labels_np)
        for label in unique_labels:
            mask = labels_np == label
            plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                      label=f"Class {label}", alpha=0.7)
        plt.legend()
    else:
        plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.7)
    
    plt.title("t-SNE Visualization of Latent Space")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, alpha=0.3)
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    else:
        plt.show()


class CompressionThresholdFinder:
    """Find optimal compression level that maintains semantic integrity."""
    
    def __init__(self, semantic_threshold: float = 0.9):
        """
        Initialize compression threshold finder.
        
        Args:
            semantic_threshold: Minimum acceptable semantic preservation score (0.0-1.0)
        """
        self.semantic_threshold = semantic_threshold
        self.metrics = SemanticMetrics()
        self.compression_results = []
    
    def evaluate_compression_level(self, 
                                 compression_level: float,
                                 original: torch.Tensor, 
                                 reconstructed: torch.Tensor) -> Dict[str, Any]:
        """
        Evaluate a specific compression level.
        
        Args:
            compression_level: Compression level in bits per dimension
            original: Original agent states
            reconstructed: Reconstructed agent states
            
        Returns:
            result: Evaluation result
        """
        # Compute semantic preservation metrics
        metrics = self.metrics.evaluate(original, reconstructed)
        
        # Create result dictionary
        result = {
            "compression_level": compression_level,
            "overall_score": metrics["overall"],
            "meets_threshold": metrics["overall"] >= self.semantic_threshold,
            "metrics": metrics
        }
        
        # Store result
        self.compression_results.append(result)
        
        return result
    
    def find_optimal_threshold(self) -> Dict[str, Any]:
        """
        Find optimal compression threshold based on recorded evaluations.
        
        Returns:
            result: Optimal compression threshold info
        """
        if not self.compression_results:
            return {"error": "No compression levels evaluated"}
        
        # Sort by compression level (ascending)
        sorted_results = sorted(self.compression_results, key=lambda x: x["compression_level"])
        
        # Find highest compression level that meets threshold
        acceptable_results = [r for r in sorted_results if r["meets_threshold"]]
        
        if not acceptable_results:
            # No compression level meets the threshold
            return {
                "optimal_level": None,
                "message": f"No compression level meets the {self.semantic_threshold} threshold",
                "recommendation": "Try lower compression or adjust threshold"
            }
        
        # Get the result with highest compression that meets threshold
        optimal = max(acceptable_results, key=lambda x: x["compression_level"])
        
        # Find next compression level (for interpolation)
        higher_levels = [r for r in sorted_results 
                       if r["compression_level"] > optimal["compression_level"]]
        
        next_level = min(higher_levels, key=lambda x: x["compression_level"]) if higher_levels else None
        
        return {
            "optimal_level": optimal["compression_level"],
            "optimal_score": optimal["overall_score"],
            "next_level": next_level["compression_level"] if next_level else None,
            "next_level_score": next_level["overall_score"] if next_level else None,
            "threshold_used": self.semantic_threshold,
            "all_evaluated_levels": [r["compression_level"] for r in sorted_results]
        } 