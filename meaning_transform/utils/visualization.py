#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization module for the meaning-preserving transformation system.

This module provides tools for:
1. Latent space visualization (t-SNE, PCA)
2. Loss curves and training dynamics visualization
3. Comparison between original and reconstructed states
4. Semantic drift tracking visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from collections import defaultdict

# Set style for matplotlib - using new style naming convention
try:
    plt.style.use('seaborn-v0_8-whitegrid')  # For newer matplotlib versions
except:
    try:
        plt.style.use('seaborn-whitegrid')  # For older matplotlib versions
    except:
        pass  # Fallback to default style

sns.set_context("talk")


class LatentSpaceVisualizer:
    """Class for visualizing the latent space of the model."""
    
    def __init__(self, output_dir: str = "results/visualizations/latent_space"):
        """
        Initialize the latent space visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_tsne(self, 
                       latent_vectors: torch.Tensor, 
                       labels: Optional[torch.Tensor] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       output_file: Optional[str] = None) -> plt.Figure:
        """
        Create t-SNE visualization of the latent space.
        
        Args:
            latent_vectors: Latent vectors [batch_size, latent_dim]
            labels: Optional labels for coloring points [batch_size]
            metadata: Optional metadata for the plot
            output_file: Optional output file path
            
        Returns:
            fig: Matplotlib figure
        """
        # Convert to numpy
        latent_np = latent_vectors.detach().cpu().numpy()
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, latent_np.shape[0] // 5)))
        embeddings = tsne.fit_transform(latent_np)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels is not None:
            labels_np = labels.detach().cpu().numpy()
            unique_labels = np.unique(labels_np)
            
            # Create scatter plot with color for each label
            for label in unique_labels:
                mask = labels_np == label
                ax.scatter(
                    embeddings[mask, 0], 
                    embeddings[mask, 1],
                    label=f"Class {label}",
                    alpha=0.7
                )
            ax.legend()
        else:
            # Create scatter plot without labels
            scatter = ax.scatter(
                embeddings[:, 0], 
                embeddings[:, 1], 
                alpha=0.7,
                c=np.arange(len(embeddings)),  # Color by index
                cmap='viridis'
            )
            
        ax.set_title("t-SNE Visualization of Latent Space")
        
        # Add metadata annotation if provided
        if metadata:
            metadata_str = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
            ax.annotate(
                metadata_str,
                xy=(0.02, 0.02),
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8)
            )
        
        # Save if output file specified
        if output_file:
            full_path = os.path.join(self.output_dir, output_file) if not os.path.isabs(output_file) else output_file
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved t-SNE visualization to {full_path}")
        
        return fig
    
    def visualize_pca(self, 
                     latent_vectors: torch.Tensor, 
                     labels: Optional[torch.Tensor] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     output_file: Optional[str] = None) -> plt.Figure:
        """
        Create PCA visualization of the latent space.
        
        Args:
            latent_vectors: Latent vectors [batch_size, latent_dim]
            labels: Optional labels for coloring points [batch_size]
            metadata: Optional metadata for the plot
            output_file: Optional output file path
            
        Returns:
            fig: Matplotlib figure
        """
        # Convert to numpy
        latent_np = latent_vectors.detach().cpu().numpy()
        
        # Apply PCA
        pca = PCA(n_components=2)
        embeddings = pca.fit_transform(latent_np)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels is not None:
            labels_np = labels.detach().cpu().numpy()
            unique_labels = np.unique(labels_np)
            
            # Create scatter plot with color for each label
            for label in unique_labels:
                mask = labels_np == label
                ax.scatter(
                    embeddings[mask, 0], 
                    embeddings[mask, 1],
                    label=f"Class {label}",
                    alpha=0.7
                )
            ax.legend()
        else:
            # Create scatter plot without labels
            scatter = ax.scatter(
                embeddings[:, 0], 
                embeddings[:, 1], 
                alpha=0.7,
                c=np.arange(len(embeddings)),  # Color by index
                cmap='viridis'
            )
            
        # Add variance explanation
        explained_variance = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({explained_variance[0]:.2%} variance)")
        ax.set_ylabel(f"PC2 ({explained_variance[1]:.2%} variance)")
        ax.set_title("PCA Visualization of Latent Space")
        
        # Add metadata annotation if provided
        if metadata:
            metadata_str = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
            ax.annotate(
                metadata_str,
                xy=(0.02, 0.02),
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8)
            )
        
        # Save if output file specified
        if output_file:
            full_path = os.path.join(self.output_dir, output_file) if not os.path.isabs(output_file) else output_file
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved PCA visualization to {full_path}")
        
        return fig
    
    def visualize_latent_interpolation(self,
                                      decode_fn: callable,
                                      state_a: torch.Tensor,
                                      state_b: torch.Tensor,
                                      encoder: callable,
                                      steps: int = 10,
                                      output_file: Optional[str] = None) -> plt.Figure:
        """
        Visualize interpolation between two states in latent space.
        
        Args:
            decode_fn: Function to decode from latent to state
            state_a: Starting state
            state_b: Ending state
            encoder: Function to encode from state to latent
            steps: Number of interpolation steps
            output_file: Optional output file path
            
        Returns:
            fig: Matplotlib figure
        """
        # Encode states to get latent vectors
        with torch.no_grad():
            latent_a = encoder(state_a.unsqueeze(0)).squeeze(0)
            latent_b = encoder(state_b.unsqueeze(0)).squeeze(0)
        
        # Create interpolation steps
        alpha_values = np.linspace(0, 1, steps)
        interpolated_latents = []
        
        for alpha in alpha_values:
            # Linear interpolation
            interpolated = (1 - alpha) * latent_a + alpha * latent_b
            interpolated_latents.append(interpolated)
        
        # Decode interpolated latents
        with torch.no_grad():
            decoded_states = [decode_fn(z.unsqueeze(0)).squeeze(0) for z in interpolated_latents]
        
        # Get a representative feature value to visualize
        # Assuming first 2 dimensions are x,y position
        x_values = [state[0].item() for state in decoded_states]
        y_values = [state[1].item() for state in decoded_states]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the decoded states
        ax.plot(x_values, y_values, 'o-', alpha=0.7)
        
        # Mark start and end points
        ax.scatter(x_values[0], y_values[0], color='green', s=100, label='Start State', zorder=5)
        ax.scatter(x_values[-1], y_values[-1], color='red', s=100, label='End State', zorder=5)
        
        # Label points with interpolation value
        for i, alpha in enumerate(alpha_values):
            ax.annotate(
                f"{alpha:.1f}",
                (x_values[i], y_values[i]),
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        ax.set_title("Latent Space Interpolation")
        ax.set_xlabel("X Position Feature")
        ax.set_ylabel("Y Position Feature")
        ax.legend()
        
        # Save if output file specified
        if output_file:
            full_path = os.path.join(self.output_dir, output_file) if not os.path.isabs(output_file) else output_file
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved interpolation visualization to {full_path}")
        
        return fig


class LossVisualizer:
    """Class for visualizing loss curves and training dynamics."""
    
    def __init__(self, output_dir: str = "results/visualizations/loss_curves"):
        """
        Initialize the loss visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.loss_history = defaultdict(list)
        
    def update(self, epoch: int, losses: Dict[str, float]) -> None:
        """
        Update loss history with new values.
        
        Args:
            epoch: Current epoch
            losses: Dictionary of loss values
        """
        for name, value in losses.items():
            self.loss_history[name].append((epoch, value))
    
    def save_history(self, filename: str = "loss_history.json") -> None:
        """
        Save loss history to a JSON file.
        
        Args:
            filename: Output filename
        """
        # Convert to a format suitable for JSON
        history_dict = {}
        for name, values in self.loss_history.items():
            history_dict[name] = {"epochs": [v[0] for v in values], "values": [v[1] for v in values]}
        
        # Save to file
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"Saved loss history to {output_path}")
    
    def load_history(self, filename: str = "loss_history.json") -> None:
        """
        Load loss history from a JSON file.
        
        Args:
            filename: Input filename
        """
        input_path = os.path.join(self.output_dir, filename)
        
        if not os.path.exists(input_path):
            print(f"Warning: History file {input_path} not found")
            return
        
        with open(input_path, 'r') as f:
            history_dict = json.load(f)
        
        # Convert back to internal format
        self.loss_history = defaultdict(list)
        for name, data in history_dict.items():
            epochs = data["epochs"]
            values = data["values"]
            self.loss_history[name] = list(zip(epochs, values))
    
    def plot_losses(self, 
                   loss_names: Optional[List[str]] = None,
                   output_file: Optional[str] = None,
                   log_scale: bool = False) -> plt.Figure:
        """
        Plot loss curves.
        
        Args:
            loss_names: Optional list of loss names to plot
            output_file: Optional output file path
            log_scale: Whether to use log scale for y-axis
            
        Returns:
            fig: Matplotlib figure
        """
        # Filter losses to plot
        if loss_names is None:
            # Plot all losses except those that contain "val_"
            loss_names = [name for name in self.loss_history.keys() if not name.startswith("val_")]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot each loss
        for name in loss_names:
            if name in self.loss_history:
                epochs, values = zip(*self.loss_history[name])
                ax.plot(epochs, values, label=name)
        
        # Plot validation losses as dashed lines
        for name in loss_names:
            val_name = f"val_{name}"
            if val_name in self.loss_history:
                epochs, values = zip(*self.loss_history[val_name])
                ax.plot(epochs, values, linestyle='--', label=val_name)
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        if log_scale:
            ax.set_yscale("log")
        ax.set_title("Training Loss Curves")
        ax.legend()
        ax.grid(True)
        
        # Save if output file specified
        if output_file:
            full_path = os.path.join(self.output_dir, output_file) if not os.path.isabs(output_file) else output_file
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved loss curves to {full_path}")
        
        return fig
    
    def plot_compression_vs_reconstruction(self,
                                         compression_levels: List[float],
                                         reconstruction_errors: List[float],
                                         semantic_losses: Optional[List[float]] = None,
                                         output_file: Optional[str] = None) -> plt.Figure:
        """
        Plot compression level vs. reconstruction error (and semantic loss).
        
        Args:
            compression_levels: List of compression levels (e.g., bits per state)
            reconstruction_errors: List of reconstruction errors
            semantic_losses: Optional list of semantic losses
            output_file: Optional output file path
            
        Returns:
            fig: Matplotlib figure
        """
        # Create figure
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot reconstruction error
        color = 'tab:blue'
        ax1.set_xlabel('Compression Level (bits per state)')
        ax1.set_ylabel('Reconstruction Error', color=color)
        ax1.plot(compression_levels, reconstruction_errors, 'o-', color=color, label='Reconstruction Error')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Add semantic loss on second y-axis if provided
        if semantic_losses is not None:
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Semantic Loss', color=color)
            ax2.plot(compression_levels, semantic_losses, 'o-', color=color, label='Semantic Loss')
            ax2.tick_params(axis='y', labelcolor=color)
        
        # Add threshold line where semantic loss crosses a certain value (e.g., 0.1)
        if semantic_losses is not None:
            threshold = 0.1
            # Find where semantic loss crosses threshold
            for i in range(len(semantic_losses) - 1):
                if (semantic_losses[i] <= threshold and semantic_losses[i+1] > threshold) or \
                   (semantic_losses[i] >= threshold and semantic_losses[i+1] < threshold):
                    # Linear interpolation to find exact crossing point
                    x1, x2 = compression_levels[i], compression_levels[i+1]
                    y1, y2 = semantic_losses[i], semantic_losses[i+1]
                    x_cross = x1 + (x2 - x1) * (threshold - y1) / (y2 - y1)
                    
                    # Add vertical line
                    plt.axvline(x=x_cross, color='black', linestyle='--', alpha=0.7)
                    plt.text(x_cross, plt.ylim()[1] * 0.9, f"Threshold: {x_cross:.2f} bits", 
                             rotation=90, verticalalignment='top')
                    break
        
        plt.title('Compression vs. Reconstruction Trade-off')
        plt.grid(True, alpha=0.3)
        
        # Create a legend with all lines
        lines1, labels1 = ax1.get_legend_handles_labels()
        if semantic_losses is not None:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        else:
            ax1.legend(loc='best')
        
        # Save if output file specified
        if output_file:
            full_path = os.path.join(self.output_dir, output_file) if not os.path.isabs(output_file) else output_file
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved compression vs. reconstruction plot to {full_path}")
        
        return fig


class StateComparisonVisualizer:
    """Class for visualizing comparison between original and reconstructed states."""
    
    def __init__(self, output_dir: str = "results/visualizations/state_comparison"):
        """
        Initialize the state comparison visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_feature_comparison(self,
                              original_features: Dict[str, np.ndarray],
                              reconstructed_features: Dict[str, np.ndarray],
                              example_indices: Optional[List[int]] = None,
                              output_file: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of features between original and reconstructed states.
        
        Args:
            original_features: Dictionary of original features
            reconstructed_features: Dictionary of reconstructed features
            example_indices: Optional list of example indices to plot
            output_file: Optional output file path
            
        Returns:
            fig: Matplotlib figure
        """
        # Get common features
        common_features = [f for f in original_features.keys() if f in reconstructed_features]
        
        if not common_features:
            raise ValueError("No common features found between original and reconstructed states")
        
        # If example_indices not provided, use the first example
        if example_indices is None:
            example_indices = [0]
        
        # Create figure with a subplot for each example
        n_examples = len(example_indices)
        fig, axes = plt.subplots(n_examples, 1, figsize=(12, 4 * n_examples), sharex=True)
        
        # Handle single subplot case
        if n_examples == 1:
            axes = [axes]
        
        for i, idx in enumerate(example_indices):
            ax = axes[i]
            
            # Create bar chart for each feature
            x = np.arange(len(common_features))
            width = 0.35
            
            # Get feature values for this example
            orig_values = []
            recon_values = []
            
            for f in common_features:
                # Handle multi-dimensional features (like position)
                if len(original_features[f].shape) > 1 and original_features[f].shape[1] > 1:
                    # For multi-dimensional features, use the norm or first dimension
                    if f == "position" and original_features[f].shape[1] == 2:
                        # For 2D position, use Euclidean norm
                        orig_val = float(np.linalg.norm(original_features[f][idx]))
                        recon_val = float(np.linalg.norm(reconstructed_features[f][idx]))
                    else:
                        # Otherwise use first dimension
                        orig_val = float(original_features[f][idx][0])
                        recon_val = float(reconstructed_features[f][idx][0])
                else:
                    # For scalar features, convert directly
                    orig_val = float(original_features[f][idx])
                    recon_val = float(reconstructed_features[f][idx])
                
                orig_values.append(orig_val)
                recon_values.append(recon_val)
            
            # Plot bars
            ax.bar(x - width/2, orig_values, width, label='Original')
            ax.bar(x + width/2, recon_values, width, label='Reconstructed')
            
            # Add labels and legend
            ax.set_ylabel('Feature Value')
            ax.set_title(f'Feature Comparison (Example {idx})')
            ax.set_xticks(x)
            ax.set_xticklabels(common_features, rotation=45, ha='right')
            ax.legend()
            
            # Add value labels on bars
            for j, v in enumerate(orig_values):
                ax.text(j - width/2, v + 0.02, f"{v:.2f}", ha='center', fontsize=8)
            for j, v in enumerate(recon_values):
                ax.text(j + width/2, v + 0.02, f"{v:.2f}", ha='center', fontsize=8)
        
        plt.tight_layout()
        
        # Save if output file specified
        if output_file:
            full_path = os.path.join(self.output_dir, output_file) if not os.path.isabs(output_file) else output_file
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved feature comparison to {full_path}")
        
        return fig
    
    def plot_state_trajectories(self,
                              original_states: List[torch.Tensor],
                              reconstructed_states: List[torch.Tensor],
                              output_file: Optional[str] = None) -> plt.Figure:
        """
        Plot trajectories of original and reconstructed states.
        
        Args:
            original_states: List of original state tensors
            reconstructed_states: List of reconstructed state tensors
            output_file: Optional output file path
            
        Returns:
            fig: Matplotlib figure
        """
        # Assuming first two dimensions are x,y position
        orig_x = [state[0].item() for state in original_states]
        orig_y = [state[1].item() for state in original_states]
        
        recon_x = [state[0].item() for state in reconstructed_states]
        recon_y = [state[1].item() for state in reconstructed_states]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot trajectories
        ax.plot(orig_x, orig_y, 'o-', label='Original', alpha=0.7)
        ax.plot(recon_x, recon_y, 'o-', label='Reconstructed', alpha=0.7)
        
        # Mark start and end points
        ax.scatter(orig_x[0], orig_y[0], color='green', s=100, label='Start', zorder=5)
        ax.scatter(orig_x[-1], orig_y[-1], color='red', s=100, label='End', zorder=5)
        
        # Add timestamp labels
        for i in range(0, len(orig_x), max(1, len(orig_x) // 5)):  # Add labels at 5 points
            ax.annotate(
                f"t={i}",
                (orig_x[i], orig_y[i]),
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        ax.set_title("Agent State Trajectories")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save if output file specified
        if output_file:
            full_path = os.path.join(self.output_dir, output_file) if not os.path.isabs(output_file) else output_file
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved trajectory comparison to {full_path}")
        
        return fig
    
    def plot_confusion_matrices(self,
                              confusion_matrices: Dict[str, np.ndarray],
                              output_file: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrices for categorical features.
        
        Args:
            confusion_matrices: Dictionary of confusion matrices by feature
            output_file: Optional output file path
            
        Returns:
            fig: Matplotlib figure
        """
        n_matrices = len(confusion_matrices)
        
        if n_matrices == 0:
            raise ValueError("No confusion matrices provided")
        
        # Calculate grid layout (aim for roughly square grid)
        n_cols = min(3, n_matrices)
        n_rows = (n_matrices + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        
        # Handle single subplot case
        if n_matrices == 1:
            axes = np.array([axes])
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Plot each confusion matrix
        for i, (feature_name, cm) in enumerate(confusion_matrices.items()):
            if i < len(axes):
                ax = axes[i]
                
                # Plot confusion matrix
                im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
                ax.set_title(f"{feature_name}")
                
                # Add colorbar
                plt.colorbar(im, ax=ax)
                
                # Add text annotations
                thresh = cm.max() / 2
                for j in range(cm.shape[0]):
                    for k in range(cm.shape[1]):
                        ax.text(k, j, f"{cm[j, k]}", 
                                ha="center", va="center",
                                color="white" if cm[j, k] > thresh else "black")
                
                # Set labels
                ax.set_ylabel('Original')
                ax.set_xlabel('Reconstructed')
        
        # Hide unused subplots
        for i in range(n_matrices, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save if output file specified
        if output_file:
            full_path = os.path.join(self.output_dir, output_file) if not os.path.isabs(output_file) else output_file
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrices to {full_path}")
        
        return fig


class DriftVisualizer:
    """Class for visualizing semantic drift over training or compression levels."""
    
    def __init__(self, output_dir: str = "results/visualizations/drift_tracking"):
        """
        Initialize the drift visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_semantic_drift(self,
                          iterations: List[int],
                          semantic_scores: Dict[str, List[float]],
                          compression_levels: Optional[List[float]] = None,
                          output_file: Optional[str] = None) -> plt.Figure:
        """
        Plot semantic drift over training iterations or compression levels.
        
        Args:
            iterations: List of iteration numbers
            semantic_scores: Dictionary of semantic scores by feature
            compression_levels: Optional list of compression levels
            output_file: Optional output file path
            
        Returns:
            fig: Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot each feature
        for feature, scores in semantic_scores.items():
            if feature != 'overall':  # Plot overall separately
                ax.plot(iterations, scores, label=feature, alpha=0.7)
        
        # Plot overall score with thicker line
        if 'overall' in semantic_scores:
            ax.plot(iterations, semantic_scores['overall'], 
                    label='overall', linewidth=3, color='black')
        
        ax.set_xlabel('Iteration' if compression_levels is None else 'Compression Level (bits)')
        ax.set_ylabel('Semantic Similarity (higher is better)')
        ax.set_title('Semantic Drift Over ' + 
                     ('Iterations' if compression_levels is None else 'Compression Levels'))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at semantic threshold
        ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.7)
        ax.text(ax.get_xlim()[1] * 0.02, 0.91, 'Semantic Threshold (0.9)', 
                color='r', horizontalalignment='left')
        
        # If compression levels provided, use them as x-ticks
        if compression_levels is not None:
            # Create a second x-axis for compression levels
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(iterations)
            ax2.set_xticklabels([f"{c:.1f}" for c in compression_levels])
            ax2.set_xlabel('Compression Level (bits per state)')
        
        # Save if output file specified
        if output_file:
            full_path = os.path.join(self.output_dir, output_file) if not os.path.isabs(output_file) else output_file
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved semantic drift plot to {full_path}")
        
        return fig
    
    def plot_threshold_finder(self,
                            compression_levels: List[float],
                            semantic_scores: List[float],
                            reconstruction_errors: List[float],
                            threshold: float = 0.9,
                            output_file: Optional[str] = None) -> plt.Figure:
        """
        Plot compression threshold finder results.
        
        Args:
            compression_levels: List of compression levels
            semantic_scores: List of semantic scores
            reconstruction_errors: List of reconstruction errors
            threshold: Semantic threshold
            output_file: Optional output file path
            
        Returns:
            fig: Matplotlib figure
        """
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot semantic scores
        color = 'tab:blue'
        ax1.set_xlabel('Compression Level (bits per state)')
        ax1.set_ylabel('Semantic Similarity', color=color)
        ax1.plot(compression_levels, semantic_scores, 'o-', color=color, label='Semantic Similarity')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim([0, 1.05])  # Scores are in [0,1]
        
        # Add second y-axis for reconstruction error
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Reconstruction Error', color=color)
        ax2.plot(compression_levels, reconstruction_errors, 'o-', color=color, label='Reconstruction Error')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Find where semantic score crosses threshold
        optimal_compression = None
        for i in range(len(semantic_scores) - 1):
            if (semantic_scores[i] >= threshold and semantic_scores[i+1] < threshold) or \
               (semantic_scores[i] <= threshold and semantic_scores[i+1] > threshold):
                # Linear interpolation to find exact crossing point
                x1, x2 = compression_levels[i], compression_levels[i+1]
                y1, y2 = semantic_scores[i], semantic_scores[i+1]
                x_cross = x1 + (x2 - x1) * (threshold - y1) / (y2 - y1)
                optimal_compression = x_cross
                
                # Add vertical line
                plt.axvline(x=x_cross, color='black', linestyle='--', alpha=0.7)
                plt.text(x_cross, ax1.get_ylim()[1] * 0.5, f"Optimal: {x_cross:.2f} bits", 
                         rotation=90, verticalalignment='center')
                break
        
        # Add horizontal line at threshold
        ax1.axhline(y=threshold, color='blue', linestyle='--', alpha=0.7)
        ax1.text(ax1.get_xlim()[1] * 0.02, threshold + 0.02, f'Threshold ({threshold})', 
                color='blue', horizontalalignment='left')
        
        plt.title('Compression Threshold Finder')
        
        # Create legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        # Save if output file specified
        if output_file:
            full_path = os.path.join(self.output_dir, output_file) if not os.path.isabs(output_file) else output_file
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Saved threshold finder plot to {full_path}")
        
        return fig, optimal_compression 