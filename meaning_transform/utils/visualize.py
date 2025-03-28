#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization interface module for the meaning-preserving transformation system.

This module provides a convenient interface to all visualization tools.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

from .visualization import (
    LatentSpaceVisualizer,
    LossVisualizer,
    StateComparisonVisualizer,
    DriftVisualizer
)


def setup_visualization_dirs(base_dir: str = "results/visualizations") -> Dict[str, str]:
    """
    Create visualization directories.
    
    Args:
        base_dir: Base visualization directory
        
    Returns:
        dirs: Dictionary of visualization directories
    """
    dirs = {
        "latent_space": os.path.join(base_dir, "latent_space"),
        "loss_curves": os.path.join(base_dir, "loss_curves"),
        "state_comparison": os.path.join(base_dir, "state_comparison"),
        "drift_tracking": os.path.join(base_dir, "drift_tracking")
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def visualize_latent_space(
    latent_vectors: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    output_dir: Optional[str] = None,
    use_tsne: bool = True,
    use_pca: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Visualize latent space using t-SNE and/or PCA.
    
    Args:
        latent_vectors: Latent vectors [batch_size, latent_dim]
        labels: Optional labels for coloring points [batch_size]
        output_dir: Optional output directory
        use_tsne: Whether to generate t-SNE visualization
        use_pca: Whether to generate PCA visualization
        metadata: Optional metadata for the plot
        
    Returns:
        paths: Dictionary of output file paths
    """
    output_dir = output_dir or "results/visualizations/latent_space"
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = LatentSpaceVisualizer(output_dir)
    output_paths = {}
    
    if use_tsne:
        tsne_file = "latent_tsne.png"
        visualizer.visualize_tsne(
            latent_vectors=latent_vectors,
            labels=labels,
            metadata=metadata,
            output_file=tsne_file
        )
        output_paths["tsne"] = os.path.join(output_dir, tsne_file)
    
    if use_pca:
        pca_file = "latent_pca.png"
        visualizer.visualize_pca(
            latent_vectors=latent_vectors,
            labels=labels,
            metadata=metadata,
            output_file=pca_file
        )
        output_paths["pca"] = os.path.join(output_dir, pca_file)
    
    return output_paths


def visualize_latent_interpolation(
    decode_fn: callable,
    state_a: torch.Tensor,
    state_b: torch.Tensor,
    encoder: callable,
    steps: int = 10,
    output_dir: Optional[str] = None
) -> str:
    """
    Visualize interpolation between two states in latent space.
    
    Args:
        decode_fn: Function to decode from latent to state
        state_a: Starting state
        state_b: Ending state
        encoder: Function to encode from state to latent
        steps: Number of interpolation steps
        output_dir: Optional output directory
        
    Returns:
        output_path: Path to the output file
    """
    output_dir = output_dir or "results/visualizations/latent_space"
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = LatentSpaceVisualizer(output_dir)
    output_file = "latent_interpolation.png"
    
    visualizer.visualize_latent_interpolation(
        decode_fn=decode_fn,
        state_a=state_a,
        state_b=state_b,
        encoder=encoder,
        steps=steps,
        output_file=output_file
    )
    
    return os.path.join(output_dir, output_file)


def visualize_loss_curves(
    loss_history: Dict[str, List[Tuple[int, float]]],
    loss_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    log_scale: bool = False
) -> str:
    """
    Visualize loss curves from training history.
    
    Args:
        loss_history: Dictionary of loss values by name (list of (epoch, value) tuples)
        loss_names: Optional list of loss names to plot
        output_dir: Optional output directory
        log_scale: Whether to use log scale for y-axis
        
    Returns:
        output_path: Path to the output file
    """
    output_dir = output_dir or "results/visualizations/loss_curves"
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = LossVisualizer(output_dir)
    
    # Update visualizer with loss history
    for name, values in loss_history.items():
        for epoch, value in values:
            visualizer.update(epoch, {name: value})
    
    output_file = "loss_curves.png"
    visualizer.plot_losses(
        loss_names=loss_names,
        output_file=output_file,
        log_scale=log_scale
    )
    
    # Also save history for later use
    visualizer.save_history()
    
    return os.path.join(output_dir, output_file)


def visualize_compression_vs_reconstruction(
    compression_levels: List[float],
    reconstruction_errors: List[float],
    semantic_losses: Optional[List[float]] = None,
    output_dir: Optional[str] = None
) -> str:
    """
    Visualize compression level vs. reconstruction error (and semantic loss).
    
    Args:
        compression_levels: List of compression levels (e.g., bits per state)
        reconstruction_errors: List of reconstruction errors
        semantic_losses: Optional list of semantic losses
        output_dir: Optional output directory
        
    Returns:
        output_path: Path to the output file
    """
    output_dir = output_dir or "results/visualizations/loss_curves"
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = LossVisualizer(output_dir)
    output_file = "compression_vs_reconstruction.png"
    
    visualizer.plot_compression_vs_reconstruction(
        compression_levels=compression_levels,
        reconstruction_errors=reconstruction_errors,
        semantic_losses=semantic_losses,
        output_file=output_file
    )
    
    return os.path.join(output_dir, output_file)


def visualize_state_comparison(
    original_features: Dict[str, np.ndarray],
    reconstructed_features: Dict[str, np.ndarray],
    example_indices: Optional[List[int]] = None,
    output_dir: Optional[str] = None
) -> str:
    """
    Visualize comparison of features between original and reconstructed states.
    
    Args:
        original_features: Dictionary of original features
        reconstructed_features: Dictionary of reconstructed features
        example_indices: Optional list of example indices to plot
        output_dir: Optional output directory
        
    Returns:
        output_path: Path to the output file
    """
    output_dir = output_dir or "results/visualizations/state_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = StateComparisonVisualizer(output_dir)
    output_file = "feature_comparison.png"
    
    visualizer.plot_feature_comparison(
        original_features=original_features,
        reconstructed_features=reconstructed_features,
        example_indices=example_indices,
        output_file=output_file
    )
    
    return os.path.join(output_dir, output_file)


def visualize_state_trajectories(
    original_states: List[torch.Tensor],
    reconstructed_states: List[torch.Tensor],
    output_dir: Optional[str] = None
) -> str:
    """
    Visualize trajectories of original and reconstructed states.
    
    Args:
        original_states: List of original state tensors
        reconstructed_states: List of reconstructed state tensors
        output_dir: Optional output directory
        
    Returns:
        output_path: Path to the output file
    """
    output_dir = output_dir or "results/visualizations/state_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = StateComparisonVisualizer(output_dir)
    output_file = "state_trajectories.png"
    
    visualizer.plot_state_trajectories(
        original_states=original_states,
        reconstructed_states=reconstructed_states,
        output_file=output_file
    )
    
    return os.path.join(output_dir, output_file)


def visualize_confusion_matrices(
    confusion_matrices: Dict[str, np.ndarray],
    output_dir: Optional[str] = None
) -> str:
    """
    Visualize confusion matrices for categorical features.
    
    Args:
        confusion_matrices: Dictionary of confusion matrices by feature
        output_dir: Optional output directory
        
    Returns:
        output_path: Path to the output file
    """
    output_dir = output_dir or "results/visualizations/state_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = StateComparisonVisualizer(output_dir)
    output_file = "confusion_matrices.png"
    
    visualizer.plot_confusion_matrices(
        confusion_matrices=confusion_matrices,
        output_file=output_file
    )
    
    return os.path.join(output_dir, output_file)


def visualize_semantic_drift(
    iterations: List[int],
    semantic_scores: Dict[str, List[float]],
    compression_levels: Optional[List[float]] = None,
    output_dir: Optional[str] = None
) -> str:
    """
    Visualize semantic drift over training iterations or compression levels.
    
    Args:
        iterations: List of iteration numbers
        semantic_scores: Dictionary of semantic scores by feature
        compression_levels: Optional list of compression levels
        output_dir: Optional output directory
        
    Returns:
        output_path: Path to the output file
    """
    output_dir = output_dir or "results/visualizations/drift_tracking"
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = DriftVisualizer(output_dir)
    output_file = "semantic_drift.png"
    
    visualizer.plot_semantic_drift(
        iterations=iterations,
        semantic_scores=semantic_scores,
        compression_levels=compression_levels,
        output_file=output_file
    )
    
    return os.path.join(output_dir, output_file)


def visualize_threshold_finder(
    compression_levels: List[float],
    semantic_scores: List[float],
    reconstruction_errors: List[float],
    threshold: float = 0.9,
    output_dir: Optional[str] = None
) -> Tuple[str, Optional[float]]:
    """
    Visualize compression threshold finder results.
    
    Args:
        compression_levels: List of compression levels
        semantic_scores: List of semantic scores
        reconstruction_errors: List of reconstruction errors
        threshold: Semantic threshold
        output_dir: Optional output directory
        
    Returns:
        output_path: Path to the output file
        optimal_compression: Optional optimal compression level
    """
    output_dir = output_dir or "results/visualizations/drift_tracking"
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = DriftVisualizer(output_dir)
    output_file = "threshold_finder.png"
    
    _, optimal_compression = visualizer.plot_threshold_finder(
        compression_levels=compression_levels,
        semantic_scores=semantic_scores,
        reconstruction_errors=reconstruction_errors,
        threshold=threshold,
        output_file=output_file
    )
    
    return os.path.join(output_dir, output_file), optimal_compression 