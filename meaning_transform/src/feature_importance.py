#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature importance analysis module for the meaning-preserving transformation system.

This module provides:
1. Tools for analyzing and measuring feature importance
2. Standardized feature grouping based on importance analysis
3. Consistent weighting schemes for feature groups
4. Utilities for feature importance visualization
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .metrics import SemanticMetrics
from .loss import SemanticLoss


class FeatureImportanceAnalyzer:
    """
    Analyzer for determining the relative importance of different features
    in agent state representations.
    """
    
    # Standard feature groups with canonical importance weights
    STANDARD_FEATURE_GROUPS = {
        "spatial": ["position"],  # 55.4% importance
        "resources": ["health", "energy"],  # 25.1% importance
        "performance": ["is_alive", "has_target", "threatened"],  # 10.5% importance
        "role": ["role"],  # <5% importance
    }
    
    CANONICAL_IMPORTANCE_WEIGHTS = {
        "spatial": 0.554,
        "resources": 0.251,
        "performance": 0.105,
        "role": 0.050,
    }
    
    def __init__(self, 
                feature_extractors: List[str] = None,
                canonical_weights: bool = True):
        """
        Initialize feature importance analyzer.
        
        Args:
            feature_extractors: List of features to extract and analyze
            canonical_weights: Whether to use canonical weights or recompute from data
        """
        self.feature_extractors = feature_extractors or [
            "position", "health", "has_target", "energy", "is_alive", 
            "role", "threatened"
        ]
        self.canonical_weights = canonical_weights
        self.semantic_loss = SemanticLoss(self.feature_extractors)
        
        # Create feature to group mapping
        self.feature_to_group = {}
        for group, features in self.STANDARD_FEATURE_GROUPS.items():
            for feature in features:
                self.feature_to_group[feature] = group
        
        # Initialize importance scores
        self._feature_importance_scores = None
        self._group_importance_scores = None
    
    def extract_feature_matrix(self, 
                              agent_states: torch.Tensor,
                              flatten: bool = True) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Extract feature matrices for importance analysis.
        
        Args:
            agent_states: Tensor of agent states
            flatten: Whether to flatten features to 2D matrix
            
        Returns:
            feature_matrices: Dictionary of feature matrices
            combined_matrix: Combined matrix of all features
        """
        # Extract semantic features
        features = self.semantic_loss.extract_semantic_features(agent_states)
        
        # Convert to numpy and flatten if needed
        feature_matrices = {}
        for name, tensor in features.items():
            if name not in self.feature_extractors:
                continue
                
            matrix = tensor.cpu().numpy()
            if flatten and matrix.ndim > 2:
                # Flatten all dimensions except batch dimension
                shape = matrix.shape
                matrix = matrix.reshape(shape[0], -1)
            
            feature_matrices[name] = matrix
        
        # Create combined matrix for full analysis
        combined_columns = []
        for name, matrix in feature_matrices.items():
            # Add name prefix to columns
            n_cols = matrix.shape[1]
            columns = [f"{name}_{i}" for i in range(n_cols)]
            combined_columns.extend(columns)
        
        # Concatenate all feature matrices
        combined_matrix = np.hstack([matrix for matrix in feature_matrices.values()])
        
        return feature_matrices, combined_matrix
    
    def analyze_importance_for_outcome(self,
                                     agent_states: torch.Tensor,
                                     outcome_values: np.ndarray,
                                     outcome_type: str = "binary",
                                     n_jobs: int = -1) -> Dict[str, float]:
        """
        Analyze feature importance for predicting a specific outcome.
        
        Args:
            agent_states: Tensor of agent states
            outcome_values: Array of outcome values to predict
            outcome_type: Type of outcome ("binary", "categorical", or "continuous")
            n_jobs: Number of parallel jobs for importance calculation
            
        Returns:
            importance_scores: Dictionary of importance scores by feature
        """
        # Extract feature matrices
        feature_matrices, combined_matrix = self.extract_feature_matrix(agent_states)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            combined_matrix, outcome_values, test_size=0.3, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Choose model based on outcome type
        if outcome_type == "binary" or outcome_type == "categorical":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  # continuous
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Compute permutation importance
        perm_importance = permutation_importance(
            model, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=n_jobs
        )
        
        # Get feature importances
        feature_names = [f"feature_{i}" for i in range(combined_matrix.shape[1])]
        importances = perm_importance.importances_mean
        
        # Map importances to original features
        feature_importance = {}
        start_idx = 0
        
        for feature_name, matrix in feature_matrices.items():
            n_cols = matrix.shape[1]
            # Average importance across all columns for this feature
            feature_imp = np.mean(importances[start_idx:start_idx+n_cols])
            feature_importance[feature_name] = feature_imp
            start_idx += n_cols
        
        # Normalize to sum to 1.0
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            for feature in feature_importance:
                feature_importance[feature] /= total_importance
        
        return feature_importance
    
    def analyze_importance_for_reconstruction(self,
                                            original_states: torch.Tensor,
                                            reconstructed_states: torch.Tensor) -> Dict[str, float]:
        """
        Analyze feature importance for reconstruction quality.
        
        Args:
            original_states: Original agent states
            reconstructed_states: Reconstructed agent states
            
        Returns:
            importance_scores: Dictionary of importance scores by feature
        """
        # Extract features from both original and reconstructed states
        original_features = self.semantic_loss.extract_semantic_features(original_states)
        reconstructed_features = self.semantic_loss.extract_semantic_features(reconstructed_states)
        
        # Calculate error for each feature
        feature_errors = {}
        for feature_name in self.feature_extractors:
            if feature_name not in original_features:
                continue
                
            # Select appropriate error metric based on feature type
            if feature_name in ["has_target", "is_alive", "threatened"]:
                # Binary features - use error rate (1 - accuracy)
                orig_binary = (original_features[feature_name] > 0.5).cpu().numpy()
                recon_binary = (reconstructed_features[feature_name] > 0.5).cpu().numpy()
                error = np.mean(orig_binary != recon_binary)
            elif feature_name in ["position", "health", "energy", "role"]:
                # Continuous features - use RMSE
                orig = original_features[feature_name].cpu().numpy()
                recon = reconstructed_features[feature_name].cpu().numpy()
                error = np.sqrt(np.mean(np.square(orig - recon)))
            else:
                # Default to MAE
                orig = original_features[feature_name].cpu().numpy()
                recon = reconstructed_features[feature_name].cpu().numpy()
                error = np.mean(np.abs(orig - recon))
                
            feature_errors[feature_name] = error
        
        # Convert errors to importance (higher error = higher importance)
        feature_importance = {}
        total_error = sum(feature_errors.values())
        if total_error > 0:
            for feature, error in feature_errors.items():
                feature_importance[feature] = error / total_error
        else:
            # Equal importance if no errors
            n_features = len(feature_errors)
            for feature in feature_errors:
                feature_importance[feature] = 1.0 / n_features
        
        return feature_importance
    
    def analyze_importance_for_behavior(self,
                                      agent_states: torch.Tensor,
                                      behavior_vectors: np.ndarray) -> Dict[str, float]:
        """
        Analyze feature importance for predicting agent behavior.
        
        Args:
            agent_states: Tensor of agent states
            behavior_vectors: Array of behavior vectors to predict
            
        Returns:
            importance_scores: Dictionary of importance scores by feature
        """
        # This is a multi-output regression problem
        
        # Extract feature matrices
        feature_matrices, combined_matrix = self.extract_feature_matrix(agent_states)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            combined_matrix, behavior_vectors, test_size=0.3, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model (regressor for multi-output)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Get feature importances from the model itself
        importances = model.feature_importances_
        
        # Map importances to original features
        feature_importance = {}
        start_idx = 0
        
        for feature_name, matrix in feature_matrices.items():
            n_cols = matrix.shape[1]
            # Average importance across all columns for this feature
            feature_imp = np.mean(importances[start_idx:start_idx+n_cols])
            feature_importance[feature_name] = feature_imp
            start_idx += n_cols
        
        # Normalize to sum to 1.0
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            for feature in feature_importance:
                feature_importance[feature] /= total_importance
        
        return feature_importance
    
    def compute_importance_weights(self,
                                 feature_importance: Dict[str, float] = None) -> Dict[str, float]:
        """
        Compute importance weights for feature groups.
        
        Args:
            feature_importance: Dictionary of feature importance scores
            
        Returns:
            group_weights: Dictionary of group importance weights
        """
        if self.canonical_weights:
            return self.CANONICAL_IMPORTANCE_WEIGHTS.copy()
        
        if feature_importance is None:
            if self._feature_importance_scores is None:
                # No scores available, return canonical weights
                return self.CANONICAL_IMPORTANCE_WEIGHTS.copy()
            feature_importance = self._feature_importance_scores
        
        # Group features and sum their importance
        group_importance = {}
        for feature, importance in feature_importance.items():
            group = self.feature_to_group.get(feature)
            if group is not None:
                group_importance[group] = group_importance.get(group, 0.0) + importance
        
        # Normalize group importance to sum to 1.0
        total_group_importance = sum(group_importance.values())
        group_weights = {}
        
        if total_group_importance > 0:
            for group, importance in group_importance.items():
                group_weights[group] = importance / total_group_importance
        else:
            # Use canonical weights if no importance data
            group_weights = self.CANONICAL_IMPORTANCE_WEIGHTS.copy()
        
        self._group_importance_scores = group_weights
        return group_weights
    
    def visualize_feature_importance(self,
                                   feature_importance: Dict[str, float] = None,
                                   title: str = "Feature Importance Analysis") -> plt.Figure:
        """
        Create visualization of feature importance scores.
        
        Args:
            feature_importance: Dictionary of feature importance scores
            title: Title for the visualization
            
        Returns:
            fig: Matplotlib figure object
        """
        if feature_importance is None:
            if self._feature_importance_scores is None:
                # No scores available
                feature_importance = {
                    "position": 0.55,
                    "health": 0.15,
                    "energy": 0.10,
                    "is_alive": 0.08,
                    "has_target": 0.07, 
                    "threatened": 0.03,
                    "role": 0.02
                }
            else:
                feature_importance = self._feature_importance_scores
        
        # Prepare data for visualization
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        
        # Sort by importance
        sorted_idx = np.argsort(importances)
        features = [features[i] for i in sorted_idx]
        importances = [importances[i] for i in sorted_idx]
        
        # Map features to groups for color coding
        feature_groups = [self.feature_to_group.get(feature, "other") for feature in features]
        
        # Create color map for groups
        group_colors = {
            "spatial": "skyblue",
            "resources": "lightgreen",
            "performance": "salmon",
            "role": "orange",
            "other": "gray"
        }
        
        bar_colors = [group_colors[group] for group in feature_groups]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create horizontal bar chart
        bars = ax.barh(features, importances, color=bar_colors)
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label = f"{width:.1%}"
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    label, va='center', fontweight='bold')
        
        # Add legend for feature groups
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, label=group)
            for group, color in group_colors.items()
            if group in feature_groups
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        # Set title and labels
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_xlim(0, max(importances) * 1.2)  # Add some padding on the right
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Ensure tight layout
        fig.tight_layout()
        
        return fig
    
    def visualize_group_importance(self,
                                 group_weights: Dict[str, float] = None,
                                 title: str = "Feature Group Importance") -> plt.Figure:
        """
        Create visualization of feature group importance.
        
        Args:
            group_weights: Dictionary of group importance weights
            title: Title for the visualization
            
        Returns:
            fig: Matplotlib figure object
        """
        if group_weights is None:
            if self._group_importance_scores is None:
                # Use canonical weights if no scores available
                group_weights = self.CANONICAL_IMPORTANCE_WEIGHTS.copy()
            else:
                group_weights = self._group_importance_scores
        
        # Prepare data for pie chart
        groups = list(group_weights.keys())
        weights = list(group_weights.values())
        
        # Define colors for groups
        group_colors = {
            "spatial": "skyblue",
            "resources": "lightgreen",
            "performance": "salmon",
            "role": "orange"
        }
        
        colors = [group_colors.get(group, "gray") for group in groups]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            weights, 
            labels=groups, 
            colors=colors,
            autopct='%1.1f%%', 
            startangle=90,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        
        # Style text elements
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
        
        # Set title
        ax.set_title(title, fontsize=14)
        
        # Ensure equal aspect ratio
        ax.axis('equal')
        
        # Add legend showing feature examples for each group
        feature_examples = {
            "spatial": "position, movement",
            "resources": "health, energy",
            "performance": "is_alive, has_target, threatened",
            "role": "role, specialization"
        }
        
        legend_labels = [f"{group}: {feature_examples.get(group, '')}" for group in groups]
        ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        # Ensure tight layout
        fig.tight_layout()
        
        return fig


def analyze_feature_importance(
    original_states: torch.Tensor,
    reconstructed_states: torch.Tensor = None,
    behavior_vectors: np.ndarray = None,
    outcome_values: np.ndarray = None,
    outcome_type: str = "binary",
    create_visualizations: bool = True,
    use_canonical_weights: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive analysis of feature importance.
    
    Args:
        original_states: Original agent states
        reconstructed_states: Reconstructed agent states (optional)
        behavior_vectors: Array of behavior vectors to predict (optional)
        outcome_values: Array of outcome values to predict (optional)
        outcome_type: Type of outcome ("binary", "categorical", or "continuous")
        create_visualizations: Whether to create and return visualizations
        use_canonical_weights: Whether to use canonical weights or compute from data
        
    Returns:
        results: Dictionary of importance analysis results
    """
    analyzer = FeatureImportanceAnalyzer(canonical_weights=use_canonical_weights)
    results = {}
    
    # Feature importance from various sources
    importance_sources = {}
    
    # Reconstruction importance (if reconstructed states provided)
    if reconstructed_states is not None:
        reconstruction_importance = analyzer.analyze_importance_for_reconstruction(
            original_states, reconstructed_states
        )
        importance_sources["reconstruction"] = reconstruction_importance
    
    # Behavior prediction importance (if behavior vectors provided)
    if behavior_vectors is not None:
        behavior_importance = analyzer.analyze_importance_for_behavior(
            original_states, behavior_vectors
        )
        importance_sources["behavior"] = behavior_importance
    
    # Outcome prediction importance (if outcome values provided)
    if outcome_values is not None:
        outcome_importance = analyzer.analyze_importance_for_outcome(
            original_states, outcome_values, outcome_type
        )
        importance_sources["outcome"] = outcome_importance
    
    # If no importance sources provided, use canonical weights
    if not importance_sources and use_canonical_weights:
        # Use canonical feature importance
        feature_importance = {
            "position": 0.554,
            "health": 0.15,
            "energy": 0.101,
            "is_alive": 0.05,
            "has_target": 0.035,
            "threatened": 0.02,
            "role": 0.05
        }
    else:
        # Average importance across all sources
        feature_importance = {}
        
        for source, importance in importance_sources.items():
            for feature, score in importance.items():
                if feature not in feature_importance:
                    feature_importance[feature] = 0.0
                feature_importance[feature] += score / len(importance_sources)
    
    # Store feature importance
    results["feature_importance"] = feature_importance
    analyzer._feature_importance_scores = feature_importance
    
    # Compute group weights
    group_weights = analyzer.compute_importance_weights(feature_importance)
    results["group_weights"] = group_weights
    
    # Create visualizations if requested
    if create_visualizations:
        # Feature importance visualization
        fig_feature = analyzer.visualize_feature_importance(
            feature_importance, "Feature Importance Analysis"
        )
        results["feature_importance_figure"] = fig_feature
        
        # Group importance visualization
        fig_group = analyzer.visualize_group_importance(
            group_weights, "Feature Group Importance"
        )
        results["group_importance_figure"] = fig_group
    
    return results 