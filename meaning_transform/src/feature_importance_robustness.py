#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature Importance Hierarchy Robustness Analysis.

This module provides tools to:
1. Implement cross-validation framework for feature importance rankings
2. Test stability of importance hierarchy across different datasets
3. Compare permutation importance with alternative importance measures
4. Perform sensitivity analysis on importance rankings
5. Validate importance against external benchmarks
6. Create visualizations for robustness analysis results
"""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

from .feature_importance import FeatureImportanceAnalyzer
from .metrics import SemanticMetrics


class FeatureImportanceRobustnessAnalyzer:
    """Analyzer for testing the robustness of feature importance hierarchies."""

    def __init__(
        self,
        feature_extractors: List[str] = None,
        n_folds: int = 5,
        random_seed: int = 42,
    ):
        """
        Initialize the robustness analyzer.
        
        Args:
            feature_extractors: List of features to analyze
            n_folds: Number of folds for cross-validation
            random_seed: Random seed for reproducibility
        """
        self.feature_extractors = feature_extractors or [
            "position", "health", "energy", "is_alive", 
            "has_target", "threatened", "role"
        ]
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.base_analyzer = FeatureImportanceAnalyzer(feature_extractors)
        self.metrics = SemanticMetrics(feature_extractors)
        
    def cross_validate_importance(
        self, 
        agent_states: torch.Tensor,
        reconstructed_states: Optional[torch.Tensor] = None,
        behavior_vectors: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Perform cross-validation of feature importance rankings.
        
        Args:
            agent_states: Original agent states
            reconstructed_states: Reconstructed agent states (optional)
            behavior_vectors: Behavior vectors for prediction (optional)
            
        Returns:
            Dictionary containing cross-validation results
        """
        # Setup cross-validation
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
        
        # Convert tensor to numpy for splitting
        states_np = agent_states.cpu().numpy()
        
        # Track importance scores across folds
        fold_importance_scores = defaultdict(list)
        fold_importance_ranks = defaultdict(list)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(states_np)):
            print(f"Processing fold {fold_idx+1}/{self.n_folds}")
            
            # Split data
            train_states = torch.tensor(states_np[train_idx], device=agent_states.device)
            val_states = torch.tensor(states_np[val_idx], device=agent_states.device)
            
            # Split reconstructed states if provided
            train_recon = None
            val_recon = None
            if reconstructed_states is not None:
                recon_np = reconstructed_states.cpu().numpy()
                train_recon = torch.tensor(recon_np[train_idx], device=reconstructed_states.device)
                val_recon = torch.tensor(recon_np[val_idx], device=reconstructed_states.device)
            
            # Split behavior vectors if provided
            train_behavior = None
            val_behavior = None
            if behavior_vectors is not None:
                train_behavior = behavior_vectors[train_idx]
                val_behavior = behavior_vectors[val_idx]
            
            # Calculate importance for this fold
            fold_results = self._calculate_fold_importance(
                train_states, val_states, train_recon, val_recon, train_behavior, val_behavior
            )
            
            # Store importance scores and ranks
            for feature, score in fold_results["importance_scores"].items():
                fold_importance_scores[feature].append(score)
            
            for feature, rank in fold_results["importance_ranks"].items():
                fold_importance_ranks[feature].append(rank)
        
        # Calculate statistics across folds
        importance_stats = self._calculate_importance_statistics(fold_importance_scores, fold_importance_ranks)
        
        # Create visualizations
        figures = self._create_robustness_visualizations(fold_importance_scores, fold_importance_ranks)
        
        # Return complete results
        return {
            "fold_importance_scores": fold_importance_scores,
            "fold_importance_ranks": fold_importance_ranks,
            "importance_stats": importance_stats,
            "figures": figures
        }
    
    def _calculate_fold_importance(
        self,
        train_states: torch.Tensor,
        val_states: torch.Tensor,
        train_recon: Optional[torch.Tensor] = None,
        val_recon: Optional[torch.Tensor] = None,
        train_behavior: Optional[np.ndarray] = None,
        val_behavior: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Calculate feature importance for a single fold.
        
        Args:
            train_states: Training set agent states
            val_states: Validation set agent states
            train_recon: Training set reconstructions
            val_recon: Validation set reconstructions
            train_behavior: Training set behavior vectors
            val_behavior: Validation set behavior vectors
            
        Returns:
            Dictionary with importance scores and ranks for this fold
        """
        # Initialize fold analyzer with same features
        fold_analyzer = FeatureImportanceAnalyzer(self.feature_extractors)
        importance_sources = {}
        
        # Get reconstruction importance if reconstructed states provided
        if val_recon is not None:
            recon_importance = fold_analyzer.analyze_importance_for_reconstruction(
                val_states, val_recon
            )
            importance_sources["reconstruction"] = recon_importance
        
        # Get behavior importance if behavior vectors provided
        if val_behavior is not None:
            behavior_importance = fold_analyzer.analyze_importance_for_behavior(
                val_states, val_behavior
            )
            importance_sources["behavior"] = behavior_importance
        
        # Combine importance sources (average)
        importance_scores = {}
        
        if importance_sources:
            for source, importance in importance_sources.items():
                for feature, score in importance.items():
                    if feature not in importance_scores:
                        importance_scores[feature] = 0.0
                    importance_scores[feature] += score / len(importance_sources)
        else:
            # If no sources available, use permutation importance on semantic drift
            # This is a fallback method
            importance_scores = self._calculate_permutation_importance(train_states, val_states)
        
        # Calculate ranks
        sorted_features = sorted(
            importance_scores.keys(), 
            key=lambda x: importance_scores[x], 
            reverse=True
        )
        importance_ranks = {feature: idx+1 for idx, feature in enumerate(sorted_features)}
        
        return {
            "importance_scores": importance_scores,
            "importance_ranks": importance_ranks
        }
    
    def _calculate_permutation_importance(
        self, 
        train_states: torch.Tensor,
        val_states: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate permutation importance as a fallback method.
        
        Args:
            train_states: Training set agent states
            val_states: Validation set agent states
            
        Returns:
            Dictionary mapping features to importance scores
        """
        # Extract features
        features = self.base_analyzer.semantic_loss.extract_semantic_features(val_states)
        
        # Calculate baseline drift
        baseline_drift = self.metrics.calculate_semantic_drift(val_states, val_states).item()
        
        # Calculate importance for each feature
        importance_scores = {}
        
        for feature_name in self.feature_extractors:
            # Create permuted states
            permuted_states = val_states.clone()
            permuted_features = self.base_analyzer.semantic_loss.extract_semantic_features(permuted_states)
            
            # Shuffle this feature
            indices = torch.randperm(permuted_states.size(0))
            permuted_features[feature_name] = permuted_features[feature_name][indices]
            
            # Calculate drift with permuted feature
            permuted_drift = self.metrics.calculate_semantic_drift(val_states, permuted_states).item()
            
            # Importance is the increase in drift when feature is permuted
            importance = permuted_drift - baseline_drift
            importance_scores[feature_name] = max(0.0, importance)  # Ensure non-negative
        
        # Normalize to sum to 1.0
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            for feature in importance_scores:
                importance_scores[feature] /= total_importance
        
        return importance_scores
    
    def _calculate_importance_statistics(
        self,
        fold_importance_scores: Dict[str, List[float]],
        fold_importance_ranks: Dict[str, List[int]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for importance scores and ranks across folds.
        
        Args:
            fold_importance_scores: Importance scores for each feature across folds
            fold_importance_ranks: Importance ranks for each feature across folds
            
        Returns:
            Dictionary with statistics for each feature
        """
        stats = {}
        
        for feature in fold_importance_scores:
            scores = np.array(fold_importance_scores[feature])
            ranks = np.array(fold_importance_ranks[feature])
            
            # Calculate statistics
            stats[feature] = {
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
                "min_score": np.min(scores),
                "max_score": np.max(scores),
                "mean_rank": np.mean(ranks),
                "std_rank": np.std(ranks),
                "min_rank": np.min(ranks),
                "max_rank": np.max(ranks),
                "rank_stability": 1.0 - (np.std(ranks) / len(fold_importance_ranks)),
            }
        
        return stats
    
    def _create_robustness_visualizations(
        self,
        fold_importance_scores: Dict[str, List[float]],
        fold_importance_ranks: Dict[str, List[int]]
    ) -> Dict[str, plt.Figure]:
        """
        Create visualizations for importance robustness.
        
        Args:
            fold_importance_scores: Importance scores for each feature across folds
            fold_importance_ranks: Importance ranks for each feature across folds
            
        Returns:
            Dictionary mapping visualization names to figure objects
        """
        figures = {}
        
        # Create DataFrame for easier plotting
        scores_data = []
        for feature, scores in fold_importance_scores.items():
            for fold, score in enumerate(scores):
                scores_data.append({
                    "Feature": feature,
                    "Fold": fold + 1,
                    "Importance Score": score
                })
        scores_df = pd.DataFrame(scores_data)
        
        ranks_data = []
        for feature, ranks in fold_importance_ranks.items():
            for fold, rank in enumerate(ranks):
                ranks_data.append({
                    "Feature": feature,
                    "Fold": fold + 1,
                    "Importance Rank": rank
                })
        ranks_df = pd.DataFrame(ranks_data)
        
        # Calculate mean scores for ordering
        mean_scores = {feature: np.mean(scores) for feature, scores in fold_importance_scores.items()}
        feature_order = sorted(mean_scores.keys(), key=lambda x: mean_scores[x], reverse=True)
        
        # Box plot of importance scores
        plt.figure(figsize=(12, 8))
        ax = sns.boxplot(x="Feature", y="Importance Score", data=scores_df, 
                         order=feature_order, palette="viridis")
        plt.title("Feature Importance Score Distribution Across Folds", fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        figures["importance_score_boxplot"] = plt.gcf()
        plt.close()
        
        # Box plot of importance ranks
        plt.figure(figsize=(12, 8))
        ax = sns.boxplot(x="Feature", y="Importance Rank", data=ranks_df,
                         order=feature_order, palette="viridis")
        plt.title("Feature Importance Rank Distribution Across Folds", fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        figures["importance_rank_boxplot"] = plt.gcf()
        plt.close()
        
        # Heatmap of importance scores across folds
        score_matrix = np.zeros((len(feature_order), self.n_folds))
        for i, feature in enumerate(feature_order):
            for j in range(self.n_folds):
                score_matrix[i, j] = fold_importance_scores[feature][j]
        
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(score_matrix, annot=True, fmt=".3f", cmap="YlGnBu",
                       xticklabels=[f"Fold {i+1}" for i in range(self.n_folds)],
                       yticklabels=feature_order)
        plt.title("Feature Importance Scores Across Folds", fontsize=14)
        plt.tight_layout()
        figures["importance_score_heatmap"] = plt.gcf()
        plt.close()
        
        # Stability analysis - std dev of ranks vs mean importance
        stability_data = []
        for feature in feature_order:
            stability_data.append({
                "Feature": feature,
                "Mean Importance": np.mean(fold_importance_scores[feature]),
                "Rank Std Dev": np.std(fold_importance_ranks[feature])
            })
        stability_df = pd.DataFrame(stability_data)
        
        plt.figure(figsize=(10, 8))
        ax = sns.scatterplot(x="Mean Importance", y="Rank Std Dev", data=stability_df, s=100)
        
        # Add feature labels to points
        for i, row in stability_df.iterrows():
            plt.text(row["Mean Importance"] + 0.01, row["Rank Std Dev"], 
                     row["Feature"], fontsize=12)
            
        plt.title("Feature Importance Stability Analysis", fontsize=14)
        plt.xlabel("Mean Importance Score", fontsize=12)
        plt.ylabel("Rank Standard Deviation (Lower = More Stable)", fontsize=12)
        plt.tight_layout()
        figures["importance_stability_scatter"] = plt.gcf()
        plt.close()
        
        return figures


def compare_importance_methods(
    agent_states: torch.Tensor,
    reconstructed_states: Optional[torch.Tensor] = None,
    behavior_vectors: Optional[np.ndarray] = None,
    feature_extractors: List[str] = None
) -> Dict[str, Any]:
    """
    Compare different importance measures on the same dataset.
    
    Args:
        agent_states: Original agent states
        reconstructed_states: Reconstructed agent states (optional)
        behavior_vectors: Behavior vectors for prediction (optional)
        feature_extractors: List of features to analyze
        
    Returns:
        Dictionary with comparison results
    """
    import shap
    from sklearn.ensemble import RandomForestRegressor
    
    feature_extractors = feature_extractors or [
        "position", "health", "energy", "is_alive", 
        "has_target", "threatened", "role"
    ]
    
    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer(feature_extractors)
    
    # Extract feature matrix
    feature_matrices, combined_matrix = analyzer.extract_feature_matrix(agent_states)
    
    # Initialize dictionary to store results
    importance_methods = {}
    
    # Method 1: Permutation importance (base method)
    if reconstructed_states is not None:
        permutation_imp = analyzer.analyze_importance_for_reconstruction(
            agent_states, reconstructed_states
        )
        importance_methods["permutation"] = permutation_imp
    
    # Method 2: Random Forest feature importance
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Create a target variable (if not provided, use reconstruction error)
    if behavior_vectors is not None:
        # Use first dimension of behavior as target
        target = behavior_vectors[:, 0]
    else:
        # Use feature drift as target
        metrics = SemanticMetrics(feature_extractors)
        target = metrics.calculate_feature_drift(
            agent_states, 
            reconstructed_states if reconstructed_states is not None else agent_states
        ).cpu().numpy()
    
    # Train model on combined matrix
    rf_model.fit(combined_matrix, target)
    
    # Get feature importances from random forest
    rf_importance = {}
    start_idx = 0
    for feature_name, matrix in feature_matrices.items():
        n_cols = matrix.shape[1]
        feature_imp = np.mean(rf_model.feature_importances_[start_idx:start_idx + n_cols])
        rf_importance[feature_name] = feature_imp
        start_idx += n_cols
    
    # Normalize
    total_rf_imp = sum(rf_importance.values())
    if total_rf_imp > 0:
        for feature in rf_importance:
            rf_importance[feature] /= total_rf_imp
    
    importance_methods["random_forest"] = rf_importance
    
    # Method 3: SHAP values (if sample size is reasonable)
    if combined_matrix.shape[0] <= 500:  # Limit to reasonable sample size
        try:
            explainer = shap.Explainer(rf_model, combined_matrix[:100])  # Use subset for efficiency
            shap_values = explainer(combined_matrix[:100])
            
            # Aggregate SHAP values by feature
            shap_importance = {}
            start_idx = 0
            for feature_name, matrix in feature_matrices.items():
                n_cols = matrix.shape[1]
                # Take absolute SHAP values and average
                feature_imp = np.mean(np.abs(shap_values.values[:, start_idx:start_idx + n_cols]).mean(0))
                shap_importance[feature_name] = feature_imp
                start_idx += n_cols
            
            # Normalize
            total_shap_imp = sum(shap_importance.values())
            if total_shap_imp > 0:
                for feature in shap_importance:
                    shap_importance[feature] /= total_shap_imp
            
            importance_methods["shap"] = shap_importance
        except:
            print("SHAP analysis failed or not available. Skipping.")
    
    # Create comparison visualization
    fig_comparison = create_method_comparison_visualization(importance_methods)
    
    # Calculate correlation between methods
    method_correlation = calculate_method_correlation(importance_methods)
    
    return {
        "importance_methods": importance_methods,
        "method_correlation": method_correlation,
        "comparison_figure": fig_comparison
    }


def create_method_comparison_visualization(
    importance_methods: Dict[str, Dict[str, float]]
) -> plt.Figure:
    """
    Create visualization comparing different importance methods.
    
    Args:
        importance_methods: Dictionary mapping method names to importance scores
        
    Returns:
        Matplotlib figure object
    """
    # Get all features across all methods
    all_features = set()
    for method_scores in importance_methods.values():
        all_features.update(method_scores.keys())
    all_features = sorted(all_features)
    
    # Create comparison data
    comparison_data = []
    for method_name, method_scores in importance_methods.items():
        for feature in all_features:
            comparison_data.append({
                "Method": method_name,
                "Feature": feature,
                "Importance": method_scores.get(feature, 0.0)
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create grouped bar chart
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x="Feature", y="Importance", hue="Method", data=comparison_df)
    plt.title("Feature Importance Comparison Across Methods", fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title="Method")
    plt.tight_layout()
    
    return plt.gcf()


def calculate_method_correlation(
    importance_methods: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Calculate correlation between different importance methods.
    
    Args:
        importance_methods: Dictionary mapping method names to importance scores
        
    Returns:
        Correlation matrix as pandas DataFrame
    """
    # Get all features across all methods
    all_features = set()
    for method_scores in importance_methods.values():
        all_features.update(method_scores.keys())
    all_features = sorted(all_features)
    
    # Create method-feature matrix
    method_feature_matrix = {}
    for method_name, method_scores in importance_methods.items():
        method_feature_matrix[method_name] = [method_scores.get(feature, 0.0) for feature in all_features]
    
    # Convert to DataFrame
    method_df = pd.DataFrame(method_feature_matrix, index=all_features)
    
    # Calculate correlation
    correlation_matrix = method_df.corr()
    
    return correlation_matrix


def run_feature_importance_robustness_analysis(
    agent_states: torch.Tensor,
    reconstructed_states: Optional[torch.Tensor] = None,
    behavior_vectors: Optional[np.ndarray] = None,
    feature_extractors: List[str] = None,
    n_folds: int = 5,
    random_seed: int = 42,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Run comprehensive feature importance robustness analysis.
    
    Args:
        agent_states: Original agent states
        reconstructed_states: Reconstructed agent states (optional)
        behavior_vectors: Behavior vectors for prediction (optional)
        feature_extractors: List of features to analyze
        n_folds: Number of folds for cross-validation
        random_seed: Random seed for reproducibility
        output_dir: Directory to save results (optional)
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    import os
    from pathlib import Path
    
    # Initialize results dictionary
    results = {}
    
    # Initialize analyzer
    analyzer = FeatureImportanceRobustnessAnalyzer(
        feature_extractors=feature_extractors,
        n_folds=n_folds,
        random_seed=random_seed
    )
    
    print("Running cross-validation analysis...")
    cv_results = analyzer.cross_validate_importance(
        agent_states, reconstructed_states, behavior_vectors
    )
    results["cross_validation"] = cv_results
    
    print("Comparing different importance methods...")
    method_comparison = compare_importance_methods(
        agent_states, reconstructed_states, behavior_vectors, feature_extractors
    )
    results["method_comparison"] = method_comparison
    
    # Save results if output directory provided
    if output_dir:
        output_path = Path(output_dir)
        os.makedirs(output_path, exist_ok=True)
        
        # Save figures
        for name, fig in cv_results["figures"].items():
            fig.savefig(output_path / f"{name}.png", dpi=300, bbox_inches="tight")
        
        method_comparison["comparison_figure"].savefig(
            output_path / "method_comparison.png", dpi=300, bbox_inches="tight"
        )
        
        # Save correlation matrix
        method_comparison["method_correlation"].to_csv(
            output_path / "method_correlation.csv"
        )
        
        # Save importance statistics
        importance_stats = pd.DataFrame.from_dict(
            cv_results["importance_stats"], orient="index"
        )
        importance_stats.to_csv(output_path / "importance_statistics.csv")
        
        print(f"Results saved to {output_path}")
    
    return results 