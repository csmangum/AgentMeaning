#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Meaning validation module for the meaning-preserving transformation system.

This module implements the operational definition of meaning and provides
validation tools to correlate semantic metrics with behavioral outcomes.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from dtw import dtw

from .standardized_metrics import StandardizedMetrics


class MeaningValidator:
    """
    Framework for validating the operational definition of meaning through
    behavioral equivalence testing and correlation with semantic metrics.
    """

    def __init__(self, semantic_metrics: Optional[StandardizedMetrics] = None):
        """
        Initialize the meaning validator.

        Args:
            semantic_metrics: StandardizedMetrics instance for semantic evaluation
        """
        self.semantic_metrics = semantic_metrics or StandardizedMetrics()
        
        # Default weights for meaning preservation score components
        self.weights = {
            "feature_preservation": 0.6,  # Semantic feature preservation
            "behavioral_equivalence": 0.3,  # Behavioral metrics
            "human_consensus": 0.1,  # Human evaluation (if available)
        }

    # ----- Behavioral Validation Methods -----

    def action_selection_agreement(
        self, 
        original_states: torch.Tensor, 
        transformed_states: torch.Tensor, 
        policy_function: Callable
    ) -> float:
        """
        Measure action selection agreement between original and transformed states.
        
        Args:
            original_states: Original agent states
            transformed_states: Transformed agent states
            policy_function: Function that maps states to actions
            
        Returns:
            agreement_rate: Proportion of matching actions
        """
        original_actions = [policy_function(state) for state in original_states]
        transformed_actions = [policy_function(state) for state in transformed_states]
        
        matches = sum(1 for o, t in zip(original_actions, transformed_actions) if o == t)
        return matches / len(original_actions)

    def trajectory_similarity(
        self, 
        original_trajectory: List[Any], 
        transformed_trajectory: List[Any]
    ) -> float:
        """
        Measure similarity between behavior trajectories using dynamic time warping.
        
        Args:
            original_trajectory: Sequence of states or actions from original states
            transformed_trajectory: Sequence of states or actions from transformed states
            
        Returns:
            similarity: Normalized similarity score (0-1)
        """
        # Convert to numpy arrays if they're not already
        if isinstance(original_trajectory, torch.Tensor):
            original_trajectory = original_trajectory.cpu().numpy()
        if isinstance(transformed_trajectory, torch.Tensor):
            transformed_trajectory = transformed_trajectory.cpu().numpy()
            
        # Calculate DTW distance
        alignment = dtw(original_trajectory, transformed_trajectory)
        dtw_distance = alignment.distance
        
        # Normalize to [0,1] where 1 is perfect similarity
        max_distance = max(len(original_trajectory), len(transformed_trajectory))
        similarity = 1 - (dtw_distance / (max_distance * 10))  # Scaling factor
        
        # Clip to [0,1] range
        return max(0, min(1, similarity))

    def decision_time_ratio(
        self, 
        original_states: torch.Tensor, 
        transformed_states: torch.Tensor, 
        policy_function: Callable
    ) -> float:
        """
        Measure the ratio of decision times between original and transformed states.
        
        Args:
            original_states: Original agent states
            transformed_states: Transformed agent states
            policy_function: Function that maps states to actions
            
        Returns:
            time_ratio: Ratio of transformed decision time to original decision time
        """
        original_times = []
        transformed_times = []
        
        for o_state, t_state in zip(original_states, transformed_states):
            start = time.time()
            policy_function(o_state)
            original_times.append(time.time() - start)
            
            start = time.time()
            policy_function(t_state)
            transformed_times.append(time.time() - start)
        
        return sum(transformed_times) / sum(original_times) if sum(original_times) > 0 else float('inf')

    def task_completion_ratio(
        self, 
        original_states: torch.Tensor, 
        transformed_states: torch.Tensor, 
        task_evaluator: Callable
    ) -> float:
        """
        Measure task completion success ratio between original and transformed states.
        
        Args:
            original_states: Original agent states
            transformed_states: Transformed agent states
            task_evaluator: Function that evaluates task success
            
        Returns:
            completion_ratio: Ratio of transformed success to original success
        """
        original_success = task_evaluator(original_states)
        transformed_success = task_evaluator(transformed_states)
        
        return transformed_success / original_success if original_success > 0 else 0.0

    # ----- Correlation Analysis Methods -----

    def feature_behavior_correlation(
        self,
        feature_preservation_levels: Dict[str, List[float]],
        behavior_metrics: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlation between feature preservation and behavior metrics.
        
        Args:
            feature_preservation_levels: Dictionary mapping feature names to preservation levels
            behavior_metrics: Dictionary mapping behavior metric names to measurements
            
        Returns:
            correlations: Nested dictionary of correlation values
        """
        correlations = {}
        p_values = {}
        
        for feature, preservation in feature_preservation_levels.items():
            correlations[feature] = {}
            p_values[feature] = {}
            
            for behavior, metrics in behavior_metrics.items():
                if len(preservation) == len(metrics):
                    corr, p_val = pearsonr(preservation, metrics)
                    correlations[feature][behavior] = corr
                    p_values[feature][behavior] = p_val
        
        return {"correlations": correlations, "p_values": p_values}

    def find_preservation_thresholds(
        self,
        preservation_levels: List[float],
        behavior_scores: Dict[str, List[float]],
        threshold: float = 0.90
    ) -> Dict[str, float]:
        """
        Find the preservation level threshold where behavior maintains above threshold % of original.
        
        Args:
            preservation_levels: List of preservation levels (0-1)
            behavior_scores: Dictionary mapping behavior metric names to scores
            threshold: Minimum acceptable behavior score (relative to maximum)
            
        Returns:
            thresholds: Dictionary mapping behavior metrics to minimum required preservation
        """
        thresholds = {}
        
        for behavior_metric, scores in behavior_scores.items():
            # Normalize behavior scores relative to maximum
            max_score = max(scores)
            if max_score > 0:
                normalized_scores = [score / max_score for score in scores]
                
                # Find first preservation level where behavior drops below threshold
                for i, score in enumerate(normalized_scores):
                    if score < threshold:
                        if i > 0:
                            thresholds[behavior_metric] = preservation_levels[i-1]
                        else:
                            thresholds[behavior_metric] = preservation_levels[0]  # Default to first level
                        break
                else:
                    # If no threshold crossing found, use the lowest tested preservation level
                    thresholds[behavior_metric] = min(preservation_levels)
        
        return thresholds

    def causal_intervention_test(
        self,
        states: torch.Tensor,
        feature: str,
        perturbation_magnitudes: List[float],
        policy_function: Callable
    ) -> List[Tuple[float, float]]:
        """
        Test causal relationship between feature and behavior through intervention.
        
        Args:
            states: Agent states to perturb
            feature: Name of the feature to perturb
            perturbation_magnitudes: List of perturbation magnitudes to test
            policy_function: Function that maps states to actions
            
        Returns:
            results: List of (magnitude, change_rate) tuples
        """
        baseline_actions = [policy_function(state) for state in states]
        
        results = []
        for magnitude in perturbation_magnitudes:
            # Create perturbed states (implementation depends on feature representation)
            perturbed_states = self._perturb_feature(states, feature, magnitude)
            perturbed_actions = [policy_function(state) for state in perturbed_states]
            
            # Calculate action change rate
            change_rate = sum(1 for b, p in zip(baseline_actions, perturbed_actions) if b != p) / len(states)
            results.append((magnitude, change_rate))
        
        return results

    def _perturb_feature(
        self, 
        states: torch.Tensor, 
        feature: str, 
        magnitude: float
    ) -> torch.Tensor:
        """
        Perturb a specific feature in the states by the given magnitude.
        
        Args:
            states: Agent states to perturb
            feature: Name of the feature to perturb
            magnitude: Perturbation magnitude
            
        Returns:
            perturbed_states: States with perturbed feature
        """
        # Deep copy the states to avoid modifying the original
        perturbed_states = states.clone()
        
        # Implementation depends on how features are represented in the state
        # This is a simplified example that assumes features are directly accessible
        if feature == "position":
            # Assuming position is stored at indices 0 and 1
            perturbed_states[:, 0:2] += magnitude * torch.randn_like(perturbed_states[:, 0:2])
        elif feature == "health":
            # Assuming health is stored at index 2
            perturbed_states[:, 2] += magnitude * torch.randn_like(perturbed_states[:, 2])
            perturbed_states[:, 2] = torch.clamp(perturbed_states[:, 2], 0, 1)  # Constrain to [0,1]
        elif feature == "is_alive":
            # Assuming is_alive is stored at index 3 as a binary value
            if magnitude > 0.5:  # Threshold for binary perturbation
                perturbed_states[:, 3] = 1 - perturbed_states[:, 3]  # Flip the binary value
        
        return perturbed_states

    # ----- Integration with Semantic Metrics -----

    def semantic_behavioral_correlation(
        self,
        semantic_metrics: Dict[str, List[float]],
        behavioral_metrics: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlation between semantic metrics and behavioral metrics.
        
        Args:
            semantic_metrics: Dictionary mapping semantic metric names to values
            behavioral_metrics: Dictionary mapping behavior metric names to values
            
        Returns:
            correlations: Nested dictionary of correlation values
        """
        correlations = {}
        p_values = {}
        
        for semantic_metric, sem_values in semantic_metrics.items():
            correlations[semantic_metric] = {}
            p_values[semantic_metric] = {}
            
            for behavioral_metric, behav_values in behavioral_metrics.items():
                if len(sem_values) == len(behav_values):
                    corr, p_val = pearsonr(sem_values, behav_values)
                    correlations[semantic_metric][behavioral_metric] = corr
                    p_values[semantic_metric][behavioral_metric] = p_val
        
        return {"correlations": correlations, "p_values": p_values}

    def build_behavior_prediction_model(
        self,
        semantic_metrics: Dict[str, List[float]],
        behavioral_outcome: List[float]
    ) -> Tuple[LinearRegression, float, Dict[str, float]]:
        """
        Build model to predict behavioral outcomes from semantic metrics.
        
        Args:
            semantic_metrics: Dictionary mapping semantic metric names to values
            behavioral_outcome: List of behavioral outcome values
            
        Returns:
            model: Trained regression model
            r2: R-squared score on test data
            feature_importance: Dictionary of feature importance scores
        """
        # Convert to numpy arrays
        X = np.array(list(semantic_metrics.values())).T
        y = np.array(behavioral_outcome)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate
        r2 = model.score(X_test, y_test)
        feature_importance = {feature: importance for feature, importance 
                             in zip(semantic_metrics.keys(), model.coef_)}
        
        return model, r2, feature_importance

    # ----- Unified Meaning Preservation Score -----

    def calculate_meaning_preservation_score(
        self,
        original_states: torch.Tensor,
        transformed_states: torch.Tensor,
        policy_function: Optional[Callable] = None,
        human_consensus: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate a unified meaning preservation score combining semantic and behavioral metrics.
        
        Args:
            original_states: Original agent states
            transformed_states: Transformed agent states
            policy_function: Optional function for behavioral evaluation
            human_consensus: Optional human consensus score
            
        Returns:
            scores: Dictionary of meaning preservation scores
        """
        # Calculate semantic preservation metrics
        semantic_results = self.semantic_metrics.evaluate(original_states, transformed_states)
        p_overall = semantic_results.get("overall_preservation", 0.0)
        
        # Calculate behavioral equivalence if policy function is provided
        b_equiv = 0.0
        if policy_function is not None:
            action_agreement = self.action_selection_agreement(
                original_states, transformed_states, policy_function
            )
            b_equiv = action_agreement
        
        # Use human consensus if provided, otherwise default to 0.0
        h_consensus = human_consensus if human_consensus is not None else 0.0
        
        # Calculate unified score using weights
        unified_score = (
            self.weights["feature_preservation"] * p_overall +
            self.weights["behavioral_equivalence"] * b_equiv +
            self.weights["human_consensus"] * h_consensus
        )
        
        # Determine qualitative category
        category = "critical"
        for threshold_name, threshold_value in sorted(
            self.semantic_metrics.THRESHOLDS["preservation"].items(),
            key=lambda x: x[1], 
            reverse=True
        ):
            if unified_score >= threshold_value:
                category = threshold_name
                break
        
        return {
            "unified_score": unified_score,
            "feature_preservation_score": p_overall,
            "behavioral_equivalence_score": b_equiv,
            "human_consensus_score": h_consensus,
            "meaning_category": category
        }

    # ----- Visualization Methods -----

    def plot_correlation_heatmap(
        self,
        correlation_matrix: Dict[str, Dict[str, float]]
    ) -> plt.Figure:
        """
        Create a heatmap visualization of correlation strengths.
        
        Args:
            correlation_matrix: Nested dictionary of correlation values
            
        Returns:
            fig: Matplotlib figure object
        """
        # Convert nested dictionary to DataFrame
        df = pd.DataFrame(correlation_matrix)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = ax.pcolor(df, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Put the major ticks at the middle of each cell
        ax.set_xticks(np.arange(df.shape[1]) + 0.5, minor=False)
        ax.set_yticks(np.arange(df.shape[0]) + 0.5, minor=False)
        
        # Want a more natural, table-like display
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        
        # Set labels
        ax.set_xticklabels(df.columns, minor=False, rotation=45)
        ax.set_yticklabels(df.index, minor=False)
        
        # Add colorbar
        cbar = plt.colorbar(heatmap)
        cbar.set_label('Correlation Strength')
        
        plt.title('Correlation between Semantic and Behavioral Metrics')
        plt.tight_layout()
        
        return fig

    def plot_threshold_analysis(
        self,
        preservation_levels: List[float],
        behavior_scores: Dict[str, List[float]]
    ) -> plt.Figure:
        """
        Create a visualization of behavior scores vs. preservation levels.
        
        Args:
            preservation_levels: List of preservation levels (0-1)
            behavior_scores: Dictionary mapping behavior metric names to scores
            
        Returns:
            fig: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for metric, scores in behavior_scores.items():
            ax.plot(preservation_levels, scores, marker='o', label=metric)
            
        # Add thresholds
        ax.axhline(y=0.90, color='r', linestyle='--', alpha=0.7, label='90% Performance')
        ax.axhline(y=0.85, color='orange', linestyle='--', alpha=0.7, label='85% Performance')
        ax.axhline(y=0.70, color='purple', linestyle='--', alpha=0.7, label='70% Performance')
        
        ax.set_xlabel('Semantic Preservation Level')
        ax.set_ylabel('Behavioral Performance')
        ax.set_title('Behavioral Performance vs. Semantic Preservation')
        ax.legend()
        ax.grid(alpha=0.3)
        
        return fig

    def plot_intervention_results(
        self,
        intervention_results: Dict[str, List[Tuple[float, float]]]
    ) -> plt.Figure:
        """
        Create a visualization of causal intervention test results.
        
        Args:
            intervention_results: Dictionary mapping feature names to lists of (magnitude, change_rate) tuples
            
        Returns:
            fig: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for feature, results in intervention_results.items():
            magnitudes, change_rates = zip(*results)
            ax.plot(magnitudes, change_rates, marker='o', label=feature)
            
        ax.set_xlabel('Perturbation Magnitude')
        ax.set_ylabel('Action Change Rate')
        ax.set_title('Feature Sensitivity Analysis')
        ax.legend()
        ax.grid(alpha=0.3)
        
        return fig 