#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standardized metrics module for the meaning-preserving transformation system.

This module provides a unified framework for measuring and tracking semantic metrics:
1. Standard definitions for semantic drift, preservation, and fidelity
2. Consistent metrics categories and normalization
3. Feature importance-weighted evaluation
4. Standardized benchmarks and thresholds
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

from .metrics import SemanticMetrics
from .loss import SemanticLoss


class StandardizedMetrics(SemanticMetrics):
    """
    Standardized framework for semantic metrics that ensures consistency
    across different experiment types and models.
    
    Extends the base SemanticMetrics class with:
    - Clear operational definitions for metrics
    - Standardized feature groups with importance weighting
    - Normalized scores for comparability across experiments
    - Categorized metrics (preservation, fidelity, drift)
    """
    
    # Standard feature groups based on importance analysis
    FEATURE_GROUPS = {
        "spatial": ["position"],  # 55.4% importance
        "resources": ["health", "energy"],  # 25.1% importance
        "performance": ["is_alive", "has_target", "threatened"],  # 10.5% importance
        "role": ["role"],  # <5% importance
    }
    
    # Standard weights for feature groups based on importance analysis
    FEATURE_GROUP_WEIGHTS = {
        "spatial": 0.554,
        "resources": 0.251,
        "performance": 0.105,
        "role": 0.050,
    }
    
    # Thresholds for acceptable performance
    THRESHOLDS = {
        "preservation": {
            "excellent": 0.95,
            "good": 0.90,
            "acceptable": 0.85,
            "poor": 0.70,
            "critical": 0.50,
        },
        "fidelity": {
            "excellent": 0.95,
            "good": 0.90,
            "acceptable": 0.85,
            "poor": 0.70,
            "critical": 0.50,
        },
        "drift": {
            "excellent": 0.05,  # Lower is better for drift
            "good": 0.10,
            "acceptable": 0.15,
            "poor": 0.30,
            "critical": 0.50,
        }
    }

    def __init__(self, feature_extractors: List[str] = None, normalize_scores: bool = True):
        """
        Initialize standardized metrics.
        
        Args:
            feature_extractors: List of semantic features to extract and compare
            normalize_scores: Whether to normalize scores to [0,1] range
        """
        # Initialize the base class
        super().__init__(feature_extractors)
        
        self.normalize_scores = normalize_scores
        
        # Create mappings for feature to group
        self.feature_to_group = {}
        for group, features in self.FEATURE_GROUPS.items():
            for feature in features:
                self.feature_to_group[feature] = group
    
    def measure_preservation(self, 
                           original: torch.Tensor, 
                           reconstructed: torch.Tensor) -> Dict[str, float]:
        """
        Measure semantic preservation between original and reconstructed states.
        
        Preservation focuses on how well the essential meaning of the original
        state is maintained in the reconstruction, with emphasis on the most
        important features defined by feature importance analysis.
        
        Args:
            original: Original agent states
            reconstructed: Reconstructed agent states
            
        Returns:
            preservation_metrics: Dictionary of preservation metrics
        """
        # Get similarity scores from base class
        similarity_scores = self.compute_equivalence_scores(original, reconstructed)
        
        # Group features by category and apply standard weights
        group_scores = {}
        for group, weight in self.FEATURE_GROUP_WEIGHTS.items():
            features_in_group = self.FEATURE_GROUPS[group]
            # Get scores for features in this group
            feature_scores = [
                similarity_scores[feature] 
                for feature in features_in_group 
                if feature in similarity_scores
            ]
            
            if feature_scores:
                group_scores[f"{group}_preservation"] = sum(feature_scores) / len(feature_scores)
            
        # Calculate overall weighted preservation score
        weighted_sum = sum(
            score * self.FEATURE_GROUP_WEIGHTS[group.split('_')[0]]
            for group, score in group_scores.items()
        )
        total_weight = sum(
            self.FEATURE_GROUP_WEIGHTS[group.split('_')[0]]
            for group in group_scores
        )
        
        overall_preservation = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Add overall preservation score
        group_scores["overall_preservation"] = overall_preservation
        
        # Add qualitative assessment based on thresholds
        threshold_category = "critical"
        for category, threshold in sorted(
            self.THRESHOLDS["preservation"].items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            if overall_preservation >= threshold:
                threshold_category = category
                break
        
        group_scores["preservation_category"] = threshold_category
        
        return group_scores
    
    def measure_fidelity(self,
                       original: torch.Tensor,
                       reconstructed: torch.Tensor) -> Dict[str, float]:
        """
        Measure the fidelity of reconstruction at the raw feature level.
        
        Fidelity focuses on the accuracy of reconstruction for specific features,
        without weighting by importance. This captures how precisely the model
        reconstructs the exact values, even for less important features.
        
        Args:
            original: Original agent states
            reconstructed: Reconstructed agent states
            
        Returns:
            fidelity_metrics: Dictionary of fidelity metrics
        """
        # Combine binary accuracy, role accuracy, and numeric feature metrics
        fidelity_metrics = {}
        
        # Get binary feature metrics
        binary_metrics = self.binary_feature_accuracy(original, reconstructed)
        for feature in ["has_target", "is_alive", "threatened"]:
            if f"{feature}_accuracy" in binary_metrics:
                fidelity_metrics[f"{feature}_fidelity"] = binary_metrics[f"{feature}_accuracy"]
        
        # Get role accuracy
        role_metrics = self.role_accuracy(original, reconstructed)
        if "role_accuracy" in role_metrics:
            fidelity_metrics["role_fidelity"] = role_metrics["role_accuracy"]
        
        # Get numeric feature errors and convert to fidelity scores
        numeric_errors = self.numeric_feature_errors(original, reconstructed)
        for feature in ["position", "health", "energy"]:
            error_key = f"{feature}_rmse"
            if error_key in numeric_errors:
                # Convert error to fidelity score (lower error = higher fidelity)
                # Use exp(-error) to map error in [0,∞) to fidelity in (0,1]
                error = numeric_errors[error_key]
                fidelity_metrics[f"{feature}_fidelity"] = float(np.exp(-error))
        
        # Group features by category
        group_fidelity = {}
        for group, features in self.FEATURE_GROUPS.items():
            group_scores = [
                fidelity_metrics[f"{feature}_fidelity"]
                for feature in features
                if f"{feature}_fidelity" in fidelity_metrics
            ]
            
            if group_scores:
                group_fidelity[f"{group}_fidelity"] = sum(group_scores) / len(group_scores)
        
        # Calculate overall fidelity (unweighted average across all features)
        feature_fidelity_scores = [
            score for key, score in fidelity_metrics.items() 
            if key.endswith("_fidelity") and not key.startswith("overall_")
        ]
        
        if feature_fidelity_scores:
            group_fidelity["overall_fidelity"] = sum(feature_fidelity_scores) / len(feature_fidelity_scores)
        else:
            group_fidelity["overall_fidelity"] = 0.0
        
        # Add qualitative assessment based on thresholds
        threshold_category = "critical"
        for category, threshold in sorted(
            self.THRESHOLDS["fidelity"].items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            if group_fidelity["overall_fidelity"] >= threshold:
                threshold_category = category
                break
        
        group_fidelity["fidelity_category"] = threshold_category
        
        # Combine individual and group metrics
        fidelity_metrics.update(group_fidelity)
        
        return fidelity_metrics
    
    def measure_drift(self,
                    original: torch.Tensor,
                    reconstructed: torch.Tensor,
                    baseline_original: Optional[torch.Tensor] = None,
                    baseline_reconstructed: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Measure semantic drift between current reconstruction and a baseline.
        
        Drift focuses on how much the reconstruction quality has changed over time
        or across different compression levels. It requires both current and baseline
        measurements to calculate the difference.
        
        Args:
            original: Current original agent states
            reconstructed: Current reconstructed agent states
            baseline_original: Baseline original agent states (if None, drift is 0)
            baseline_reconstructed: Baseline reconstructed agent states (if None, drift is 0)
            
        Returns:
            drift_metrics: Dictionary of drift metrics
        """
        # If no baseline provided, return zero drift
        if baseline_original is None or baseline_reconstructed is None:
            return {
                "overall_drift": 0.0,
                "spatial_drift": 0.0,
                "resources_drift": 0.0,
                "performance_drift": 0.0,
                "role_drift": 0.0,
                "drift_category": "excellent"
            }
        
        # Measure current preservation
        current_preservation = self.measure_preservation(original, reconstructed)
        
        # Measure baseline preservation
        baseline_preservation = self.measure_preservation(baseline_original, baseline_reconstructed)
        
        # Calculate drift as the difference in preservation
        drift_metrics = {}
        for key, value in current_preservation.items():
            if key.endswith("_preservation") and key in baseline_preservation:
                # Drift is decrease in preservation (positive means worse performance)
                drift_name = key.replace("_preservation", "_drift")
                drift_value = max(0.0, baseline_preservation[key] - value)
                drift_metrics[drift_name] = drift_value
        
        # Add qualitative assessment based on thresholds
        threshold_category = "critical"
        for category, threshold in sorted(
            self.THRESHOLDS["drift"].items(), 
            key=lambda x: x[1]
        ):
            if drift_metrics.get("overall_drift", 1.0) <= threshold:
                threshold_category = category
                break
        
        drift_metrics["drift_category"] = threshold_category
        
        return drift_metrics
    
    def evaluate(self,
                original: torch.Tensor,
                reconstructed: torch.Tensor,
                baseline_original: Optional[torch.Tensor] = None,
                baseline_reconstructed: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Comprehensive standardized evaluation combining all metric categories.
        
        Args:
            original: Original agent states
            reconstructed: Reconstructed agent states
            baseline_original: Optional baseline original states for drift measurement
            baseline_reconstructed: Optional baseline reconstructed states for drift measurement
            
        Returns:
            evaluation: Dictionary of all evaluation metrics organized by category
        """
        # Measure all metrics
        preservation_metrics = self.measure_preservation(original, reconstructed)
        fidelity_metrics = self.measure_fidelity(original, reconstructed)
        drift_metrics = self.measure_drift(
            original, reconstructed, baseline_original, baseline_reconstructed
        )
        
        # Build comprehensive evaluation with clear categories
        evaluation = {
            "preservation": preservation_metrics,
            "fidelity": fidelity_metrics,
            "drift": drift_metrics,
            
            # Include top-level summary metrics for convenience
            "overall_preservation": preservation_metrics["overall_preservation"],
            "overall_fidelity": fidelity_metrics["overall_fidelity"],
            "overall_drift": drift_metrics["overall_drift"],
            
            # Include categorical assessments
            "preservation_category": preservation_metrics["preservation_category"],
            "fidelity_category": fidelity_metrics["fidelity_category"],
            "drift_category": drift_metrics["drift_category"],
        }
        
        return evaluation


class MetricsConverter:
    """
    Utility class for converting legacy metrics to standardized formats
    and providing backward compatibility.
    """
    
    def __init__(self, standardized_metrics: StandardizedMetrics):
        """
        Initialize metrics converter.
        
        Args:
            standardized_metrics: Instance of StandardizedMetrics to use for conversion
        """
        self.standardized_metrics = standardized_metrics
    
    def legacy_to_standardized(self, legacy_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert legacy metrics format to standardized format.
        
        Args:
            legacy_metrics: Metrics in the legacy format
            
        Returns:
            standardized: Metrics in the standardized format
        """
        # Initialize output structure
        standardized = {
            "preservation": {},
            "fidelity": {},
            "drift": {"overall_drift": 0.0, "drift_category": "excellent"},
        }
        
        # Map legacy overall score to preservation
        if "overall" in legacy_metrics:
            standardized["preservation"]["overall_preservation"] = legacy_metrics["overall"]
            
            # Determine preservation category
            threshold_category = "critical"
            for category, threshold in sorted(
                StandardizedMetrics.THRESHOLDS["preservation"].items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
                if legacy_metrics["overall"] >= threshold:
                    threshold_category = category
                    break
            standardized["preservation"]["preservation_category"] = threshold_category
        
        # Map binary accuracies to fidelity
        for feature in ["has_target", "is_alive", "threatened"]:
            key = f"{feature}_accuracy"
            if key in legacy_metrics:
                standardized["fidelity"][f"{feature}_fidelity"] = legacy_metrics[key]
        
        # Map role accuracy to fidelity
        if "role_accuracy" in legacy_metrics:
            standardized["fidelity"]["role_fidelity"] = legacy_metrics["role_accuracy"]
        
        # Map numeric errors to fidelity
        for feature in ["position", "health", "energy"]:
            error_key = f"{feature}_rmse"
            if error_key in legacy_metrics:
                error = legacy_metrics[error_key]
                standardized["fidelity"][f"{feature}_fidelity"] = float(np.exp(-error))
        
        # Calculate group scores and overall metrics
        self._calculate_group_scores(standardized)
        
        # Add top-level summary metrics
        standardized["overall_preservation"] = standardized["preservation"].get("overall_preservation", 0.0)
        standardized["overall_fidelity"] = standardized["fidelity"].get("overall_fidelity", 0.0)
        standardized["overall_drift"] = 0.0  # Legacy metrics don't have drift
        
        # Add categorical assessments
        standardized["preservation_category"] = standardized["preservation"].get("preservation_category", "critical")
        standardized["fidelity_category"] = standardized["fidelity"].get("fidelity_category", "critical")
        standardized["drift_category"] = "excellent"  # Legacy metrics don't have drift
        
        return standardized
    
    def _calculate_group_scores(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Calculate group scores for preservation and fidelity.
        
        Args:
            metrics: Metrics dictionary to update with group scores
        """
        # Calculate group scores for fidelity
        fidelity = metrics["fidelity"]
        for group, features in StandardizedMetrics.FEATURE_GROUPS.items():
            group_scores = [
                fidelity[f"{feature}_fidelity"]
                for feature in features
                if f"{feature}_fidelity" in fidelity
            ]
            
            if group_scores:
                fidelity[f"{group}_fidelity"] = sum(group_scores) / len(group_scores)
        
        # Calculate overall fidelity if not already present
        if "overall_fidelity" not in fidelity:
            feature_scores = [
                score for key, score in fidelity.items() 
                if key.endswith("_fidelity") and not key.startswith("overall_")
            ]
            
            if feature_scores:
                fidelity["overall_fidelity"] = sum(feature_scores) / len(feature_scores)
            else:
                fidelity["overall_fidelity"] = 0.0
        
        # Add fidelity category if not present
        if "fidelity_category" not in fidelity:
            threshold_category = "critical"
            for category, threshold in sorted(
                StandardizedMetrics.THRESHOLDS["fidelity"].items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
                if fidelity.get("overall_fidelity", 0.0) >= threshold:
                    threshold_category = category
                    break
            fidelity["fidelity_category"] = threshold_category


# Legacy compatibility function
def convert_legacy_metrics(legacy_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to convert legacy metrics to standardized format.
    
    Args:
        legacy_metrics: Metrics in the legacy format
        
    Returns:
        standardized: Metrics in the standardized format
    """
    converter = MetricsConverter(StandardizedMetrics())
    return converter.legacy_to_standardized(legacy_metrics) 