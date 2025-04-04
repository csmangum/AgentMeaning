#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robustness pipeline for the meaning-preserving transformation system.

This module provides a pipeline that tests the robustness of the meaning representation
by introducing various types of perturbations:
1. Gaussian noise (random noise)
2. Structured noise (targeting specific feature dimensions)
3. Feature dropout (zeroing out random features)
4. Feature swapping (scrambling feature order)
5. Outlier injection (adding extreme values)

The pipeline measures how well meaning is preserved under these perturbations.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from meaning_transform.src.config import Config
from meaning_transform.src.data import AgentState, AgentStateDataset
from meaning_transform.src.models import MeaningVAE
from meaning_transform.src.pipelines.pipeline import (
    BranchComponent,
    ConditionalComponent,
    CustomComponent,
    Pipeline,
    PipelineAdapter,
    PipelineComponent,
    SemanticEvaluationComponent,
)
from meaning_transform.src.standardized_metrics import StandardizedMetrics


class PerturbationComponent(PipelineComponent):
    """Component that applies perturbations to input data."""

    def __init__(
        self,
        perturbation_type: str = "gaussian",
        intensity: float = 0.1,
        feature_indices: Optional[List[int]] = None,
        dropout_prob: float = 0.2,
        name: Optional[str] = None,
    ):
        """
        Initialize perturbation component.

        Args:
            perturbation_type: Type of perturbation to apply
                ("gaussian", "structured", "dropout", "swap", "outlier")
            intensity: Intensity of the perturbation (standard deviation for noise)
            feature_indices: Specific feature indices to perturb (for structured noise)
            dropout_prob: Probability of dropping a feature (for dropout)
            name: Component name (optional)
        """
        self.perturbation_type = perturbation_type
        self.intensity = intensity
        self.feature_indices = feature_indices
        self.dropout_prob = dropout_prob
        self._name = name

    @property
    def name(self) -> str:
        """Get component name."""
        return self._name or f"Perturbation[{self.perturbation_type}]"

    def process(
        self, data: torch.Tensor, context: Dict[str, Any] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply perturbation to input data.

        Args:
            data: Input tensor
            context: Processing context

        Returns:
            output: Perturbed tensor
            context: Updated context
        """
        if context is None:
            context = {}

        # Store original data
        context["original_data"] = data.clone()

        # Only perturb tensor data
        if not isinstance(data, torch.Tensor):
            return data, context

        # Apply perturbation based on type
        if self.perturbation_type == "gaussian":
            # Apply Gaussian noise to all features
            noise = torch.randn_like(data) * self.intensity
            perturbed = data + noise

        elif self.perturbation_type == "structured":
            # Apply noise to specific feature dimensions
            perturbed = data.clone()
            indices = self.feature_indices

            if indices is None:
                # If no specific indices provided, select random 30% of features
                num_features = data.shape[1]
                num_to_perturb = max(1, int(0.3 * num_features))
                indices = np.random.choice(num_features, num_to_perturb, replace=False)

            # Apply structured noise to selected features
            perturbed[:, indices] += (
                torch.randn_like(perturbed[:, indices]) * self.intensity
            )

        elif self.perturbation_type == "dropout":
            # Randomly zero out features
            perturbed = data.clone()
            mask = torch.rand_like(data) > self.dropout_prob
            perturbed = perturbed * mask

        elif self.perturbation_type == "swap":
            # Randomly swap feature positions
            perturbed = data.clone()
            num_features = data.shape[1]
            # Determine how many features to swap based on intensity
            num_swaps = max(1, int(self.intensity * num_features))

            for _ in range(num_swaps):
                # Select two random features to swap
                i, j = np.random.choice(num_features, 2, replace=False)
                perturbed[:, [i, j]] = perturbed[:, [j, i]]

        elif self.perturbation_type == "outlier":
            # Inject outliers into random positions
            perturbed = data.clone()
            batch_size, num_features = data.shape

            # Determine how many outliers to inject
            outlier_count = max(1, int(self.dropout_prob * batch_size))

            # Select random samples and features for outlier injection
            sample_indices = np.random.choice(batch_size, outlier_count, replace=False)
            feature_indices = np.random.choice(
                num_features, outlier_count, replace=True
            )

            # Calculate outlier values (mean + several std devs)
            feature_means = data.mean(dim=0)
            feature_stds = data.std(dim=0)
            outlier_values = feature_means[feature_indices] + feature_stds[
                feature_indices
            ] * (5.0 * self.intensity)

            # Inject outliers
            for i, (sample_idx, feature_idx) in enumerate(
                zip(sample_indices, feature_indices)
            ):
                perturbed[sample_idx, feature_idx] = outlier_values[i]

        else:
            raise ValueError(f"Unknown perturbation type: {self.perturbation_type}")

        # Store perturbation info in context
        context["perturbation_type"] = self.perturbation_type
        context["perturbation_intensity"] = self.intensity
        context["perturbed_data"] = perturbed

        # Calculate perturbation magnitude
        if (
            self.perturbation_type != "swap"
        ):  # Swap doesn't change values, just positions
            perturbation_magnitude = torch.norm(perturbed - data) / torch.norm(data)
            context["perturbation_magnitude"] = perturbation_magnitude.item()

        return perturbed, context


class MeaningPreservationMetricComponent(PipelineComponent):
    """Component that measures how well meaning is preserved under perturbation."""

    def __init__(
        self, metrics: StandardizedMetrics, name: str = "MeaningPreservationMetric"
    ):
        """
        Initialize meaning preservation metric component.

        Args:
            metrics: Standardized metrics for evaluation
            name: Component name
        """
        self.metrics = metrics
        self._name = name

    @property
    def name(self) -> str:
        """Get component name."""
        return self._name

    def process(
        self, data: Any, context: Dict[str, Any] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Measure meaning preservation.

        Args:
            data: Model output
            context: Processing context

        Returns:
            output: Unchanged model output
            context: Updated context with metrics
        """
        if context is None:
            context = {}

        # Need original data and perturbed data for comparison
        original_data = context.get("original_data")
        perturbed_data = context.get("perturbed_data")

        if original_data is None or perturbed_data is None:
            raise ValueError("Original or perturbed data not found in context")

        # Measure input perturbation impact
        input_impact = self.metrics.evaluate(original_data, perturbed_data)
        context["input_perturbation_impact"] = input_impact

        # Measure output preservation (how well the model handles perturbation)
        output_preservation = self.metrics.evaluate(original_data, data)
        context["output_preservation"] = output_preservation

        # Calculate robustness score (ratio of preservation to perturbation)
        if (
            "perturbation_magnitude" in context
            and context["perturbation_magnitude"] > 0
        ):
            robustness_score = 1.0 - (
                output_preservation["drift"]["overall_drift"]
                / context["perturbation_magnitude"]
            )
            context["robustness_score"] = max(0.0, min(1.0, robustness_score))

        return data, context


def create_perturbation_pipeline(
    perturbation_types: List[str], model: MeaningVAE
) -> Dict[str, Pipeline]:
    """
    Create multiple pipelines for different perturbation types.

    Args:
        perturbation_types: List of perturbation types to test
        model: MeaningVAE model

    Returns:
        pipelines: Dictionary of perturbation pipelines
    """
    pipelines = {}
    metrics = StandardizedMetrics()

    for p_type in perturbation_types:
        # Create pipeline for this perturbation type
        pipeline = Pipeline(name=f"{p_type.capitalize()}PerturbationPipeline")

        # Add perturbation component
        pipeline.add(PerturbationComponent(perturbation_type=p_type, intensity=0.1))

        # Add model adapter
        pipeline.add(PipelineAdapter(model))

        # Add meaning preservation metric
        pipeline.add(MeaningPreservationMetricComponent(metrics))

        pipelines[p_type] = pipeline

    return pipelines


def create_robustness_pipeline(
    model: MeaningVAE,
    config: Optional[Config] = None,
    perturbation_types: Optional[List[str]] = None,
    intensities: Optional[List[float]] = None,
) -> Pipeline:
    """
    Create a comprehensive robustness testing pipeline.

    This pipeline applies multiple perturbation types in parallel branches and
    compares their impact on meaning preservation.

    Args:
        model: MeaningVAE model
        config: Configuration object (optional)
        perturbation_types: Types of perturbation to test
        intensities: Intensities to test for each perturbation

    Returns:
        pipeline: Configured pipeline
    """
    if config is None:
        config = Config()

    if perturbation_types is None:
        perturbation_types = ["gaussian", "structured", "dropout", "swap", "outlier"]

    if intensities is None:
        intensities = [0.05, 0.1, 0.2]

    pipeline = Pipeline(name="RobustnessPipeline")

    # Create a branch for each perturbation type and intensity
    branches = {}

    for p_type in perturbation_types:
        for intensity in intensities:
            branch_name = f"{p_type}_{intensity}"

            # Create a sub-pipeline for this perturbation
            branch_pipeline = Pipeline(name=f"{p_type}_{intensity}_Pipeline")

            # Add perturbation component
            branch_pipeline.add(
                PerturbationComponent(
                    perturbation_type=p_type,
                    intensity=intensity,
                    name=f"{p_type}_{intensity}",
                )
            )

            # Add model adapter
            branch_pipeline.add(PipelineAdapter(model))

            # Add to branches
            branches[branch_name] = branch_pipeline

    # Add branch component to main pipeline
    pipeline.add(BranchComponent(branches))

    # Add component to compare results across branches
    def compare_robustness(
        data: Dict[str, Any], context: Dict[str, Any] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if context is None:
            context = {}

        # Extract branch contexts
        branch_contexts = context.get("branch_contexts", {})

        # Analyze robustness across perturbations
        robustness_results = {}

        metrics = StandardizedMetrics()

        for branch_name, branch_context in branch_contexts.items():
            # Extract perturbation details
            p_type, intensity = branch_name.split("_")
            intensity = float(intensity)

            # Get original and perturbed data
            original = branch_context.get("original_data")
            perturbed = branch_context.get("perturbed_data")

            if original is None or perturbed is None:
                continue

            # Get model output
            branch_output = data.get(branch_name)

            # Measure preservation
            input_impact = metrics.evaluate(original, perturbed)
            output_preservation = metrics.evaluate(original, branch_output)

            # Calculate perturbation magnitude
            perturbation_magnitude = torch.norm(perturbed - original) / torch.norm(
                original
            )

            # Calculate robustness score
            robustness_score = 1.0 - (
                output_preservation["drift"]["overall_drift"]
                / perturbation_magnitude.item()
            )
            robustness_score = max(0.0, min(1.0, robustness_score))

            # Store results
            robustness_results[branch_name] = {
                "perturbation_type": p_type,
                "intensity": intensity,
                "input_impact": input_impact["drift"]["overall_drift"],
                "output_preservation": output_preservation["drift"]["overall_drift"],
                "robustness_score": robustness_score,
            }

        # Calculate average robustness by perturbation type
        avg_by_type = {}
        for p_type in perturbation_types:
            type_results = [
                r
                for name, r in robustness_results.items()
                if r["perturbation_type"] == p_type
            ]
            if type_results:
                avg_by_type[p_type] = sum(
                    r["robustness_score"] for r in type_results
                ) / len(type_results)

        # Find most and least robust perturbations
        if robustness_results:
            most_robust = max(
                robustness_results.items(), key=lambda x: x[1]["robustness_score"]
            )
            least_robust = min(
                robustness_results.items(), key=lambda x: x[1]["robustness_score"]
            )

            context["most_robust"] = most_robust[0]
            context["most_robust_score"] = most_robust[1]["robustness_score"]
            context["least_robust"] = least_robust[0]
            context["least_robust_score"] = least_robust[1]["robustness_score"]

        # Store overall results
        context["robustness_results"] = robustness_results
        context["average_by_type"] = avg_by_type
        context["overall_robustness"] = (
            sum(avg_by_type.values()) / len(avg_by_type) if avg_by_type else 0
        )

        return data, context

    # Add comparison component
    pipeline.add(CustomComponent(compare_robustness, name="RobustnessComparison"))

    return pipeline


def analyze_feature_robustness(
    model: MeaningVAE,
    dataset: AgentStateDataset,
    num_features: int = 10,
    feature_indices: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Analyze the robustness of individual features.

    Args:
        model: MeaningVAE model
        dataset: Dataset to analyze
        num_features: Number of features to analyze (if feature_indices not provided)
        feature_indices: Specific feature indices to analyze (optional)

    Returns:
        feature_robustness: Dictionary of feature robustness scores
    """
    # Get a batch of data
    batch = dataset.get_batch()

    # Determine which features to analyze
    if feature_indices is None:
        feature_indices = list(range(min(num_features, batch.shape[1])))

    # Create standardized metrics
    metrics = StandardizedMetrics()

    # Set up model in evaluation mode
    model.eval()

    # Create results dictionary
    feature_robustness = {}

    # Create baseline reconstruction
    baseline_output = model(batch)
    if isinstance(baseline_output, dict):
        baseline_recon = baseline_output.get("reconstruction", baseline_output)
    else:
        baseline_recon = baseline_output

    # For each feature, perturb it and measure impact
    for feature_idx in feature_indices:
        # Create perturbed data by modifying only this feature
        perturbed_batch = batch.clone()

        # Apply significant perturbation to just this feature
        feature_std = batch[:, feature_idx].std()
        perturbed_batch[:, feature_idx] += feature_std * 2.0

        # Process through model
        perturbed_output = model(perturbed_batch)
        if isinstance(perturbed_output, dict):
            perturbed_recon = perturbed_output.get("reconstruction", perturbed_output)
        else:
            perturbed_recon = perturbed_output

        # Measure semantic drift
        evaluation = metrics.evaluate(baseline_recon, perturbed_recon)

        # Calculate robustness (lower drift means higher robustness)
        feature_drift = evaluation["drift"]["overall_drift"]
        feature_robustness[f"feature_{feature_idx}"] = 1.0 - min(1.0, feature_drift)

    return feature_robustness


if __name__ == "__main__":
    # Example usage
    config = Config()

    # Create model
    model = MeaningVAE(
        input_dim=config.model.input_dim,
        latent_dim=config.model.latent_dim,
        compression_type=getattr(config.model, "compression_type", None),
        compression_level=getattr(config.model, "compression_level", None),
    )

    # Generate synthetic data
    dataset = AgentStateDataset(batch_size=config.training.batch_size)
    dataset.generate_synthetic_data(10)
    batch = dataset.get_batch()

    # Create robustness pipeline
    pipeline = create_robustness_pipeline(
        model=model,
        config=config,
        perturbation_types=["gaussian", "dropout", "swap"],
        intensities=[0.1, 0.2],
    )

    # Process data through pipeline
    result, context = pipeline.process(batch)

    # Print results
    print(f"Robustness Pipeline: {pipeline}")
    print(f"Overall robustness score: {context['overall_robustness']:.4f}")
    print(
        f"Most robust to: {context['most_robust']} (score: {context['most_robust_score']:.4f})"
    )
    print(
        f"Least robust to: {context['least_robust']} (score: {context['least_robust_score']:.4f})"
    )

    # Analyze feature robustness
    feature_robustness = analyze_feature_robustness(model, dataset, num_features=5)

    print("\nFeature Robustness:")
    for feature, score in sorted(
        feature_robustness.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {feature}: {score:.4f}")
