#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example usage of the pipeline module for building custom processing flows.

This example demonstrates:
1. Creating pipelines with different components
2. Extending the pipeline with custom components
3. Combining different processing branches
4. Creating feature-specific processing
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from meaning_transform.src.config import Config
from meaning_transform.src.data import AgentState, AgentStateDataset
from meaning_transform.src.models import MeaningVAE
from meaning_transform.src.pipelines.pipeline import (
    BranchComponent,
    CustomComponent,
    Pipeline,
    PipelineAdapter,
    PipelineComponent,
    PipelineFactory,
    SemanticEvaluationComponent,
)
from meaning_transform.src.standardized_metrics import StandardizedMetrics


def main():
    """Main example function."""
    print("Pipeline Architecture Example")
    print("-" * 50)

    # Create configuration and model
    config = Config()
    model = MeaningVAE(
        input_dim=config.model.input_dim,
        latent_dim=config.model.latent_dim,
        compression_type=config.model.compression_type,
        compression_level=config.model.compression_level,
    )

    # Generate synthetic data
    dataset = AgentStateDataset(batch_size=config.training.batch_size)
    dataset.generate_synthetic_data(10)
    batch = dataset.get_batch()

    # Example 1: Basic VAE Pipeline
    basic_pipeline = PipelineFactory.create_vae_pipeline(model)
    print("\nExample 1: Basic VAE Pipeline")
    print(basic_pipeline)

    # Process data through basic pipeline
    result, context = basic_pipeline.process(batch)
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Context keys: {list(context.keys())}")

    # Example 2: Custom Component
    # Create a custom normalization component
    class NormalizationComponent(PipelineComponent):
        """Component that normalizes tensor data."""

        def __init__(self, mean: float = 0.0, std: float = 1.0):
            self.mean = mean
            self.std = std

        def process(
            self, data: torch.Tensor, context: Dict[str, Any] = None
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
            if context is None:
                context = {}

            # Store original statistics
            context["original_mean"] = data.mean().item()
            context["original_std"] = data.std().item()

            # Normalize data
            normalized = (data - self.mean) / self.std

            # Store normalization parameters
            context["normalization_mean"] = self.mean
            context["normalization_std"] = self.std

            return normalized, context

    # Create custom pipeline with normalization
    custom_pipeline = Pipeline(name="NormalizedVAEPipeline")
    custom_pipeline.add(NormalizationComponent(mean=0.5, std=2.0))
    custom_pipeline.add(PipelineAdapter(model))

    print("\nExample 2: Custom Pipeline with Normalization")
    print(custom_pipeline)

    # Process data through custom pipeline
    result, context = custom_pipeline.process(batch)
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {result.shape}")
    print(
        f"Original mean: {context['original_mean']:.4f}, std: {context['original_std']:.4f}"
    )
    print(
        f"Normalized with mean: {context['normalization_mean']}, std: {context['normalization_std']}"
    )

    # Example 3: Feature-Specific Processing
    # Create custom components for feature-specific processing

    # Function to extract spatial features (assuming first 3 dimensions are spatial)
    def extract_spatial_features(
        data: torch.Tensor, context: Dict[str, Any] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if context is None:
            context = {}

        # Extract spatial features (first 3 dimensions)
        spatial = data[:, :3]
        context["feature_type"] = "spatial"

        return spatial, context

    # Function to extract resource features (assuming dimensions 3-6 are resources)
    def extract_resource_features(
        data: torch.Tensor, context: Dict[str, Any] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if context is None:
            context = {}

        # Extract resource features (dimensions 3-6)
        resources = data[:, 3:7]
        context["feature_type"] = "resources"

        return resources, context

    # Function to combine processed features
    def combine_features(
        data: Dict[str, Any], context: Dict[str, Any] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if context is None:
            context = {}

        # Get branch results
        spatial = data["spatial"]
        resources = data["resources"]

        # Combine features
        combined = torch.cat([spatial, resources], dim=1)

        return combined, context

    # Create feature-specific pipeline
    feature_pipeline = Pipeline(name="FeatureSpecificPipeline")

    # Add branch component for parallel processing
    feature_pipeline.add(
        BranchComponent(
            {
                "spatial": CustomComponent(
                    extract_spatial_features, name="ExtractSpatial"
                ),
                "resources": CustomComponent(
                    extract_resource_features, name="ExtractResources"
                ),
            }
        )
    )

    # Add combiner component
    feature_pipeline.add(CustomComponent(combine_features, name="CombineFeatures"))

    print("\nExample 3: Feature-Specific Pipeline")
    print(feature_pipeline)

    # Process data through feature pipeline
    result, context = feature_pipeline.process(batch)
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Branch results: {[k for k in context['branch_results'].keys()]}")

    # Example 4: Semantic Evaluation Pipeline
    # Create a semantic evaluation component using standardized metrics
    semantic_metrics = StandardizedMetrics()

    # Create evaluation function
    def evaluate_semantics(
        original: torch.Tensor, reconstructed: torch.Tensor
    ) -> Dict[str, Any]:
        return semantic_metrics.evaluate(original, reconstructed)

    # Create pipeline with semantic evaluation
    evaluation_pipeline = Pipeline(name="SemanticEvaluationPipeline")
    evaluation_pipeline.add(PipelineAdapter(model))
    evaluation_pipeline.add(SemanticEvaluationComponent(evaluate_semantics))

    print("\nExample 4: Semantic Evaluation Pipeline")
    print(evaluation_pipeline)

    # Process data through evaluation pipeline
    result, context = evaluation_pipeline.process(batch)
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Semantic evaluation results: {list(context['semantic_evaluation'].keys())}")
    print(
        f"Overall drift: {context['semantic_evaluation']['drift']['overall_drift']:.4f}"
    )

    # Example 5: Composition of Pipelines
    # Create encoder and decoder pipelines
    encoder_pipeline = PipelineFactory.create_encoder_only_pipeline(model)
    decoder_pipeline = PipelineFactory.create_decoder_only_pipeline(model)

    # Process through encoder
    latent, encoder_context = encoder_pipeline.process(batch)

    # Process through decoder
    reconstruction, decoder_context = decoder_pipeline.process(latent)

    print("\nExample 5: Pipeline Composition")
    print(f"Encoder: {encoder_pipeline}")
    print(f"Decoder: {decoder_pipeline}")
    print(f"Input shape: {batch.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")

    print("\nAll examples completed successfully!")


if __name__ == "__main__":
    main()
