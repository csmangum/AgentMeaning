#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Default pipeline implementation for the meaning-preserving transformation system.

This module provides a comprehensive default pipeline that combines:
1. Data normalization
2. VAE model processing
3. Semantic evaluation
4. Feature-specific processing (optional)
5. Graph conversion (optional)

The default pipeline is designed to be flexible and can be customized
for specific use cases.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from meaning_transform.src.config import Config
from meaning_transform.src.data import AgentState, AgentStateDataset
from meaning_transform.src.knowledge_graph import AgentStateToGraph
from meaning_transform.src.models import MeaningVAE
from meaning_transform.src.pipelines.pipeline import (
    BranchComponent,
    ConditionalComponent,
    CustomComponent,
    GraphConversionComponent,
    Pipeline,
    PipelineAdapter,
    PipelineComponent,
    PipelineFactory,
    SemanticEvaluationComponent,
)
from meaning_transform.src.standardized_metrics import StandardizedMetrics


class NormalizationComponent(PipelineComponent):
    """Component that normalizes tensor data."""

    def __init__(
        self, mean: float = 0.0, std: float = 1.0, name: str = "Normalization"
    ):
        """
        Initialize normalization component.

        Args:
            mean: Normalization mean
            std: Normalization standard deviation
            name: Component name
        """
        self.mean = mean
        self.std = std
        self._name = name

    @property
    def name(self) -> str:
        """Get component name."""
        return self._name

    def process(
        self, data: torch.Tensor, context: Dict[str, Any] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Normalize input data.

        Args:
            data: Input tensor
            context: Processing context

        Returns:
            output: Normalized tensor
            context: Updated context
        """
        if context is None:
            context = {}

        # Only normalize tensor data
        if isinstance(data, torch.Tensor):
            # Store original statistics
            context["original_mean"] = data.mean().item()
            context["original_std"] = data.std().item()

            # Normalize data
            normalized = (data - self.mean) / self.std

            # Store normalization parameters
            context["normalization_mean"] = self.mean
            context["normalization_std"] = self.std

            return normalized, context

        # Pass through non-tensor data unchanged
        return data, context


def extract_spatial_features(
    data: torch.Tensor, context: Dict[str, Any] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Extract spatial features from data.

    Args:
        data: Input tensor
        context: Processing context

    Returns:
        output: Spatial features
        context: Updated context
    """
    if context is None:
        context = {}

    # Extract first 3 dimensions (assumed to be spatial)
    # This should be customized based on actual data format
    spatial_dim = min(3, data.shape[1])
    spatial = data[:, :spatial_dim]
    context["feature_type"] = "spatial"

    return spatial, context


def extract_resource_features(
    data: torch.Tensor, context: Dict[str, Any] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Extract resource features from data.

    Args:
        data: Input tensor
        context: Processing context

    Returns:
        output: Resource features
        context: Updated context
    """
    if context is None:
        context = {}

    # Extract dimensions 3-6 (assumed to be resources)
    # This should be customized based on actual data format
    if data.shape[1] > 3:
        resources_dim = min(4, data.shape[1] - 3)
        resources = data[:, 3 : 3 + resources_dim]
    else:
        resources = torch.zeros((data.shape[0], 1), device=data.device)

    context["feature_type"] = "resources"

    return resources, context


def combine_features(
    data: Dict[str, Any], context: Dict[str, Any] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Combine processed features.

    Args:
        data: Dictionary of processed features
        context: Processing context

    Returns:
        output: Combined features
        context: Updated context
    """
    if context is None:
        context = {}

    # Get branch results
    spatial = data.get("spatial", None)
    resources = data.get("resources", None)

    if spatial is None or resources is None:
        raise ValueError("Missing feature branches in data")

    # Combine features
    combined = torch.cat([spatial, resources], dim=1)

    return combined, context


def create_default_pipeline(
    model: MeaningVAE,
    config: Optional[Config] = None,
    use_normalization: bool = True,
    use_feature_specific: bool = False,
    use_graph_conversion: bool = False,
    use_semantic_evaluation: bool = True,
) -> Pipeline:
    """
    Create a default pipeline for the meaning-preserving transformation system.

    Args:
        model: MeaningVAE model
        config: Configuration object (optional)
        use_normalization: Whether to use normalization
        use_feature_specific: Whether to use feature-specific processing
        use_graph_conversion: Whether to use graph conversion
        use_semantic_evaluation: Whether to use semantic evaluation

    Returns:
        pipeline: Configured pipeline
    """
    if config is None:
        config = Config()

    pipeline = Pipeline(name="DefaultPipeline")

    # 1. Add normalization if requested
    if use_normalization:
        pipeline.add(NormalizationComponent(mean=0.0, std=1.0))

    # 2. Add feature-specific processing if requested
    if use_feature_specific:
        pipeline.add(
            ConditionalComponent(
                predicate=lambda data, ctx: isinstance(data, torch.Tensor),
                component=BranchComponent(
                    {
                        "spatial": CustomComponent(
                            extract_spatial_features, name="ExtractSpatial"
                        ),
                        "resources": CustomComponent(
                            extract_resource_features, name="ExtractResources"
                        ),
                    }
                ),
            )
        )
        pipeline.add(
            ConditionalComponent(
                predicate=lambda data, ctx: isinstance(data, dict)
                and "spatial" in data
                and "resources" in data,
                component=CustomComponent(combine_features, name="CombineFeatures"),
            )
        )

    # 3. Add graph conversion if requested
    if use_graph_conversion:
        graph_converter = AgentStateToGraph(
            relationship_threshold=getattr(config.model, "relationship_threshold", 0.5),
            include_relations=getattr(config.model, "include_relations", True),
            property_as_node=getattr(config.model, "property_as_node", True),
        )

        pipeline.add(
            ConditionalComponent(
                predicate=lambda data, ctx: isinstance(data, (AgentState, list))
                and (
                    isinstance(data, list)
                    and len(data) > 0
                    and isinstance(data[0], AgentState)
                ),
                component=GraphConversionComponent(
                    graph_converter, direction="to_graph"
                ),
            )
        )

    # 4. Add model adapter
    pipeline.add(PipelineAdapter(model))

    # 5. Add semantic evaluation if requested
    if use_semantic_evaluation:
        metrics = StandardizedMetrics()

        def evaluate_semantics(original: Any, reconstructed: Any) -> Dict[str, Any]:
            return metrics.evaluate(original, reconstructed)

        pipeline.add(SemanticEvaluationComponent(evaluate_semantics))

    return pipeline


def load_default_pipeline(
    model_path: str, config_path: Optional[str] = None
) -> Tuple[Pipeline, MeaningVAE]:
    """
    Load a model and create a default pipeline.

    Args:
        model_path: Path to saved model
        config_path: Path to configuration file (optional)

    Returns:
        pipeline: Configured pipeline
        model: Loaded model
    """
    # Load configuration if provided
    if config_path is not None:
        config = Config.load(config_path)
    else:
        config = Config()

    # Load model
    model = MeaningVAE(
        input_dim=config.model.input_dim,
        latent_dim=config.model.latent_dim,
        compression_type=getattr(config.model, "compression_type", None),
        compression_level=getattr(config.model, "compression_level", None),
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create default pipeline
    pipeline = create_default_pipeline(
        model=model,
        config=config,
        use_normalization=True,
        use_feature_specific=False,
        use_graph_conversion=getattr(config.model, "use_graph", False),
        use_semantic_evaluation=True,
    )

    return pipeline, model


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

    # Create default pipeline
    pipeline = create_default_pipeline(
        model=model,
        config=config,
        use_normalization=True,
        use_feature_specific=False,
        use_graph_conversion=False,
        use_semantic_evaluation=True,
    )

    # Print pipeline structure
    print(f"Default Pipeline: {pipeline}")

    # Generate synthetic data
    dataset = AgentStateDataset(batch_size=config.training.batch_size)
    dataset.generate_synthetic_data(10)
    batch = dataset.get_batch()

    # Process data through pipeline
    result, context = pipeline.process(batch)

    # Print results
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Context keys: {list(context.keys())}")

    if "semantic_evaluation" in context:
        print(
            f"Semantic drift: {context['semantic_evaluation']['drift']['overall_drift']:.4f}"
        )
