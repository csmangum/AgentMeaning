#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline architecture for meaning-preserving transformation system.

This module defines:
1. Base classes for pipeline components
2. A flexible pipeline that can chain components together
3. Various transformation stages (encoding, compression, etc.)
4. Factory methods for common pipeline configurations
"""

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from ..data import AgentState
from ..graph_model import VGAE, GraphDecoder, GraphEncoder
from ..knowledge_graph import AgentStateToGraph
from ..models import Decoder, Encoder, EntropyBottleneck, MeaningVAE, VectorQuantizer


class PipelineComponent(ABC):
    """Base class for all pipeline components."""

    @abstractmethod
    def process(
        self, data: Any, context: Dict[str, Any] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Process input data and return processed output and context.

        Args:
            data: Input data
            context: Processing context (metrics, intermediate results, etc.)

        Returns:
            output: Processed output
            context: Updated context with additional information
        """
        pass

    @property
    def name(self) -> str:
        """Get component name (defaults to class name)."""
        return self.__class__.__name__

    def __str__(self) -> str:
        """String representation of component."""
        return f"{self.name}"


class Pipeline:
    """
    Flexible pipeline that chains components together for processing.

    The pipeline maintains a sequence of components that process data in order,
    passing the output of each component to the next one along with a context dictionary
    that can be used to store intermediate results, metrics, and other information.
    """

    def __init__(self, name: str = "Pipeline"):
        """
        Initialize pipeline.

        Args:
            name: Name of the pipeline
        """
        self.name = name
        self.components = []

    def add(self, component: PipelineComponent) -> "Pipeline":
        """
        Add component to pipeline.

        Args:
            component: Component to add

        Returns:
            self: For method chaining
        """
        self.components.append(component)
        return self

    def add_many(self, components: List[PipelineComponent]) -> "Pipeline":
        """
        Add multiple components to pipeline.

        Args:
            components: List of components to add

        Returns:
            self: For method chaining
        """
        self.components.extend(components)
        return self

    def replace(self, index: int, component: PipelineComponent) -> "Pipeline":
        """
        Replace component at specified index.

        Args:
            index: Index of component to replace
            component: New component

        Returns:
            self: For method chaining
        """
        if 0 <= index < len(self.components):
            self.components[index] = component
        return self

    def insert(self, index: int, component: PipelineComponent) -> "Pipeline":
        """
        Insert component at specified index.

        Args:
            index: Index to insert component at
            component: Component to insert

        Returns:
            self: For method chaining
        """
        self.components.insert(index, component)
        return self

    def remove(self, index: int) -> "Pipeline":
        """
        Remove component at specified index.

        Args:
            index: Index of component to remove

        Returns:
            self: For method chaining
        """
        if 0 <= index < len(self.components):
            self.components.pop(index)
        return self

    def process(
        self, data: Any, context: Dict[str, Any] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Process data through pipeline.

        Args:
            data: Input data
            context: Initial context (optional)

        Returns:
            output: Final output
            context: Final context with information from all processing stages
        """
        if context is None:
            context = {}

        # Store original input in context
        context["input"] = data

        # Initialize result with input data
        result = data

        # Process through each component
        for i, component in enumerate(self.components):
            # Add component index to context
            context["current_component"] = i
            context["component_name"] = component.name

            # Process data
            result, context = component.process(result, context)

            # Store intermediate result in context
            context[f"result_{i}"] = result

        # Store final result in context
        context["output"] = result

        return result, context

    def __str__(self) -> str:
        """String representation of pipeline."""
        components_str = " -> ".join(str(c) for c in self.components)
        return f"{self.name}: {components_str}"


class EncoderComponent(PipelineComponent):
    """Component that encodes input using an encoder model."""

    def __init__(self, encoder: Union[Encoder, GraphEncoder], name: str = None):
        """
        Initialize encoder component.

        Args:
            encoder: Encoder model
            name: Component name (optional)
        """
        self.encoder = encoder
        self._name = name

    @property
    def name(self) -> str:
        """Get component name."""
        return self._name or f"Encoder[{self.encoder.__class__.__name__}]"

    def process(
        self, data: torch.Tensor, context: Dict[str, Any] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """
        Encode input data.

        Args:
            data: Input tensor
            context: Processing context

        Returns:
            output: Tuple of (mu, log_var)
            context: Updated context
        """
        if context is None:
            context = {}

        # Handle different encoder types
        if isinstance(self.encoder, Encoder):
            mu, log_var = self.encoder(data)
        elif isinstance(self.encoder, GraphEncoder):
            # For graph encoder
            mu, log_var = self.encoder(data)
        else:
            raise TypeError(f"Unsupported encoder type: {type(self.encoder)}")

        # Store in context
        context["mu"] = mu
        context["log_var"] = log_var

        return (mu, log_var), context


class ReparameterizationComponent(PipelineComponent):
    """Component that applies the reparameterization trick."""

    def __init__(self, name: str = "Reparameterize"):
        """
        Initialize reparameterization component.

        Args:
            name: Component name
        """
        self._name = name

    @property
    def name(self) -> str:
        """Get component name."""
        return self._name

    def process(
        self, data: Tuple[torch.Tensor, torch.Tensor], context: Dict[str, Any] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply reparameterization trick.

        Args:
            data: Tuple of (mu, log_var)
            context: Processing context

        Returns:
            output: Sampled latent vector
            context: Updated context
        """
        if context is None:
            context = {}

        mu, log_var = data

        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Compute KL loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / mu.size(0)  # Normalize by batch size

        # Store in context
        context["z"] = z
        context["kl_loss"] = kl_loss

        return z, context


class CompressionComponent(PipelineComponent):
    """Component that compresses latent representation."""

    def __init__(
        self, compression: Union[EntropyBottleneck, VectorQuantizer], name: str = None
    ):
        """
        Initialize compression component.

        Args:
            compression: Compression model
            name: Component name (optional)
        """
        self.compression = compression
        self._name = name

    @property
    def name(self) -> str:
        """Get component name."""
        return self._name or f"Compression[{self.compression.__class__.__name__}]"

    def process(
        self, data: torch.Tensor, context: Dict[str, Any] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compress latent representation.

        Args:
            data: Latent tensor
            context: Processing context

        Returns:
            output: Compressed latent tensor
            context: Updated context
        """
        if context is None:
            context = {}

        # Handle different compression types
        if isinstance(self.compression, EntropyBottleneck):
            z_compressed, compression_loss = self.compression(data)
            context["compression_loss"] = compression_loss
        elif isinstance(self.compression, VectorQuantizer):
            z_compressed, vq_loss, perplexity = self.compression(data)
            context["vq_loss"] = vq_loss
            context["perplexity"] = perplexity
        else:
            raise TypeError(f"Unsupported compression type: {type(self.compression)}")

        # Store in context
        context["z_compressed"] = z_compressed

        return z_compressed, context


class DecoderComponent(PipelineComponent):
    """Component that decodes latent representation."""

    def __init__(self, decoder: Union[Decoder, GraphDecoder], name: str = None):
        """
        Initialize decoder component.

        Args:
            decoder: Decoder model
            name: Component name (optional)
        """
        self.decoder = decoder
        self._name = name

    @property
    def name(self) -> str:
        """Get component name."""
        return self._name or f"Decoder[{self.decoder.__class__.__name__}]"

    def process(
        self, data: torch.Tensor, context: Dict[str, Any] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Decode latent representation.

        Args:
            data: Latent tensor
            context: Processing context

        Returns:
            output: Reconstructed output
            context: Updated context
        """
        if context is None:
            context = {}

        # Handle different decoder types
        if isinstance(self.decoder, Decoder):
            reconstruction = self.decoder(data)
            context["reconstruction"] = reconstruction
            return reconstruction, context
        elif isinstance(self.decoder, GraphDecoder):
            # For graph decoder
            node_features, edge_index, edge_attr = self.decoder(data)
            context["reconstruction"] = {
                "node_features": node_features,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
            }
            return (node_features, edge_index, edge_attr), context
        else:
            raise TypeError(f"Unsupported decoder type: {type(self.decoder)}")


class GraphConversionComponent(PipelineComponent):
    """Component that converts between tensor and graph representations."""

    def __init__(
        self,
        converter: AgentStateToGraph,
        direction: str = "to_graph",
        name: str = None,
    ):
        """
        Initialize graph conversion component.

        Args:
            converter: Graph conversion utility
            direction: Conversion direction ("to_graph" or "to_tensor")
            name: Component name (optional)
        """
        self.converter = converter
        self.direction = direction
        self._name = name

    @property
    def name(self) -> str:
        """Get component name."""
        return self._name or f"GraphConversion[{self.direction}]"

    def process(
        self, data: Any, context: Dict[str, Any] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Convert between tensor and graph representations.

        Args:
            data: Input data
            context: Processing context

        Returns:
            output: Converted output
            context: Updated context
        """
        if context is None:
            context = {}

        if self.direction == "to_graph":
            if isinstance(data, list) and all(
                isinstance(item, AgentState) for item in data
            ):
                # Convert multiple agent states to graph
                graph = self.converter.agents_to_graph(data)
                result = self.converter.to_torch_geometric(graph)
            elif isinstance(data, AgentState):
                # Convert single agent state to graph
                graph = self.converter.agent_to_graph(data)
                result = self.converter.to_torch_geometric(graph)
            else:
                raise TypeError(f"Cannot convert to graph: {type(data)}")

            context["graph"] = result
            return result, context

        elif self.direction == "to_tensor":
            if isinstance(data, (Data, Batch)):
                # Convert graph to tensor representation
                result = data.x
                context["tensor"] = result
                return result, context
            else:
                raise TypeError(f"Cannot convert to tensor: {type(data)}")

        else:
            raise ValueError(f"Unsupported direction: {self.direction}")


class SemanticEvaluationComponent(PipelineComponent):
    """Component that evaluates semantic preservation."""

    def __init__(self, evaluator: Callable, name: str = "SemanticEvaluation"):
        """
        Initialize semantic evaluation component.

        Args:
            evaluator: Function that evaluates semantic preservation
            name: Component name
        """
        self.evaluator = evaluator
        self._name = name

    @property
    def name(self) -> str:
        """Get component name."""
        return self._name

    def process(
        self, data: Any, context: Dict[str, Any] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Evaluate semantic preservation.

        Args:
            data: Input data
            context: Processing context

        Returns:
            output: Unchanged input data
            context: Updated context with evaluation results
        """
        if context is None:
            context = {}

        # Need both original input and current result
        original = context.get("input")

        if original is None:
            raise ValueError("Original input not found in context")

        # Evaluate semantic preservation
        evaluation_results = self.evaluator(original, data)

        # Store in context
        context["semantic_evaluation"] = evaluation_results

        # Pass through the data unchanged
        return data, context


class ConditionalComponent(PipelineComponent):
    """Component that conditionally processes data based on a predicate."""

    def __init__(
        self,
        predicate: Callable[[Any, Dict[str, Any]], bool],
        component: PipelineComponent,
        name: str = None,
    ):
        """
        Initialize conditional component.

        Args:
            predicate: Function that determines whether to process
            component: Component to conditionally apply
            name: Component name (optional)
        """
        self.predicate = predicate
        self.component = component
        self._name = name

    @property
    def name(self) -> str:
        """Get component name."""
        return self._name or f"If[{self.component.name}]"

    def process(
        self, data: Any, context: Dict[str, Any] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Conditionally process data.

        Args:
            data: Input data
            context: Processing context

        Returns:
            output: Processed or unchanged data
            context: Updated context
        """
        if context is None:
            context = {}

        # Evaluate predicate
        if self.predicate(data, context):
            # Apply component if predicate is true
            return self.component.process(data, context)
        else:
            # Otherwise pass through unchanged
            return data, context


class BranchComponent(PipelineComponent):
    """Component that branches processing into multiple parallel paths."""

    def __init__(self, branches: Dict[str, PipelineComponent], name: str = "Branch"):
        """
        Initialize branch component.

        Args:
            branches: Dictionary of branch name to component
            name: Component name
        """
        self.branches = branches
        self._name = name

    @property
    def name(self) -> str:
        """Get component name."""
        return self._name

    def process(
        self, data: Any, context: Dict[str, Any] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process data through multiple branches.

        Args:
            data: Input data
            context: Processing context

        Returns:
            output: Dictionary of branch results
            context: Updated context
        """
        if context is None:
            context = {}

        # Process each branch
        results = {}
        branch_contexts = {}

        for branch_name, component in self.branches.items():
            # Create a copy of the context for each branch
            branch_context = context.copy()

            # Process data through branch
            branch_result, branch_context = component.process(data, branch_context)

            # Store results
            results[branch_name] = branch_result
            branch_contexts[branch_name] = branch_context

        # Store branch results in context
        context["branch_results"] = results
        context["branch_contexts"] = branch_contexts

        return results, context


class PipelineAdapter(PipelineComponent):
    """Component that adapts a model to the pipeline interface."""

    def __init__(
        self, model: nn.Module, forward_kwargs: Dict[str, Any] = None, name: str = None
    ):
        """
        Initialize pipeline adapter.

        Args:
            model: Model to adapt
            forward_kwargs: Additional arguments to pass to forward method
            name: Component name (optional)
        """
        self.model = model
        self.forward_kwargs = forward_kwargs or {}
        self._name = name

    @property
    def name(self) -> str:
        """Get component name."""
        return self._name or f"Model[{self.model.__class__.__name__}]"

    def process(
        self, data: Any, context: Dict[str, Any] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Process data using the adapted model.

        Args:
            data: Input data
            context: Processing context

        Returns:
            output: Model output
            context: Updated context
        """
        if context is None:
            context = {}

        # Get model output
        with torch.set_grad_enabled(self.model.training):
            output = self.model(data, **self.forward_kwargs)

        # If output is a dictionary, merge with context
        if isinstance(output, dict):
            for key, value in output.items():
                context[key] = value

            # Return the 'reconstruction' as the primary output if it exists
            primary_output = output.get("reconstruction", output)
            return primary_output, context

        # Otherwise return output directly
        return output, context


class CustomComponent(PipelineComponent):
    """Component that applies a custom function."""

    def __init__(
        self,
        func: Callable[[Any, Dict[str, Any]], Tuple[Any, Dict[str, Any]]],
        name: str = None,
    ):
        """
        Initialize custom component.

        Args:
            func: Custom processing function
            name: Component name (optional)
        """
        self.func = func
        self._name = name

    @property
    def name(self) -> str:
        """Get component name."""
        return self._name or f"Custom[{self.func.__name__}]"

    def process(
        self, data: Any, context: Dict[str, Any] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Apply custom function.

        Args:
            data: Input data
            context: Processing context

        Returns:
            output: Processed output
            context: Updated context
        """
        if context is None:
            context = {}

        return self.func(data, context)


@dataclass
class PipelineFactory:
    """Factory for creating common pipeline configurations."""

    @staticmethod
    def create_vae_pipeline(vae: MeaningVAE) -> Pipeline:
        """
        Create a pipeline for a VAE model.

        Args:
            vae: VAE model

        Returns:
            pipeline: Configured pipeline
        """
        pipeline = Pipeline(name="VAEPipeline")

        # If using graph representation, add graph conversion
        if getattr(vae, "use_graph", False):
            graph_converter = AgentStateToGraph(
                relationship_threshold=0.5,
                include_relations=True,
                property_as_node=True,
            )

            # Add conditional graph conversion for agent states
            pipeline.add(
                ConditionalComponent(
                    predicate=lambda data, ctx: isinstance(data, (AgentState, list))
                    and not isinstance(data, (torch.Tensor, Data, Batch)),
                    component=GraphConversionComponent(
                        graph_converter, direction="to_graph"
                    ),
                )
            )

            # Add model adapter
            pipeline.add(PipelineAdapter(vae))

        else:
            # Standard VAE pipeline
            pipeline.add(EncoderComponent(vae.encoder))
            pipeline.add(ReparameterizationComponent())

            # Add compression if available
            if vae.compression is not None:
                pipeline.add(CompressionComponent(vae.compression))

            pipeline.add(DecoderComponent(vae.decoder))

        return pipeline

    @staticmethod
    def create_encoder_only_pipeline(vae: MeaningVAE) -> Pipeline:
        """
        Create a pipeline for encoding only.

        Args:
            vae: VAE model

        Returns:
            pipeline: Configured pipeline
        """
        pipeline = Pipeline(name="EncoderPipeline")

        # If using graph representation, add graph conversion
        if getattr(vae, "use_graph", False):
            graph_converter = AgentStateToGraph(
                relationship_threshold=0.5,
                include_relations=True,
                property_as_node=True,
            )

            # Add conditional graph conversion for agent states
            pipeline.add(
                ConditionalComponent(
                    predicate=lambda data, ctx: isinstance(data, (AgentState, list))
                    and not isinstance(data, (torch.Tensor, Data, Batch)),
                    component=GraphConversionComponent(
                        graph_converter, direction="to_graph"
                    ),
                )
            )

            # Use the encode method directly
            pipeline.add(
                CustomComponent(
                    func=lambda data, ctx: (vae.encode(data), ctx), name="Encode"
                )
            )

        else:
            # Standard encoder pipeline
            pipeline.add(EncoderComponent(vae.encoder))
            pipeline.add(ReparameterizationComponent())

            # Add compression if available
            if vae.compression is not None:
                pipeline.add(CompressionComponent(vae.compression))

        return pipeline

    @staticmethod
    def create_decoder_only_pipeline(vae: MeaningVAE) -> Pipeline:
        """
        Create a pipeline for decoding only.

        Args:
            vae: VAE model

        Returns:
            pipeline: Configured pipeline
        """
        pipeline = Pipeline(name="DecoderPipeline")

        # If using graph representation
        if getattr(vae, "use_graph", False):
            # Use the decode method directly
            pipeline.add(
                CustomComponent(
                    func=lambda data, ctx: (vae.decode(data, is_graph=True), ctx),
                    name="Decode",
                )
            )
        else:
            # Standard decoder pipeline
            pipeline.add(DecoderComponent(vae.decoder))

        return pipeline

    @staticmethod
    def create_custom_pipeline(
        components: List[PipelineComponent], name: str = "CustomPipeline"
    ) -> Pipeline:
        """
        Create a custom pipeline with specified components.

        Args:
            components: List of components
            name: Pipeline name

        Returns:
            pipeline: Configured pipeline
        """
        pipeline = Pipeline(name=name)
        pipeline.add_many(components)
        return pipeline


if __name__ == "__main__":
    # Example usage
    from meaning_transform.src.config import Config
    from meaning_transform.src.data import AgentStateDataset
    from meaning_transform.src.models import MeaningVAE

    # Create configuration
    config = Config()

    # Create model
    model = MeaningVAE(
        input_dim=config.model.input_dim,
        latent_dim=config.model.latent_dim,
        compression_type=config.model.compression_type,
        compression_level=config.model.compression_level,
    )

    # Create pipeline
    pipeline = PipelineFactory.create_vae_pipeline(model)

    # Generate synthetic data
    dataset = AgentStateDataset(batch_size=config.training.batch_size)
    dataset.generate_synthetic_data(10)

    # Get a batch
    batch = dataset.get_batch()

    # Process through pipeline
    result, context = pipeline.process(batch)

    print(f"Pipeline: {pipeline}")
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Context keys: {list(context.keys())}")
