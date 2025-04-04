#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standard compression experiment pipeline for meaning-preserving transformation system.

This module provides a standard pipeline that implements the compression experiment
sequence:
1. Input preparation (converts agent states to tensors)
2. Encoding via encoder component
3. Reparameterization of latent space 
4. Compression with configurable compression levels
5. Decoding via decoder component
6. Semantic evaluation of reconstructions

This pipeline closely mirrors the process used in compression experiments.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from meaning_transform.src.config import Config
from meaning_transform.src.data import AgentState
from meaning_transform.src.knowledge_graph import AgentStateToGraph
from meaning_transform.src.models import MeaningVAE, AdaptiveMeaningVAE
from meaning_transform.src.pipelines.pipeline import (
    BranchComponent,
    CompressionComponent,
    ConditionalComponent,
    CustomComponent,
    DecoderComponent,
    EncoderComponent,
    GraphConversionComponent,
    Pipeline,
    PipelineAdapter,
    PipelineComponent,
    ReparameterizationComponent,
    SemanticEvaluationComponent,
)
from meaning_transform.src.standardized_metrics import StandardizedMetrics


class AgentStateToTensorComponent(PipelineComponent):
    """Component that converts agent states to tensors."""

    def __init__(self, name: str = "AgentStateToTensor", device=None):
        """
        Initialize agent state to tensor component.

        Args:
            name: Component name
            device: Device to place tensors on (default: None, uses CPU)
        """
        self._name = name
        self.device = device

    @property
    def name(self) -> str:
        """Get component name."""
        return self._name

    def process(
        self, data: Union[AgentState, List[AgentState]], context: Dict[str, Any] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Convert agent states to tensors.

        Args:
            data: Agent state or list of agent states
            context: Processing context

        Returns:
            output: Tensor representation
            context: Updated context
        """
        if context is None:
            context = {}

        # Store original states for reference
        if isinstance(data, AgentState):
            context["original_states"] = [data]
        elif isinstance(data, list) and all(isinstance(item, AgentState) for item in data):
            context["original_states"] = data
            
        # Handle single agent state
        if isinstance(data, AgentState):
            try:
                # Try to convert to tensor
                tensor = data.to_tensor().unsqueeze(0)  # Add batch dimension
                if self.device is not None:
                    tensor = tensor.to(self.device)
                return tensor, context
            except Exception as e:
                print(f"Error converting AgentState to tensor: {e}")
                # Return a default tensor
                import torch
                dim = getattr(data, "dimension", 64)  # Default to 64 if dimension unknown
                tensor = torch.zeros((1, dim), dtype=torch.float32)
                if self.device is not None:
                    tensor = tensor.to(self.device)
                return tensor, context

        # Handle list of agent states
        elif isinstance(data, list) and all(isinstance(item, AgentState) for item in data):
            try:
                # Convert each state to tensor
                tensors = []
                for state in data:
                    try:
                        tensor = state.to_tensor().unsqueeze(0)
                        tensors.append(tensor)
                    except Exception as e:
                        print(f"Error converting AgentState to tensor: {e}")
                        # Use zero tensor as fallback
                        dim = getattr(state, "dimension", 64)
                        tensors.append(torch.zeros((1, dim), dtype=torch.float32))
                
                # Concatenate tensors if available
                if tensors:
                    tensor = torch.cat(tensors, dim=0)
                    if self.device is not None:
                        tensor = tensor.to(self.device)
                    return tensor, context
                else:
                    # Return default tensor if no states could be converted
                    dim = getattr(data[0], "dimension", 64) if data else 64
                    tensor = torch.zeros((len(data), dim), dtype=torch.float32)
                    if self.device is not None:
                        tensor = tensor.to(self.device)
                    return tensor, context
            except Exception as e:
                print(f"Error processing agent states: {e}")
                # Return a default tensor
                dim = getattr(data[0], "dimension", 64) if data else 64
                tensor = torch.zeros((len(data), dim), dtype=torch.float32)
                if self.device is not None:
                    tensor = tensor.to(self.device)
                return tensor, context

        # Pass through non-agent state data unchanged
        else:
            return data, context


def evaluate_semantics(
    original: torch.Tensor, reconstructed: torch.Tensor
) -> Dict[str, Any]:
    """
    Evaluate semantic preservation metrics.
    
    Args:
        original: Original tensor
        reconstructed: Reconstructed tensor
        
    Returns:
        Dictionary of semantic evaluation metrics
    """
    # Import metrics
    from meaning_transform.src.standardized_metrics import StandardizedMetrics
    metrics = StandardizedMetrics()
    
    # Make sure inputs are tensors
    if not isinstance(original, torch.Tensor):
        original = torch.tensor(original)
    if not isinstance(reconstructed, torch.Tensor):
        reconstructed = torch.tensor(reconstructed)
    
    # Add batch dimension if needed
    if len(original.shape) == 1:
        original = original.unsqueeze(0)
    if len(reconstructed.shape) == 1:
        reconstructed = reconstructed.unsqueeze(0)
    
    # Make sure they're on the same device
    if original.device != reconstructed.device:
        reconstructed = reconstructed.to(original.device)
    
    return metrics.evaluate(original, reconstructed)


def create_compression_pipeline(
    model: Union[MeaningVAE, AdaptiveMeaningVAE],
    config: Optional[Config] = None,
    compression_level: float = 1.0,
    use_graph: bool = False,
    evaluate_semantics_fn: Optional[Callable] = None,
    device = None,
) -> Pipeline:
    """
    Create a standard compression experiment pipeline.

    Args:
        model: MeaningVAE or AdaptiveMeaningVAE model
        config: Configuration object (optional)
        compression_level: Compression level to use
        use_graph: Whether to use graph-based modeling
        evaluate_semantics_fn: Custom semantic evaluation function (optional)
        device: Device to use for tensors (default: None, will use model's device)
        
    Returns:
        pipeline: Configured compression pipeline
    """
    if config is None:
        config = Config()
        
    # Set compression level on model if it has the attribute
    if hasattr(model, "compression_level"):
        model.compression_level = compression_level
    
    # Determine the device to use
    if device is None:
        # Get the model's device
        device = next(model.parameters()).device
    
    # Create pipeline
    pipeline = Pipeline(name=f"CompressionPipeline_level{compression_level}")
    
    # Add agent state to tensor conversion component with device
    pipeline.add(
        ConditionalComponent(
            predicate=lambda data, ctx: isinstance(data, (AgentState, list)) and not 
                                      isinstance(data, (torch.Tensor)),
            component=AgentStateToTensorComponent(device=device)
        )
    )
    
    # If using graph representation, add graph conversion
    if use_graph:
        graph_converter = AgentStateToGraph(
            relationship_threshold=getattr(config.model, "relationship_threshold", 0.5),
            include_relations=getattr(config.model, "include_relations", True),
            property_as_node=getattr(config.model, "property_as_node", True),
        )
        
        pipeline.add(
            ConditionalComponent(
                predicate=lambda data, ctx: not getattr(ctx, "is_graph", False),
                component=GraphConversionComponent(
                    graph_converter, direction="to_graph"
                ),
            )
        )
        
        # Use the model as a whole for graph-based models
        pipeline.add(PipelineAdapter(model))
    else:
        # For standard vector-based models, add components individually
        
        # Encoding component
        pipeline.add(EncoderComponent(model.encoder))
        
        # Reparameterization component
        pipeline.add(ReparameterizationComponent())
        
        # Compression component if available
        if hasattr(model, "compression") and model.compression is not None:
            pipeline.add(CompressionComponent(model.compression))
        
        # Decoding component
        pipeline.add(DecoderComponent(model.decoder))
    
    # Add semantic evaluation if provided or use default
    eval_fn = evaluate_semantics_fn or evaluate_semantics
    pipeline.add(SemanticEvaluationComponent(eval_fn))
    
    return pipeline


def load_compression_pipeline(
    model_path: str, 
    compression_level: float = 1.0,
    config_path: Optional[str] = None,
    use_graph: bool = False,
    use_adaptive_model: bool = False,
) -> Tuple[Pipeline, Union[MeaningVAE, AdaptiveMeaningVAE]]:
    """
    Load a model and create a standard compression pipeline.

    Args:
        model_path: Path to saved model
        compression_level: Compression level to use
        config_path: Path to config (optional)
        use_graph: Whether the model uses graph-based modeling
        use_adaptive_model: Whether the model is an adaptive model
        
    Returns:
        Tuple of (pipeline, model)
    """
    # Load configuration if provided
    config = None
    if config_path:
        config = Config.load(config_path)
    else:
        config = Config()
    
    # Determine model class
    model_class = AdaptiveMeaningVAE if use_adaptive_model else MeaningVAE
    
    # Load model
    model = model_class.load(model_path)
    
    # Set compression level
    if hasattr(model, "compression_level"):
        model.compression_level = compression_level
    
    # Create pipeline
    pipeline = create_compression_pipeline(
        model=model,
        config=config,
        compression_level=compression_level,
        use_graph=use_graph
    )
    
    return pipeline, model


if __name__ == "__main__":
    # Example usage
    from meaning_transform.src.data import AgentStateDataset
    
    # Create configuration
    config = Config()
    
    # Create model
    model = MeaningVAE(
        input_dim=config.model.input_dim,
        latent_dim=config.model.latent_dim,
        compression_type=config.model.compression_type,
        compression_level=2.0,  # Set compression level
    )
    
    # Create pipeline
    pipeline = create_compression_pipeline(model, compression_level=2.0)
    
    # Generate synthetic data
    dataset = AgentStateDataset(batch_size=config.training.batch_size)
    dataset.generate_synthetic_data(10)
    states = dataset.states[:5]  # Take 5 states
    
    # Process through pipeline
    result, context = pipeline.process(states)
    
    print(f"Pipeline: {pipeline}")
    print(f"Number of input states: {len(states)}")
    print(f"Output shape: {result.shape}")
    print(f"Semantic drift: {context['semantic_evaluation']['drift']['overall_drift']:.4f}") 