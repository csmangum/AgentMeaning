"""
Meaning-preserving transformation framework for agent state representation.

This package provides tools for compressing and representing agent states
while preserving their semantic meaning across various transformations.
"""

__version__ = "0.1.0"

from .src.data import AgentState
from .src.model import Encoder, Decoder, EntropyBottleneck
from .src.knowledge_graph import AgentStateToGraph, KnowledgeGraphDataset
from .src.graph_model import (
    GraphEncoder, GraphDecoder, VGAE, GraphCompressionModel, 
    GraphVAELoss, GraphSemanticLoss
)
from .src.explainability import GraphVisualizer, LatentSpaceVisualizer, ModelExplainer
from .src.interactive import AgentStateDashboard, LatentSpaceExplorer, run_dashboard 