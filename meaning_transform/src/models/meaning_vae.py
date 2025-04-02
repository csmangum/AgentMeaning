from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from meaning_transform.src.graph_model import VGAE, GraphDecoder, GraphEncoder
from meaning_transform.src.knowledge_graph import AgentStateToGraph
from meaning_transform.src.models.adaptive_entropy_bottleneck import (
    AdaptiveEntropyBottleneck,
)
from meaning_transform.src.models.decoder import Decoder
from meaning_transform.src.models.encoder import Encoder
from meaning_transform.src.models.entropy_bottleneck import EntropyBottleneck
from meaning_transform.src.models.utils import BaseModelIO, set_temp_seed
from meaning_transform.src.models.vector_quantizer import VectorQuantizer


class MeaningVAE(nn.Module, BaseModelIO):
    """VAE model for meaning-preserving transformations."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        compression_type: str = "entropy",
        compression_level: float = 1.0,
        vq_num_embeddings: int = 512,
        use_batch_norm: bool = True,
        use_graph: bool = False,
        graph_hidden_dim: int = 128,
        gnn_type: str = "GCN",
        graph_num_layers: int = 3,
        seed: int = None,
    ):
        """
        Initialize MeaningVAE model.

        Args:
            input_dim: Dimension of input agent state
            latent_dim: Dimension of latent space
            compression_type: Type of compression ('entropy', 'adaptive_entropy', or 'vq')
            compression_level: Level of compression
            vq_num_embeddings: Number of embeddings for vector quantization
            use_batch_norm: Whether to use batch normalization
            use_graph: Whether to use graph-based representation
            graph_hidden_dim: Hidden dimension for graph neural networks
            gnn_type: Type of graph neural network ('GCN', 'GAT', 'SAGE', 'GIN')
            graph_num_layers: Number of layers in graph neural networks
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.compression_type = compression_type
        self.compression_level = compression_level
        self.use_batch_norm = use_batch_norm
        self.use_graph = use_graph
        self.seed = seed

        # Standard vector encoder/decoder for non-graph inputs
        self.encoder = Encoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=None,
            use_batch_norm=use_batch_norm,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=None,
            use_batch_norm=use_batch_norm,
        )

        # Add graph encoder/decoder if using graphs
        if self.use_graph:
            # Standard feature dimensions for graph nodes and edges
            node_dim = 15  # Based on AgentState features
            edge_dim = 5  # Based on relationship types

            # Graph conversion utility
            self.graph_converter = AgentStateToGraph(
                relationship_threshold=0.5,
                include_relations=True,
                property_as_node=True,
            )

            # Graph-based encoder and decoder
            self.graph_encoder = GraphEncoder(
                in_channels=node_dim,
                hidden_channels=graph_hidden_dim,
                out_channels=latent_dim,
                num_layers=graph_num_layers,
                gnn_type=gnn_type,
                use_edge_attr=True,
                edge_dim=edge_dim,
            )

            self.graph_decoder = GraphDecoder(
                embedding_dim=latent_dim,
                hidden_channels=graph_hidden_dim,
                feature_dim=node_dim,
                edge_dim=edge_dim,
            )

            # Graph VAE for direct graph processing
            self.graph_vae = VGAE(
                in_channels=node_dim,
                hidden_channels=graph_hidden_dim,
                latent_dim=latent_dim,
                feature_dim=node_dim,
                edge_dim=edge_dim,
                num_layers=graph_num_layers,
                gnn_type=gnn_type,
            )

        # Set up compression module based on compression_type
        if compression_type == "entropy":
            self.compression = EntropyBottleneck(
                latent_dim=latent_dim, compression_level=compression_level, seed=seed
            )
        elif compression_type == "adaptive_entropy":
            self.compression = AdaptiveEntropyBottleneck(
                latent_dim=latent_dim, compression_level=compression_level, seed=seed
            )
        elif compression_type == "vq":
            self.compression = VectorQuantizer(
                latent_dim=latent_dim, num_embeddings=vq_num_embeddings, seed=seed
            )
        else:
            self.compression = None

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from latent distribution.

        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution

        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)

        if self.training:
            with set_temp_seed(self.seed):
                eps = torch.randn_like(std)
            z = mu + eps * std
            return z
        else:
            # During evaluation, just use the mean for deterministic results
            return mu

    def forward(self, x: Union[torch.Tensor, Data, Batch]) -> Dict[str, Any]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of agent state or graph data

        Returns:
            results: Dictionary of results
                - mu: Mean of latent representation
                - log_var: Log variance of latent representation
                - z: Latent representation after reparameterization
                - z_compressed: Compressed latent representation
                - reconstruction: Reconstructed agent state
                - kl_loss: KL divergence loss
                - compression_loss: Loss from compression (if applicable)
                - quantization_loss: Loss from vector quantization (if applicable)
                - perplexity: Perplexity of codebook usage (if applicable)
        """
        # Validate input
        if not isinstance(x, (torch.Tensor, Data, Batch)):
            raise TypeError(f"Expected torch.Tensor, Data, or Batch, got {type(x)}")

        if isinstance(x, torch.Tensor) and (
            x.dim() != 2 or x.size(1) != self.input_dim
        ):
            raise ValueError(
                f"Expected tensor shape (batch_size, {self.input_dim}), got {x.shape}"
            )

        results = {}

        # Handle graph data
        if self.use_graph and isinstance(x, (Data, Batch)):
            # Use graph VAE directly
            graph_results = self.graph_vae(x)

            # Combine results with standard VAE format
            results["mu"] = graph_results["mu"]
            results["log_var"] = graph_results["log_var"]
            results["z"] = graph_results["z"]
            results["reconstruction"] = graph_results["node_features"]
            results["edge_pred"] = graph_results["edge_pred"]
            results["edge_attr_pred"] = graph_results["edge_attr_pred"]

            # Calculate KL loss
            kl_loss = -0.5 * torch.sum(
                1 + results["log_var"] - results["mu"].pow(2) - results["log_var"].exp()
            )
            results["kl_loss"] = kl_loss / results["mu"].size(
                0
            )  # Normalize by batch size

            # Apply compression if enabled
            if self.compression is not None:
                if self.compression_type in ["entropy", "adaptive_entropy"]:
                    z_compressed, compression_loss = self.compression(results["z"])
                    results["z_compressed"] = z_compressed
                    results["compression_loss"] = compression_loss
                elif self.compression_type == "vq":
                    z_quantized, vq_loss, perplexity = self.compression(results["z"])
                    results["z_compressed"] = z_quantized
                    results["quantization_loss"] = vq_loss
                    results["perplexity"] = perplexity

            return results

        # Standard VAE forward pass for tensor input
        mu, log_var = self.encoder(x)
        results["mu"] = mu
        results["log_var"] = log_var

        # Calculate KL loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        results["kl_loss"] = kl_loss / mu.size(0)  # Normalize by batch size

        # Reparameterization trick
        z = self.reparameterize(mu, log_var)
        results["z"] = z

        # Apply compression if enabled
        if self.compression is not None:
            if self.compression_type in ["entropy", "adaptive_entropy"]:
                z_compressed, compression_loss = self.compression(z)
                results["z_compressed"] = z_compressed
                results["compression_loss"] = compression_loss

                # Decode from compressed representation
                reconstruction = self.decoder(z_compressed)
            elif self.compression_type == "vq":
                z_quantized, vq_loss, perplexity = self.compression(z)
                results["z_compressed"] = z_quantized
                results["quantization_loss"] = vq_loss
                results["perplexity"] = perplexity

                # Decode from quantized representation
                reconstruction = self.decoder(z_quantized)
        else:
            # No compression, decode directly from z
            reconstruction = self.decoder(z)

        results["reconstruction"] = reconstruction
        return results

    def encode(self, x: Union[torch.Tensor, Data, Batch]) -> torch.Tensor:
        """
        Encode input to latent representation.

        Args:
            x: Input tensor of agent state or graph data

        Returns:
            z: Latent representation
        """
        # Validate input
        if not isinstance(x, (torch.Tensor, Data, Batch)):
            raise TypeError(f"Expected torch.Tensor, Data, or Batch, got {type(x)}")

        if isinstance(x, torch.Tensor) and (
            x.dim() != 2 or x.size(1) != self.input_dim
        ):
            raise ValueError(
                f"Expected tensor shape (batch_size, {self.input_dim}), got {x.shape}"
            )

        # Handle graph data
        if self.use_graph and isinstance(x, (Data, Batch)):
            graph_z = self.graph_vae.encode(x)
            if self.compression is not None:
                if self.compression_type in ["entropy", "adaptive_entropy"]:
                    compressed_z, _ = self.compression(graph_z)
                    return compressed_z
                elif self.compression_type == "vq":
                    quantized_z, _, _ = self.compression(graph_z)
                    return quantized_z
            return graph_z

        # Handle tensor data
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)

        # Apply compression if enabled
        if self.compression is not None:
            if self.compression_type in ["entropy", "adaptive_entropy"]:
                compressed_z, _ = self.compression(z)
                return compressed_z
            elif self.compression_type == "vq":
                quantized_z, _, _ = self.compression(z)
                return quantized_z

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output.

        Args:
            z: Latent representation

        Returns:
            reconstruction: Reconstructed output
        """
        # Validate input
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(z)}")

        if z.dim() != 2 or z.size(1) != self.latent_dim:
            raise ValueError(
                f"Expected shape (batch_size, {self.latent_dim}), got {z.shape}"
            )

        return self.decoder(z)

    def get_compression_rate(self) -> float:
        """Get the effective compression rate."""
        if self.compression is None:
            return 1.0

        return self.compression.get_compression_rate()

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration as a dictionary."""
        config = super().get_config()
        # Add MeaningVAE specific configuration
        config.update(
            {
                "vq_num_embeddings": getattr(self, "vq_num_embeddings", 512),
                "use_batch_norm": self.use_batch_norm,
                "use_graph": self.use_graph,
            }
        )
        return config

    def convert_agent_to_graph(self, agent_state):
        """
        Convert an agent state to graph representation.

        Args:
            agent_state: AgentState object

        Returns:
            graph_data: PyTorch Geometric Data object
        """
        if not self.use_graph:
            raise ValueError("Model not configured for graph representation")

        # Convert agent state to NetworkX graph
        nx_graph = self.graph_converter.agent_to_graph(agent_state)

        # Convert to PyTorch Geometric Data
        return self.graph_converter.to_torch_geometric(nx_graph)

    def convert_agents_to_graph(self, agent_states):
        """
        Convert multiple agent states to a single graph representation.

        Args:
            agent_states: List of AgentState objects

        Returns:
            graph_data: PyTorch Geometric Data object
        """
        if not self.use_graph:
            raise ValueError("Model not configured for graph representation")

        # Convert agent states to NetworkX graph
        nx_graph = self.graph_converter.agents_to_graph(agent_states)

        # Convert to PyTorch Geometric Data
        return self.graph_converter.to_torch_geometric(nx_graph)
