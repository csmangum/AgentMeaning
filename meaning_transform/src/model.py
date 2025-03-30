#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VAE model architecture for meaning-preserving transformations.

This module defines:
1. Encoder architecture for embedding agent states into latent space
2. Decoder architecture for reconstructing agent states from latent space
3. Compression mechanisms (entropy bottleneck or vector quantization)
4. The full pipeline: agent state → binary → latent → compressed → reconstructed
5. Graph-based representation and transformation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List, Optional, Union
import numpy as np
from torch_geometric.data import Data, Batch

# Import graph-related components
from .knowledge_graph import AgentStateToGraph
from .graph_model import GraphEncoder, GraphDecoder, VGAE


class Encoder(nn.Module):
    """Encoder network that maps agent states to latent representations."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = None, use_batch_norm: bool = True):
        """
        Initialize encoder network.
        
        Args:
            input_dim: Dimension of input agent state
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.use_batch_norm = use_batch_norm
        
        # Build encoder architecture
        modules = []
        in_dim = input_dim
        
        for h_dim in self.hidden_dims:
            if self.use_batch_norm:
                modules.append(nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                ))
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.LeakyReLU()
                ))
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.mu = nn.Linear(self.hidden_dims[-1], latent_dim)
        self.log_var = nn.Linear(self.hidden_dims[-1], latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of agent state
            
        Returns:
            mu: Mean of latent representation
            log_var: Log variance of latent representation
        """
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        
        return mu, log_var


class Decoder(nn.Module):
    """Decoder network that maps latent representations back to agent states."""
    
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: List[int] = None, use_batch_norm: bool = True):
        """
        Initialize decoder network.
        
        Args:
            latent_dim: Dimension of latent space
            output_dim: Dimension of output agent state
            hidden_dims: List of hidden layer dimensions (in reverse order from encoder)
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [64, 128, 256]
        self.use_batch_norm = use_batch_norm
        
        # Build decoder architecture
        modules = []
        in_dim = latent_dim
        
        for h_dim in self.hidden_dims:
            if self.use_batch_norm:
                modules.append(nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                ))
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.LeakyReLU()
                ))
            in_dim = h_dim
        
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Linear(self.hidden_dims[-1], output_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            z: Latent representation
            
        Returns:
            x_reconstructed: Reconstructed agent state
        """
        x = self.decoder(z)
        x_reconstructed = self.final_layer(x)
        
        return x_reconstructed


class EntropyBottleneck(nn.Module):
    """Entropy bottleneck for compressing latent representations."""
    
    def __init__(self, latent_dim: int, compression_level: float = 1.0):
        """
        Initialize entropy bottleneck.
        
        Args:
            latent_dim: Dimension of latent space
            compression_level: Level of compression (higher = more compression)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.compression_level = compression_level
        
        # Learnable parameters for the bottleneck
        self.compress_mu = nn.Parameter(torch.zeros(latent_dim))
        self.compress_log_scale = nn.Parameter(torch.zeros(latent_dim))
        
        # Projection layers for adaptive compression
        self.proj_compress = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, latent_dim * 2)  # mu and log_scale
        )
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress latent representation using entropy bottleneck.
        
        Args:
            z: Latent representation [B, D]
            
        Returns:
            z_compressed: Compressed latent representation
            compression_loss: Loss term for compression entropy
        """
        # Project to get adaptive mu and log_scale
        projection = self.proj_compress(z)
        mu, log_scale = torch.chunk(projection, 2, dim=-1)
        
        # Apply base parameters with adaptive adjustments
        mu = mu + self.compress_mu
        log_scale = log_scale + self.compress_log_scale
        
        # Scale compression based on compression_level
        log_scale = log_scale - np.log(self.compression_level)
        
        # Add noise for quantization
        if self.training:
            # Reparameterization trick during training
            epsilon = torch.randn_like(mu)
            z_compressed = mu + torch.exp(log_scale) * epsilon
        else:
            # Deterministic rounding during inference
            z_compressed = torch.round(mu) / self.compression_level
        
        # Compute entropy loss (bits per dimension)
        compression_loss = 0.5 * log_scale.exp().pow(2) + 0.5 * torch.log(2 * torch.tensor(np.pi, device=z.device))
        compression_loss = compression_loss.mean()
        
        return z_compressed, compression_loss


class VectorQuantizer(nn.Module):
    """Vector quantization layer for discrete latent representation."""
    
    def __init__(self, latent_dim: int, num_embeddings: int, commitment_cost: float = 0.25):
        """
        Initialize vector quantizer.
        
        Args:
            latent_dim: Dimension of latent vectors
            num_embeddings: Number of embedding vectors
            commitment_cost: Weight for commitment loss
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Embedding table
        self.embedding = nn.Embedding(num_embeddings, latent_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """
        Quantize latent vectors.
        
        Args:
            z: Latent vectors [B, D]
            
        Returns:
            quantized: Quantized latent vectors
            vq_loss: Vector quantization loss
            perplexity: Perplexity of the codebook usage
        """
        # Reshape z -> (batch, latent_dim)
        input_shape = z.shape
        flat_z = z.reshape(-1, self.latent_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_z**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_z, self.embedding.weight.t()))
        
        # Get the indices of the closest embedding vectors
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Quantize the latent vectors
        quantized = self.embedding(encoding_indices).reshape(input_shape)
        
        # Compute the VQ loss
        q_latent_loss = F.mse_loss(quantized.detach(), z)
        commitment_loss = F.mse_loss(quantized, z.detach())
        vq_loss = q_latent_loss + self.commitment_cost * commitment_loss
        
        # Compute the perplexity (measure of codebook usage)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Straight-through estimator
        # Pass the gradient from quantized to input z
        quantized = z + (quantized - z).detach()
        
        return quantized, vq_loss, perplexity


class MeaningVAE(nn.Module):
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
        graph_num_layers: int = 3
    ):
        """
        Initialize MeaningVAE model.
        
        Args:
            input_dim: Dimension of input agent state
            latent_dim: Dimension of latent space
            compression_type: Type of compression ('entropy' or 'vq')
            compression_level: Level of compression
            vq_num_embeddings: Number of embeddings for vector quantization
            use_batch_norm: Whether to use batch normalization
            use_graph: Whether to use graph-based representation
            graph_hidden_dim: Hidden dimension for graph neural networks
            gnn_type: Type of graph neural network ('GCN', 'GAT', 'SAGE', 'GIN')
            graph_num_layers: Number of layers in graph neural networks
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.compression_type = compression_type
        self.compression_level = compression_level
        self.use_batch_norm = use_batch_norm
        self.use_graph = use_graph
        
        # Standard vector encoder/decoder for non-graph inputs
        self.encoder = Encoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=None,
            use_batch_norm=use_batch_norm
        )
        
        self.decoder = Decoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=None,
            use_batch_norm=use_batch_norm
        )
        
        # Add graph encoder/decoder if using graphs
        if self.use_graph:
            # Standard feature dimensions for graph nodes and edges
            node_dim = 15  # Based on AgentState features
            edge_dim = 5   # Based on relationship types
            
            # Graph conversion utility
            self.graph_converter = AgentStateToGraph(
                relationship_threshold=0.5,
                include_relations=True,
                property_as_node=True
            )
            
            # Graph-based encoder and decoder
            self.graph_encoder = GraphEncoder(
                in_channels=node_dim,
                hidden_channels=graph_hidden_dim,
                out_channels=latent_dim,
                num_layers=graph_num_layers,
                gnn_type=gnn_type,
                use_edge_attr=True,
                edge_dim=edge_dim
            )
            
            self.graph_decoder = GraphDecoder(
                embedding_dim=latent_dim,
                hidden_channels=graph_hidden_dim,
                feature_dim=node_dim,
                edge_dim=edge_dim
            )
            
            # Graph VAE for direct graph processing
            self.graph_vae = VGAE(
                in_channels=node_dim,
                hidden_channels=graph_hidden_dim,
                latent_dim=latent_dim,
                feature_dim=node_dim,
                edge_dim=edge_dim,
                num_layers=graph_num_layers,
                gnn_type=gnn_type
            )
        
        # Set up compression module based on compression_type
        if compression_type == "entropy":
            self.compression = EntropyBottleneck(
                latent_dim=latent_dim,
                compression_level=compression_level
            )
        elif compression_type == "vq":
            self.compression = VectorQuantizer(
                latent_dim=latent_dim,
                num_embeddings=vq_num_embeddings
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
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
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
                - compression_loss: Loss from compression (if applicable)
                - quantization_loss: Loss from vector quantization (if applicable)
                - perplexity: Perplexity of codebook usage (if applicable)
        """
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
            
            # Apply compression if enabled
            if self.compression is not None:
                if self.compression_type == "entropy":
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
        
        # Reparameterization trick
        z = self.reparameterize(mu, log_var)
        results["z"] = z
        
        # Apply compression if enabled
        if self.compression is not None:
            if self.compression_type == "entropy":
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
        # Handle graph data
        if self.use_graph and isinstance(x, (Data, Batch)):
            # Use graph encoder
            return self.graph_vae.encode_to_latent(x)
        
        # Standard encoding for tensor input
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        
        # Apply compression if enabled
        if self.compression is not None:
            if self.compression_type == "entropy":
                z, _ = self.compression(z)
            elif self.compression_type == "vq":
                z, _, _ = self.compression(z)
        
        return z
    
    def decode(self, z: torch.Tensor, is_graph: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Decode latent representation to agent state or graph components.
        
        Args:
            z: Latent representation
            is_graph: Whether to decode as graph components
            
        Returns:
            Decoded representation (tensor for standard, tuple for graph)
        """
        if is_graph and self.use_graph:
            # Use graph decoder
            return self.graph_decoder(z)
        
        # Standard decoding for non-graph
        return self.decoder(z)
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath: str) -> None:
        """Load model from file."""
        self.load_state_dict(torch.load(filepath))
    
    def get_compression_rate(self) -> float:
        """
        Calculate and return the effective compression rate.
        
        Returns:
            compression_rate: Effective bits per dimension
        """
        if self.compression_type == "entropy":
            # Estimate bits per dimension from the entropy bottleneck
            return torch.exp(self.compression.compress_log_scale).mean().item()
        elif self.compression_type == "vq":
            # Estimate bits per dimension for VQ as log2(codebook size) / dimension
            return np.log2(self.compression.num_embeddings) / self.latent_dim
        else:
            # No compression
            return 1.0

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