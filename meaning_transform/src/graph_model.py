#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Graph Neural Network and Variational Graph Autoencoder implementations.

This module handles:
1. GNN implementations for processing knowledge graphs
2. Graph-level encodings of agent states
3. Variational Graph Autoencoder for compression
4. Graph reconstruction from latent space
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    SAGEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from meaning_transform.src.metrics import SemanticMetrics

# Graph Neural Network model implementations


class GraphEncoder(nn.Module):
    """
    Graph Neural Network encoder for processing agent state graphs.

    This encoder processes graph structured data into node embeddings
    and can produce graph-level embeddings through pooling.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        gnn_type: str = "GCN",
        pool_type: str = "mean",
        use_edge_attr: bool = True,
        edge_dim: Optional[int] = None,
    ):
        """
        Initialize the graph encoder.

        Args:
            in_channels: Input feature dimensions
            hidden_channels: Hidden layer dimensions
            out_channels: Output embedding dimensions
            num_layers: Number of GNN layers
            dropout: Dropout probability
            gnn_type: Type of GNN layer ('GCN', 'GAT', 'SAGE', 'GIN')
            pool_type: Graph pooling method ('mean', 'max', 'add')
            use_edge_attr: Whether to use edge attributes in GNN
            edge_dim: Edge feature dimensions (required if use_edge_attr=True)
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.pool_type = pool_type
        self.use_edge_attr = use_edge_attr
        self.edge_dim = edge_dim

        # Create GNN layers
        self.convs = nn.ModuleList()

        # Input layer
        self.convs.append(self._create_conv_layer(in_channels, hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(self._create_conv_layer(hidden_channels, hidden_channels))

        # Output layer
        self.convs.append(self._create_conv_layer(hidden_channels, out_channels))

        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Output embedding normalization
        self.norm = nn.LayerNorm(out_channels)

    def _create_conv_layer(self, in_dim: int, out_dim: int) -> nn.Module:
        """
        Create a GNN layer of the specified type.

        Args:
            in_dim: Input dimension
            out_dim: Output dimension

        Returns:
            conv: GNN convolutional layer
        """
        if self.gnn_type == "GCN":
            return GCNConv(in_dim, out_dim, add_self_loops=True)

        elif self.gnn_type == "GAT":
            return GATConv(in_dim, out_dim, heads=4, concat=False, dropout=self.dropout)

        elif self.gnn_type == "SAGE":
            return SAGEConv(in_dim, out_dim)

        elif self.gnn_type == "GIN":
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )
            return GINConv(mlp)

        else:
            raise ValueError(f"Unknown GNN type: {self.gnn_type}")

    def forward(self, data: Union[Data, Batch]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the graph encoder.

        Args:
            data: PyTorch Geometric Data or Batch object

        Returns:
            node_embeddings: Node-level embeddings
            graph_embedding: Graph-level embedding
        """
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Initialize batch if None
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Process through GNN layers
        for i, conv in enumerate(self.convs):
            # For last layer, don't apply activation or batch norm
            if (
                self.use_edge_attr
                and hasattr(conv, "supports_edge_attr")
                and conv.supports_edge_attr
            ):
                # Use edge attributes if available and supported
                x = conv(x, edge_index, edge_attr)
            else:
                # Fallback to not using edge attributes
                x = conv(x, edge_index)

            if i < len(self.convs) - 1:
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply final normalization
        node_embeddings = self.norm(x)

        # Apply pooling to get graph-level embeddings
        if self.pool_type == "mean":
            graph_embedding = global_mean_pool(node_embeddings, batch)
        elif self.pool_type == "max":
            graph_embedding = global_max_pool(node_embeddings, batch)
        elif self.pool_type == "add":
            graph_embedding = global_add_pool(node_embeddings, batch)
        else:
            raise ValueError(f"Unknown pool type: {self.pool_type}")

        return node_embeddings, graph_embedding


class GraphDecoder(nn.Module):
    """
    Graph decoder for reconstructing graphs from embeddings.

    This decoder takes node embeddings and reconstructs
    the adjacency matrix and node features.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_channels: int,
        feature_dim: int,
        edge_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize the graph decoder.

        Args:
            embedding_dim: Dimension of input node embeddings
            hidden_channels: Hidden layer dimensions
            feature_dim: Original node feature dimensions to reconstruct
            edge_dim: Edge feature dimensions to reconstruct
            num_layers: Number of layers in MLPs
            dropout: Dropout probability
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.edge_dim = edge_dim

        # Node feature decoder
        node_decoder = []
        node_decoder.append(nn.Linear(embedding_dim, hidden_channels))
        node_decoder.append(nn.BatchNorm1d(hidden_channels))
        node_decoder.append(nn.ReLU())
        node_decoder.append(nn.Dropout(dropout))

        for _ in range(num_layers - 2):
            node_decoder.append(nn.Linear(hidden_channels, hidden_channels))
            node_decoder.append(nn.BatchNorm1d(hidden_channels))
            node_decoder.append(nn.ReLU())
            node_decoder.append(nn.Dropout(dropout))

        node_decoder.append(nn.Linear(hidden_channels, feature_dim))

        self.node_decoder = nn.Sequential(*node_decoder)

        # Edge predictor - takes pair of node embeddings and predicts existence and attributes
        self.edge_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1 + edge_dim),  # 1 for existence + edge features
        )

    def forward(
        self, node_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through graph decoder.

        Args:
            node_embeddings: Node embeddings from encoder

        Returns:
            node_features: Reconstructed node features
            edge_logits: Predicted edges and edge features
        """
        # Reconstruct node features
        node_features = self.node_decoder(node_embeddings)

        # Create all possible node pairs for edge prediction
        num_nodes = node_embeddings.size(0)
        node_i = node_embeddings.repeat_interleave(num_nodes, dim=0)
        node_j = node_embeddings.repeat(num_nodes, 1)

        # Concatenate node pairs
        edge_inputs = torch.cat([node_i, node_j], dim=1)

        # Predict edge existence and features
        edge_outputs = self.edge_predictor(edge_inputs)

        # Split into edge existence logits and edge features
        edge_logits = edge_outputs[:, 0].view(num_nodes, num_nodes)
        edge_features = edge_outputs[:, 1:].view(num_nodes, num_nodes, -1)

        return node_features, (edge_logits, edge_features)

    def reconstruct_graph(
        self, node_embeddings: torch.Tensor, threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstruct a graph from node embeddings.

        Args:
            node_embeddings: Node embeddings from encoder
            threshold: Threshold for edge prediction

        Returns:
            node_features: Reconstructed node features
            edge_index: Reconstructed edge indices
            edge_attr: Reconstructed edge attributes
        """
        # Reconstruct node features and edge predictions
        node_features, (edge_logits, edge_features) = self.forward(node_embeddings)

        # Apply threshold to create edge index
        edge_probs = torch.sigmoid(edge_logits)
        edge_mask = edge_probs > threshold

        # Create edge index from mask
        edge_index = torch.nonzero(edge_mask).t().contiguous()

        # Get edge attributes for selected edges
        if edge_index.size(1) > 0:
            edge_attr = torch.stack([edge_features[i, j] for i, j in edge_index.t()])
        else:
            edge_attr = torch.empty((0, self.edge_dim), device=node_features.device)

        return node_features, edge_index, edge_attr


class VGAE(nn.Module):
    """
    Variational Graph Autoencoder for agent state graphs.

    This model combines GNN-based encoding with variational
    sampling and graph reconstruction.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int,
        feature_dim: int,
        edge_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        gnn_type: str = "GCN",
        pool_type: str = "mean",
        use_edge_attr: bool = True,
    ):
        """
        Initialize VGAE model.

        Args:
            in_channels: Input node feature dimensions
            hidden_channels: Hidden layer dimensions
            latent_dim: Latent space dimensions
            feature_dim: Original node feature dimensions
            edge_dim: Edge feature dimensions
            num_layers: Number of GNN layers
            dropout: Dropout probability
            gnn_type: Type of GNN ('GCN', 'GAT', 'SAGE', 'GIN')
            pool_type: Graph pooling type ('mean', 'max', 'add')
            use_edge_attr: Whether to use edge attributes
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.edge_dim = edge_dim

        # Encoder for node embeddings
        self.encoder = GraphEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            pool_type=pool_type,
            use_edge_attr=use_edge_attr,
            edge_dim=edge_dim if use_edge_attr else None,
        )

        # Variational parameters
        self.mu = nn.Linear(hidden_channels, latent_dim)
        self.log_var = nn.Linear(hidden_channels, latent_dim)

        # Decoder for graph reconstruction
        self.node_proj = nn.Linear(latent_dim, hidden_channels)

        self.decoder = GraphDecoder(
            embedding_dim=hidden_channels,
            hidden_channels=hidden_channels,
            feature_dim=feature_dim,
            edge_dim=edge_dim,
            num_layers=num_layers - 1,
            dropout=dropout,
        )

    def encode(
        self, data: Union[Data, Batch]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode graph to latent space.

        Args:
            data: PyTorch Geometric Data or Batch

        Returns:
            mu: Mean of latent encoding
            log_var: Log variance of latent encoding
            node_embeddings: Intermediate node embeddings
        """
        # Get node embeddings from graph encoder
        node_embeddings, _ = self.encoder(data)

        # Project to latent parameters
        mu = self.mu(node_embeddings)
        log_var = self.log_var(node_embeddings)

        return mu, log_var, node_embeddings

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Apply reparameterization trick.

        Args:
            mu: Mean vectors
            log_var: Log variance vectors

        Returns:
            z: Sampled latent vectors
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode latent vectors to reconstructed graph.

        Args:
            z: Latent vectors

        Returns:
            node_features: Reconstructed node features
            edge_logits: Edge prediction logits
            edge_features: Edge feature predictions
        """
        # Project latent to hidden dimension
        h = self.node_proj(z)
        h = F.relu(h)

        # Decode to node features and edges
        node_features, (edge_logits, edge_features) = self.decoder(h)

        return node_features, edge_logits, edge_features

    def forward(self, data: Union[Data, Batch]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VGAE.

        Args:
            data: PyTorch Geometric Data or Batch

        Returns:
            outputs: Dictionary with model outputs
                - mu: Latent mean
                - log_var: Latent log variance
                - z: Sampled latent vectors
                - node_features: Reconstructed node features
                - edge_logits: Edge prediction logits
                - edge_features: Edge feature predictions
        """
        # Encode
        mu, log_var, node_embeddings = self.encode(data)

        # Sample latent
        z = self.reparameterize(mu, log_var)

        # Decode
        node_features, edge_logits, edge_features = self.decode(z)

        return {
            "mu": mu,
            "log_var": log_var,
            "z": z,
            "node_features": node_features,
            "edge_logits": edge_logits,
            "edge_features": edge_features,
            "node_embeddings": node_embeddings,
        }

    def reconstruct_graph(
        self, data: Union[Data, Batch], threshold: float = 0.5
    ) -> Data:
        """
        Reconstruct graph from input data.

        Args:
            data: Input graph data
            threshold: Edge prediction threshold

        Returns:
            reconstructed_data: Reconstructed graph data
        """
        # Forward pass through model
        outputs = self.forward(data)

        # Use decoder to reconstruct graph structure
        node_features, edge_index, edge_attr = self.decoder.reconstruct_graph(
            outputs["node_embeddings"], threshold=threshold
        )

        # Create new Data object
        reconstructed_data = Data(
            x=node_features, edge_index=edge_index, edge_attr=edge_attr
        )

        return reconstructed_data

    def encode_graph(self, data: Union[Data, Batch]) -> torch.Tensor:
        """
        Encode graph to single graph-level embedding.

        Args:
            data: Input graph data

        Returns:
            graph_embedding: Graph-level embedding vector
        """
        # Get node-level embeddings and graph embedding
        _, graph_embedding = self.encoder(data)

        return graph_embedding

    def encode_to_latent(self, data: Union[Data, Batch]) -> torch.Tensor:
        """
        Encode graph directly to latent space.

        Args:
            data: Input graph data

        Returns:
            z: Latent representation
        """
        mu, log_var, _ = self.encode(data)
        z = self.reparameterize(mu, log_var)
        return z


class GraphCompressionModel(nn.Module):
    """
    Complete graph-based agent state compression model.

    This model combines VGAE with additional components for
    meaningful compression and state reconstruction.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        gnn_type: str = "GCN",
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Initialize graph compression model.

        Args:
            node_feature_dim: Input node feature dimensions
            edge_feature_dim: Input edge feature dimensions
            hidden_dim: Hidden layer dimensions
            latent_dim: Latent space dimensions
            gnn_type: Type of GNN to use
            num_layers: Number of GNN layers
            dropout: Dropout probability
        """
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Variational Graph Autoencoder
        self.vgae = VGAE(
            in_channels=node_feature_dim,
            hidden_channels=hidden_dim,
            latent_dim=latent_dim,
            feature_dim=node_feature_dim,
            edge_dim=edge_feature_dim,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            use_edge_attr=True,
        )

        # Additional adaptive components
        # These can be used for domain-specific adaptations
        self.semantic_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def encode(self, data: Union[Data, Batch]) -> Dict[str, torch.Tensor]:
        """
        Encode graph data to latent space.

        Args:
            data: Input graph data

        Returns:
            encodings: Dictionary with encoding outputs
                - mu: Latent mean vectors
                - log_var: Latent log variance vectors
                - z: Sampled latent vectors
                - graph_z: Graph-level latent representation
        """
        # Get VGAE encoding
        mu, log_var, _ = self.vgae.encode(data)
        z = self.vgae.reparameterize(mu, log_var)

        # Get graph-level encoding
        graph_embedding = self.vgae.encode_graph(data)

        # Create semantic projection for downstream tasks
        semantic_features = self.semantic_proj(z)

        return {
            "mu": mu,
            "log_var": log_var,
            "z": z,
            "graph_embedding": graph_embedding,
            "semantic_features": semantic_features,
        }

    def decode(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode latent vectors to graph components.

        Args:
            z: Latent vectors

        Returns:
            node_features: Reconstructed node features
            edge_logits: Edge prediction logits
            edge_features: Edge feature predictions
        """
        return self.vgae.decode(z)

    def forward(self, data: Union[Data, Batch]) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through the model.

        Args:
            data: Input graph data

        Returns:
            outputs: Dictionary with all model outputs
        """
        # Get VGAE outputs
        vgae_outputs = self.vgae(data)

        # Get graph-level encoding
        graph_embedding = self.vgae.encode_graph(data)

        # Add graph embedding to outputs
        outputs = {**vgae_outputs, "graph_embedding": graph_embedding}

        # Add semantic features
        semantic_features = self.semantic_proj(outputs["z"])
        outputs["semantic_features"] = semantic_features

        return outputs

    def compress(self, data: Union[Data, Batch]) -> torch.Tensor:
        """
        Compress graph to latent representation.

        Args:
            data: Input graph data

        Returns:
            latent: Compressed latent representation
        """
        with torch.no_grad():
            # Get latent representation (deterministic in eval mode)
            self.eval()
            z = self.vgae.encode_to_latent(data)
            return z

    def decompress(self, z: torch.Tensor, threshold: float = 0.5) -> Data:
        """
        Decompress latent vector to graph data.

        Args:
            z: Latent vector
            threshold: Edge prediction threshold

        Returns:
            graph_data: Reconstructed graph data
        """
        with torch.no_grad():
            self.eval()

            # Project latent to node embeddings
            h = self.vgae.node_proj(z)
            h = F.relu(h)

            # Reconstruct graph
            node_features, edge_index, edge_attr = self.vgae.decoder.reconstruct_graph(
                h, threshold=threshold
            )

            # Create new Data object
            graph_data = Data(
                x=node_features, edge_index=edge_index, edge_attr=edge_attr
            )

            return graph_data


# Losses for graph models


class GraphVAELoss(nn.Module):
    """
    Loss function for Variational Graph Autoencoder.

    Combines reconstruction loss for nodes, edges, and edge attributes
    with KL divergence regularization and semantic preservation.
    """

    def __init__(
        self,
        node_weight: float = 1.0,
        edge_weight: float = 1.0,
        kl_weight: float = 0.1,
        edge_attr_weight: float = 0.5,
        semantic_weight: float = 0.5,
    ):
        """
        Initialize VAE loss function.

        Args:
            node_weight: Weight for node feature reconstruction loss
            edge_weight: Weight for edge reconstruction loss
            kl_weight: Weight for KL divergence regularization
            edge_attr_weight: Weight for edge attribute reconstruction loss
            semantic_weight: Weight for semantic preservation loss
        """
        super().__init__()
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.kl_weight = kl_weight
        self.edge_attr_weight = edge_attr_weight
        self.semantic_weight = semantic_weight
        
        # Initialize semantic loss if semantic weight is non-zero
        self.use_semantic_metrics = semantic_weight > 0
        if self.use_semantic_metrics:
            try:
                self.semantic_metrics = SemanticMetrics()
            except (ImportError, ModuleNotFoundError):
                print("Warning: SemanticMetrics not found, semantic loss disabled")
                self.use_semantic_metrics = False

    def forward(
        self, outputs: Dict[str, torch.Tensor], data: Union[Data, Batch]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for graph VAE.

        Args:
            outputs: Dict of model outputs including
                     x_reconstructed, edge_pred, edge_attr_pred, mu, log_var
            data: PyTorch Geometric Data or Batch object

        Returns:
            loss_dict: Dictionary of loss components and total loss
        """
        # Extract model outputs
        x_reconstructed = outputs.get("x_reconstructed")
        edge_pred = outputs.get("edge_pred")
        edge_attr_pred = outputs.get("edge_attr_pred")
        mu = outputs.get("mu")
        log_var = outputs.get("log_var")

        # Extract data
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        loss_dict = {}

        # Node feature reconstruction loss (MSE)
        if x_reconstructed is not None and x is not None:
            node_loss = F.mse_loss(x_reconstructed, x)
            loss_dict["node_loss"] = node_loss * self.node_weight
        else:
            loss_dict["node_loss"] = torch.tensor(0.0, device=mu.device)

        # Edge prediction loss (BCE)
        if edge_pred is not None and edge_index is not None:
            # Create target adjacency matrix
            num_nodes = x.size(0)
            adj_target = torch.zeros(
                (num_nodes, num_nodes), dtype=torch.float, device=edge_pred.device
            )
            adj_target[edge_index[0], edge_index[1]] = 1.0

            # Compute edge prediction loss
            edge_loss = F.binary_cross_entropy_with_logits(edge_pred, adj_target)
            loss_dict["edge_loss"] = edge_loss * self.edge_weight
        else:
            loss_dict["edge_loss"] = torch.tensor(0.0, device=mu.device)

        # Edge attribute reconstruction loss (MSE)
        if edge_attr_pred is not None and edge_attr is not None:
            # Create source and target node indices from edge_index
            src, dst = edge_index[0], edge_index[1]

            # Extract predicted edge attributes for actual edges
            pred_edge_attr = edge_attr_pred[src, dst]

            # Compute edge attribute loss
            edge_attr_loss = F.mse_loss(pred_edge_attr, edge_attr)
            loss_dict["edge_attr_loss"] = edge_attr_loss * self.edge_attr_weight
        else:
            loss_dict["edge_attr_loss"] = torch.tensor(0.0, device=mu.device)

        # KL divergence
        if mu is not None and log_var is not None:
            kl_loss = -0.5 * torch.mean(
                torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
            )
            loss_dict["kl_loss"] = kl_loss * self.kl_weight
        else:
            loss_dict["kl_loss"] = torch.tensor(0.0, device=x.device)
            
        # Semantic preservation loss
        if self.use_semantic_metrics and x_reconstructed is not None and x is not None:
            try:
                # Calculate semantic metrics
                with torch.no_grad():
                    metrics = self.semantic_metrics.evaluate(x, x_reconstructed)
                    
                # Convert to loss (1 - similarity score)
                semantic_loss = 1.0 - torch.tensor(metrics.get('overall', 0.0), device=x.device)
                loss_dict["semantic_loss"] = semantic_loss * self.semantic_weight
            except Exception as e:
                print(f"Error computing semantic loss: {e}")
                loss_dict["semantic_loss"] = torch.tensor(0.0, device=x.device)
        else:
            loss_dict["semantic_loss"] = torch.tensor(0.0, device=x.device)

        # Compute total loss
        total_loss = sum(loss_dict.values())
        loss_dict["loss"] = total_loss

        return loss_dict


class GraphSemanticLoss(nn.Module):
    """
    Semantic consistency loss for graph embeddings.

    Encourages embeddings to maintain semantic relationships.
    """

    def __init__(self, margin: float = 0.5, similarity_weight: float = 1.0):
        """
        Initialize semantic loss.

        Args:
            margin: Margin for triplet loss
            similarity_weight: Weight for similarity preservation
        """
        super().__init__()
        self.margin = margin
        self.similarity_weight = similarity_weight
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

    def forward(
        self, embeddings: torch.Tensor, similarity_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate semantic consistency loss.

        Args:
            embeddings: Graph embeddings [batch_size, embed_dim]
            similarity_matrix: Ground truth similarity matrix [batch_size, batch_size]

        Returns:
            loss: Semantic consistency loss
        """
        batch_size = embeddings.size(0)

        # Compute pairwise distances in embedding space
        embed_dist = torch.cdist(embeddings, embeddings, p=2)

        # Normalize to [0, 1]
        embed_dist = embed_dist / embed_dist.max()

        # Compute loss based on similarity preservation
        # Similar items should be closer in embedding space
        similarity_loss = torch.mean(
            (1 - similarity_matrix) * embed_dist - similarity_matrix * (1 - embed_dist)
        )

        # Generate triplets for triplet loss
        # We need anchor, positive, negative samples
        triplet_loss = torch.tensor(0.0, device=embeddings.device)
        triplet_count = 0

        for i in range(batch_size):
            # Get similarity scores for this item
            sim_scores = similarity_matrix[i]

            # Find positive and negative examples
            pos_indices = torch.where(sim_scores > 0.7)[0]
            neg_indices = torch.where(sim_scores < 0.3)[0]

            # Skip if not enough samples
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            # Sample positive and negative
            for pos_idx in pos_indices:
                for neg_idx in neg_indices:
                    anchor = embeddings[i].unsqueeze(0)
                    positive = embeddings[pos_idx].unsqueeze(0)
                    negative = embeddings[neg_idx].unsqueeze(0)

                    triplet_loss += self.triplet_loss(anchor, positive, negative)
                    triplet_count += 1

        # Average triplet loss
        if triplet_count > 0:
            triplet_loss = triplet_loss / triplet_count

        # Combined loss
        combined_loss = triplet_loss + self.similarity_weight * similarity_loss

        return combined_loss
