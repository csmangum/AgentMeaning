#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script demonstrating knowledge graph-based agent state compression.

This example shows:
1. Creating agent states
2. Converting to knowledge graphs
3. Training a VGAE model
4. Compressing and reconstructing agent states
5. Visualizing the results
6. Tracking semantic drift as a reconstruction metric
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
import torch.nn.functional as F

# Import from meaning_transform
from meaning_transform.src.data import AgentState
from meaning_transform.src.explainability import GraphVisualizer, LatentSpaceVisualizer
from meaning_transform.src.graph_model import GraphCompressionModel, GraphVAELoss
from meaning_transform.src.knowledge_graph import (
    AgentStateToGraph,
    KnowledgeGraphDataset,
)
# Import semantic metrics for drift tracking
from meaning_transform.src.metrics import SemanticMetrics, DriftTracker


def generate_sample_agents(n_agents=20):
    """Generate sample agent states for demonstration."""
    agents = []

    # Create sample roles
    roles = ["explorer", "gatherer", "defender", "builder", "leader"]

    # Create some sample agents
    for i in range(n_agents):
        agent_id = f"agent_{i}"
        role = roles[i % len(roles)]
        position = (
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(-1, 1),
        )
        health = np.random.uniform(0.5, 1.0)
        energy = np.random.uniform(0.3, 1.0)

        # Create inventory based on role
        inventory = {}
        if role == "gatherer":
            inventory = {
                "wood": np.random.randint(5, 20),
                "stone": np.random.randint(5, 15),
            }
        elif role == "explorer":
            inventory = {"map": 1, "food": np.random.randint(1, 5)}
        elif role == "defender":
            inventory = {"sword": 1, "shield": np.random.randint(0, 2)}
        elif role == "builder":
            inventory = {
                "wood": np.random.randint(10, 30),
                "stone": np.random.randint(10, 20),
                "tools": np.random.randint(1, 3),
            }
        elif role == "leader":
            inventory = {"plan": 1, "food": np.random.randint(3, 8)}

        # Create goals based on role
        goals = []
        if role == "gatherer":
            goals = ["gather resources", "return to base"]
        elif role == "explorer":
            goals = ["explore map", "find resources"]
        elif role == "defender":
            goals = ["patrol area", "protect agents"]
        elif role == "builder":
            goals = ["build structure", "gather materials"]
        elif role == "leader":
            goals = ["coordinate agents", "plan strategy"]

        # Create agent
        agent = AgentState(
            position=position,
            health=health,
            energy=energy,
            inventory=inventory,
            role=role,
            goals=goals,
            agent_id=agent_id,
            step_number=0,
            resource_level=np.random.uniform(0.2, 0.9),
            current_health=health * np.random.uniform(0.8, 1.0),
            is_defending=role == "defender",
            age=np.random.randint(10, 100),
            total_reward=np.random.uniform(10, 100),
        )

        agents.append(agent)

    return agents


def create_agent_graphs(agents):
    """Create graph representations of agent states."""
    # Create knowledge graph converter
    converter = AgentStateToGraph(
        relationship_threshold=0.6, include_relations=True, property_as_node=True
    )

    # Create dataset
    dataset = KnowledgeGraphDataset(converter)

    # Add agents to dataset
    dataset.add_agent_states(agents)

    # Convert to PyTorch Geometric format
    data_batch = dataset.to_torch_geometric_batch()

    return dataset, data_batch


def train_model(data_loader, model, num_epochs, device, output_dir):
    """Train the graph compression model."""
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create loss function with semantic drift component
    loss_fn = GraphVAELoss(
        node_weight=1.0, edge_weight=1.0, kl_weight=0.1, edge_attr_weight=0.5
    )
    
    # Set up drift tracking folder
    drift_dir = os.path.join(output_dir, "drift_tracking")
    os.makedirs(drift_dir, exist_ok=True)
    
    # Training loop
    train_losses = []
    semantic_drifts = []

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_semantic_drift = 0.0
        num_batches = 0

        for batch in data_loader:
            # Move batch to device
            batch = batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)

            # Calculate loss
            loss_dict = loss_fn(outputs, batch)
            loss = loss_dict["loss"]

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            
            # Calculate semantic drift with cosine similarity if we have node features
            if 'x_reconstructed' in outputs and batch.x is not None:
                # Get features
                original = batch.x
                reconstructed = outputs['x_reconstructed']
                
                # Calculate cosine similarity between flattened features
                try:
                    # Make sure dimensions match
                    min_nodes = min(original.size(0), reconstructed.size(0))
                    original_subset = original[:min_nodes].flatten().unsqueeze(0)
                    reconstructed_subset = reconstructed[:min_nodes].flatten().unsqueeze(0)
                    
                    # Compute cosine similarity (higher is better)
                    cos_sim = F.cosine_similarity(original_subset, reconstructed_subset).item()
                    
                    # Semantic drift is the inverse of similarity (lower is better)
                    current_drift = 1.0 - cos_sim
                    epoch_semantic_drift += current_drift
                except Exception as e:
                    print(f"Error calculating semantic drift: {e}")

        # Calculate average loss and drift for the epoch
        avg_loss = epoch_loss / num_batches
        avg_semantic_drift = epoch_semantic_drift / num_batches if num_batches > 0 else 0.0
        
        train_losses.append(avg_loss)
        semantic_drifts.append(avg_semantic_drift)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Semantic Drift: {avg_semantic_drift:.4f}")

    # Save model
    model_path = os.path.join(output_dir, "graph_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot training loss and semantic drift
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(semantic_drifts)
    plt.title("Semantic Drift")
    plt.xlabel("Epoch")
    plt.ylabel("Drift (1 - cosine similarity)")
    plt.grid(True)
    
    plt.tight_layout()
    loss_path = os.path.join(output_dir, "training_metrics.png")
    plt.savefig(loss_path)
    print(f"Training metrics plot saved to {loss_path}")
    
    # Save drift data to a CSV file
    try:
        import pandas as pd
        drift_data = {
            'epoch': list(range(1, num_epochs + 1)),
            'loss': train_losses,
            'semantic_drift': semantic_drifts
        }
        df = pd.DataFrame(drift_data)
        drift_csv_path = os.path.join(drift_dir, "drift_data.csv")
        df.to_csv(drift_csv_path, index=False)
        print(f"Drift data saved to {drift_csv_path}")
        
        # Create a simple drift report
        drift_report = f"""# Semantic Drift Analysis Report

## Training Overview
- Total Epochs: {num_epochs}
- Initial Semantic Drift: {semantic_drifts[0]:.4f}
- Final Semantic Drift: {semantic_drifts[-1]:.4f}
- Change in Semantic Drift: {semantic_drifts[-1] - semantic_drifts[0]:.4f}

## Key Observations
- {'Semantic drift decreased' if semantic_drifts[-1] < semantic_drifts[0] else 'Semantic drift increased'} over training
- Lowest drift: {min(semantic_drifts):.4f} (epoch {semantic_drifts.index(min(semantic_drifts)) + 1})
- Highest drift: {max(semantic_drifts):.4f} (epoch {semantic_drifts.index(max(semantic_drifts)) + 1})

## Interpretation
The semantic drift measure used here (1 - cosine similarity) represents how much meaning is lost during compression and reconstruction.
Lower drift values indicate better preservation of semantics across the transformation.
"""
        
        drift_report_path = os.path.join(output_dir, "drift_report.md")
        with open(drift_report_path, 'w') as f:
            f.write(drift_report)
        print(f"Drift report saved to {drift_report_path}")
    except Exception as e:
        print(f"Error saving drift data: {e}")

    return model


def compress_and_visualize(model, agents, device, output_dir):
    """Compress agent states and visualize the results."""
    # Create graphs
    dataset, data_batch = create_agent_graphs(agents)

    # Move to device
    data_batch = data_batch.to(device)

    # Get original graphs
    graphs = dataset.graphs

    # Create graph visualizer
    graph_viz = GraphVisualizer()

    # Visualize first agent graph
    first_graph = graphs[0]
    fig = graph_viz.plot_knowledge_graph(
        first_graph, title="Original Agent Graph", show_labels=True
    )
    original_path = os.path.join(output_dir, "original_graph.png")
    plt.savefig(original_path)
    print(f"Original graph visualization saved to {original_path}")

    # Compress and reconstruct
    model.eval()
    with torch.no_grad():
        # Compress
        latent = model.compress(data_batch)

        # Decompress first agent
        first_latent = latent[0].unsqueeze(0)
        reconstructed_data = model.decompress(first_latent)

    # Convert reconstructed data to NetworkX
    reconstructed_graph = nx.Graph()

    # Add nodes
    for i in range(reconstructed_data.x.size(0)):
        reconstructed_graph.add_node(i, features=reconstructed_data.x[i].cpu().numpy())

    # Add edges
    edge_index = reconstructed_data.edge_index.cpu().numpy()
    edge_attr = reconstructed_data.edge_attr.cpu().numpy()

    for j in range(edge_index.shape[1]):
        source, target = edge_index[0, j], edge_index[1, j]
        reconstructed_graph.add_edge(source, target, features=edge_attr[j])

    # Visualize reconstructed graph
    fig = graph_viz.plot_knowledge_graph(
        reconstructed_graph, title="Reconstructed Agent Graph", show_labels=False
    )
    reconstructed_path = os.path.join(output_dir, "reconstructed_graph.png")
    plt.savefig(reconstructed_path)
    print(f"Reconstructed graph visualization saved to {reconstructed_path}")

    # Calculate semantic drift for a subset of nodes if possible
    try:
        # Calculate simple metrics instead of using semantic metrics
        print("Calculating reconstruction quality metrics:")
        
        # Get original and reconstructed features
        original_features = data_batch.x
        reconstructed_features = reconstructed_data.x
        
        # Make sure number of nodes match or use just the common nodes
        min_nodes = min(original_features.size(0), reconstructed_features.size(0))
        original_subset = original_features[:min_nodes]
        reconstructed_subset = reconstructed_features[:min_nodes]
        
        # Calculate mean squared error
        mse = torch.nn.functional.mse_loss(original_subset, reconstructed_subset).item()
        print(f"Mean Squared Error: {mse:.4f}")
        
        # Calculate mean absolute error
        mae = torch.nn.functional.l1_loss(original_subset, reconstructed_subset).item()
        print(f"Mean Absolute Error: {mae:.4f}")
        
        # Calculate cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            original_subset.flatten().unsqueeze(0), 
            reconstructed_subset.flatten().unsqueeze(0)
        ).item()
        print(f"Cosine Similarity: {cos_sim:.4f}")
        
        # Calculate semantic drift (1 - cosine similarity)
        semantic_drift = 1.0 - cos_sim
        print(f"Estimated Semantic Drift: {semantic_drift:.4f}")
        
        # Save metrics to file
        metrics = {
            "mse": mse,
            "mae": mae,
            "cosine_similarity": cos_sim,
            "semantic_drift": semantic_drift,
            "num_nodes_original": original_features.size(0),
            "num_nodes_reconstructed": reconstructed_features.size(0),
            "num_nodes_compared": min_nodes,
            "feature_dim": original_features.size(1),
        }
        
        metrics_path = os.path.join(output_dir, "reconstruction_metrics.json")
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Reconstruction metrics saved to {metrics_path}")
    
    except Exception as e:
        print(f"Error calculating semantic metrics: {e}")

    # Create latent space visualizer
    latent_viz = LatentSpaceVisualizer()

    # Visualize latent space - handle potential batch/dimensionality issues
    latent_np = latent.cpu().numpy()

    # If we have more latent vectors than agents, reshape or slice appropriately
    # This ensures the labels and data array have matching dimensions
    if latent_np.shape[0] != len(agents):
        # Option 1: If the latent array has a batch dimension that doesn't match agents
        # Extract just the first latent per agent or reshape as needed
        if latent_np.shape[0] > len(agents):
            # Take only the latent vectors corresponding to number of agents
            latent_np = latent_np[: len(agents)]
        else:
            # In case we have fewer latent vectors than agents, replicate last vector
            # (This is just a fallback solution)
            padding = np.repeat(
                latent_np[-1:], len(agents) - latent_np.shape[0], axis=0
            )
            latent_np = np.vstack([latent_np, padding])

    # Get role labels for agents
    labels = [agent.role for agent in agents]

    # Now visualization should work as dimensions match
    fig = latent_viz.visualize_latent_space(
        latent_np, labels=labels, method="tsne", title="Latent Space Visualization"
    )
    latent_path = os.path.join(output_dir, "latent_space.png")
    plt.savefig(latent_path)
    print(f"Latent space visualization saved to {latent_path}")


def main(args):
    """Main function."""
    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    )
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate sample agents
    agents = generate_sample_agents(args.num_agents)
    print(f"Generated {len(agents)} sample agents")

    # Create graph dataset
    dataset, data_batch = create_agent_graphs(agents)
    print(f"Created graph dataset with {len(dataset.graphs)} graphs")

    # Get the node feature dimension and edge dimension
    node_feature_dim = data_batch.x.size(1)
    edge_feature_dim = (
        data_batch.edge_attr.size(1) if data_batch.edge_attr is not None else 1
    )

    # Create model
    model = GraphCompressionModel(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        gnn_type=args.gnn_type,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = model.to(device)
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

    # Create data loader
    data_loader = DataLoader([data_batch], batch_size=1)

    # Train model
    if args.train:
        model = train_model(data_loader, model, args.num_epochs, device, output_dir)

    # Compress and visualize
    compress_and_visualize(model, agents, device, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Agent Compression Example"
    )

    # Dataset parameters
    parser.add_argument(
        "--num_agents", type=int, default=20, help="Number of agents to generate"
    )

    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument(
        "--gnn_type",
        type=str,
        default="GCN",
        choices=["GCN", "GAT", "SAGE", "GIN"],
        help="GNN type",
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="Number of GNN layers"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability"
    )

    # Training parameters
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training")
    parser.add_argument(
        "--track_semantic_drift", 
        action="store_true", 
        help="Track semantic drift during training and evaluation"
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/graph_example",
        help="Output directory",
    )

    args = parser.parse_args()
    main(args)
