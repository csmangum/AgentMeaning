#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for explainability.py module.

This script tests:
1. Knowledge graph visualization
2. Latent space visualization
3. Model explainability components
"""

import sys
import os
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data

# Add the project root to the path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from meaning_transform.src.explainability import (
    GraphVisualizer, 
    LatentSpaceVisualizer,
    ModelExplainer
)
from meaning_transform.src.knowledge_graph import AgentStateToGraph

# Test helper functions
def create_test_graph():
    """Create a test knowledge graph for visualization."""
    G = nx.Graph()
    
    # Add agent nodes
    G.add_node("agent/1", type="agent", label="Agent 1")
    G.add_node("agent/2", type="agent", label="Agent 2")
    
    # Add property nodes
    G.add_node("property/health_1", type="property", name="health", value="0.8")
    G.add_node("property/energy_1", type="property", name="energy", value="0.7")
    G.add_node("property/position_1", type="property", name="position", value="(10, 15)")
    
    G.add_node("property/health_2", type="property", name="health", value="0.5")
    G.add_node("property/energy_2", type="property", name="energy", value="0.9")
    G.add_node("property/position_2", type="property", name="position", value="(12, 18)")
    
    # Add inventory item nodes
    G.add_node("item/1", type="inventory_item", name="wood", quantity="10")
    G.add_node("item/2", type="inventory_item", name="stone", quantity="5")
    
    # Add goal nodes
    G.add_node("goal/1", type="goal", description="Collect resources")
    
    # Add edges
    G.add_edge("agent/1", "property/health_1", relation="has_health", weight=0.8)
    G.add_edge("agent/1", "property/energy_1", relation="has_energy", weight=0.7)
    G.add_edge("agent/1", "property/position_1", relation="has_position", weight=1.0)
    G.add_edge("agent/1", "item/1", relation="has_item", weight=0.9)
    G.add_edge("agent/1", "goal/1", relation="has_goal", weight=0.7)
    
    G.add_edge("agent/2", "property/health_2", relation="has_health", weight=0.5)
    G.add_edge("agent/2", "property/energy_2", relation="has_energy", weight=0.9)
    G.add_edge("agent/2", "property/position_2", relation="has_position", weight=1.0)
    G.add_edge("agent/2", "item/2", relation="has_item", weight=0.6)
    
    G.add_edge("agent/1", "agent/2", relation="proximity", weight=0.4)
    
    return G

def create_test_pyg_data():
    """Create a test PyTorch Geometric Data object for explainability testing."""
    # Create a simple graph with 5 nodes and 10 edges
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],  # Source nodes
        [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]   # Target nodes
    ], dtype=torch.long)
    
    # Node features (5 nodes, 8 features each)
    x = torch.randn(5, 8)
    
    # Edge features (10 edges, 4 features each)
    edge_attr = torch.randn(10, 4)
    
    # Create PyG Data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data

class MockModel(torch.nn.Module):
    """Mock model for testing explainability methods."""
    def __init__(self):
        super(MockModel, self).__init__()
        self.conv1 = torch.nn.Linear(8, 16)
        self.conv2 = torch.nn.Linear(16, 8)
        
    def forward(self, x, edge_index=None, edge_attr=None):
        # Simple forward pass that ignores graph structure for testing
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x

def test_graph_visualizer():
    """Test the graph visualizer functionality."""
    print("\n=== Testing Graph Visualizer ===")
    
    # Create temporary results directory
    log_dir = "test_results/explainability"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create test graph
    G = create_test_graph()
    
    # Initialize graph visualizer
    visualizer = GraphVisualizer(node_size=300, figsize=(10, 8))
    
    # Test basic visualization
    output_file = os.path.join(log_dir, "test_graph.png")
    fig = visualizer.plot_knowledge_graph(
        G, 
        title="Test Knowledge Graph", 
        show_labels=True,
        save_path=output_file
    )
    print(f"Basic graph visualization saved to: {output_file}")
    plt.close(fig)
    
    # Test visualization with node highlighting
    output_file = os.path.join(log_dir, "test_graph_highlighted.png")
    fig = visualizer.plot_knowledge_graph(
        G, 
        title="Test Knowledge Graph with Highlights", 
        highlight_nodes=["agent/1", "property/health_1"],
        highlight_edges=[("agent/1", "property/health_1")],
        save_path=output_file
    )
    print(f"Highlighted graph visualization saved to: {output_file}")
    plt.close(fig)
    
    # Test interactive visualization
    output_file = os.path.join(log_dir, "test_graph_interactive.html")
    fig = visualizer.create_interactive_graph(
        G, 
        title="Interactive Knowledge Graph",
        node_attributes=["type", "name", "value"],
        save_path=output_file
    )
    print(f"Interactive graph saved to: {output_file}")
    
    return visualizer

def test_latent_space_visualizer():
    """Test the latent space visualizer functionality."""
    print("\n=== Testing Latent Space Visualizer ===")
    
    # Create temporary results directory
    log_dir = "test_results/explainability"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create test embeddings
    num_samples = 100
    latent_dim = 32
    
    # Create two clusters in latent space
    cluster1 = torch.randn(num_samples // 2, latent_dim) + torch.tensor([2.0, 1.0] + [0.0] * (latent_dim - 2))
    cluster2 = torch.randn(num_samples // 2, latent_dim) + torch.tensor([-2.0, -1.0] + [0.0] * (latent_dim - 2))
    embeddings = torch.cat([cluster1, cluster2], dim=0)
    
    # Create labels
    labels = ["Cluster A"] * (num_samples // 2) + ["Cluster B"] * (num_samples // 2)
    
    # Initialize latent space visualizer
    visualizer = LatentSpaceVisualizer(figsize=(10, 8), cmap="viridis")
    
    # Test static visualization
    output_file = os.path.join(log_dir, "latent_space_tsne.png")
    fig = visualizer.visualize_latent_space(
        embeddings, 
        labels=labels,
        method="tsne",
        title="t-SNE Visualization of Latent Space",
        save_path=output_file
    )
    print(f"t-SNE visualization saved to: {output_file}")
    plt.close(fig)
    
    # Test PCA visualization
    output_file = os.path.join(log_dir, "latent_space_pca.png")
    fig = visualizer.visualize_latent_space(
        embeddings, 
        labels=labels,
        method="pca",
        title="PCA Visualization of Latent Space",
        save_path=output_file
    )
    print(f"PCA visualization saved to: {output_file}")
    plt.close(fig)
    
    # Test interactive visualization
    output_file = os.path.join(log_dir, "latent_space_interactive.html")
    fig = visualizer.visualize_latent_space(
        embeddings, 
        labels=labels,
        method="tsne",
        title="Interactive t-SNE Visualization",
        interactive=True,
        save_path=output_file
    )
    print(f"Interactive visualization saved to: {output_file}")
    
    # Test latent space interpolation
    start_vector = cluster1[0]
    end_vector = cluster2[0]
    output_file = os.path.join(log_dir, "latent_interpolation.png")
    fig = visualizer.interpolate_latent_space(
        start_vector,
        end_vector,
        steps=10,
        title="Latent Space Interpolation",
        save_dir=log_dir
    )
    print(f"Latent space interpolation visualization saved to: {log_dir}")
    plt.close(fig)
    
    return visualizer

def test_model_explainer():
    """Test the model explainer functionality."""
    print("\n=== Testing Model Explainer ===")
    
    # Create temporary results directory
    log_dir = "test_results/explainability"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create test PyG data
    data = create_test_pyg_data()
    
    # Create mock model
    model = MockModel()
    
    # Initialize model explainer
    explainer = ModelExplainer(model)
    
    # Test node importance explanation
    output_file = os.path.join(log_dir, "node_importance.png")
    attributions, fig = explainer.explain_node_importance(
        data,
        method="saliency",
        visualize=True,
        save_path=output_file
    )
    print(f"Node importance visualization saved to: {output_file}")
    print(f"Attribution shape: {attributions.shape}")
    plt.close(fig)
    
    # Test feature importance explanation
    output_file = os.path.join(log_dir, "feature_importance.png")
    feature_names = [f"Feature {i}" for i in range(data.x.size(1))]
    fig = explainer.explain_feature_importance(
        data,
        feature_names=feature_names,
        top_k=5,
        save_path=output_file
    )
    print(f"Feature importance visualization saved to: {output_file}")
    plt.close(fig)
    
    return explainer

def main():
    """Run all tests."""
    print("=== Running Tests for Explainability Module ===")
    
    # Create results directory
    os.makedirs("test_results/explainability", exist_ok=True)
    
    # Run tests
    test_graph_visualizer()
    test_latent_space_visualizer()
    test_model_explainer()
    
    print("\n=== All Explainability Tests Completed Successfully ===")

if __name__ == "__main__":
    main() 