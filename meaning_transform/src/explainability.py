#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Explainability module for agent state representations.

This module handles:
1. Visualizing knowledge graphs of agent states
2. Interpreting latent space representations
3. Generating explanations for model decisions
4. Feature importance and attribution methods
"""

import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
import captum.attr as attr
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx

from .knowledge_graph import AgentStateToGraph


class GraphVisualizer:
    """
    Visualizer for knowledge graphs of agent states.
    """
    
    def __init__(self, 
                 node_size: int = 300, 
                 figsize: Tuple[int, int] = (12, 10),
                 edge_width_scale: float = 2.0):
        """
        Initialize knowledge graph visualizer.
        
        Args:
            node_size: Base size for nodes in visualization
            figsize: Figure size (width, height) in inches
            edge_width_scale: Scaling factor for edge widths
        """
        self.node_size = node_size
        self.figsize = figsize
        self.edge_width_scale = edge_width_scale
        
        # Color maps for different node and edge types
        self.node_color_map = {
            'agent': '#1f77b4',  # blue
            'property': '#ff7f0e',  # orange
            'inventory_item': '#2ca02c',  # green
            'goal': '#d62728',  # red
            'unknown': '#7f7f7f'  # gray
        }
        
        self.edge_color_map = {
            'has_health': '#1f77b4',
            'has_energy': '#aec7e8',
            'has_position': '#ff7f0e',
            'has_role': '#ffbb78',
            'has_item': '#2ca02c',
            'has_goal': '#98df8a',
            'proximity': '#d62728',
            'inventory_similarity': '#ff9896',
            'cooperation': '#9467bd',
            'unknown': '#c5b0d5'
        }
    
    def plot_knowledge_graph(self, 
                            G: nx.Graph, 
                            title: str = "Agent Knowledge Graph",
                            show_labels: bool = True,
                            highlight_nodes: Optional[List[str]] = None,
                            highlight_edges: Optional[List[Tuple]] = None,
                            node_attributes: Optional[str] = None,
                            layout: str = 'spring',
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a knowledge graph visualization.
        
        Args:
            G: NetworkX graph to visualize
            title: Plot title
            show_labels: Whether to show node labels
            highlight_nodes: List of node IDs to highlight
            highlight_edges: List of edge tuples (source, target) to highlight
            node_attributes: Node attribute to use for sizing/coloring
            layout: Graph layout algorithm ('spring', 'kamada_kawai', 'circular')
            save_path: Path to save figure (if provided)
            
        Returns:
            fig: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, seed=42)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Collect node attributes
        node_colors = []
        node_sizes = []
        node_labels = {}
        
        for node in G.nodes():
            # Get node type and determine color
            node_type = G.nodes[node].get('type', 'unknown')
            node_color = self.node_color_map.get(node_type, self.node_color_map['unknown'])
            
            # Adjust color if node is highlighted
            if highlight_nodes and str(node) in highlight_nodes:
                # Make highlighted nodes brighter
                node_color = self._adjust_brightness(node_color, 1.3)
            
            node_colors.append(node_color)
            
            # Determine node size
            size = self.node_size
            if node_attributes and node_attributes in G.nodes[node]:
                # Scale size by attribute
                attr_value = G.nodes[node][node_attributes]
                if isinstance(attr_value, (int, float)):
                    size = self.node_size * (0.5 + attr_value)
            
            # Make agent nodes larger
            if node_type == 'agent':
                size *= 1.5
                
            # Make highlighted nodes larger
            if highlight_nodes and str(node) in highlight_nodes:
                size *= 1.5
                
            node_sizes.append(size)
            
            # Create labels
            if node_type == 'agent':
                label = G.nodes[node].get('label', str(node).split('/')[-1])
            elif node_type == 'property':
                prop_name = G.nodes[node].get('name', '')
                prop_value = G.nodes[node].get('value', '')
                label = f"{prop_name}: {prop_value}"
            elif node_type == 'inventory_item':
                item_name = G.nodes[node].get('name', '')
                quantity = G.nodes[node].get('quantity', '')
                label = f"{item_name} ({quantity})"
            elif node_type == 'goal':
                label = G.nodes[node].get('description', '')
            else:
                label = str(node).split('/')[-1]
            
            # Truncate long labels
            if len(label) > 20:
                label = label[:17] + "..."
                
            node_labels[node] = label
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
            ax=ax
        )
        
        # Collect edge attributes
        edge_colors = []
        edge_widths = []
        
        for u, v, data in G.edges(data=True):
            # Get edge type and determine color
            edge_type = data.get('relation', 'unknown')
            edge_color = self.edge_color_map.get(edge_type, self.edge_color_map['unknown'])
            
            # Adjust color if edge is highlighted
            if highlight_edges and (str(u), str(v)) in highlight_edges:
                edge_color = self._adjust_brightness(edge_color, 1.3)
                
            edge_colors.append(edge_color)
            
            # Determine edge width
            width = 1.0
            if 'weight' in data:
                width = 1.0 + (data['weight'] * self.edge_width_scale)
                
            # Make highlighted edges wider
            if highlight_edges and (str(u), str(v)) in highlight_edges:
                width *= 2
                
            edge_widths.append(width)
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.7,
            ax=ax
        )
        
        # Draw labels if requested
        if show_labels:
            nx.draw_networkx_labels(
                G, pos,
                labels=node_labels,
                font_size=8,
                font_weight='bold',
                ax=ax
            )
        
        # Set title and remove axis
        plt.title(title, fontsize=16)
        plt.axis('off')
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _adjust_brightness(self, color_hex: str, factor: float) -> str:
        """
        Adjust brightness of a hex color.
        
        Args:
            color_hex: Hex color string
            factor: Brightness factor (>1 for brighter, <1 for darker)
            
        Returns:
            adjusted_color: Adjusted hex color
        """
        # Convert hex to RGB
        color_rgb = mcolors.hex2color(color_hex)
        
        # Adjust brightness
        adjusted_rgb = tuple(min(1.0, c * factor) for c in color_rgb)
        
        # Convert back to hex
        return mcolors.rgb2hex(adjusted_rgb)
    
    def create_interactive_graph(self, 
                               G: nx.Graph, 
                               title: str = "Interactive Agent Knowledge Graph",
                               node_attributes: Optional[List[str]] = None,
                               save_path: Optional[str] = None) -> go.Figure:
        """
        Create an interactive plot of the knowledge graph using Plotly.
        
        Args:
            G: NetworkX graph to visualize
            title: Plot title
            node_attributes: List of node attributes to include in hover text
            save_path: Path to save HTML figure (if provided)
            
        Returns:
            fig: Plotly figure
        """
        # Create layout
        pos = nx.spring_layout(G, seed=42, dim=3)
        
        # Extract node positions
        x_nodes = [pos[node][0] for node in G.nodes()]
        y_nodes = [pos[node][1] for node in G.nodes()]
        z_nodes = [pos[node][2] for node in G.nodes()]
        
        # Create node traces
        node_traces = {}
        
        # Group nodes by type
        node_groups = {}
        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'unknown')
            if node_type not in node_groups:
                node_groups[node_type] = []
            node_groups[node_type].append(node)
        
        # Create trace for each node type
        for node_type, nodes in node_groups.items():
            # Extract positions for this node type
            x = [pos[node][0] for node in nodes]
            y = [pos[node][1] for node in nodes]
            z = [pos[node][2] for node in nodes]
            
            # Create hover text
            hover_text = []
            for node in nodes:
                text = f"ID: {str(node).split('/')[-1]}<br>Type: {node_type}"
                
                # Add additional attributes
                if node_attributes:
                    for attr in node_attributes:
                        if attr in G.nodes[node]:
                            text += f"<br>{attr}: {G.nodes[node][attr]}"
                
                hover_text.append(text)
            
            # Create node trace
            node_color = self.node_color_map.get(node_type, self.node_color_map['unknown'])
            
            # Determine size
            marker_size = 8
            if node_type == 'agent':
                marker_size = 12
            
            node_trace = go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=node_color,
                    opacity=0.8
                ),
                text=hover_text,
                hoverinfo='text',
                name=node_type
            )
            
            node_traces[node_type] = node_trace
        
        # Create edge traces
        edge_traces = {}
        
        # Group edges by relation type
        edge_groups = {}
        for u, v, data in G.edges(data=True):
            relation = data.get('relation', 'unknown')
            if relation not in edge_groups:
                edge_groups[relation] = []
            edge_groups[relation].append((u, v, data))
        
        # Create trace for each edge type
        for relation, edges in edge_groups.items():
            # Create lists to hold edge coordinates
            edge_x = []
            edge_y = []
            edge_z = []
            hover_text = []
            
            # Add coordinates for each edge
            for u, v, data in edges:
                # Add line coordinates
                x0, y0, z0 = pos[u]
                x1, y1, z1 = pos[v]
                
                # Add coordinates with None to create separation between edges
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])
                
                # Add hover text
                text = f"Relation: {relation}"
                if 'weight' in data:
                    text += f"<br>Weight: {data['weight']:.2f}"
                hover_text.append(text)
                hover_text.append(text)
                hover_text.append(None)
            
            # Create edge trace
            edge_color = self.edge_color_map.get(relation, self.edge_color_map['unknown'])
            
            edge_trace = go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(
                    color=edge_color,
                    width=2
                ),
                text=hover_text,
                hoverinfo='text',
                name=relation
            )
            
            edge_traces[relation] = edge_trace
        
        # Create figure
        fig = go.Figure(
            data=list(edge_traces.values()) + list(node_traces.values()),
            layout=go.Layout(
                title=title,
                showlegend=True,
                scene=dict(
                    xaxis=dict(showticklabels=False, title=''),
                    yaxis=dict(showticklabels=False, title=''),
                    zaxis=dict(showticklabels=False, title='')
                ),
                margin=dict(l=0, r=0, b=0, t=40)
            )
        )
        
        # Save figure if requested
        if save_path:
            fig.write_html(save_path)
        
        return fig


class LatentSpaceVisualizer:
    """
    Visualizer for latent space representations of agent states.
    """
    
    def __init__(self, 
                 figsize: Tuple[int, int] = (10, 8),
                 cmap: str = 'viridis'):
        """
        Initialize latent space visualizer.
        
        Args:
            figsize: Figure size (width, height) in inches
            cmap: Colormap for visualizations
        """
        self.figsize = figsize
        self.cmap = cmap
    
    def visualize_latent_space(self, 
                              embeddings: torch.Tensor, 
                              labels: Optional[List[Any]] = None,
                              method: str = 'tsne',
                              title: str = "Latent Space Visualization",
                              colormap: Optional[str] = None,
                              interactive: bool = False,
                              save_path: Optional[str] = None) -> Union[plt.Figure, go.Figure]:
        """
        Visualize latent space representations.
        
        Args:
            embeddings: Latent embeddings [n_samples, n_dimensions]
            labels: Optional labels for coloring points
            method: Dimensionality reduction method ('tsne', 'pca', 'mds')
            title: Plot title
            colormap: Optional colormap override
            interactive: Whether to create an interactive plot
            save_path: Path to save figure (if provided)
            
        Returns:
            fig: Matplotlib or Plotly figure
        """
        # Convert to numpy if tensor
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        # Apply dimensionality reduction
        if embeddings.shape[1] > 2:
            if method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, embeddings.shape[0] - 1))
            elif method == 'pca':
                reducer = PCA(n_components=2, random_state=42)
            elif method == 'mds':
                reducer = MDS(n_components=2, random_state=42)
            else:
                raise ValueError(f"Unknown reduction method: {method}")
                
            reduced_data = reducer.fit_transform(embeddings)
        else:
            reduced_data = embeddings
            
        # Create visualization
        if interactive:
            return self._create_interactive_latent_viz(
                reduced_data, labels, title, colormap or self.cmap, save_path
            )
        else:
            return self._create_static_latent_viz(
                reduced_data, labels, title, colormap or self.cmap, save_path
            )
    
    def _create_static_latent_viz(self, 
                                 data: np.ndarray, 
                                 labels: Optional[List[Any]],
                                 title: str,
                                 cmap: str,
                                 save_path: Optional[str]) -> plt.Figure:
        """
        Create static Matplotlib visualization.
        
        Args:
            data: Reduced data [n_samples, 2]
            labels: Optional labels for coloring points
            title: Plot title
            cmap: Colormap
            save_path: Path to save figure
            
        Returns:
            fig: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot with or without labels
        if labels is not None:
            unique_labels = sorted(set(labels))
            cmap_obj = plt.get_cmap(cmap, len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                ax.scatter(
                    data[mask, 0], data[mask, 1],
                    c=[cmap_obj(i)],
                    label=label,
                    alpha=0.8,
                    s=60
                )
            
            plt.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter(
                data[:, 0], data[:, 1],
                c=np.arange(len(data)),
                cmap=cmap,
                alpha=0.8,
                s=60
            )
        
        plt.title(title, fontsize=14)
        plt.xlabel(f"Dimension 1")
        plt.ylabel(f"Dimension 2")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _create_interactive_latent_viz(self, 
                                      data: np.ndarray, 
                                      labels: Optional[List[Any]],
                                      title: str,
                                      cmap: str,
                                      save_path: Optional[str]) -> go.Figure:
        """
        Create interactive Plotly visualization.
        
        Args:
            data: Reduced data [n_samples, 2]
            labels: Optional labels for coloring points
            title: Plot title
            cmap: Colormap
            save_path: Path to save figure
            
        Returns:
            fig: Plotly figure
        """
        # Create DataFrame for Plotly Express
        import pandas as pd
        df = pd.DataFrame({
            'x': data[:, 0],
            'y': data[:, 1],
            'label': labels if labels is not None else ['Point'] * len(data),
            'index': range(len(data))
        })
        
        # Create plot
        fig = px.scatter(
            df, x='x', y='y', color='label',
            title=title,
            color_discrete_sequence=px.colors.qualitative.Plotly,
            hover_data=['index', 'label']
        )
        
        fig.update_layout(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            legend_title="Labels",
            font=dict(size=12)
        )
        
        # Save figure if requested
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def interpolate_latent_space(self, 
                                start_vector: torch.Tensor, 
                                end_vector: torch.Tensor,
                                steps: int = 10,
                                decode_fn: Optional[callable] = None,
                                title: str = "Latent Space Interpolation",
                                save_dir: Optional[str] = None) -> plt.Figure:
        """
        Visualize interpolation in latent space.
        
        Args:
            start_vector: Starting point in latent space
            end_vector: Ending point in latent space
            steps: Number of interpolation steps
            decode_fn: Function to decode latent vectors (if provided)
            title: Plot title
            save_dir: Directory to save visualization
            
        Returns:
            fig: Matplotlib figure with interpolation
        """
        # Ensure vectors are tensors
        if not isinstance(start_vector, torch.Tensor):
            start_vector = torch.tensor(start_vector)
        if not isinstance(end_vector, torch.Tensor):
            end_vector = torch.tensor(end_vector)
            
        # Generate interpolated points
        alphas = np.linspace(0, 1, steps)
        vectors = []
        
        for alpha in alphas:
            interp_vec = (1 - alpha) * start_vector + alpha * end_vector
            vectors.append(interp_vec)
            
        # Convert to tensor for batch processing
        interpolated = torch.stack(vectors)
        
        # If we have a decoder function, decode the vectors
        decoded = None
        if decode_fn is not None:
            with torch.no_grad():
                decoded = decode_fn(interpolated)
        
        # Create visualization
        if decoded is not None:
            # Visualization depends on the decoded output type
            # For graphs, visualize network structure changes
            if isinstance(decoded[0], (nx.Graph, Data)):
                return self._visualize_graph_interpolation(
                    decoded, alphas, title, save_dir
                )
            else:
                # Default to latent space point visualization
                fig, ax = plt.subplots(figsize=self.figsize)
                
                # Reduce all vectors to 2D for visualization
                if interpolated.shape[1] > 2:
                    reducer = TSNE(n_components=2, random_state=42)
                    reduced = reducer.fit_transform(interpolated.detach().cpu().numpy())
                else:
                    reduced = interpolated.detach().cpu().numpy()
                
                # Plot interpolation path
                ax.plot(reduced[:, 0], reduced[:, 1], 'o-', alpha=0.7)
                
                # Highlight start and end points
                ax.scatter(reduced[0, 0], reduced[0, 1], c='green', s=100, label='Start')
                ax.scatter(reduced[-1, 0], reduced[-1, 1], c='red', s=100, label='End')
                
                # Add labels for intermediate points
                for i, (x, y) in enumerate(reduced):
                    ax.annotate(f"{alphas[i]:.1f}", (x, y), 
                                textcoords="offset points", 
                                xytext=(0, 10), 
                                ha='center')
                
                plt.title(title, fontsize=14)
                plt.legend()
                
                if save_dir:
                    save_path = Path(save_dir) / f"{title.replace(' ', '_')}.png"
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
                return fig
        else:
            # Just visualize points in latent space
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Reduce all vectors to 2D for visualization
            if interpolated.shape[1] > 2:
                # Use PCA for interpretability of the interpolation
                reducer = PCA(n_components=2, random_state=42)
                reduced = reducer.fit_transform(interpolated.detach().cpu().numpy())
            else:
                reduced = interpolated.detach().cpu().numpy()
            
            # Plot interpolation path
            ax.plot(reduced[:, 0], reduced[:, 1], 'o-', alpha=0.7)
            
            # Highlight start and end points
            ax.scatter(reduced[0, 0], reduced[0, 1], c='green', s=100, label='Start')
            ax.scatter(reduced[-1, 0], reduced[-1, 1], c='red', s=100, label='End')
            
            # Add labels for intermediate points
            for i, (x, y) in enumerate(reduced):
                ax.annotate(f"{alphas[i]:.1f}", (x, y), 
                            textcoords="offset points", 
                            xytext=(0, 10), 
                            ha='center')
            
            plt.title(title, fontsize=14)
            plt.legend()
            
            if save_dir:
                save_path = Path(save_dir) / f"{title.replace(' ', '_')}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
    
    def _visualize_graph_interpolation(self, 
                                      graphs: List[Union[nx.Graph, Data]], 
                                      alphas: List[float],
                                      title: str,
                                      save_dir: Optional[str]) -> plt.Figure:
        """
        Visualize interpolation of graph structures.
        
        Args:
            graphs: List of graph structures from interpolation
            alphas: Interpolation alphas
            title: Plot title
            save_dir: Directory to save visualization
            
        Returns:
            fig: Matplotlib figure
        """
        # Convert PyTorch Geometric Data to NetworkX if needed
        nx_graphs = []
        for g in graphs:
            if isinstance(g, Data):
                nx_graphs.append(to_networkx(g, to_undirected=True))
            else:
                nx_graphs.append(g)
        
        # Determine number of rows and columns for subplot grid
        n = len(nx_graphs)
        cols = min(5, n)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        
        # Flatten axes for easier indexing
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        
        # Spring layout tends to move nodes around too much between steps
        # Use the same layout for all graphs to better visualize structural changes
        combined_graph = nx.compose_all(nx_graphs)
        pos = nx.spring_layout(combined_graph, seed=42)
        
        # Plot each graph
        for i, G in enumerate(nx_graphs):
            ax = axes[i] if rows * cols > 1 else axes
            
            # Extract subgraph positions
            sub_pos = {node: pos[node] for node in G.nodes()}
            
            # Draw the graph
            nx.draw_networkx(
                G, 
                pos=sub_pos,
                ax=ax,
                with_labels=False,
                node_size=50,
                node_color='skyblue',
                edge_color='gray',
                alpha=0.8
            )
            
            # Add title with interpolation value
            ax.set_title(f"α = {alphas[i]:.2f}")
            ax.axis('off')
        
        # Hide empty subplots
        for i in range(len(nx_graphs), rows * cols):
            ax = axes[i] if rows * cols > 1 else axes
            ax.axis('off')
        
        # Add overall title
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save figure if requested
        if save_dir:
            save_path = Path(save_dir) / f"{title.replace(' ', '_')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class ModelExplainer:
    """
    Explainer for graph-based agent state models using attribution techniques.
    """
    
    def __init__(self, model: torch.nn.Module):
        """
        Initialize model explainer.
        
        Args:
            model: Trained PyTorch model to explain
        """
        self.model = model
        
    def explain_node_importance(self, 
                               data: Data, 
                               method: str = 'integrated_gradients',
                               target_node: Optional[int] = None,
                               visualize: bool = True,
                               save_path: Optional[str] = None) -> Tuple[torch.Tensor, Optional[plt.Figure]]:
        """
        Explain node importance in graph.
        
        Args:
            data: PyTorch Geometric Data
            method: Attribution method ('integrated_gradients', 'saliency', 'deeplift')
            target_node: Node to explain (use None for all nodes)
            visualize: Whether to create visualization
            save_path: Path to save visualization
            
        Returns:
            attributions: Node importance scores
            fig: Matplotlib figure if visualize=True
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Define target nodes
        if target_node is None:
            target_nodes = list(range(data.x.size(0)))
        else:
            target_nodes = [target_node]
        
        # Define attribution method
        if method == 'integrated_gradients':
            attribution_method = attr.IntegratedGradients(self.model)
        elif method == 'saliency':
            attribution_method = attr.Saliency(self.model)
        elif method == 'deeplift':
            attribution_method = attr.DeepLift(self.model)
        else:
            raise ValueError(f"Unknown attribution method: {method}")
        
        # Compute attributions
        attributions = attribution_method.attribute(
            data.x, target=target_nodes, additional_forward_args=(data.edge_index, data.edge_attr)
        )
        
        # Create visualization if requested
        fig = None
        if visualize:
            fig = self._visualize_node_importance(data, attributions, target_nodes, save_path)
        
        return attributions, fig
    
    def _visualize_node_importance(self, 
                                 data: Data,
                                 attributions: torch.Tensor,
                                 target_nodes: List[int],
                                 save_path: Optional[str]) -> plt.Figure:
        """
        Visualize node importance in graph.
        
        Args:
            data: PyTorch Geometric Data
            attributions: Node importance scores
            target_nodes: List of target nodes
            save_path: Path to save visualization
            
        Returns:
            fig: Matplotlib figure
        """
        # Convert to NetworkX for visualization
        G = to_networkx(data, to_undirected=True)
        
        # Compute aggregated attributions (mean absolute value across features)
        agg_attributions = attributions.abs().mean(dim=1).detach().cpu().numpy()
        
        # Create node sizes based on attributions
        node_sizes = 300 + 700 * agg_attributions / (agg_attributions.max() + 1e-10)
        
        # Create node colors based on attributions
        norm = plt.Normalize(agg_attributions.min(), agg_attributions.max())
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        node_colors = sm.to_rgba(agg_attributions)
        
        # Highlight target nodes
        for node in target_nodes:
            if node < len(node_sizes):
                node_sizes[node] *= 1.5
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw graph
        nx.draw_networkx(
            G, pos=pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color='gray',
            alpha=0.8,
            with_labels=False
        )
        
        # Highlight target nodes
        if len(target_nodes) < len(G.nodes()):
            nx.draw_networkx_nodes(
                G, pos=pos, ax=ax,
                nodelist=target_nodes,
                node_color='red',
                node_size=[node_sizes[i] for i in target_nodes],
                alpha=0.6
            )
        
        # Add colorbar
        plt.colorbar(sm, ax=ax, label='Importance Score')
        
        plt.title('Node Importance in Graph', fontsize=14)
        plt.axis('off')
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def explain_feature_importance(self, 
                                 data: Data,
                                 feature_names: Optional[List[str]] = None,
                                 top_k: int = 10,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Explain feature importance in graph.
        
        Args:
            data: PyTorch Geometric Data
            feature_names: Names of node features
            top_k: Number of top features to show
            save_path: Path to save visualization
            
        Returns:
            fig: Matplotlib figure
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Define attribution method
        attribution_method = attr.FeaturePermutation(self.model)
        
        # Compute attributions
        attributions = attribution_method.attribute(
            data.x, additional_forward_args=(data.edge_index, data.edge_attr)
        )
        
        # Aggregate attributions across nodes
        agg_attributions = attributions.abs().mean(dim=0).detach().cpu().numpy()
        
        # Generate feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(agg_attributions))]
        
        # Get top-k features
        top_indices = np.argsort(agg_attributions)[-top_k:]
        top_attributions = agg_attributions[top_indices]
        top_features = [feature_names[i] for i in top_indices]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(top_features)), top_attributions, color='skyblue')
        
        # Add feature names as y-tick labels
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        
        # Add value labels
        for i, v in enumerate(top_attributions):
            ax.text(v + 0.01, i, f"{v:.3f}", va='center')
        
        plt.title('Feature Importance', fontsize=14)
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig 