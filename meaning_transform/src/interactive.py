#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive components for exploring agent states and latent representations.

This module handles:
1. Interactive dashboards for exploring agent states
2. Visualization of latent space traversal
3. What-if analysis of agent state modifications
4. Real-time monitoring of agent state compression
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import dash
import dash_cytoscape as cyto
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from dash import Input, Output, State, callback, dcc, html
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .data import AgentState
from .knowledge_graph import AgentStateToGraph, KnowledgeGraphDataset

# Load Cytoscape extension for graph visualization
cyto.load_extra_layouts()


class AgentStateDashboard:
    """
    Interactive dashboard for exploring agent states and their representations.
    """

    def __init__(
        self, external_stylesheets: Optional[List[str]] = None, port: int = 8050
    ):
        """
        Initialize the dashboard.

        Args:
            external_stylesheets: CSS stylesheets for Dash app
            port: Port for serving the dashboard
        """
        self.port = port
        self.graph_converter = AgentStateToGraph()

        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=external_stylesheets
            or [
                "https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"
            ],
        )

        # Set up layout
        self._setup_layout()

        # Set up callbacks
        self._setup_callbacks()

        # Setup default data
        self.agent_states = []
        self.graphs = []
        self.latent_vectors = []

    def _setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div(
            [
                html.H1(
                    "Agent State Exploration Dashboard",
                    className="mt-4 mb-4 text-center",
                ),
                # Control panel
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4("Controls", className="mb-3"),
                                html.Button(
                                    "Load Sample Data",
                                    id="load-sample-btn",
                                    className="btn btn-primary mr-2",
                                ),
                                html.Button(
                                    "Load Custom Data",
                                    id="load-custom-btn",
                                    className="btn btn-secondary mr-2",
                                ),
                                dcc.Upload(
                                    id="upload-data",
                                    children=html.Div(
                                        ["Drag and Drop or ", html.A("Select Files")]
                                    ),
                                    style={
                                        "width": "100%",
                                        "height": "60px",
                                        "lineHeight": "60px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "5px",
                                        "textAlign": "center",
                                        "margin": "10px 0",
                                    },
                                    multiple=False,
                                ),
                                html.Hr(),
                                # Agent selection
                                html.H5("Agent Selection", className="mt-3"),
                                dcc.Dropdown(
                                    id="agent-selector",
                                    options=[],
                                    placeholder="Select an agent",
                                    className="mb-3",
                                ),
                                # Visualization options
                                html.H5("Visualization Options", className="mt-3"),
                                dcc.Checklist(
                                    id="display-options",
                                    options=[
                                        {
                                            "label": " Show Agent Properties",
                                            "value": "show_props",
                                        },
                                        {
                                            "label": " Show Relationships",
                                            "value": "show_rels",
                                        },
                                        {"label": " 3D View", "value": "3d_view"},
                                    ],
                                    value=["show_props", "show_rels"],
                                    className="mb-3",
                                ),
                                # Latent space options
                                html.H5("Latent Space Options", className="mt-3"),
                                dcc.RadioItems(
                                    id="latent-viz-method",
                                    options=[
                                        {"label": " t-SNE", "value": "tsne"},
                                        {"label": " PCA", "value": "pca"},
                                    ],
                                    value="tsne",
                                    className="mb-3",
                                ),
                                # Property filter
                                html.H5("Property Filter", className="mt-3"),
                                dcc.Dropdown(
                                    id="property-filter",
                                    options=[],
                                    placeholder="Color by property",
                                    className="mb-3",
                                ),
                                # What-if analysis
                                html.H5("What-If Analysis", className="mt-3"),
                                html.Div(
                                    [
                                        html.Label("Modify Property:"),
                                        dcc.Dropdown(
                                            id="modify-property",
                                            options=[],
                                            placeholder="Select property",
                                            className="mb-2",
                                        ),
                                        html.Label("New Value:"),
                                        dcc.Input(
                                            id="new-value",
                                            type="text",
                                            placeholder="Enter new value",
                                            className="form-control mb-2",
                                        ),
                                        html.Button(
                                            "Apply Change",
                                            id="apply-change-btn",
                                            className="btn btn-info btn-sm",
                                        ),
                                    ]
                                ),
                            ],
                            className="col-md-3",
                        ),
                        # Visualization area
                        html.Div(
                            [
                                # Tabs for different visualizations
                                dcc.Tabs(
                                    [
                                        dcc.Tab(
                                            label="Knowledge Graph",
                                            children=[
                                                html.Div(
                                                    id="graph-view-container",
                                                    children=[
                                                        cyto.Cytoscape(
                                                            id="agent-graph",
                                                            layout={"name": "cose"},
                                                            style={
                                                                "width": "100%",
                                                                "height": "600px",
                                                            },
                                                            elements=[],
                                                            stylesheet=[
                                                                {
                                                                    "selector": "node",
                                                                    "style": {
                                                                        "label": "data(label)",
                                                                        "background-color": "data(color)",
                                                                        "shape": "data(shape)",
                                                                    },
                                                                },
                                                                {
                                                                    "selector": "edge",
                                                                    "style": {
                                                                        "label": "data(label)",
                                                                        "curve-style": "bezier",
                                                                        "line-color": "data(color)",
                                                                        "width": "data(width)",
                                                                    },
                                                                },
                                                            ],
                                                        ),
                                                        html.Div(
                                                            id="graph-details",
                                                            className="mt-3",
                                                        ),
                                                    ],
                                                )
                                            ],
                                        ),
                                        dcc.Tab(
                                            label="Latent Space",
                                            children=[
                                                dcc.Graph(
                                                    id="latent-space-viz",
                                                    style={"height": "600px"},
                                                )
                                            ],
                                        ),
                                        dcc.Tab(
                                            label="Agent Properties",
                                            children=[
                                                dcc.Graph(
                                                    id="agent-properties-viz",
                                                    style={"height": "600px"},
                                                )
                                            ],
                                        ),
                                        dcc.Tab(
                                            label="Comparison",
                                            children=[
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                dcc.Dropdown(
                                                                    id="compare-agent1",
                                                                    options=[],
                                                                    placeholder="Select first agent",
                                                                    className="mb-2",
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="compare-agent2",
                                                                    options=[],
                                                                    placeholder="Select second agent",
                                                                    className="mb-2",
                                                                ),
                                                                html.Button(
                                                                    "Compare",
                                                                    id="compare-btn",
                                                                    className="btn btn-primary btn-sm",
                                                                ),
                                                            ],
                                                            className="mb-3",
                                                        ),
                                                        html.Div(
                                                            id="comparison-results"
                                                        ),
                                                    ]
                                                )
                                            ],
                                        ),
                                    ]
                                )
                            ],
                            className="col-md-9",
                        ),
                    ],
                    className="row",
                ),
            ],
            className="container-fluid",
        )

    def _setup_callbacks(self):
        """Set up the dashboard callbacks."""

        @self.app.callback(
            [
                Output("agent-selector", "options"),
                Output("property-filter", "options"),
                Output("modify-property", "options"),
                Output("compare-agent1", "options"),
                Output("compare-agent2", "options"),
            ],
            [Input("load-sample-btn", "n_clicks"), Input("upload-data", "contents")],
            [State("upload-data", "filename")],
        )
        def load_data(sample_btn, upload_contents, upload_filename):
            """Load data from either sample data or uploaded file."""
            ctx = dash.callback_context
            trigger_id = (
                ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
            )

            if trigger_id == "load-sample-btn" and sample_btn:
                # Load sample data
                self.agent_states = self._generate_sample_agents()
            elif trigger_id == "upload-data" and upload_contents:
                # Load uploaded data
                try:
                    self.agent_states = self._parse_uploaded_data(
                        upload_contents, upload_filename
                    )
                except Exception as e:
                    print(f"Error loading data: {e}")
                    self.agent_states = []

            # Convert to graphs
            if self.agent_states:
                self.graphs = [
                    self.graph_converter.agent_to_graph(agent)
                    for agent in self.agent_states
                ]

                # Create agent options for dropdowns
                agent_options = [
                    {"label": f"Agent {i} ({agent.agent_id})", "value": i}
                    for i, agent in enumerate(self.agent_states)
                ]

                # Create property options for filters
                sample_agent = self.agent_states[0]
                property_options = [
                    {"label": key, "value": key}
                    for key in sample_agent.to_dict().keys()
                    if key not in ["agent_id", "step_number", "properties"]
                ]

                return (
                    agent_options,
                    property_options,
                    property_options,
                    agent_options,
                    agent_options,
                )

            return [], [], [], [], []

        @self.app.callback(
            Output("agent-graph", "elements"),
            [Input("agent-selector", "value"), Input("display-options", "value")],
        )
        def update_graph(agent_idx, display_options):
            """Update the graph visualization based on selected agent."""
            if agent_idx is None or not self.graphs:
                return []

            # Get the graph for selected agent
            G = self.graphs[int(agent_idx)]

            # Convert NetworkX graph to Cytoscape elements
            return self._networkx_to_cytoscape(G, display_options)

        @self.app.callback(
            Output("latent-space-viz", "figure"),
            [
                Input("agent-selector", "value"),
                Input("latent-viz-method", "value"),
                Input("property-filter", "value"),
            ],
        )
        def update_latent_space(agent_idx, viz_method, color_property):
            """Update the latent space visualization."""
            if not self.agent_states:
                return go.Figure()

            # Create feature vectors for all agents
            feature_vectors = []
            for agent in self.agent_states:
                # Extract numerical features (simplistic approach)
                features = []
                (
                    features.extend(agent.position)
                    if agent.position
                    else features.extend([0, 0, 0])
                )
                features.append(agent.health)
                features.append(agent.energy)
                features.append(float(agent.is_defending))
                features.append(agent.age)
                features.append(agent.total_reward)
                feature_vectors.append(features)

            # Convert to numpy array
            features_array = np.array(feature_vectors)

            # Apply dimensionality reduction
            if viz_method == "tsne":
                reducer = TSNE(n_components=2, random_state=42)
                reduced_data = reducer.fit_transform(features_array)
            else:  # PCA
                reducer = PCA(n_components=2, random_state=42)
                reduced_data = reducer.fit_transform(features_array)

            # Create dataframe for plotting
            df = pd.DataFrame(
                {
                    "x": reduced_data[:, 0],
                    "y": reduced_data[:, 1],
                    "agent_id": [agent.agent_id for agent in self.agent_states],
                    "index": list(range(len(self.agent_states))),
                }
            )

            # Add color property if specified
            if color_property and color_property in self.agent_states[0].to_dict():
                df[color_property] = [
                    getattr(agent, color_property) for agent in self.agent_states
                ]
                color_col = color_property
            else:
                color_col = None

            # Highlight selected agent
            selected_point = int(agent_idx) if agent_idx is not None else None

            # Create plot
            fig = px.scatter(
                df,
                x="x",
                y="y",
                color=color_col,
                hover_data=["agent_id", "index"],
                title=f"Agent Latent Space ({viz_method.upper()})",
            )

            # Highlight selected agent
            if selected_point is not None:
                selected_point_data = df.iloc[selected_point]
                fig.add_trace(
                    go.Scatter(
                        x=[selected_point_data["x"]],
                        y=[selected_point_data["y"]],
                        mode="markers",
                        marker=dict(
                            size=15,
                            color="red",
                            symbol="circle-open",
                            line=dict(width=2),
                        ),
                        name="Selected Agent",
                        hoverinfo="skip",
                    )
                )

            fig.update_layout(
                xaxis_title=f"{viz_method.upper()} Dimension 1",
                yaxis_title=f"{viz_method.upper()} Dimension 2",
                legend_title=color_property if color_property else "Agent",
                font=dict(size=12),
            )

            return fig

        @self.app.callback(
            Output("agent-properties-viz", "figure"), [Input("agent-selector", "value")]
        )
        def update_properties_viz(agent_idx):
            """Update the agent properties visualization."""
            if agent_idx is None or not self.agent_states:
                return go.Figure()

            # Get selected agent
            agent = self.agent_states[int(agent_idx)]
            agent_dict = agent.to_dict()

            # Filter properties for visualization
            numeric_props = {}
            categorical_props = {}

            for key, value in agent_dict.items():
                if key in [
                    "agent_id",
                    "step_number",
                    "properties",
                    "goals",
                    "inventory",
                    "position",
                ]:
                    continue

                if isinstance(value, (int, float)):
                    numeric_props[key] = value
                else:
                    categorical_props[key] = value

            # Create subplots: numeric bar chart and categorical pie chart
            if numeric_props:
                # Create bar chart for numeric properties
                fig = go.Figure()

                fig.add_trace(
                    go.Bar(
                        x=list(numeric_props.keys()),
                        y=list(numeric_props.values()),
                        marker_color="skyblue",
                    )
                )

                fig.update_layout(
                    title=f"Properties of Agent {agent.agent_id}",
                    yaxis_title="Value",
                    font=dict(size=12),
                )

                return fig
            else:
                # Fallback if no numeric properties
                fig = go.Figure()
                fig.update_layout(
                    title="No numeric properties available", font=dict(size=12)
                )
                return fig

        @self.app.callback(
            Output("comparison-results", "children"),
            [Input("compare-btn", "n_clicks")],
            [State("compare-agent1", "value"), State("compare-agent2", "value")],
        )
        def compare_agents(n_clicks, agent1_idx, agent2_idx):
            """Compare two selected agents."""
            if (
                n_clicks is None
                or agent1_idx is None
                or agent2_idx is None
                or not self.agent_states
            ):
                return html.Div("Select two agents to compare")

            # Get selected agents
            agent1 = self.agent_states[int(agent1_idx)]
            agent2 = self.agent_states[int(agent2_idx)]

            # Compare properties
            comparison_results = []

            # Add header
            comparison_results.append(
                html.H5(f"Comparison: {agent1.agent_id} vs {agent2.agent_id}")
            )

            # Create comparison table
            rows = []
            for key in agent1.to_dict().keys():
                if key in ["agent_id", "step_number", "properties"]:
                    continue

                val1 = getattr(agent1, key)
                val2 = getattr(agent2, key)

                # Format values for display
                if isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
                    val1_str = str(val1)
                    val2_str = str(val2)
                    # Calculate similarity for lists
                    if key == "goals" and val1 and val2:
                        similarity = len(set(val1).intersection(set(val2))) / len(
                            set(val1).union(set(val2))
                        )
                        similarity_str = f"{similarity:.2f}"
                    elif key == "inventory" and val1 and val2:
                        common_items = set(val1.keys()).intersection(set(val2.keys()))
                        all_items = set(val1.keys()).union(set(val2.keys()))
                        similarity = (
                            len(common_items) / len(all_items) if all_items else 0
                        )
                        similarity_str = f"{similarity:.2f}"
                    elif key == "position":
                        # Euclidean distance
                        similarity = np.sqrt(
                            sum((p1 - p2) ** 2 for p1, p2 in zip(val1, val2))
                        )
                        similarity_str = f"{similarity:.2f} (distance)"
                    else:
                        similarity_str = "N/A"
                else:
                    val1_str = str(val1)
                    val2_str = str(val2)
                    if isinstance(val1, (int, float)) and isinstance(
                        val2, (int, float)
                    ):
                        # Numeric difference
                        similarity = abs(val1 - val2)
                        similarity_str = f"{similarity:.2f} (diff)"
                    else:
                        similarity_str = "Same" if val1 == val2 else "Different"

                rows.append(
                    html.Tr(
                        [
                            html.Td(key),
                            html.Td(val1_str),
                            html.Td(val2_str),
                            html.Td(similarity_str),
                        ]
                    )
                )

            # Create table
            table = html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("Property"),
                                html.Th(agent1.agent_id),
                                html.Th(agent2.agent_id),
                                html.Th("Similarity/Difference"),
                            ]
                        )
                    ),
                    html.Tbody(rows),
                ],
                className="table table-striped table-hover",
            )

            comparison_results.append(table)

            return html.Div(comparison_results)

        @self.app.callback(
            Output("graph-details", "children"), [Input("agent-graph", "tapNode")]
        )
        def show_node_details(tap_data):
            """Show details for a tapped node."""
            if tap_data is None:
                return html.Div("Click on a node to see details")

            node_data = tap_data["data"]
            node_id = node_data.get("id", "unknown")
            node_type = node_data.get("node_type", "unknown")

            details = [html.H5(f"Node Details: {node_data.get('label', node_id)}")]

            # Add property table
            rows = []
            for key, value in node_data.items():
                if key not in ["id", "label", "color", "shape", "node_type"]:
                    rows.append(html.Tr([html.Td(key), html.Td(str(value))]))

            if rows:
                details.append(
                    html.Table(
                        [
                            html.Thead(
                                html.Tr([html.Th("Property"), html.Th("Value")])
                            ),
                            html.Tbody(rows),
                        ],
                        className="table table-sm",
                    )
                )
            else:
                details.append(html.Div("No additional properties"))

            return html.Div(details)

    def _networkx_to_cytoscape(
        self, G: nx.Graph, display_options: List[str]
    ) -> List[Dict]:
        """
        Convert NetworkX graph to Cytoscape elements.

        Args:
            G: NetworkX graph
            display_options: Display options

        Returns:
            elements: Cytoscape elements
        """
        show_props = "show_props" in display_options
        show_rels = "show_rels" in display_options

        elements = []

        # Add nodes
        for node, data in G.nodes(data=True):
            node_type = data.get("type", "unknown")

            # Skip property nodes if show_props is False
            if not show_props and node_type != "agent":
                continue

            # Determine node shape and color
            if node_type == "agent":
                shape = "ellipse"
                color = "#1f77b4"  # blue
            elif node_type == "property":
                shape = "round-rectangle"
                color = "#ff7f0e"  # orange
            elif node_type == "inventory_item":
                shape = "diamond"
                color = "#2ca02c"  # green
            elif node_type == "goal":
                shape = "star"
                color = "#d62728"  # red
            else:
                shape = "square"
                color = "#7f7f7f"  # gray

            # Create label
            if node_type == "agent":
                label = data.get("label", str(node).split("/")[-1])
            elif node_type == "property":
                prop_name = data.get("name", "")
                prop_value = data.get("value", "")
                label = f"{prop_name}: {prop_value}"
            elif node_type == "inventory_item":
                item_name = data.get("name", "")
                quantity = data.get("quantity", "")
                label = f"{item_name} ({quantity})"
            elif node_type == "goal":
                label = data.get("description", "")
            else:
                label = str(node).split("/")[-1]

            # Truncate long labels
            if len(label) > 20:
                label = label[:17] + "..."

            # Create node element
            node_element = {
                "data": {
                    "id": str(node),
                    "label": label,
                    "node_type": node_type,
                    "color": color,
                    "shape": shape,
                }
            }

            # Add all other data
            for key, value in data.items():
                if key not in ["type", "label", "features"]:
                    node_element["data"][key] = value

            elements.append(node_element)

        # Add edges
        if show_props or show_rels:
            for u, v, data in G.edges(data=True):
                source_type = G.nodes[u].get("type", "unknown")
                target_type = G.nodes[v].get("type", "unknown")

                # Skip property edges if show_props is False
                if not show_props and (
                    source_type != "agent" or target_type != "agent"
                ):
                    continue

                # Skip relationship edges if show_rels is False
                if not show_rels and source_type == "agent" and target_type == "agent":
                    continue

                # Determine edge color and width
                relation = data.get("relation", "unknown")

                if relation == "proximity":
                    color = "#d62728"  # red
                elif relation == "inventory_similarity":
                    color = "#ff9896"  # light red
                elif relation == "cooperation":
                    color = "#9467bd"  # purple
                elif relation.startswith("has_"):
                    color = "#1f77b4"  # blue
                else:
                    color = "#c5b0d5"  # light purple

                # Determine edge width
                width = 1.0
                if "weight" in data:
                    width = 1.0 + (data["weight"] * 2.0)

                # Create edge element
                edge_element = {
                    "data": {
                        "source": str(u),
                        "target": str(v),
                        "label": relation,
                        "color": color,
                        "width": width,
                    }
                }

                # Add all other data
                for key, value in data.items():
                    if key not in ["relation"]:
                        edge_element["data"][key] = value

                elements.append(edge_element)

        return elements

    def _generate_sample_agents(self) -> List[AgentState]:
        """
        Generate sample agent states for demonstration.

        Returns:
            agents: List of sample agent states
        """
        agents = []

        # Create sample roles
        roles = ["explorer", "gatherer", "defender", "builder", "leader"]

        # Create some sample agents
        for i in range(5):
            agent_id = f"agent_{i}"
            role = roles[i]
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

    def _parse_uploaded_data(self, contents: str, filename: str) -> List[AgentState]:
        """
        Parse uploaded data.

        Args:
            contents: File contents
            filename: File name

        Returns:
            agents: List of agent states
        """
        import base64
        import io

        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        agents = []

        if "json" in filename:
            # Parse JSON
            try:
                data = json.loads(decoded.decode("utf-8"))

                if isinstance(data, list):
                    for agent_data in data:
                        agent = AgentState(**agent_data)
                        agents.append(agent)
                elif isinstance(data, dict):
                    agent = AgentState(**data)
                    agents.append(agent)
            except Exception as e:
                print(f"Error parsing JSON: {e}")

        return agents

    def run(self):
        """Run the dashboard."""
        self.app.run(debug=True, port=self.port)


class LatentSpaceExplorer:
    """
    Tool for exploring the latent space of agent states.

    This class allows for visualizing latent trajectories,
    interpolating between states, and analyzing latent clusters.
    """

    def __init__(self, model: Optional[torch.nn.Module] = None):
        """
        Initialize latent space explorer.

        Args:
            model: Trained model for encoding agent states
        """
        self.model = model

    def visualize_trajectories(
        self,
        agent_sequences: List[List[AgentState]],
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Visualize trajectories of agent states in latent space.

        Args:
            agent_sequences: List of sequences of agent states
            labels: Labels for each sequence
            save_path: Path to save visualization

        Returns:
            fig: Plotly figure
        """
        # If no model is provided, create embeddings directly from agent states
        trajectories = []

        for sequence in agent_sequences:
            # Extract features
            features = []
            for agent in sequence:
                # Extract numerical features (simplistic approach)
                agent_features = []
                (
                    agent_features.extend(agent.position)
                    if agent.position
                    else agent_features.extend([0, 0, 0])
                )
                agent_features.append(agent.health)
                agent_features.append(agent.energy)
                agent_features.append(float(agent.is_defending))
                agent_features.append(agent.age)
                agent_features.append(agent.total_reward)
                features.append(agent_features)

            trajectories.append(np.array(features))

        # Apply PCA to all features combined
        all_features = np.vstack([traj for traj in trajectories])
        reducer = PCA(n_components=2)
        all_reduced = reducer.fit_transform(all_features)

        # Split back into trajectories
        reduced_trajectories = []
        start_idx = 0
        for traj in trajectories:
            traj_len = len(traj)
            reduced_trajectories.append(all_reduced[start_idx : start_idx + traj_len])
            start_idx += traj_len

        # Create visualization
        fig = go.Figure()

        # Add trajectories
        for i, reduced_traj in enumerate(reduced_trajectories):
            label = labels[i] if labels and i < len(labels) else f"Trajectory {i+1}"

            # Add line
            fig.add_trace(
                go.Scatter(
                    x=reduced_traj[:, 0],
                    y=reduced_traj[:, 1],
                    mode="lines+markers",
                    name=label,
                    line=dict(width=2),
                )
            )

            # Highlight start and end points
            fig.add_trace(
                go.Scatter(
                    x=[reduced_traj[0, 0]],
                    y=[reduced_traj[0, 1]],
                    mode="markers",
                    marker=dict(size=10, symbol="circle", color="green"),
                    name=f"{label} Start",
                    showlegend=False,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[reduced_traj[-1, 0]],
                    y=[reduced_traj[-1, 1]],
                    mode="markers",
                    marker=dict(size=10, symbol="star", color="red"),
                    name=f"{label} End",
                    showlegend=False,
                )
            )

        # Update layout
        fig.update_layout(
            title="Agent State Trajectories in Latent Space",
            xaxis_title="PCA Dimension 1",
            yaxis_title="PCA Dimension 2",
            font=dict(size=12),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            template="plotly_white",
        )

        # Save figure if requested
        if save_path:
            fig.write_html(save_path)

        return fig


# Function to run the dashboard as a standalone app
def run_dashboard(port: int = 8050):
    """
    Run the agent state dashboard as a standalone app.

    Args:
        port: Port to run the dashboard on
    """
    dashboard = AgentStateDashboard(port=port)
    dashboard.run()
