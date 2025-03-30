#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Knowledge Graph module for agent state representation.

This module handles:
1. Converting agent states to knowledge graph structures
2. Defining relationships between agents and properties
3. Building and manipulating knowledge graphs
4. Serialization and deserialization of graph structures
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import rdflib
import torch
from rdflib import Graph, Literal, Namespace, URIRef
from torch_geometric.data import Batch, Data

from .data import AgentState

# Define namespaces for RDF
AGENT = Namespace("http://agent-meaning.org/agent/")
PROP = Namespace("http://agent-meaning.org/property/")
REL = Namespace("http://agent-meaning.org/relation/")


class AgentStateToGraph:
    """
    Converter class to transform agent states into graph representations.
    """

    def __init__(
        self,
        relationship_threshold: float = 0.5,
        include_relations: bool = True,
        property_as_node: bool = True,
    ):
        """
        Initialize the converter with configuration.

        Args:
            relationship_threshold: Threshold for creating relationships between agents
            include_relations: Whether to include inter-agent relationships
            property_as_node: Whether to represent properties as nodes (True) or edge attributes (False)
        """
        self.relationship_threshold = relationship_threshold
        self.include_relations = include_relations
        self.property_as_node = property_as_node

        # Feature mapping for node features
        self.feature_map = {
            "position": 0,
            "health": 1,
            "energy": 2,
            "role": 3,
            "resource_level": 4,
            "is_defending": 5,
            "age": 6,
        }

        # Role mapping - convert string roles to one-hot encodings
        self.role_map = {
            "explorer": 0,
            "gatherer": 1,
            "defender": 2,
            "builder": 3,
            "leader": 4,
        }

        # Relationship types between agents
        self.relation_types = {
            "proximity": 0,  # Based on physical distance
            "resource_competition": 1,  # Competing for same resources
            "cooperation": 2,  # Working together on goals
            "hierarchy": 3,  # Leadership/follower relationship
            "inventory_similarity": 4,  # Similar inventory items
        }

    def agent_to_graph(self, agent: AgentState) -> nx.Graph:
        """
        Convert a single agent state to a graph representation.

        Args:
            agent: AgentState object

        Returns:
            G: NetworkX graph representation
        """
        G = nx.Graph()

        # Create agent node
        agent_id = agent.agent_id or "agent_default"
        agent_uri = URIRef(f"{AGENT}{agent_id}")

        # Add agent node with features
        agent_features = self._extract_agent_features(agent)
        G.add_node(agent_uri, type="agent", features=agent_features, label=agent_id)

        # Handle properties based on configuration
        if self.property_as_node:
            # Add properties as nodes with edges to agent
            self._add_property_nodes(G, agent, agent_uri)
        else:
            # Add properties as attributes on agent node
            for key, value in agent.to_dict().items():
                if key not in ["agent_id", "step_number"]:
                    G.nodes[agent_uri][key] = value

        return G

    def _extract_agent_features(self, agent: AgentState) -> np.ndarray:
        """
        Extract numerical features from agent state.

        Args:
            agent: AgentState object

        Returns:
            features: NumPy array of features
        """
        # Initialize features array
        features = np.zeros(15)  # Adjust size based on feature dimensions

        # Position (x, y, z)
        if agent.position:
            features[0:3] = agent.position

        # Health and energy
        features[3] = agent.health
        features[4] = agent.energy

        # One-hot encoding for role
        role_idx = self.role_map.get(agent.role, 0)
        features[5 + role_idx] = 1.0  # One-hot encoding (5-9)

        # Resource level
        features[10] = agent.resource_level if agent.resource_level is not None else 0.0

        # Current health
        features[11] = (
            agent.current_health if agent.current_health is not None else agent.health
        )

        # Is defending (boolean to float)
        features[12] = float(agent.is_defending)

        # Age
        features[13] = agent.age

        # Total reward
        features[14] = agent.total_reward

        return features

    def _add_property_nodes(
        self, G: nx.Graph, agent: AgentState, agent_uri: URIRef
    ) -> None:
        """
        Add property nodes to the graph.

        Args:
            G: NetworkX graph
            agent: AgentState object
            agent_uri: URI reference for agent
        """
        # Add nodes for scalar properties
        for prop_name in [
            "health",
            "energy",
            "resource_level",
            "current_health",
            "age",
            "total_reward",
        ]:
            if hasattr(agent, prop_name) and getattr(agent, prop_name) is not None:
                prop_value = getattr(agent, prop_name)
                prop_uri = URIRef(f"{PROP}{prop_name}_{agent.agent_id}")
                G.add_node(prop_uri, type="property", name=prop_name, value=prop_value)
                G.add_edge(agent_uri, prop_uri, relation=f"has_{prop_name}")

        # Add node for position (vector property)
        if agent.position:
            pos_uri = URIRef(f"{PROP}position_{agent.agent_id}")
            G.add_node(pos_uri, type="property", name="position", value=agent.position)
            G.add_edge(agent_uri, pos_uri, relation="has_position")

        # Add node for role (categorical property)
        if agent.role:
            role_uri = URIRef(f"{PROP}role_{agent.agent_id}")
            G.add_node(role_uri, type="property", name="role", value=agent.role)
            G.add_edge(agent_uri, role_uri, relation="has_role")

        # Add nodes for inventory items
        if agent.inventory:
            for item, quantity in agent.inventory.items():
                item_uri = URIRef(f"{PROP}inventory_{item}_{agent.agent_id}")
                G.add_node(
                    item_uri, type="inventory_item", name=item, quantity=quantity
                )
                G.add_edge(agent_uri, item_uri, relation="has_item")

        # Add nodes for goals
        if agent.goals:
            for idx, goal in enumerate(agent.goals):
                goal_uri = URIRef(f"{PROP}goal_{idx}_{agent.agent_id}")
                G.add_node(goal_uri, type="goal", description=goal)
                G.add_edge(agent_uri, goal_uri, relation="has_goal")

    def agents_to_graph(self, agents: List[AgentState]) -> nx.Graph:
        """
        Convert multiple agent states to a unified graph representation.

        Args:
            agents: List of AgentState objects

        Returns:
            G: NetworkX graph representation with agent relationships
        """
        # Create initial graph with all agents
        G = nx.Graph()

        # Add each agent to the graph
        for agent in agents:
            agent_graph = self.agent_to_graph(agent)
            G = nx.compose(G, agent_graph)

        # Add relationships between agents if enabled
        if self.include_relations:
            self._add_agent_relationships(G, agents)

        return G

    def _add_agent_relationships(self, G: nx.Graph, agents: List[AgentState]) -> None:
        """
        Add relationships between agents based on various metrics.

        Args:
            G: NetworkX graph
            agents: List of AgentState objects
        """
        # Get list of agent nodes (filtering out property nodes)
        agent_nodes = [
            node for node, data in G.nodes(data=True) if data.get("type") == "agent"
        ]

        for i, agent1 in enumerate(agents):
            agent1_uri = URIRef(f"{AGENT}{agent1.agent_id or 'agent_default'}")

            for j, agent2 in enumerate(agents[i + 1 :], start=i + 1):
                agent2_uri = URIRef(f"{AGENT}{agent2.agent_id or 'agent_default'}")

                # Skip self-relationships
                if agent1.agent_id == agent2.agent_id:
                    continue

                # Calculate proximity if positions are available
                if agent1.position and agent2.position:
                    proximity = self._calculate_proximity(
                        agent1.position, agent2.position
                    )

                    # Add proximity edge if close enough
                    if proximity < self.relationship_threshold:
                        G.add_edge(
                            agent1_uri,
                            agent2_uri,
                            relation="proximity",
                            weight=1.0 - proximity,
                        )

                # Calculate inventory similarity
                if agent1.inventory and agent2.inventory:
                    similarity = self._calculate_inventory_similarity(
                        agent1.inventory, agent2.inventory
                    )

                    if similarity > self.relationship_threshold:
                        G.add_edge(
                            agent1_uri,
                            agent2_uri,
                            relation="inventory_similarity",
                            weight=similarity,
                        )

                # Check for cooperation based on shared goals
                if agent1.goals and agent2.goals:
                    cooperation = self._calculate_goal_overlap(
                        agent1.goals, agent2.goals
                    )

                    if cooperation > self.relationship_threshold:
                        G.add_edge(
                            agent1_uri,
                            agent2_uri,
                            relation="cooperation",
                            weight=cooperation,
                        )

    def _calculate_proximity(
        self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]
    ) -> float:
        """
        Calculate normalized proximity between two positions.

        Args:
            pos1: Position of first agent (x, y, z)
            pos2: Position of second agent (x, y, z)

        Returns:
            proximity: Normalized distance (0-1 where 0 is closest)
        """
        # Calculate Euclidean distance
        distance = np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pos1, pos2)))

        # Normalize to 0-1 range (assuming max world size of 100)
        max_distance = 100.0
        proximity = min(distance / max_distance, 1.0)

        return proximity

    def _calculate_inventory_similarity(
        self, inv1: Dict[str, int], inv2: Dict[str, int]
    ) -> float:
        """
        Calculate similarity between two inventories.

        Args:
            inv1: Inventory of first agent
            inv2: Inventory of second agent

        Returns:
            similarity: Normalized similarity score (0-1)
        """
        # Get all unique items
        all_items = set(inv1.keys()).union(set(inv2.keys()))

        if not all_items:
            return 0.0

        # Count shared items
        shared_items = sum(1 for item in all_items if item in inv1 and item in inv2)

        # Calculate similarity
        similarity = shared_items / len(all_items)

        return similarity

    def _calculate_goal_overlap(self, goals1: List[str], goals2: List[str]) -> float:
        """
        Calculate goal overlap between two agents.

        Args:
            goals1: Goals of first agent
            goals2: Goals of second agent

        Returns:
            overlap: Normalized overlap score (0-1)
        """
        # Get all unique goals
        all_goals = set(goals1).union(set(goals2))

        if not all_goals:
            return 0.0

        # Count shared goals
        shared_goals = set(goals1).intersection(set(goals2))

        # Calculate overlap
        overlap = len(shared_goals) / len(all_goals)

        return overlap

    def to_torch_geometric(self, G: nx.Graph) -> Data:
        """
        Convert NetworkX graph to PyTorch Geometric Data format.

        Args:
            G: NetworkX graph

        Returns:
            data: PyTorch Geometric Data
        """
        # Extract node features
        nodes = list(G.nodes())
        node_types = [G.nodes[node].get("type", "unknown") for node in nodes]

        # Create node type mapping
        node_type_map = {
            "agent": 0,
            "property": 1,
            "inventory_item": 2,
            "goal": 3,
            "unknown": 4,
        }

        # Initialize feature matrix - for now using fixed size feature vector
        node_features = []

        for node in nodes:
            node_data = G.nodes[node]
            node_type = node_data.get("type", "unknown")

            if node_type == "agent" and "features" in node_data:
                # Use agent features
                features = node_data["features"]
                # Add node type one-hot encoding (first 5 elements)
                type_one_hot = np.zeros(5)
                type_one_hot[node_type_map[node_type]] = 1.0
                node_features.append(np.concatenate([type_one_hot, features]))
            else:
                # Create features for non-agent nodes with one-hot encoding for type
                features = np.zeros(15)  # Same size as agent features
                type_one_hot = np.zeros(5)
                type_one_hot[node_type_map[node_type]] = 1.0

                # Add property value if available
                if "value" in node_data and isinstance(
                    node_data["value"], (int, float)
                ):
                    features[0] = float(node_data["value"])
                elif "quantity" in node_data and isinstance(
                    node_data["quantity"], (int, float)
                ):
                    features[0] = float(node_data["quantity"])

                node_features.append(np.concatenate([type_one_hot, features]))

        # Convert to tensor
        x = torch.tensor(np.array(node_features), dtype=torch.float)

        # Create edge index
        edge_index = []
        edge_attr = []

        for source, target, data in G.edges(data=True):
            # Map node IDs to indices
            source_idx = nodes.index(source)
            target_idx = nodes.index(target)

            # Add edge in both directions (undirected graph)
            edge_index.append([source_idx, target_idx])
            edge_index.append([target_idx, source_idx])

            # Extract edge attributes
            relation = data.get("relation", "unknown")
            weight = data.get("weight", 1.0)

            # Relation type mapping
            relation_map = {
                "has_health": 0,
                "has_energy": 1,
                "has_position": 2,
                "has_role": 3,
                "has_item": 4,
                "has_goal": 5,
                "proximity": 6,
                "inventory_similarity": 7,
                "cooperation": 8,
                "unknown": 9,
            }

            # Create one-hot encoding for relation type
            relation_one_hot = np.zeros(len(relation_map))
            relation_one_hot[relation_map.get(relation, 9)] = 1.0

            # Add weight as additional feature
            edge_features = np.append(relation_one_hot, weight)

            # Add edge attributes (same for both directions)
            edge_attr.append(edge_features)
            edge_attr.append(edge_features)

        # Convert to tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # Create PyTorch Geometric Data
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data

    def to_rdf(self, G: nx.Graph, format: str = "turtle") -> str:
        """
        Convert NetworkX graph to RDF representation.

        Args:
            G: NetworkX graph
            format: RDF serialization format ('turtle', 'xml', 'json-ld', etc.)

        Returns:
            rdf_str: RDF serialization as string
        """
        # Create RDF graph
        rdf_graph = rdflib.Graph()

        # Bind namespaces
        rdf_graph.bind("agent", AGENT)
        rdf_graph.bind("prop", PROP)
        rdf_graph.bind("rel", REL)

        # Add all nodes and edges to RDF graph
        for node, data in G.nodes(data=True):
            # Skip nodes that are already URIRefs
            if isinstance(node, URIRef):
                node_uri = node
            else:
                node_uri = URIRef(str(node))

            # Add node type
            node_type = data.get("type", "unknown")
            rdf_graph.add((node_uri, rdflib.RDF.type, URIRef(f"{REL}{node_type}")))

            # Add node attributes
            for key, value in data.items():
                if key in ["type", "features"]:
                    continue

                # Handle different value types
                if isinstance(value, (str, int, float, bool)):
                    rdf_graph.add((node_uri, URIRef(f"{PROP}{key}"), Literal(value)))
                elif isinstance(value, (list, tuple)):
                    for i, item in enumerate(value):
                        rdf_graph.add(
                            (node_uri, URIRef(f"{PROP}{key}_{i}"), Literal(item))
                        )

        # Add edges
        for source, target, data in G.edges(data=True):
            # Convert to URIRefs if needed
            source_uri = source if isinstance(source, URIRef) else URIRef(str(source))
            target_uri = target if isinstance(target, URIRef) else URIRef(str(target))

            # Get relation type
            relation = data.get("relation", "connected_to")
            relation_uri = URIRef(f"{REL}{relation}")

            # Add relationship
            rdf_graph.add((source_uri, relation_uri, target_uri))

            # Add edge attributes
            for key, value in data.items():
                if key != "relation":
                    edge_bnode = rdflib.BNode()
                    rdf_graph.add((source_uri, relation_uri, edge_bnode))
                    rdf_graph.add(
                        (edge_bnode, rdflib.RDF.type, URIRef(f"{REL}EdgeProperty"))
                    )
                    rdf_graph.add((edge_bnode, URIRef(f"{PROP}{key}"), Literal(value)))

        # Serialize to requested format
        return rdf_graph.serialize(format=format)


class KnowledgeGraphDataset:
    """Dataset for handling collections of agent state graphs."""

    def __init__(self, converter: Optional[AgentStateToGraph] = None):
        """
        Initialize dataset.

        Args:
            converter: AgentStateToGraph converter to use
        """
        self.converter = converter or AgentStateToGraph()
        self.graphs = []

    def add_agent_states(self, agents: List[AgentState]) -> None:
        """
        Add agent states to dataset.

        Args:
            agents: List of agent states
        """
        graph = self.converter.agents_to_graph(agents)
        self.graphs.append(graph)

    def to_torch_geometric_batch(self) -> Batch:
        """
        Convert all graphs to PyTorch Geometric batch.

        Returns:
            batch: PyTorch Geometric batch of graphs
        """
        # Convert each graph to PyTorch Geometric Data
        data_list = [self.converter.to_torch_geometric(G) for G in self.graphs]

        # Create batch
        batch = Batch.from_data_list(data_list)

        return batch

    def save(self, filename: str) -> None:
        """
        Save dataset to file.

        Args:
            filename: Path to output file
        """
        # Create directory if it doesn't exist
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        # Serialize graphs as list of adjacency data
        graph_data = []

        for G in self.graphs:
            # Convert graph to dictionary data
            nodes_data = {node: data for node, data in G.nodes(data=True)}
            edges_data = {(str(u), str(v)): data for u, v, data in G.edges(data=True)}

            # Handle non-serializable objects
            for node, data in nodes_data.items():
                for key, value in data.items():
                    if key == "features" and isinstance(value, np.ndarray):
                        nodes_data[node][key] = value.tolist()

            graph_data.append({"nodes": nodes_data, "edges": edges_data})

        # Save to file
        with open(filename, "w") as f:
            json.dump(graph_data, f, indent=2)

    @classmethod
    def load(cls, filename: str) -> "KnowledgeGraphDataset":
        """
        Load dataset from file.

        Args:
            filename: Path to input file

        Returns:
            dataset: Loaded dataset
        """
        # Create dataset
        dataset = cls()

        # Load from file
        with open(filename, "r") as f:
            graph_data = json.load(f)

        # Reconstruct graphs
        for data in graph_data:
            G = nx.Graph()

            # Add nodes
            for node, node_data in data["nodes"].items():
                # Convert features back to numpy if needed
                if "features" in node_data and isinstance(node_data["features"], list):
                    node_data["features"] = np.array(node_data["features"])

                G.add_node(node, **node_data)

            # Add edges
            for edge_str, edge_data in data["edges"].items():
                u, v = eval(edge_str)  # Convert string tuple back to tuple
                G.add_edge(u, v, **edge_data)

            dataset.graphs.append(G)

        return dataset


def serialize_knowledge_graph(G: nx.Graph) -> bytes:
    """
    Serialize NetworkX graph to binary format.

    Args:
        G: NetworkX graph

    Returns:
        data: Binary serialization
    """
    # Convert nodes and edges to serializable format
    node_data = []
    for node, attrs in G.nodes(data=True):
        # Convert node ID to string if it's a URIRef
        node_str = str(node)

        # Handle numpy arrays in attributes
        node_attrs = {}
        for key, value in attrs.items():
            if isinstance(value, np.ndarray):
                node_attrs[key] = value.tolist()
            else:
                node_attrs[key] = value

        node_data.append((node_str, node_attrs))

    edge_data = []
    for u, v, attrs in G.edges(data=True):
        edge_data.append((str(u), str(v), attrs))

    # Create serializable data structure
    graph_data = {"nodes": node_data, "edges": edge_data}

    # Serialize using pickle
    binary_data = pickle.dumps(graph_data)

    return binary_data


def deserialize_knowledge_graph(data: bytes) -> nx.Graph:
    """
    Deserialize binary data to NetworkX graph.

    Args:
        data: Binary serialization

    Returns:
        G: NetworkX graph
    """
    # Deserialize using pickle
    graph_data = pickle.loads(data)

    # Create graph
    G = nx.Graph()

    # Add nodes
    for node_str, attrs in graph_data["nodes"]:
        # Handle URIRef nodes
        if node_str.startswith("http://"):
            node = URIRef(node_str)
        else:
            node = node_str

        # Handle features array
        if "features" in attrs and isinstance(attrs["features"], list):
            attrs["features"] = np.array(attrs["features"])

        G.add_node(node, **attrs)

    # Add edges
    for u_str, v_str, attrs in graph_data["edges"]:
        # Handle URIRef nodes
        if u_str.startswith("http://"):
            u = URIRef(u_str)
        else:
            u = u_str

        if v_str.startswith("http://"):
            v = URIRef(v_str)
        else:
            v = v_str

        G.add_edge(u, v, **attrs)

    return G
