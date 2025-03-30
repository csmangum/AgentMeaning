#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the data module implementation.

This script tests the functionality of the data module, including:
- AgentState class
- AgentStateDataset class
- Serialization and deserialization
- Conversion to and from various formats (dict, binary, tensor, graph)
- Data loading and batching
"""

import os
import sys
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

from meaning_transform.src.knowledge_graph import AgentStateToGraph

# Add the project root to the path so imports work correctly
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from meaning_transform.src.data import (
    AgentState,
    AgentStateDataset,
    deserialize_states,
    determine_role,
    generate_agent_states,
    serialize_states,
)


# Test AgentState class
class TestAgentState:
    """Tests for the AgentState class."""

    def test_init(self):
        """Test that AgentState initializes correctly."""
        state = AgentState(
            position=(1.0, 2.0, 3.0),
            health=0.8,
            energy=0.7,
            inventory={"wood": 5, "stone": 3},
            role="explorer",
            goals=["find_resources", "explore_territory"],
            agent_id="test_agent",
            step_number=42,
            resource_level=0.6,
            current_health=0.75,
            is_defending=True,
            age=100,
            total_reward=50.0,
            custom_attr="test",
        )

        assert state.position == (1.0, 2.0, 3.0)
        assert state.health == 0.8
        assert state.energy == 0.7
        assert state.inventory == {"wood": 5, "stone": 3}
        assert state.role == "explorer"
        assert state.goals == ["find_resources", "explore_territory"]
        assert state.agent_id == "test_agent"
        assert state.step_number == 42
        assert state.resource_level == 0.6
        assert state.current_health == 0.75
        assert state.is_defending is True
        assert state.age == 100
        assert state.total_reward == 50.0
        assert state.properties["custom_attr"] == "test"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        state = AgentState(
            position=(1.0, 2.0, 3.0), health=0.8, energy=0.7, role="explorer"
        )

        state_dict = state.to_dict()
        assert isinstance(state_dict, dict)
        assert state_dict["position"] == (1.0, 2.0, 3.0)
        assert state_dict["health"] == 0.8
        assert state_dict["energy"] == 0.7
        assert state_dict["role"] == "explorer"

    def test_to_binary_and_from_binary(self):
        """Test binary serialization and deserialization."""
        original_state = AgentState(
            position=(1.0, 2.0, 3.0),
            health=0.8,
            energy=0.7,
            inventory={"wood": 5, "stone": 3},
            role="explorer",
            goals=["find_resources"],
            is_defending=True,
        )

        # Convert to binary
        binary_data = original_state.to_binary()
        assert isinstance(binary_data, bytes)

        # Convert back from binary
        reconstructed_state = AgentState.from_binary(binary_data)

        # Verify all properties are preserved
        assert reconstructed_state.position == original_state.position
        assert reconstructed_state.health == original_state.health
        assert reconstructed_state.energy == original_state.energy
        assert reconstructed_state.inventory == original_state.inventory
        assert reconstructed_state.role == original_state.role
        assert reconstructed_state.goals == original_state.goals
        assert reconstructed_state.is_defending == original_state.is_defending

    def test_to_tensor_and_from_tensor(self):
        """Test tensor conversion in both directions."""
        original_state = AgentState(
            position=(1.0, 2.0, 3.0),
            health=0.8,
            energy=0.7,
            role="explorer",
            resource_level=0.6,
            current_health=0.75,
            is_defending=True,
            age=100,
            total_reward=50.0,
        )

        # Convert to tensor
        tensor = original_state.to_tensor()
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == 15  # Expected feature count

        # Convert back from tensor
        reconstructed_state = AgentState.from_tensor(tensor)

        # Check key properties are preserved
        assert reconstructed_state.position[0] == pytest.approx(
            original_state.position[0], abs=0.01
        )
        assert reconstructed_state.position[1] == pytest.approx(
            original_state.position[1], abs=0.01
        )
        assert reconstructed_state.position[2] == pytest.approx(
            original_state.position[2], abs=0.01
        )
        assert reconstructed_state.health == pytest.approx(
            original_state.health, abs=0.01
        )
        assert reconstructed_state.energy == pytest.approx(
            original_state.energy, abs=0.01
        )
        assert reconstructed_state.role == original_state.role
        assert reconstructed_state.is_defending == original_state.is_defending
        assert reconstructed_state.age == pytest.approx(
            original_state.age, abs=20
        )  # Less precise due to normalization

    def test_from_db_record(self):
        """Test creating agent state from database record."""
        db_record = {
            "agent_id": "agent_123",
            "step_number": 42,
            "position_x": 10.0,
            "position_y": 20.0,
            "position_z": 5.0,
            "resource_level": 0.75,
            "current_health": 0.9,
            "is_defending": True,
            "total_reward": 150.0,
            "age": 200,
        }

        state = AgentState.from_db_record(db_record)

        assert state.agent_id == "agent_123"
        assert state.step_number == 42
        assert state.position == (10.0, 20.0, 5.0)
        assert state.resource_level == 0.75
        assert state.current_health == 0.9
        assert state.is_defending is True
        assert state.total_reward == 150.0
        assert state.age == 200
        assert state.role == "defender"  # Based on is_defending=True
        assert state.health == 0.9  # Should equal current_health
        assert state.energy == 0.75  # Should equal resource_level

    def test_get_feature_names(self):
        """Test getting feature names for tensor representation."""
        state = AgentState()
        feature_names = state.get_feature_names()

        assert len(feature_names) == 15  # Expected number of features
        assert "position_x" in feature_names
        assert "position_y" in feature_names
        assert "position_z" in feature_names
        assert "health" in feature_names
        assert "energy" in feature_names
        assert "role_explorer" in feature_names
        assert "role_defender" in feature_names

    @pytest.mark.skipif(
        AgentStateToGraph is None, reason="knowledge_graph module not available"
    )
    def test_to_graph_and_from_graph(self):
        """Test conversion to graph and back."""
        try:
            original_state = AgentState(
                position=(1.0, 2.0, 3.0), health=0.8, energy=0.7, role="explorer"
            )

            # Convert to graph
            graph = original_state.to_graph()
            assert isinstance(graph, nx.Graph)
            assert len(graph.nodes) > 0

            # Convert back from graph
            reconstructed_state = AgentState.from_graph(graph)
            assert reconstructed_state.agent_id is not None
        except ImportError:
            pytest.skip("knowledge_graph module not available")

    @pytest.mark.skipif(
        AgentStateToGraph is None, reason="knowledge_graph module not available"
    )
    def test_to_torch_geometric(self):
        """Test conversion to PyTorch Geometric Data object."""
        try:
            from torch_geometric.data import Data

            state = AgentState(
                position=(1.0, 2.0, 3.0), health=0.8, energy=0.7, role="explorer"
            )

            data_obj = state.to_torch_geometric()
            assert isinstance(data_obj, Data)
            assert hasattr(data_obj, "x")
            assert hasattr(data_obj, "edge_index")
        except (ImportError, ModuleNotFoundError):
            pytest.skip("torch_geometric or knowledge_graph module not available")


# Test AgentStateDataset class
class TestAgentStateDataset:
    """Tests for the AgentStateDataset class."""

    def test_init_and_batch(self):
        """Test initialization and getting batches."""
        states = generate_agent_states(count=50)
        dataset = AgentStateDataset(states=states, batch_size=16)

        # Test batch retrieval
        batch = dataset.get_batch()
        assert isinstance(batch, torch.Tensor)
        assert batch.shape[0] == 16  # Batch size
        assert batch.shape[1] == 15  # Feature count

    def test_loading_saving(self, tmp_path):
        """Test saving and loading agent states."""
        # Create temporary file
        temp_file = tmp_path / "test_states.pkl"

        # Generate states
        original_states = generate_agent_states(count=20)

        # Save to file using serialization
        with open(temp_file, "wb") as f:
            f.write(serialize_states(original_states))

        # Create new dataset and load from file
        dataset = AgentStateDataset()
        dataset.load_from_file(temp_file)

        assert len(dataset.states) == 20
        assert isinstance(dataset.states[0], AgentState)

    @pytest.mark.skipif(
        AgentStateToGraph is None, reason="knowledge_graph module not available"
    )
    def test_to_graph_dataset(self):
        """Test conversion to graph dataset."""
        try:
            from torch_geometric.data import Data

            states = generate_agent_states(count=10)
            dataset = AgentStateDataset(states=states)

            graph_dataset = dataset.to_graph_dataset()
            assert len(graph_dataset) == 10
            assert all(isinstance(data, Data) for data in graph_dataset)
        except (ImportError, ModuleNotFoundError):
            pytest.skip("torch_geometric or knowledge_graph module not available")

    @pytest.mark.skipif(
        AgentStateToGraph is None, reason="knowledge_graph module not available"
    )
    def test_to_multi_agent_graph(self):
        """Test conversion to multi-agent graph."""
        try:
            from torch_geometric.data import Data

            states = generate_agent_states(count=5)
            dataset = AgentStateDataset(states=states)

            multi_graph = dataset.to_multi_agent_graph()
            assert isinstance(multi_graph, Data)
            assert hasattr(multi_graph, "x")
            assert hasattr(multi_graph, "edge_index")
        except (ImportError, ModuleNotFoundError):
            pytest.skip("torch_geometric or knowledge_graph module not available")

    @pytest.mark.skipif(
        AgentStateToGraph is None, reason="knowledge_graph module not available"
    )
    def test_get_graph_batch(self):
        """Test getting graph batches."""
        try:
            from torch_geometric.data import Batch, Data

            states = generate_agent_states(count=20)
            dataset = AgentStateDataset(states=states, batch_size=5)

            # Test with multi-agent graph (small batch)
            batch1 = dataset.get_graph_batch(batch_size=5)
            assert isinstance(batch1, Data)

            # Test with batched individual graphs (larger batch)
            batch2 = dataset.get_graph_batch(batch_size=15)
            assert isinstance(batch2, Batch)
        except (ImportError, ModuleNotFoundError):
            pytest.skip("torch_geometric or knowledge_graph module not available")


# Test helper functions
class TestHelperFunctions:
    """Tests for helper functions in the data module."""

    def test_determine_role(self):
        """Test role determination from record."""
        defender_record = {"is_defending": True}
        explorer_record = {"is_defending": False}

        assert determine_role(defender_record) == "defender"
        assert determine_role(explorer_record) == "explorer"

    def test_serialize_deserialize_states(self):
        """Test serialization and deserialization of multiple states."""
        original_states = generate_agent_states(count=10)

        # Serialize
        binary_data = serialize_states(original_states)
        assert isinstance(binary_data, bytes)

        # Deserialize
        reconstructed_states = deserialize_states(binary_data)
        assert len(reconstructed_states) == 10
        assert all(isinstance(state, AgentState) for state in reconstructed_states)

        # Verify first state properties match
        assert reconstructed_states[0].position == original_states[0].position
        assert reconstructed_states[0].health == original_states[0].health
        assert reconstructed_states[0].role == original_states[0].role

    def test_generate_agent_states(self):
        """Test synthetic agent state generation."""
        # Test with random seed for reproducibility
        states1 = generate_agent_states(count=5, random_seed=42)
        states2 = generate_agent_states(count=5, random_seed=42)

        assert len(states1) == 5
        assert len(states2) == 5
        assert all(isinstance(state, AgentState) for state in states1)

        # With same random seed, states should be identical
        for i in range(5):
            assert states1[i].position == states2[i].position
            assert states1[i].health == states2[i].health
            assert states1[i].energy == states2[i].energy
            assert states1[i].role == states2[i].role


if __name__ == "__main__":
    pytest.main(["-v", __file__])
