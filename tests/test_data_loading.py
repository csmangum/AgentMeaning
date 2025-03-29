#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for data handling functionality.

This script tests:
1. Loading agent states from the simulation database
2. Converting agent states to binary format and back
3. Creating tensor representations of agent states
4. Generating synthetic agent states
"""

import os
import random
import sys
from pathlib import Path

import torch

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meaning_transform.src.data import (
    AgentState,
    AgentStateDataset,
    deserialize_states,
    generate_agent_states,
    load_from_simulation_db,
    serialize_states,
)


def test_load_from_db():
    """Test loading agent states from the simulation database."""
    db_path = "../simulation.db"

    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}")
        return False

    # Load a limited number of states for testing
    dataset = AgentStateDataset()
    dataset.load_from_db(db_path, limit=100)

    if not dataset.states:
        print("Failed to load any states from the database")
        return False

    print(f"Successfully loaded {len(dataset.states)} states from the database")

    # Print sample state
    sample_state = dataset.states[0]
    print(f"Sample state: {sample_state.to_dict()}")

    return True


def test_binary_serialization():
    """Test binary serialization and deserialization of agent states."""
    # Generate a synthetic state
    states = generate_agent_states(10)

    # Convert to binary
    binary_data = serialize_states(states)
    print(f"Serialized {len(states)} states to {len(binary_data)} bytes of binary data")

    # Convert back from binary
    reconstructed_states = deserialize_states(binary_data)
    print(f"Deserialized {len(reconstructed_states)} states from binary data")

    # Check if we got the same number of states back
    assert len(states) == len(
        reconstructed_states
    ), "State count mismatch after deserialization"

    # Check a few fields to verify data integrity
    for i, (orig, recon) in enumerate(zip(states, reconstructed_states)):
        assert orig.position == recon.position, f"Position mismatch in state {i}"
        assert orig.health == recon.health, f"Health mismatch in state {i}"
        assert orig.energy == recon.energy, f"Energy mismatch in state {i}"

    print("Binary serialization test passed")
    return True


def test_individual_binary_conversion():
    """Test binary conversion for individual agent states."""
    # Create a test state
    state = AgentState(
        position=(10.5, -20.3, 0.0),
        health=0.75,
        energy=0.8,
        inventory={"food": 5, "medicine": 2},
        role="defender",
        goals=["protect_base", "gather_intelligence"],
        agent_id="test_agent_1",
        step_number=42,
        resource_level=65.3,
        current_health=0.7,
        is_defending=True,
        age=123,
        total_reward=45.2,
    )

    # Convert to binary
    binary_data = state.to_binary()
    print(f"Serialized state to {len(binary_data)} bytes of binary data")

    # Convert back from binary
    reconstructed_state = AgentState.from_binary(binary_data)

    # Check fields
    assert state.position == reconstructed_state.position, "Position mismatch"
    assert state.health == reconstructed_state.health, "Health mismatch"
    assert state.role == reconstructed_state.role, "Role mismatch"
    assert state.goals == reconstructed_state.goals, "Goals mismatch"
    assert state.agent_id == reconstructed_state.agent_id, "Agent ID mismatch"
    assert (
        state.is_defending == reconstructed_state.is_defending
    ), "Is defending mismatch"

    print("Individual binary conversion test passed")
    return True


def test_tensor_conversion():
    """Test converting agent states to tensor representation."""
    # Generate some states
    states = generate_agent_states(5)

    # Convert to tensors
    tensors = [state.to_tensor() for state in states]

    # Check tensor shapes are consistent
    tensor_shape = tensors[0].shape
    for i, tensor in enumerate(tensors):
        assert tensor.shape == tensor_shape, f"Tensor shape mismatch for state {i}"

    # Create a batch tensor
    batch_tensor = torch.stack(tensors)
    expected_shape = (5, tensor_shape[0])
    assert (
        batch_tensor.shape == expected_shape
    ), f"Batch tensor shape {batch_tensor.shape} doesn't match expected {expected_shape}"

    print(f"Tensor conversion test passed. Tensor shape: {tensor_shape}")
    return True


def test_dataset_batch_retrieval():
    """Test retrieving batches from the dataset."""
    # Create dataset with synthetic states
    dataset = AgentStateDataset(batch_size=4)
    dataset.generate_synthetic_data(num_states=10)

    # Get a batch
    batch1 = dataset.get_batch()
    print(f"Retrieved batch with shape: {batch1.shape}")

    # Get another batch
    batch2 = dataset.get_batch()
    print(f"Retrieved second batch with shape: {batch2.shape}")

    # Should have retrieved all states now, next batch should loop back
    batch3 = dataset.get_batch()
    print(f"Retrieved third batch with shape: {batch3.shape}")

    return True


if __name__ == "__main__":
    print("Running data handling tests...")

    # Skip DB loading test if no database is available
    db_path = "../simulation.db"
    has_db = os.path.exists(db_path)

    test_results = {
        "DB Loading": test_load_from_db() if has_db else "SKIPPED",
        "Batch Binary Serialization": test_binary_serialization(),
        "Individual Binary Conversion": test_individual_binary_conversion(),
        "Tensor Conversion": test_tensor_conversion(),
        "Dataset Batch Retrieval": test_dataset_batch_retrieval(),
    }

    print("\nTest Results:")
    for test_name, result in test_results.items():
        if result == "SKIPPED":
            status = "SKIPPED"
        else:
            status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
