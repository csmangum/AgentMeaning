#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agent state generation and serialization module.

This module handles:
1. Synthetic agent state generation
2. Serialization/deserialization of agent states
3. Data loading and batching for training
4. Loading real agent states from simulation database
"""

import json
import pickle
import random
import sqlite3
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class AgentState:
    """Class representing an agent's state with semantic properties."""

    def __init__(
        self,
        position: Tuple[float, float, float] = None,
        health: float = None,
        energy: float = None,
        inventory: Dict[str, int] = None,
        role: str = None,
        goals: List[str] = None,
        agent_id: str = None,
        step_number: int = None,
        resource_level: float = None,
        current_health: float = None,
        is_defending: bool = False,
        age: int = None,
        total_reward: float = None,
        **kwargs,
    ):
        """
        Initialize an agent state with semantic properties.

        Args:
            position: 3D position coordinates (x, y, z)
            health: Health level (0.0-1.0)
            energy: Energy level (0.0-1.0)
            inventory: Dictionary of items and quantities
            role: Agent's role in the environment
            goals: List of current goals
            agent_id: Unique identifier for the agent
            step_number: Simulation step number
            resource_level: Available resources
            current_health: Current health level
            is_defending: Whether agent is in defensive stance
            age: Agent's age in simulation steps
            total_reward: Cumulative reward received
            **kwargs: Additional properties
        """
        self.position = position or (0.0, 0.0, 0.0)
        self.health = health or 1.0
        self.energy = energy or 1.0
        self.inventory = inventory or {}
        self.role = role or "explorer"
        self.goals = goals or []
        self.agent_id = agent_id
        self.step_number = step_number
        self.resource_level = resource_level
        self.current_health = current_health or self.health
        self.is_defending = is_defending
        self.age = age or 0
        self.total_reward = total_reward or 0.0
        self.properties = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent state to dictionary representation."""
        return {
            "position": self.position,
            "health": self.health,
            "energy": self.energy,
            "inventory": self.inventory,
            "role": self.role,
            "goals": self.goals,
            "agent_id": self.agent_id,
            "step_number": self.step_number,
            "resource_level": self.resource_level,
            "current_health": self.current_health,
            "is_defending": self.is_defending,
            "age": self.age,
            "total_reward": self.total_reward,
            **self.properties,
        }

    def to_binary(self) -> bytes:
        """Serialize agent state to binary format."""
        # Convert to JSON string first
        state_dict = self.to_dict()

        # JSON doesn't natively support tuples, so we need to mark them
        # Convert position tuple to a special format for proper deserialization
        if isinstance(state_dict["position"], tuple):
            state_dict["position"] = {
                "_type": "tuple",
                "value": list(state_dict["position"]),
            }

        json_str = json.dumps(state_dict)

        # Create a binary header with format:
        # 4 bytes: length of JSON data
        header = struct.pack("!I", len(json_str))

        # Combine header and JSON data
        return header + json_str.encode("utf-8")

    @classmethod
    def from_binary(cls, data: bytes) -> "AgentState":
        """Deserialize agent state from binary format."""
        # Extract header
        header_size = struct.calcsize("!I")
        json_size = struct.unpack("!I", data[:header_size])[0]

        # Extract and decode JSON string
        json_bytes = data[header_size : header_size + json_size]
        state_dict = json.loads(json_bytes.decode("utf-8"))

        # Convert special position format back to tuple
        if (
            isinstance(state_dict.get("position"), dict)
            and state_dict["position"].get("_type") == "tuple"
        ):
            state_dict["position"] = tuple(state_dict["position"]["value"])

        # Extract standard fields
        position = state_dict.pop("position", None)
        health = state_dict.pop("health", None)
        energy = state_dict.pop("energy", None)
        inventory = state_dict.pop("inventory", None)
        role = state_dict.pop("role", None)
        goals = state_dict.pop("goals", None)
        agent_id = state_dict.pop("agent_id", None)
        step_number = state_dict.pop("step_number", None)
        resource_level = state_dict.pop("resource_level", None)
        current_health = state_dict.pop("current_health", None)
        is_defending = state_dict.pop("is_defending", False)
        age = state_dict.pop("age", 0)
        total_reward = state_dict.pop("total_reward", 0.0)

        # Create object with remaining properties in kwargs
        return cls(
            position=position,
            health=health,
            energy=energy,
            inventory=inventory,
            role=role,
            goals=goals,
            agent_id=agent_id,
            step_number=step_number,
            resource_level=resource_level,
            current_health=current_health,
            is_defending=is_defending,
            age=age,
            total_reward=total_reward,
            **state_dict,
        )

    def to_tensor(self) -> torch.Tensor:
        """Convert agent state to tensor representation for model input."""
        # Create fixed-size feature vector
        features = []

        # Position (3 features)
        # Handle None position or None components
        if self.position is None:
            features.extend([0.0, 0.0, 0.0])
        else:
            # Handle None values in position tuple
            x = self.position[0] if self.position[0] is not None else 0.0
            y = self.position[1] if self.position[1] is not None else 0.0
            z = self.position[2] if self.position[2] is not None else 0.0
            features.extend([x, y, z])

        # Health and energy (2 features)
        features.append(self.health if self.health is not None else 1.0)
        features.append(self.energy if self.energy is not None else 1.0)

        # Resource level (1 feature)
        features.append(self.resource_level if self.resource_level is not None else 0.0)

        # Current health (1 feature)
        features.append(
            self.current_health if self.current_health is not None else (self.health if self.health is not None else 1.0)
        )

        # Is defending (1 feature)
        features.append(1.0 if self.is_defending else 0.0)

        # Age (1 feature, normalized)
        features.append(min(self.age / 1000.0, 1.0) if self.age is not None else 0.0)

        # Total reward (1 feature, normalized)
        features.append(
            max(min(self.total_reward / 100.0, 1.0), -1.0)
            if self.total_reward is not None
            else 0.0
        )

        # Role one-hot encoding (assuming 5 possible roles)
        roles = ["explorer", "gatherer", "defender", "attacker", "builder"]
        role_features = [1.0 if self.role == r else 0.0 for r in roles]
        features.extend(role_features)

        # Convert to tensor
        return torch.tensor(features, dtype=torch.float32)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "AgentState":
        """Convert tensor representation back to agent state.
        
        This is the inverse of to_tensor method.
        
        Args:
            tensor: Tensor representation of agent state
            
        Returns:
            Reconstructed AgentState object
        """
        # Convert tensor to list if it's not already
        if isinstance(tensor, torch.Tensor):
            features = tensor.tolist()
        else:
            features = tensor
            
        # Extract position (first 3 features)
        position = (features[0], features[1], features[2])
        
        # Extract health and energy (next 2 features)
        health = features[3]
        energy = features[4]
        
        # Extract resource level (next 1 feature)
        resource_level = features[5]
        
        # Extract current health (next 1 feature)
        current_health = features[6]
        
        # Extract is_defending (next 1 feature)
        is_defending = features[7] > 0.5
        
        # Extract age (next 1 feature, denormalize)
        age = int(features[8] * 1000)
        
        # Extract total reward (next 1 feature, denormalize)
        total_reward = features[9] * 100.0
        
        # Extract role from one-hot encoding (next 5 features)
        roles = ["explorer", "gatherer", "defender", "attacker", "builder"]
        role_features = features[10:15]
        role_index = role_features.index(max(role_features))
        role = roles[role_index]
        
        # Create agent state with extracted features
        return cls(
            position=position,
            health=health,
            energy=energy,
            role=role,
            resource_level=resource_level,
            current_health=current_health,
            is_defending=is_defending,
            age=age,
            total_reward=total_reward
        )

    @classmethod
    def from_db_record(cls, record: Dict[str, Any]) -> "AgentState":
        """Create an agent state from a database record."""
        return cls(
            position=(
                record.get("position_x", 0.0),
                record.get("position_y", 0.0),
                record.get("position_z", 0.0),
            ),
            health=record.get("current_health", 1.0),
            energy=record.get("resource_level", 1.0),  # Using resource_level as energy
            agent_id=record.get("agent_id"),
            step_number=record.get("step_number"),
            resource_level=record.get("resource_level"),
            current_health=record.get("current_health"),
            is_defending=record.get("is_defending", False),
            age=record.get("age", 0),
            total_reward=record.get("total_reward", 0.0),
            role=determine_role(record),  # Helper function to determine role
        )

    def get_feature_names(self) -> List[str]:
        """Return names of features in the tensor representation.
        
        Returns:
            List of feature names in the same order as the tensor representation
        """
        # Position features
        features = ['position_x', 'position_y', 'position_z']
        
        # Health and energy
        features.extend(['health', 'energy'])
        
        # Other numeric features
        features.extend(['resource_level', 'current_health', 'is_defending', 'age', 'total_reward'])
        
        # Role one-hot features
        roles = ["explorer", "gatherer", "defender", "attacker", "builder"]
        role_features = [f'role_{role}' for role in roles]
        features.extend(role_features)
        
        return features


class AgentStateDataset:
    """Dataset for loading and batching agent states."""

    def __init__(self, states=None, batch_size=32):
        """
        Initialize dataset with agent states.

        Args:
            states: List of AgentState objects
            batch_size: Batch size for training
        """
        self.states = states or []
        self.batch_size = batch_size
        self._current_idx = 0

    def get_batch(self) -> torch.Tensor:
        """Get a batch of agent states as tensors."""
        if not self.states:
            raise ValueError(
                "Dataset is empty. Load data from database first."
            )

        if self._current_idx >= len(self.states):
            self._current_idx = 0
            random.shuffle(self.states)

        batch_end = min(self._current_idx + self.batch_size, len(self.states))
        batch_states = self.states[self._current_idx : batch_end]
        self._current_idx = batch_end

        # Convert states to tensors
        tensors = [state.to_tensor() for state in batch_states]
        return torch.stack(tensors)

    def load_from_db(self, db_path: str, limit: Optional[int] = None) -> None:
        """
        Load agent states from simulation database.

        Args:
            db_path: Path to the simulation.db file
            limit: Maximum number of states to load
        """
        # Connect to database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build query with optional limit
        query = """
            SELECT * FROM agent_states
            ORDER BY step_number
        """
        if limit:
            query += f" LIMIT {limit}"

        # Execute query and convert results to AgentState objects
        cursor.execute(query)
        self.states = [
            AgentState.from_db_record(dict(row)) for row in cursor.fetchall()
        ]

        conn.close()
        print(f"Loaded {len(self.states)} agent states from database")


# Helper functions


def determine_role(record: Dict[str, Any]) -> str:
    """Determine agent role based on its attributes."""
    # Example logic - can be adjusted based on actual data patterns
    if record.get("is_defending", False):
        return "defender"

    # Other logic could be added based on agent attributes
    return "explorer"  # Default role


# Functions for data serialization and loading


def serialize_states(states: List[AgentState]) -> bytes:
    """Serialize a list of agent states to binary format."""
    # Convert states to list of dictionaries
    state_dicts = [state.to_dict() for state in states]

    # Use pickle for efficient binary serialization
    return pickle.dumps(state_dicts)


def deserialize_states(data: bytes) -> List[AgentState]:
    """Deserialize a list of agent states from binary format."""
    # Load state dictionaries from binary
    state_dicts = pickle.loads(data)

    # Convert dictionaries back to AgentState objects
    return [AgentState(**state_dict) for state_dict in state_dicts]


def load_from_simulation_db(
    db_path: str, limit: Optional[int] = None
) -> List[AgentState]:
    """
    Load agent states directly from simulation database.

    Args:
        db_path: Path to the simulation.db file
        limit: Maximum number of states to load

    Returns:
        List of AgentState objects
    """
    dataset = AgentStateDataset()
    dataset.load_from_db(db_path, limit)
    return dataset.states
