#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pytest configuration file with fixtures for testing.
"""

import os
import pytest
import tempfile

from meaning_transform.src.config import (
    Config, ModelConfig, TrainingConfig, DataConfig, MetricsConfig
)


@pytest.fixture
def custom_model_config():
    """Return a custom ModelConfig for testing."""
    return ModelConfig(
        input_dim=256,
        latent_dim=64,
        encoder_hidden_dims=[512, 256, 128],
        compression_type="vq"
    )


@pytest.fixture
def custom_training_config():
    """Return a custom TrainingConfig for testing."""
    return TrainingConfig(
        batch_size=128,
        learning_rate=5e-4,
        optimizer="sgd",
        num_epochs=200
    )


@pytest.fixture
def custom_data_config():
    """Return a custom DataConfig for testing."""
    return DataConfig(
        num_states=5000,
        validation_split=0.15,
        db_path="custom_simulation.db",
        position_range=(-20.0, 20.0)
    )


@pytest.fixture
def custom_metrics_config():
    """Return a custom MetricsConfig for testing."""
    return MetricsConfig(
        drift_tracking_interval=10,
        feature_extractors=["role", "health", "inventory"],
        semantic_similarity_type="euclidean"
    )


@pytest.fixture
def temp_config_file():
    """Create a temporary file for config I/O testing and clean it up afterward."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_file:
        temp_path = temp_file.name
        
    yield temp_path
    
    if os.path.exists(temp_path):
        os.remove(temp_path) 