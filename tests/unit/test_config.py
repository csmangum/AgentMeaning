#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the configuration module.
"""

import os
import pytest
import tempfile
from meaning_transform.src.config import (
    Config, ModelConfig, TrainingConfig, DataConfig, MetricsConfig, 
    default_config, load_config, save_config
)

class TestDefaultConfig:
    """Test the default configuration values."""
    
    def test_default_model_config(self):
        """Test that default model config values are set correctly."""
        config = default_config.model
        
        assert config.input_dim == 128
        assert config.latent_dim == 32
        assert config.encoder_hidden_dims == [256, 128, 64]
        assert config.decoder_hidden_dims == [64, 128, 256]
        assert config.compression_type == "entropy"
        assert config.compression_level == 1.0
        assert config.vq_num_embeddings == 512
        assert config.vq_commitment_cost == 0.25
    
    def test_default_training_config(self):
        """Test that default training config values are set correctly."""
        config = default_config.training
        
        assert config.batch_size == 64
        assert config.learning_rate == 1e-3
        assert config.weight_decay == 1e-6
        assert config.num_epochs == 100
        assert config.recon_loss_weight == 1.0
        assert config.kl_loss_weight == 0.1
        assert config.semantic_loss_weight == 0.5
        assert config.optimizer == "adam"
        assert config.scheduler == "cosine"
        assert config.scheduler_step_size == 30
        assert config.scheduler_gamma == 0.5
        assert config.checkpoint_dir == "../results/checkpoints"
        assert config.checkpoint_interval == 10
        assert config.patience == 10
        assert config.min_delta == 1e-4
    
    def test_default_data_config(self):
        """Test that default data config values are set correctly."""
        config = default_config.data
        
        assert config.num_states == 10000
        assert config.validation_split == 0.2
        assert config.test_split == 0.1
        assert config.db_path == "simulation.db"
        assert config.position_range == (-10.0, 10.0)
        assert config.health_range == (0.0, 1.0)
        assert config.energy_range == (0.0, 1.0)
        assert config.num_roles == 5
        assert config.max_goals == 3
        assert config.max_inventory_items == 10
        assert config.use_augmentation is True
        assert config.augmentation_noise_level == 0.05
    
    def test_default_metrics_config(self):
        """Test that default metrics config values are set correctly."""
        config = default_config.metrics
        
        assert config.drift_tracking_interval == 5
        assert config.drift_threshold == 0.1
        assert config.feature_extractors == ["role", "position", "goals"]
        assert config.semantic_similarity_type == "cosine"
        assert config.use_tsne is True
        assert config.tsne_perplexity == 30.0
        assert config.use_pca is True
        assert config.pca_components == 2


class TestCustomConfig:
    """Test custom configuration overrides."""
    
    def test_custom_model_config(self, custom_model_config):
        """Test that custom model config values override defaults."""
        config = Config(model=custom_model_config)
        
        assert config.model.input_dim == 256
        assert config.model.latent_dim == 64
        assert config.model.encoder_hidden_dims == [512, 256, 128]
        assert config.model.compression_type == "vq"
        # Unchanged values should remain as defaults
        assert config.model.decoder_hidden_dims == [64, 128, 256]
        assert config.model.compression_level == 1.0
    
    def test_custom_training_config(self, custom_training_config):
        """Test that custom training config values override defaults."""
        config = Config(training=custom_training_config)
        
        assert config.training.batch_size == 128
        assert config.training.learning_rate == 5e-4
        assert config.training.optimizer == "sgd"
        assert config.training.num_epochs == 200
        # Unchanged values should remain as defaults
        assert config.training.weight_decay == 1e-6
        assert config.training.recon_loss_weight == 1.0
    
    def test_custom_data_config(self, custom_data_config):
        """Test that custom data config values override defaults."""
        config = Config(data=custom_data_config)
        
        assert config.data.num_states == 5000
        assert config.data.validation_split == 0.15
        assert config.data.db_path == "custom_simulation.db"
        assert config.data.position_range == (-20.0, 20.0)
        # Unchanged values should remain as defaults
        assert config.data.test_split == 0.1
        assert config.data.max_goals == 3
    
    def test_custom_metrics_config(self, custom_metrics_config):
        """Test that custom metrics config values override defaults."""
        config = Config(metrics=custom_metrics_config)
        
        assert config.metrics.drift_tracking_interval == 10
        assert config.metrics.feature_extractors == ["role", "health", "inventory"]
        assert config.metrics.semantic_similarity_type == "euclidean"
        # Unchanged values should remain as defaults
        assert config.metrics.drift_threshold == 0.1
        assert config.metrics.use_tsne is True


class TestConfigIO:
    """Test configuration serialization and deserialization."""
    
    @pytest.mark.skip(reason="load_config not fully implemented yet")
    def test_load_config(self, temp_config_file):
        """Test loading configuration from file."""
        # This test should be implemented once load_config is completed
        pass
    
    @pytest.mark.skip(reason="save_config not fully implemented yet")
    def test_save_config(self, temp_config_file):
        """Test saving configuration to file."""
        # This test should be implemented once save_config is completed
        pass
    
    @pytest.mark.skip(reason="save_config and load_config not fully implemented yet")
    def test_round_trip(self, temp_config_file, custom_model_config):
        """Test that saving and loading a config preserves all values."""
        # Create a custom config
        original_config = Config(model=custom_model_config, debug=True)
        
        # Save and reload
        save_config(original_config, temp_config_file)
        loaded_config = load_config(temp_config_file)
        
        # Verify all values are preserved
        assert loaded_config.model.input_dim == 256
        assert loaded_config.model.latent_dim == 64
        assert loaded_config.debug is True
        
        # Check other configs remain as defaults
        assert loaded_config.training.batch_size == 64
        assert loaded_config.data.num_states == 10000
        assert loaded_config.metrics.drift_threshold == 0.1


if __name__ == "__main__":
    pytest.main() 