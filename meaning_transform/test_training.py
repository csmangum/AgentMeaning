#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the training infrastructure.

This script verifies that:
1. The trainer initializes correctly
2. Training works for a small synthetic dataset
3. Checkpointing and resuming work properly
4. Semantic drift tracking functions as expected
"""

import sys
import torch
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path

# Add the project root to the path so imports work correctly
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from meaning_transform.src.config import Config
    from meaning_transform.src.train import Trainer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("\nTrying alternative import paths...")
    try:
        from src.config import Config
        from src.train import Trainer
        print("Successfully imported using relative paths.")
    except ImportError:
        print("Import failed. Check the project structure and import paths.")
        sys.exit(1)


def setup_test_config():
    """Create a minimal configuration for testing."""
    config = Config()
    
    # Override with minimal values for quick testing
    config.experiment_name = "test_training"
    
    # Model config
    config.model.input_dim = 15
    config.model.latent_dim = 8
    config.model.encoder_hidden_dims = [64, 32]
    config.model.decoder_hidden_dims = [32, 64]
    
    # Training config
    config.training.batch_size = 16
    config.training.learning_rate = 1e-3
    config.training.num_epochs = 2
    config.training.checkpoint_interval = 1
    
    # Data config
    config.data.num_states = 100
    config.data.validation_split = 0.2
    
    # Use temporary directory for checkpoints
    temp_dir = tempfile.mkdtemp()
    config.training.checkpoint_dir = temp_dir
    
    # Ensure deterministic behavior
    config.seed = 42
    
    # Force CPU usage for testing to avoid CUDA errors
    config.use_gpu = False
    
    return config, temp_dir


def test_trainer_initialization():
    """Test that the trainer initializes correctly."""
    print("\n=== Testing trainer initialization ===")
    
    config, temp_dir = setup_test_config()
    
    try:
        trainer = Trainer(config)
        print("✓ Trainer initialized successfully")
        
        # Check that model was created correctly
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.loss_fn is not None
        
        print("✓ Model, optimizer, and loss function created successfully")
        
        # Check experiment directory
        assert trainer.experiment_dir.exists()
        assert (trainer.experiment_dir / "config.json").exists()
        
        print("✓ Experiment directory and config file created successfully")
        
    except Exception as e:
        print(f"✗ Trainer initialization failed: {e}")
        raise
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_training_loop():
    """Test that the training loop runs without errors."""
    print("\n=== Testing training loop ===")
    
    config, temp_dir = setup_test_config()
    
    try:
        trainer = Trainer(config)
        
        # Run training for a minimal number of epochs
        trainer.prepare_data()
        print("✓ Data preparation successful")
        
        # Run a single training epoch
        train_metrics = trainer.train_epoch()
        print("✓ Training epoch completed")
        
        # Check metrics
        assert "loss" in train_metrics
        assert "recon_loss" in train_metrics
        assert "kl_loss" in train_metrics
        assert "semantic_loss" in train_metrics
        
        print("✓ Training metrics computed correctly")
        
        # Test validation
        val_metrics = trainer.validate()
        print("✓ Validation completed")
        
        # Check semantic drift tracking
        drift_metrics = trainer.track_semantic_drift()
        assert "total_semantic_loss" in drift_metrics
        assert "feature_losses" in drift_metrics
        print("✓ Semantic drift tracking works")
        
        # Test checkpointing
        trainer.save_checkpoint(1, val_metrics, is_best=True)
        assert (trainer.experiment_dir / "checkpoint_epoch_1.pt").exists()
        assert (trainer.experiment_dir / "best_model.pt").exists()
        print("✓ Checkpointing works")
        
    except Exception as e:
        print(f"✗ Training loop test failed: {e}")
        raise
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_full_training():
    """Test a full training run with checkpointing and resuming."""
    print("\n=== Testing full training ===")
    
    config, temp_dir = setup_test_config()
    
    try:
        # First training run
        trainer = Trainer(config)
        training_history = trainer.train()
        
        print("✓ First training run completed")
        
        # Check that history contains expected data
        assert "train_losses" in training_history
        assert "val_losses" in training_history
        assert "semantic_drift" in training_history
        assert "best_val_loss" in training_history
        
        best_model_path = str(trainer.experiment_dir / "best_model.pt")
        
        # Modify config for resumed training to do only one more epoch
        new_config = Config()
        new_config.experiment_name = "test_training_resumed"
        new_config.model.input_dim = config.model.input_dim
        new_config.model.latent_dim = config.model.latent_dim
        new_config.model.encoder_hidden_dims = config.model.encoder_hidden_dims
        new_config.model.decoder_hidden_dims = config.model.decoder_hidden_dims
        new_config.training.batch_size = config.training.batch_size
        new_config.training.learning_rate = config.training.learning_rate
        new_config.training.num_epochs = 3  # One more epoch
        new_config.training.checkpoint_interval = config.training.checkpoint_interval
        new_config.data.num_states = config.data.num_states
        new_config.data.validation_split = config.data.validation_split
        new_config.training.checkpoint_dir = temp_dir
        new_config.seed = config.seed
        new_config.use_gpu = config.use_gpu
        
        # Ensure epoch integrity
        checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'))
        start_epoch = checkpoint["epoch"]
        
        # Resume training
        trainer2 = Trainer(new_config)
        training_history2 = trainer2.train(resume_from=best_model_path)
        
        print("✓ Resumed training completed")
        
        # Check that the resumed training history is valid
        assert "train_losses" in training_history2
        assert "val_losses" in training_history2
        assert "semantic_drift" in training_history2
        assert "best_val_loss" in training_history2
        assert "experiment_dir" in training_history2
        
    except Exception as e:
        print(f"✗ Full training test failed: {e}")
        raise
    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def run_all_tests():
    """Run all tests in sequence."""
    print("\n=== Running all training infrastructure tests ===")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        test_trainer_initialization()
        test_training_loop()
        test_full_training()
        
        print("\n=== All tests passed successfully! ===")
    except Exception as e:
        print(f"\n=== Tests failed: {e} ===")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests() 