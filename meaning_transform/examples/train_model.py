#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script for training the meaning-preserving transformation model.

This script demonstrates how to:
1. Configure the training process
2. Train a model with different compression methods
3. Resume training from a checkpoint
4. Analyze training results
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from meaning_transform.src.config import Config
from meaning_transform.src.train import Trainer
from meaning_transform.src.data import load_from_simulation_db, AgentState


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train meaning-preserving transformation model")
    
    parser.add_argument("--experiment", type=str, default="default", 
                        help="Experiment name")
    parser.add_argument("--compression", type=str, default="entropy", 
                        choices=["entropy", "vq", "none"],
                        help="Compression type: entropy, vq, or none")
    parser.add_argument("--compression-level", type=float, default=1.0,
                        help="Compression level (for entropy bottleneck)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--latent-dim", type=int, default=32,
                        help="Dimension of latent space")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--use-real-data", action="store_true",
                        help="Use real data from simulation.db instead of synthetic data")
    parser.add_argument("--num-states", type=int, default=10000,
                        help="Number of synthetic states to generate (if not using real data)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for training")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    
    return parser.parse_args()


def create_config(args):
    """Create configuration from command line arguments."""
    config = Config()
    
    # Set experiment name
    config.experiment_name = args.experiment
    
    # Determine the input dimension by creating a sample agent state and checking its tensor size
    sample_state = AgentState()
    input_dim = sample_state.to_tensor().shape[0]
    
    # Model configuration
    config.model.input_dim = input_dim  # Set input dimension based on actual agent state tensor
    config.model.latent_dim = args.latent_dim
    config.model.compression_type = args.compression
    config.model.compression_level = args.compression_level
    
    # Set smaller hidden dimensions for the encoder and decoder to fix dimension mismatch
    config.model.encoder_hidden_dims = [64, 48, 32]
    config.model.decoder_hidden_dims = [32, 48, 64]
    
    # Training configuration
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    
    # Data configuration
    config.data.num_states = args.num_states
    
    # Runtime flags
    config.debug = args.debug
    config.use_gpu = args.gpu
    
    if config.debug:
        print(f"Using input dimension: {input_dim}")
        print(f"Using encoder hidden dimensions: {config.model.encoder_hidden_dims}")
        print(f"Using decoder hidden dimensions: {config.model.decoder_hidden_dims}")
    
    return config


def main():
    """Run training with specified configuration."""
    args = parse_args()
    config = create_config(args)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Use real data if specified
    if args.use_real_data:
        if os.path.exists("simulation.db"):
            print("Loading agent states from simulation.db...")
            # Custom data loading code here if needed
            pass
        else:
            print("Warning: simulation.db not found. Using synthetic data instead.")
    
    # Train model
    resume_path = args.resume
    if resume_path is not None and not os.path.exists(resume_path):
        print(f"Warning: Checkpoint {resume_path} not found. Starting from scratch.")
        resume_path = None
        
    training_results = trainer.train(resume_from=resume_path)
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {training_results['experiment_dir']}")
    print(f"Best validation loss: {training_results['best_val_loss']:.4f}")
    
    return training_results


if __name__ == "__main__":
    main() 