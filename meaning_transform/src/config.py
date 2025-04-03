#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration module for the meaning-preserving transformation system.

This module defines all hyperparameters, runtime flags, and configuration options
used throughout the project.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple


@dataclass
class ModelConfig:
    """Configuration for the VAE model architecture."""
    
    # Input/output dimensions
    input_dim: int = 128
    latent_dim: int = 32
    
    # Network architecture
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    decoder_hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    
    # Compression options
    compression_type: str = "entropy"  # "entropy" or "vq"
    compression_level: float = 1.0
    vq_num_embeddings: int = 512
    vq_commitment_cost: float = 0.25


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Basic training parameters
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    num_epochs: int = 100
    
    # Loss weights
    recon_loss_weight: float = 1.0
    kl_loss_weight: float = 0.1
    semantic_loss_weight: float = 0.5
    
    # Optimizer and scheduler
    optimizer: str = "adam"  # "adam" or "sgd"
    scheduler: str = "cosine"  # "cosine", "step", or "none"
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.5
    
    # Checkpointing
    checkpoint_dir: str = "../results/checkpoints"
    checkpoint_interval: int = 10
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4


@dataclass
class DataConfig:
    """Configuration for data generation and processing."""
    
    # Agent state parameters
    num_states: int = 10000
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Database configuration
    db_path: str = "simulation.db"
    
    # Agent state properties
    position_range: Tuple[float, float] = (-10.0, 10.0)
    health_range: Tuple[float, float] = (0.0, 1.0)
    energy_range: Tuple[float, float] = (0.0, 1.0)
    num_roles: int = 5
    max_goals: int = 3
    max_inventory_items: int = 10
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_noise_level: float = 0.05


@dataclass
class MetricsConfig:
    """Configuration for evaluation metrics and semantic feature extraction."""
    
    # Drift tracking
    drift_tracking_interval: int = 5
    drift_threshold: float = 0.1
    
    # Semantic feature extractors
    feature_extractors: List[str] = field(default_factory=lambda: ["role", "position", "goals"])
    semantic_similarity_type: str = "cosine"  # "cosine", "euclidean", or "manhattan"
    
    # Visualization
    visualization_interval: int = 5
    use_tsne: bool = True
    tsne_perplexity: float = 30.0
    use_pca: bool = True
    pca_components: int = 2


@dataclass
class Config:
    """Master configuration class that contains all other configs."""
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    
    # Runtime flags
    debug: bool = False
    verbose: bool = True
    use_gpu: bool = True
    seed: int = 42
    experiment_name: str = "default"
    

# Default configuration
default_config = Config()


def load_config(config_path: str) -> Config:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
        
    Returns:
        Loaded configuration object
    """
    # TODO: Implement config loading from file
    return default_config


def save_config(config: Config, config_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration object to save
        config_path: Path to save configuration to
    """
    # TODO: Implement config saving to file
    pass 