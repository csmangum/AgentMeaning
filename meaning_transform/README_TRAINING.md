# Training Infrastructure for Meaning-Preserving Transformations

This document explains how to use the training infrastructure to train and evaluate meaning-preserving transformation models.

## Overview

The training infrastructure provides:

1. A complete training loop with metrics tracking
2. Checkpointing and resumption of training
3. Semantic drift tracking
4. Visualization of training progress
5. Early stopping based on validation metrics

## Quick Start

To train a model with default settings:

```bash
# From the project root directory
python -m meaning_transform.examples.train_model --experiment my_first_experiment
```

To train with specific configuration:

```bash
python -m meaning_transform.examples.train_model \
    --experiment custom_training \
    --compression entropy \
    --compression-level 2.0 \
    --epochs 100 \
    --batch-size 64 \
    --latent-dim 32 \
    --learning-rate 0.001 \
    --gpu
```

## Training Options

The training script supports the following options:

- `--experiment`: Experiment name (used for organization)
- `--compression`: Compression type ("entropy", "vq", or "none")
- `--compression-level`: Compression level for entropy bottleneck (higher = more compression)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--latent-dim`: Dimension of latent space
- `--learning-rate`: Learning rate for optimizer
- `--use-real-data`: Use real data from simulation.db instead of synthetic data
- `--num-states`: Number of synthetic states to generate (if not using real data)
- `--resume`: Path to checkpoint to resume training from
- `--gpu`: Use GPU for training if available
- `--debug`: Enable debug mode with verbose logging

## Training Output

The training process creates an experiment directory with:

1. Checkpoints (saved every N epochs and on best validation loss)
2. Configuration JSON file
3. Training curve plots
4. Semantic drift visualization

Example output directory structure:

```
results/
└── checkpoints/
    └── my_experiment_20230328_120000/
        ├── config.json
        ├── checkpoint_epoch_10.pt
        ├── checkpoint_epoch_20.pt
        ├── ...
        ├── best_model.pt
        ├── latest_model.pt
        ├── training_curves.png
        └── semantic_drift.png
```

## Resuming Training

To resume training from a checkpoint:

```bash
python -m meaning_transform.examples.train_model \
    --experiment resumed_training \
    --resume /path/to/checkpoint.pt \
    --epochs 150  # Total epochs to train
```

## Using the Trainer API

You can also use the Trainer API directly in your code:

```python
from meaning_transform.src.config import Config
from meaning_transform.src.train import Trainer

# Create configuration
config = Config()
config.model.latent_dim = 32
config.training.num_epochs = 100
config.data.num_states = 10000

# Create trainer
trainer = Trainer(config)

# Train model
results = trainer.train()

# Access training history
train_losses = results["train_losses"]
val_losses = results["val_losses"]
semantic_drift = results["semantic_drift"]
best_val_loss = results["best_val_loss"]
```

## Testing the Training Infrastructure

To run tests that verify the training infrastructure works correctly:

```bash
python -m meaning_transform.test_training
```

## Customizing Training

The training infrastructure is designed to be customizable:

1. Modify `Config` in `src/config.py` to add new options
2. Extend `Trainer` class in `src/train.py` to add custom functionality
3. Create custom loss functions in `src/loss.py`

## Advanced Usage

### Multi-Stage Training

For advanced training workflows:

```python
# Create trainer with initial config
trainer = Trainer(config)

# Stage 1: Train with high KL weight
config.training.kl_loss_weight = 1.0
stage1_results = trainer.train()

# Stage 2: Fine-tune with high semantic weight
config.training.kl_loss_weight = 0.1
config.training.semantic_loss_weight = 1.0
best_model_path = f"{stage1_results['experiment_dir']}/best_model.pt"
stage2_results = trainer.train(resume_from=best_model_path)
```

### Tracking Custom Metrics

You can extend the trainer to track custom metrics during training. 