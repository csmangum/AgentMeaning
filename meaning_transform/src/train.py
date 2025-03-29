#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training infrastructure for the meaning-preserving transformation system.

This module defines:
1. Training loop
2. Logging and checkpointing
3. Semantic drift tracking
"""

import os
import time
import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import shutil

from .model import MeaningVAE
from .data import AgentStateDataset, AgentState
from .loss import CombinedLoss
from .config import Config
from .metrics import DriftTracker, SemanticMetrics, generate_t_sne_visualization


class Trainer:
    """Training infrastructure for the meaning-preserving transformation system."""
    
    def __init__(self, config: Config, device: str = None):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
            device: Device to train on ('cuda', 'cpu', or None for auto-detection)
        """
        self.config = config
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = MeaningVAE(
            input_dim=self.config.model.input_dim,
            latent_dim=self.config.model.latent_dim,
            compression_type=self.config.model.compression_type,
            compression_level=self.config.model.compression_level,
            vq_num_embeddings=self.config.model.vq_num_embeddings
        ).to(self.device)
        
        # Create loss function
        self.loss_fn = CombinedLoss(
            recon_loss_weight=self.config.training.recon_loss_weight,
            kl_loss_weight=self.config.training.kl_loss_weight,
            semantic_loss_weight=self.config.training.semantic_loss_weight
        )
        
        # Create optimizer
        if self.config.training.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer}")
            
        # Create learning rate scheduler
        if self.config.training.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs
            )
        elif self.config.training.scheduler == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.config.training.scheduler_step_size,
                gamma=self.config.training.scheduler_gamma
            )
        else:
            self.scheduler = None
            
        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.semantic_drift_history = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Create timestamped experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{self.config.experiment_name}_{timestamp}"
        self.experiment_dir = self.checkpoint_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize semantic metrics and drift tracker
        self.drift_tracker = DriftTracker(log_dir=str(self.experiment_dir / "drift_tracking"))
        self.semantic_metrics = SemanticMetrics()
        
        # Create subdirectories for results
        (self.experiment_dir / "visualizations").mkdir(exist_ok=True)
        
        # Save config
        self.save_config()
        
    def save_config(self):
        """Save configuration to JSON file."""
        config_dict = {
            "model": {key: value for key, value in vars(self.config.model).items() 
                      if not key.startswith('_')},
            "training": {key: value for key, value in vars(self.config.training).items() 
                        if not key.startswith('_')},
            "data": {key: value for key, value in vars(self.config.data).items() 
                    if not key.startswith('_')},
            "metrics": {key: value for key, value in vars(self.config.metrics).items() 
                      if not key.startswith('_')},
            "experiment_name": self.config.experiment_name,
            "debug": self.config.debug,
            "verbose": self.config.verbose,
            "use_gpu": self.config.use_gpu,
            "seed": self.config.seed
        }
        
        # Handle non-serializable types
        for section in config_dict:
            if isinstance(config_dict[section], dict):
                for key, value in config_dict[section].items():
                    if isinstance(value, tuple):
                        config_dict[section][key] = list(value)
        
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
    def prepare_data(self):
        """Prepare training and validation datasets."""
        if self.train_dataset is not None and self.val_dataset is not None:
            return  # Data already prepared
        
        # Generate or load agent states
        dataset = AgentStateDataset(batch_size=self.config.training.batch_size)
        
        # Check if simulation.db exists and use it if available
        db_path = "simulation.db"
        if os.path.exists(db_path):
            if self.config.debug:
                print(f"Loading agent states from {db_path}...")
            dataset.load_from_db(db_path, limit=self.config.data.num_states)
            if not dataset.states:
                if self.config.debug:
                    print("No states loaded from database. Generating synthetic data instead.")
                dataset.generate_synthetic_data(self.config.data.num_states)
        else:
            if self.config.debug:
                print(f"Database file not found: {db_path}")
                print(f"Generating {self.config.data.num_states} synthetic agent states...")
            dataset.generate_synthetic_data(self.config.data.num_states)
        
        # Split into train and validation sets
        total_states = len(dataset.states)
        val_size = int(total_states * self.config.data.validation_split)
        train_size = total_states - val_size
        
        train_states = dataset.states[:train_size]
        val_states = dataset.states[train_size:]
        
        self.train_dataset = AgentStateDataset(train_states, batch_size=self.config.training.batch_size)
        self.val_dataset = AgentStateDataset(val_states, batch_size=self.config.training.batch_size)
        
        if self.config.debug:
            print(f"Training set: {len(self.train_dataset.states)} states")
            print(f"Validation set: {len(self.val_dataset.states)} states")
            
        # Set aside a small set of states for tracking semantic drift
        self.drift_tracking_states = val_states[:min(10, len(val_states))]
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train model for one epoch.
        
        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()
        
        # Track metrics
        total_loss = 0.0
        recon_loss_total = 0.0
        kl_loss_total = 0.0
        semantic_loss_total = 0.0
        compression_loss_total = 0.0
        num_batches = 0
        
        # Shuffle at the start of each epoch
        random_idx = torch.randperm(len(self.train_dataset.states))
        self.train_dataset.states = [self.train_dataset.states[i] for i in random_idx]
        
        # Reset dataset index
        self.train_dataset._current_idx = 0
        
        # Calculate number of batches
        num_total_batches = len(self.train_dataset.states) // self.config.training.batch_size
        
        start_time = time.time()
        
        while self.train_dataset._current_idx < len(self.train_dataset.states):
            # Get batch
            batch = self.train_dataset.get_batch()
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            results = self.model(batch)
            
            # Compute loss
            loss_results = self.loss_fn(results, batch)
            loss = loss_results["total_loss"]
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            recon_loss_total += loss_results["recon_loss"].item()
            kl_loss_total += loss_results["kl_loss"].item()
            semantic_loss_total += loss_results["semantic_loss"].item()
            
            if "compression_loss" in results:
                compression_loss_total += results["compression_loss"].item()
                
            num_batches += 1
            
            # Print progress
            if self.config.verbose and num_batches % 10 == 0:
                progress = num_batches / num_total_batches * 100
                print(f"Training: {progress:.1f}% ({num_batches}/{num_total_batches}) "
                      f"Loss: {loss.item():.4f}")
        
        # Calculate metrics
        metrics = {
            "loss": total_loss / num_batches,
            "recon_loss": recon_loss_total / num_batches,
            "kl_loss": kl_loss_total / num_batches,
            "semantic_loss": semantic_loss_total / num_batches,
            "compression_loss": compression_loss_total / num_batches,
            "time": time.time() - start_time
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            metrics: Dictionary of validation metrics
        """
        self.model.eval()
        
        # Track metrics
        total_loss = 0.0
        recon_loss_total = 0.0
        kl_loss_total = 0.0
        semantic_loss_total = 0.0
        compression_loss_total = 0.0
        num_batches = 0
        
        # Reset dataset index
        self.val_dataset._current_idx = 0
        
        with torch.no_grad():
            while self.val_dataset._current_idx < len(self.val_dataset.states):
                # Get batch
                batch = self.val_dataset.get_batch()
                batch = batch.to(self.device)
                
                # Forward pass
                results = self.model(batch)
                
                # Compute loss
                loss_results = self.loss_fn(results, batch)
                loss = loss_results["total_loss"]
                
                # Update metrics
                total_loss += loss.item()
                recon_loss_total += loss_results["recon_loss"].item()
                kl_loss_total += loss_results["kl_loss"].item()
                semantic_loss_total += loss_results["semantic_loss"].item()
                
                if "compression_loss" in results:
                    compression_loss_total += results["compression_loss"].item()
                    
                num_batches += 1
        
        # Calculate metrics
        metrics = {
            "loss": total_loss / num_batches,
            "recon_loss": recon_loss_total / num_batches,
            "kl_loss": kl_loss_total / num_batches,
            "semantic_loss": semantic_loss_total / num_batches,
            "compression_loss": compression_loss_total / num_batches
        }
        
        return metrics
    
    def track_semantic_drift(self):
        """Track semantic drift for a fixed set of states across training."""
        self.model.eval()
        
        with torch.no_grad():
            # Stack drift tracking states into a tensor
            state_tensors = [state.to_tensor() for state in self.drift_tracking_states]
            batch = torch.stack(state_tensors).to(self.device)
            
            # Forward pass
            results = self.model(batch)
            
            # Get current compression level (bits per dimension)
            if hasattr(self.model, 'bits_per_dim'):
                compression_level = self.model.bits_per_dim
            else:
                # Default to configurable compression level 
                compression_level = self.config.model.compression_level
            
            # Track metrics using our drift tracker
            drift_metrics = self.drift_tracker.log_iteration(
                iteration=len(self.train_losses),
                compression_level=compression_level,
                original=batch,
                reconstructed=results["x_reconstructed"]
            )
            
            # Extract feature-specific losses for backward compatibility
            detailed_losses = self.semantic_metrics.compute_equivalence_scores(
                batch, results["x_reconstructed"]
            )
            
            # Create backward-compatible drift metrics
            legacy_metrics = {
                "epoch": len(self.train_losses),
                "total_semantic_loss": 1.0 - detailed_losses["overall"],  # Convert similarity to loss
                "feature_losses": {
                    k: 1.0 - v for k, v in detailed_losses.items() if k != "overall"
                }
            }
            
            # Save to history
            self.semantic_drift_history.append(legacy_metrics)
            
            # Generate latent space visualization periodically
            if len(self.train_losses) % 10 == 0 and hasattr(results, "z"):
                # Extract latent vectors
                latent_vectors = results["z"]
                
                # Create role labels for visualization
                role_indices = torch.argmax(batch[:, 5:10], dim=1)
                
                # Generate t-SNE visualization
                vis_path = self.experiment_dir / "visualizations" / f"latent_tsne_epoch_{len(self.train_losses)}.png"
                generate_t_sne_visualization(
                    latent_vectors, 
                    labels=role_indices,
                    output_file=str(vis_path)
                )
        
        return legacy_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_path = self.experiment_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save as best model if applicable
        if is_best:
            best_path = self.experiment_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            
        # Save the most recent model
        latest_path = self.experiment_dir / "latest_model.pt"
        torch.save(checkpoint, latest_path)
            
        # Cleanup old checkpoints if needed
        if not self.config.debug:
            self._cleanup_checkpoints(epoch)
    
    def _cleanup_checkpoints(self, current_epoch: int):
        """
        Clean up old checkpoints, keeping only the last few and checkpoints at regular intervals.
        
        Args:
            current_epoch: Current epoch
        """
        # Keep:
        # 1. Latest checkpoint
        # 2. Best model
        # 3. One checkpoint every 10 epochs
        # 4. The 5 most recent checkpoints
        
        keep_epochs = set()
        
        # Keep one checkpoint every 10 epochs
        for e in range(0, current_epoch + 1, 10):
            keep_epochs.add(e)
            
        # Keep the 5 most recent checkpoints
        for e in range(max(0, current_epoch - 4), current_epoch + 1):
            keep_epochs.add(e)
            
        # Find all checkpoint files
        checkpoint_files = list(self.experiment_dir.glob("checkpoint_epoch_*.pt"))
        
        for checkpoint_file in checkpoint_files:
            # Extract epoch number from filename
            filename = checkpoint_file.name
            try:
                epoch_str = filename.split("_")[-1].split(".")[0]
                epoch = int(epoch_str)
                
                # Remove if not in the set of epochs to keep
                if epoch not in keep_epochs:
                    checkpoint_file.unlink()
            except (IndexError, ValueError):
                # Skip files with unexpected naming
                continue
    
    def plot_training_curves(self):
        """Plot training and validation loss curves."""
        epochs = list(range(1, len(self.train_losses) + 1))
        
        plt.figure(figsize=(12, 8))
        
        # Plot total loss
        plt.subplot(2, 2, 1)
        plt.plot(epochs, [m["loss"] for m in self.train_losses], label="Train")
        plt.plot(epochs, [m["loss"] for m in self.val_losses], label="Validation")
        plt.title("Total Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        # Plot reconstruction loss
        plt.subplot(2, 2, 2)
        plt.plot(epochs, [m["recon_loss"] for m in self.train_losses], label="Train")
        plt.plot(epochs, [m["recon_loss"] for m in self.val_losses], label="Validation")
        plt.title("Reconstruction Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        # Plot KL loss
        plt.subplot(2, 2, 3)
        plt.plot(epochs, [m["kl_loss"] for m in self.train_losses], label="Train")
        plt.plot(epochs, [m["kl_loss"] for m in self.val_losses], label="Validation")
        plt.title("KL Divergence Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        # Plot semantic loss
        plt.subplot(2, 2, 4)
        plt.plot(epochs, [m["semantic_loss"] for m in self.train_losses], label="Train")
        plt.plot(epochs, [m["semantic_loss"] for m in self.val_losses], label="Validation")
        plt.title("Semantic Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.experiment_dir / "training_curves.png"
        plt.savefig(plot_path)
        plt.close()
    
    def plot_semantic_drift(self):
        """Plot semantic drift over time."""
        if not self.semantic_drift_history:
            return
            
        # Use the drift tracker's visualization instead of custom plotting
        output_file = self.experiment_dir / "semantic_drift.png"
        self.drift_tracker.visualize_drift(str(output_file))
        
        # Also generate a comprehensive report
        report_file = self.experiment_dir / "drift_report.md"
        report = self.drift_tracker.generate_report(str(report_file))
    
    def train(self, resume_from: Optional[str] = None):
        """
        Train the model.
        
        Args:
            resume_from: Path to checkpoint to resume from
        """
        # Prepare data
        self.prepare_data()
        
        # Resume from checkpoint if specified
        start_epoch = 0
        last_metrics = None
        if resume_from is not None:
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                
            start_epoch = checkpoint["epoch"] + 1
            self.train_losses = checkpoint.get("train_losses", [])
            self.val_losses = checkpoint.get("val_losses", [])
            self.semantic_drift_history = checkpoint.get("semantic_drift_history", [])
            self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            
            # Get last metrics for cases where we don't enter the training loop
            if self.val_losses:
                last_metrics = self.val_losses[-1]
            
            print(f"Resumed from checkpoint at epoch {start_epoch}")
        
        # Initialize last_epoch to handle case where loop doesn't run
        last_epoch = start_epoch - 1
        
        # Training loop
        for epoch in range(start_epoch, self.config.training.num_epochs):
            last_epoch = epoch  # Update last_epoch in each iteration
            print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            self.val_losses.append(val_metrics)
            last_metrics = val_metrics  # Store metrics for use after loop
            
            # Track semantic drift
            drift_metrics = self.track_semantic_drift()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Reconstruction: {train_metrics['recon_loss']:.4f}, "
                  f"KL: {train_metrics['kl_loss']:.4f}, "
                  f"Semantic: {train_metrics['semantic_loss']:.4f}")
            print(f"Validation Loss: {val_metrics['loss']:.4f}, "
                  f"Reconstruction: {val_metrics['recon_loss']:.4f}, "
                  f"KL: {val_metrics['kl_loss']:.4f}, "
                  f"Semantic: {val_metrics['semantic_loss']:.4f}")
            print(f"Semantic Drift: {drift_metrics['total_semantic_loss']:.4f}")
            
            # Check if this is the best model
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
                print("New best model!")
            else:
                self.patience_counter += 1
                
            # Save checkpoint
            if (epoch + 1) % self.config.training.checkpoint_interval == 0 or is_best:
                self.save_checkpoint(epoch + 1, val_metrics, is_best)
                
            # Plot training curves
            self.plot_training_curves()
            self.plot_semantic_drift()
                
            # Early stopping
            if self.patience_counter >= self.config.training.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Skip saving if we didn't run any training epochs and don't have metrics
        if last_metrics is not None:
            # Save final model - use the last epoch we processed
            self.save_checkpoint(
                last_epoch + 1, last_metrics, 
                is_best=(last_metrics["loss"] < self.best_val_loss)
            )
            
            print(f"Training completed. Best validation loss: {self.best_val_loss:.4f}")
        else:
            print("No training was performed (num_epochs may be less than or equal to start_epoch)")
        
        # Return training history
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "semantic_drift": self.semantic_drift_history,
            "best_val_loss": self.best_val_loss,
            "experiment_dir": str(self.experiment_dir)
        }


if __name__ == "__main__":
    # Example usage
    from meaning_transform.src.config import Config
    
    # Create configuration
    config = Config()
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train model
    training_history = trainer.train()
    
    print(f"Training completed. Results saved to {training_history['experiment_dir']}") 