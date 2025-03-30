#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training infrastructure for the meaning-preserving transformation system.

This module defines:
1. Training loop
2. Logging and checkpointing
3. Semantic drift tracking
4. Graph-based training support
"""

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch_geometric.data import Batch, Data

from .config import Config
from .data import AgentState, AgentStateDataset
from .graph_model import GraphVAELoss
from .loss import CombinedLoss
from .metrics import DriftTracker, generate_t_sne_visualization
from .model import MeaningVAE
from .standardized_metrics import StandardizedMetrics


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
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu"
            )
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Create model
        self.model = MeaningVAE(
            input_dim=self.config.model.input_dim,
            latent_dim=self.config.model.latent_dim,
            compression_type=self.config.model.compression_type,
            compression_level=self.config.model.compression_level,
            vq_num_embeddings=self.config.model.vq_num_embeddings,
            use_graph=getattr(self.config.model, "use_graph", False),
            graph_hidden_dim=getattr(self.config.model, "graph_hidden_dim", 128),
            gnn_type=getattr(self.config.model, "gnn_type", "GCN"),
            graph_num_layers=getattr(self.config.model, "graph_num_layers", 3),
        ).to(self.device)

        # Create loss function
        if getattr(self.config.model, "use_graph", False):
            # Use graph-specific loss if using graph-based model
            self.loss_fn = GraphVAELoss(
                node_weight=getattr(self.config.training, "node_recon_weight", 1.0),
                edge_weight=getattr(self.config.training, "edge_recon_weight", 1.0),
                kl_weight=self.config.training.kl_loss_weight,
                edge_attr_weight=getattr(self.config.training, "edge_attr_weight", 0.5),
                semantic_weight=self.config.training.semantic_loss_weight,
            )
        else:
            # Use standard combined loss for vector-based model
            self.loss_fn = CombinedLoss(
                recon_loss_weight=self.config.training.recon_loss_weight,
                kl_loss_weight=self.config.training.kl_loss_weight,
                semantic_loss_weight=self.config.training.semantic_loss_weight,
            )

        # Create optimizer
        if self.config.training.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        elif self.config.training.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer}")

        # Create learning rate scheduler
        if self.config.training.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.config.training.num_epochs
            )
        elif self.config.training.scheduler == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.config.training.scheduler_step_size,
                gamma=self.config.training.scheduler_gamma,
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
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Create timestamped experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{self.config.experiment_name}_{timestamp}"
        self.experiment_dir = self.checkpoint_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize semantic metrics and drift tracker
        self.drift_tracker = DriftTracker(
            log_dir=str(self.experiment_dir / "drift_tracking")
        )
        self.semantic_metrics = StandardizedMetrics()

        # Create subdirectories for results
        (self.experiment_dir / "visualizations").mkdir(exist_ok=True)

        # Flag to determine if we're using graph-based representation
        self.use_graph = getattr(self.config.model, "use_graph", False)

        # Save config
        self.save_config()

    def save_config(self):
        """Save configuration to JSON file."""
        config_dict = {
            "model": {
                key: value
                for key, value in vars(self.config.model).items()
                if not key.startswith("_")
            },
            "training": {
                key: value
                for key, value in vars(self.config.training).items()
                if not key.startswith("_")
            },
            "data": {
                key: value
                for key, value in vars(self.config.data).items()
                if not key.startswith("_")
            },
            "metrics": {
                key: value
                for key, value in vars(self.config.metrics).items()
                if not key.startswith("_")
            },
            "experiment_name": self.config.experiment_name,
            "debug": self.config.debug,
            "verbose": self.config.verbose,
            "use_gpu": self.config.use_gpu,
            "seed": self.config.seed,
        }

        # Handle non-serializable types
        for section in config_dict:
            if isinstance(config_dict[section], dict):
                for key, value in config_dict[section].items():
                    if isinstance(value, tuple):
                        config_dict[section][key] = list(value)

        config_path = self.experiment_dir / "config.json"
        with open(config_path, "w") as f:
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
                    print(
                        "No states loaded from database. Generating synthetic data instead."
                    )
                dataset.generate_synthetic_data(self.config.data.num_states)
        else:
            if self.config.debug:
                print(f"Database file not found: {db_path}")
                print(
                    f"Generating {self.config.data.num_states} synthetic agent states..."
                )
            dataset.generate_synthetic_data(self.config.data.num_states)

        # Split into train and validation sets
        total_states = len(dataset.states)
        val_size = int(total_states * self.config.data.validation_split)
        train_size = total_states - val_size

        train_states = dataset.states[:train_size]
        val_states = dataset.states[train_size:]

        self.train_dataset = AgentStateDataset(
            train_states, batch_size=self.config.training.batch_size
        )
        self.val_dataset = AgentStateDataset(
            val_states, batch_size=self.config.training.batch_size
        )

        if self.config.debug:
            print(f"Training set: {len(self.train_dataset.states)} states")
            print(f"Validation set: {len(self.val_dataset.states)} states")

        # Set aside a small set of states for tracking semantic drift
        self.drift_tracking_states = val_states[: min(10, len(val_states))]

        # If using graph-based representation, prepare graph versions as well
        if self.use_graph:
            if self.config.debug:
                print("Preparing graph-based datasets...")

            # Create graph-based versions of the test states for evaluation
            try:
                self.drift_tracking_graphs = [
                    state.to_torch_geometric() for state in self.drift_tracking_states
                ]
                if self.config.debug:
                    print(
                        f"Created {len(self.drift_tracking_graphs)} graph representations for drift tracking"
                    )
            except Exception as e:
                print(f"Warning: Could not create graph representations: {e}")
                self.drift_tracking_graphs = None

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
        edge_loss_total = 0.0  # For graph models
        num_batches = 0

        # Shuffle at the start of each epoch
        random_idx = torch.randperm(len(self.train_dataset.states))
        self.train_dataset.states = [self.train_dataset.states[i] for i in random_idx]

        # Reset dataset index
        self.train_dataset._current_idx = 0

        # Calculate number of batches
        num_total_batches = (
            len(self.train_dataset.states) // self.config.training.batch_size
        )

        start_time = time.time()

        while self.train_dataset._current_idx < len(self.train_dataset.states):
            # Get batch - either graph or tensor based on configuration
            if self.use_graph:
                batch = self.train_dataset.get_graph_batch()
            else:
                batch = self.train_dataset.get_batch()

            # Move batch to device
            if isinstance(batch, (Data, Batch)):
                # Graph data needs special handling to move to device
                batch = batch.to(self.device)
            else:
                # Standard tensor data
                batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            results = self.model(batch)

            # Compute loss
            if self.use_graph:
                # Graph-specific loss calculation
                loss_results = self.loss_fn(results, batch)
            else:
                # Standard tensor loss calculation
                loss_results = self.loss_fn(results, batch)

            loss = loss_results["total_loss"]

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()

            # Standard metrics
            if "recon_loss" in loss_results:
                recon_loss_total += loss_results["recon_loss"].item()
            if "kl_loss" in loss_results:
                kl_loss_total += loss_results["kl_loss"].item()
            if "semantic_loss" in loss_results:
                semantic_loss_total += loss_results["semantic_loss"].item()
            if "compression_loss" in loss_results:
                compression_loss_total += loss_results["compression_loss"].item()

            # Graph-specific metrics
            if "edge_loss" in loss_results:
                edge_loss_total += loss_results["edge_loss"].item()

            num_batches += 1

            # Progress update
            if self.config.verbose and num_batches % 10 == 0:
                elapsed = time.time() - start_time
                progress = num_batches / num_total_batches
                remaining = elapsed / progress - elapsed if progress > 0 else 0

                print(
                    f"Batch {num_batches}/{num_total_batches} "
                    f"[{progress:.0%}] - "
                    f"Loss: {loss.item():.4f} - "
                    f"Elapsed: {elapsed:.0f}s - "
                    f"Remaining: {remaining:.0f}s"
                )

        # Compute average metrics
        metrics = {
            "train_loss": total_loss / num_batches,
            "train_recon_loss": recon_loss_total / num_batches,
            "train_kl_loss": kl_loss_total / num_batches,
            "train_semantic_loss": semantic_loss_total / num_batches,
            "train_compression_loss": compression_loss_total / num_batches,
        }

        # Add graph-specific metrics if available
        if self.use_graph:
            metrics["train_edge_loss"] = edge_loss_total / num_batches

        return metrics

    def validate(self) -> Dict[str, float]:
        """
        Validate model on validation set.

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
        edge_loss_total = 0.0  # For graph models
        num_batches = 0

        # Reset dataset index
        self.val_dataset._current_idx = 0

        with torch.no_grad():
            while self.val_dataset._current_idx < len(self.val_dataset.states):
                # Get batch - either graph or tensor based on configuration
                if self.use_graph:
                    batch = self.val_dataset.get_graph_batch()
                else:
                    batch = self.val_dataset.get_batch()

                # Move batch to device
                if isinstance(batch, (Data, Batch)):
                    # Graph data needs special handling to move to device
                    batch = batch.to(self.device)
                else:
                    # Standard tensor data
                    batch = batch.to(self.device)

                # Forward pass
                results = self.model(batch)

                # Compute loss
                if self.use_graph:
                    # Graph-specific loss calculation
                    loss_results = self.loss_fn(results, batch)
                else:
                    # Standard tensor loss calculation
                    loss_results = self.loss_fn(results, batch)

                loss = loss_results["total_loss"]

                # Update metrics
                total_loss += loss.item()

                # Standard metrics
                if "recon_loss" in loss_results:
                    recon_loss_total += loss_results["recon_loss"].item()
                if "kl_loss" in loss_results:
                    kl_loss_total += loss_results["kl_loss"].item()
                if "semantic_loss" in loss_results:
                    semantic_loss_total += loss_results["semantic_loss"].item()
                if "compression_loss" in loss_results:
                    compression_loss_total += loss_results["compression_loss"].item()

                # Graph-specific metrics
                if "edge_loss" in loss_results:
                    edge_loss_total += loss_results["edge_loss"].item()

                num_batches += 1

        # Compute average metrics
        metrics = {
            "val_loss": total_loss / num_batches,
            "val_recon_loss": recon_loss_total / num_batches,
            "val_kl_loss": kl_loss_total / num_batches,
            "val_semantic_loss": semantic_loss_total / num_batches,
            "val_compression_loss": compression_loss_total / num_batches,
        }

        # Add graph-specific metrics if available
        if self.use_graph:
            metrics["val_edge_loss"] = edge_loss_total / num_batches

        return metrics

    def track_semantic_drift(self):
        """Track semantic drift of agent states through VAE transformation."""
        self.model.eval()

        with torch.no_grad():
            # Track drift for either graph or tensor representations
            if (
                self.use_graph
                and hasattr(self, "drift_tracking_graphs")
                and self.drift_tracking_graphs
            ):
                # Process graph-based drift tracking
                self._track_graph_semantic_drift()
            else:
                # Process standard tensor-based drift tracking
                self._track_tensor_semantic_drift()

    def _track_tensor_semantic_drift(self):
        """Track semantic drift using tensor representations."""
        # Convert states to tensors
        tensors = [state.to_tensor() for state in self.drift_tracking_states]
        states_tensor = torch.stack(tensors).to(self.device)

        # Pass through model
        results = self.model(states_tensor)

        # Extract reconstructions
        reconstructions = results["reconstruction"].cpu()

        # Calculate semantic metrics using standardized metrics
        semantic_metrics = self.semantic_metrics.evaluate(
            states_tensor.cpu(), reconstructions
        )

        # Add metrics to tracking history using standardized metric names
        current_drift = {
            "fidelity": semantic_metrics["overall_fidelity"],
            "meaning_preservation": semantic_metrics["overall_preservation"],
            "reconstruction_error": 1.0 - semantic_metrics["overall_fidelity"],
        }

        self.semantic_drift_history.append(current_drift)

        # Log drift metrics
        self.drift_tracker.log_semantic_drift(current_drift)

    def _track_graph_semantic_drift(self):
        """Track semantic drift using graph representations."""
        results_list = []

        # Process each graph individually to avoid batch size issues
        for graph_data in self.drift_tracking_graphs:
            # Move to device
            graph_data = graph_data.to(self.device)

            # Process through model
            results = self.model(graph_data)
            results_list.append(results)

        # Calculate graph-based semantic metrics
        node_features_orig = [g.x.cpu() for g in self.drift_tracking_graphs]
        node_features_recon = [r["reconstruction"].cpu() for r in results_list]

        # Use standardized metrics for graph evaluation if available
        try:
            # Try to use dedicated graph metrics method if available
            semantic_metrics = self.semantic_metrics.evaluate_graph(
                self.drift_tracking_graphs, results_list, self.model
            )
        except AttributeError:
            # Fall back to node feature comparison if graph metrics not available
            # Convert node features to tensors for standardized metrics
            orig_tensors = torch.cat([feat for feat in node_features_orig], dim=0)
            recon_tensors = torch.cat([feat for feat in node_features_recon], dim=0)

            # Use standard evaluation
            semantic_metrics = self.semantic_metrics.evaluate(
                orig_tensors, recon_tensors
            )

        # Add metrics to tracking history using standardized metric names
        current_drift = {
            "fidelity": semantic_metrics.get("overall_fidelity", 0.0),
            "meaning_preservation": semantic_metrics.get("overall_preservation", 0.0),
            "reconstruction_error": 1.0 - semantic_metrics.get("overall_fidelity", 0.0),
        }

        self.semantic_drift_history.append(current_drift)

        # Log drift metrics
        self.drift_tracker.log_semantic_drift(current_drift)

    def save_checkpoint(
        self, epoch: int, metrics: Dict[str, float], is_best: bool = False
    ):
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
            "metrics": metrics,
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
        plt.plot(epochs, [m["train_loss"] for m in self.train_losses], label="Train")
        plt.plot(epochs, [m["val_loss"] for m in self.val_losses], label="Validation")
        plt.title("Total Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Plot reconstruction loss
        plt.subplot(2, 2, 2)
        plt.plot(
            epochs, [m["train_recon_loss"] for m in self.train_losses], label="Train"
        )
        plt.plot(
            epochs, [m["val_recon_loss"] for m in self.val_losses], label="Validation"
        )
        plt.title("Reconstruction Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Plot KL loss
        plt.subplot(2, 2, 3)
        plt.plot(epochs, [m["train_kl_loss"] for m in self.train_losses], label="Train")
        plt.plot(
            epochs, [m["val_kl_loss"] for m in self.val_losses], label="Validation"
        )
        plt.title("KL Divergence Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Plot semantic loss
        plt.subplot(2, 2, 4)
        plt.plot(
            epochs, [m["train_semantic_loss"] for m in self.train_losses], label="Train"
        )
        plt.plot(
            epochs,
            [m["val_semantic_loss"] for m in self.val_losses],
            label="Validation",
        )
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

        # Initialize or load model
        start_epoch = 0
        if resume_from:
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            self.train_losses = checkpoint["train_losses"]
            self.val_losses = checkpoint["val_losses"]
            self.semantic_drift_history = checkpoint.get("semantic_drift_history", [])
            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            self.patience_counter = checkpoint.get("patience_counter", 0)

            print(f"Resuming from epoch {start_epoch}")

        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Model has {total_params:,} parameters ({trainable_params:,} trainable)")

        # Print configuration summary
        print(f"Training configuration:")
        print(f"  - Epochs: {self.config.training.num_epochs}")
        print(f"  - Batch size: {self.config.training.batch_size}")
        print(f"  - Learning rate: {self.config.training.learning_rate}")
        print(f"  - Using graph: {self.use_graph}")
        if self.use_graph:
            print(f"  - GNN type: {getattr(self.config.model, 'gnn_type', 'GCN')}")
            print(
                f"  - Graph layers: {getattr(self.config.model, 'graph_num_layers', 3)}"
            )

        # Training loop
        for epoch in range(start_epoch, self.config.training.num_epochs):
            epoch_start_time = time.time()

            # Train for one epoch
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Track semantic drift
            self.track_semantic_drift()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Track losses
            self.train_losses.append(train_metrics["train_loss"])
            self.val_losses.append(val_metrics["val_loss"])

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            # Print progress
            print(
                f"Epoch {epoch+1}/{self.config.training.num_epochs} "
                f"[{(epoch+1)/self.config.training.num_epochs:.0%}] - "
                f"Train Loss: {train_metrics['train_loss']:.4f} - "
                f"Val Loss: {val_metrics['val_loss']:.4f} - "
                f"Time: {epoch_time:.1f}s"
            )

            # More detailed metrics if in verbose mode
            if self.config.verbose:
                print(
                    f"  - Train Recon Loss: {train_metrics['train_recon_loss']:.4f} - "
                    f"Val Recon Loss: {val_metrics['val_recon_loss']:.4f}"
                )
                print(
                    f"  - Train KL Loss: {train_metrics['train_kl_loss']:.4f} - "
                    f"Val KL Loss: {val_metrics['val_kl_loss']:.4f}"
                )

                # Graph-specific metrics
                if self.use_graph and "train_edge_loss" in train_metrics:
                    print(
                        f"  - Train Edge Loss: {train_metrics['train_edge_loss']:.4f} - "
                        f"Val Edge Loss: {val_metrics['val_edge_loss']:.4f}"
                    )

                # Semantic drift metrics
                current_drift = self.semantic_drift_history[-1]
                print(
                    f"  - Fidelity: {current_drift['fidelity']:.4f} - "
                    f"Meaning Preservation: {current_drift['meaning_preservation']:.4f}"
                )

            # Save model
            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(epoch, {**train_metrics, **val_metrics}, is_best)

            # Cleanup old checkpoints
            self._cleanup_checkpoints(epoch)

            # Plot training curves
            if (epoch + 1) % self.config.metrics.visualization_interval == 0:
                self.plot_training_curves()
                self.plot_semantic_drift()

                # Generate t-SNE visualization of latent space
                if (
                    hasattr(self, "drift_tracking_graphs")
                    and self.drift_tracking_graphs
                    and self.use_graph
                ):
                    # Generate visualization for graph embeddings
                    graph_batch = Batch.from_data_list(
                        [g.to(self.device) for g in self.drift_tracking_graphs[:30]]
                    )
                    with torch.no_grad():
                        embeddings = self.model.encode(graph_batch).cpu().numpy()

                    generate_t_sne_visualization(
                        embeddings,
                        labels=[
                            state.role for state in self.drift_tracking_states[:30]
                        ],
                        save_path=str(
                            self.experiment_dir
                            / "visualizations"
                            / f"tsne_latent_epoch_{epoch+1}.png"
                        ),
                        title=f"t-SNE of Graph Latent Space (Epoch {epoch+1})",
                    )
                else:
                    # Generate visualization for tensor embeddings
                    tensors = [
                        state.to_tensor() for state in self.drift_tracking_states[:30]
                    ]
                    states_tensor = torch.stack(tensors).to(self.device)

                    with torch.no_grad():
                        embeddings = self.model.encode(states_tensor).cpu().numpy()

                    generate_t_sne_visualization(
                        embeddings,
                        labels=[
                            state.role for state in self.drift_tracking_states[:30]
                        ],
                        save_path=str(
                            self.experiment_dir
                            / "visualizations"
                            / f"tsne_latent_epoch_{epoch+1}.png"
                        ),
                        title=f"t-SNE of Latent Space (Epoch {epoch+1})",
                    )

            # Early stopping
            if self.patience_counter >= self.config.training.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Skip saving if we didn't run any training epochs and don't have metrics
        if self.val_losses:
            # Save final model - use the last epoch we processed
            self.save_checkpoint(
                epoch,
                self.val_losses[-1],
                is_best=(self.val_losses[-1]["val_loss"] < self.best_val_loss),
            )

            print(f"Training completed. Best validation loss: {self.best_val_loss:.4f}")
        else:
            print(
                "No training was performed (num_epochs may be less than or equal to start_epoch)"
            )

        # Return training history
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "semantic_drift": self.semantic_drift_history,
            "best_val_loss": self.best_val_loss,
            "experiment_dir": str(self.experiment_dir),
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
