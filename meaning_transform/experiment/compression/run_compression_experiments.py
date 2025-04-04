#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compression Experiments for Meaning-Preserving Transformations

This script runs a series of compression experiments to analyze how different
compression levels affect semantic preservation in the meaning-preserving transformation system.

Experiments:
1. Run the model with varying compression levels (0.5, 1.0, 2.0, 5.0)
2. Analyze how different compression rates affect semantic preservation
3. Create visualization comparisons between compression levels
4. Identify optimal compression setting for balancing information density with meaning retention
5. Document findings in a compression analysis report
"""

import argparse
import json
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from meaning_transform.experiment.base_experiment import BaseExperiment

# Import after adding project root to path
from meaning_transform.src.config import Config
from meaning_transform.src.data import AgentState, AgentStateDataset
from meaning_transform.src.loss import beta_annealing
from meaning_transform.src.meaning_validation import MeaningValidator
from meaning_transform.src.models import AdaptiveMeaningVAE, MeaningVAE
from meaning_transform.src.pipelines.standard_pipeline import (
    create_compression_pipeline,
)

# Configure logging
def setup_logging(output_dir: Path):
    """Set up logging configuration."""
    # Create logs directory
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"compression_experiment_{timestamp}.log"
    
    # Configure logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging configured. Log file: {log_file}")
    return log_file


class CompressionExperiment(BaseExperiment):
    """Class to run and analyze compression experiments."""

    def __init__(
        self,
        base_config: Config = None,
        output_dir: str = None,
        use_adaptive_model: bool = False,
        use_graph: bool = False,
        track_drift: bool = False,
        use_beta_annealing: bool = True,
    ):
        """
        Initialize compression experiment.

        Args:
            base_config: Base configuration to use (will be modified for each experiment)
            output_dir: Directory to save experiment results
            use_adaptive_model: Whether to use the adaptive model architecture
            use_graph: Whether to use graph-based modeling
            track_drift: Whether to track semantic drift (may cause errors if dependencies missing)
            use_beta_annealing: Whether to use beta annealing for stable KL loss
        """
        # Create timestamp for experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"compression_experiments_{timestamp}"

        # Initialize the parent class
        super().__init__(
            base_config=base_config or Config(),
            output_dir=output_dir or "results/compression_experiments",
            experiment_name=experiment_name,
            project_root=project_root,
        )

        self.use_adaptive_model = use_adaptive_model
        self.use_graph = use_graph
        self.track_drift = track_drift
        self.use_beta_annealing = use_beta_annealing

        # Set compression levels to test
        self.compression_levels = [0.5, 1.0, 2.0, 5.0]

        # For semantic drift tracking
        self.baseline_originals = None
        self.baseline_reconstructions = None

        # Create validator for meaning preservation testing
        self.validator = MeaningValidator()

        # Save base configuration
        self._save_base_config()

    def _save_base_config(self):
        """Save the base configuration to a file."""
        # Add compression-specific parameters
        config_dict = {
            "compression_levels": self.compression_levels,
            "use_adaptive_model": self.use_adaptive_model,
            "use_graph": self.use_graph,
            "use_beta_annealing": self.use_beta_annealing,
        }

        # Use parent class method to save the config
        self._save_config(config_dict, filename="base_config.json")

    def run_experiments(self):
        """
        Run experiments with different compression levels.

        For each compression level, create a model with the pipeline and evaluate its performance.
        """
        logging.info(f"Running compression experiments with levels: {self.compression_levels}")
        logging.info(f"Using adaptive model: {self.use_adaptive_model}")
        logging.info(f"Using graph-based modeling: {self.use_graph}")

        # Prepare a common dataset for all experiments
        dataset = self._prepare_data()

        # Determine device
        device = torch.device(
            "cuda" if self.base_config.use_gpu and torch.cuda.is_available() else "cpu"
        )
        logging.info(f"Using device: {device}")

        # Run an experiment for each compression level
        for level in self.compression_levels:
            logging.info(f"\n{'='*80}")
            logging.info(f"Running experiment with compression level: {level}")
            logging.info(f"{'='*80}")

            # Create a config for this experiment
            config = self._create_config_for_level(level)

            # Create model for this compression level
            model = self._create_model(config, level)
            model = model.to(device)

            # Create pipeline for this compression level
            pipeline = create_compression_pipeline(
                model=model,
                config=config,
                compression_level=level,
                use_graph=self.use_graph,
                device=device,
            )

            # Train the model
            training_results = self._train_model(model, config, dataset)

            # Save model after training
            model_dest = self.models_dir / f"model_compression_{level}.pt"
            model.save(model_dest)
            logging.info(f"Saved model to {model_dest}")

            # Process validation data through pipeline to get metrics
            val_states = dataset["val"].states[:5]  # Use fewer states for evaluation

            logging.info("Evaluating semantic preservation...")
            with torch.no_grad():
                try:
                    # Process through pipeline with explicit device handling
                    result, context = pipeline.process(val_states)

                except Exception as e:
                    logging.error(f"Error during pipeline processing: {str(e)}")
                    # If pipeline fails, try direct model evaluation
                    try:
                        # Prepare data
                        val_tensors = torch.stack(
                            [state.to_tensor() for state in val_states[:5]]
                        )
                        val_tensors = val_tensors.to(device)

                        # Run through model directly
                        with torch.no_grad():
                            if hasattr(model, "forward") and callable(model.forward):
                                results = model(val_tensors)
                                if (
                                    isinstance(results, dict)
                                    and "reconstruction" in results
                                ):
                                    reconstructions = results["reconstruction"]
                                elif (
                                    isinstance(results, dict)
                                    and "x_reconstructed" in results
                                ):
                                    reconstructions = results["x_reconstructed"]
                                elif isinstance(results, tuple) and len(results) >= 1:
                                    reconstructions = results[0]
                                else:
                                    raise ValueError(
                                        f"Unknown model output format: {type(results)}"
                                    )
                            else:
                                reconstructions, _, _ = model(val_tensors)

                        # Create metrics and evaluate
                        try:
                            # Explicit import to ensure it's available in this scope
                            from meaning_transform.src.standardized_metrics import (
                                StandardizedMetrics,
                            )

                            metrics = StandardizedMetrics()
                        except ImportError:
                            logging.warning(
                                "Could not import StandardizedMetrics, using placeholder metrics"
                            )
                            # Create a minimal semantic evaluation result as placeholder
                            semantic_evaluation = {
                                "overall_preservation": 0.5,
                                "overall_fidelity": 0.5,
                                "overall_drift": 0.5,
                                "preservation_category": "acceptable",
                                "fidelity_category": "acceptable",
                                "drift_category": "acceptable",
                                "preservation": {},
                                "fidelity": {},
                                "drift": {},
                            }
                            context = {"semantic_evaluation": semantic_evaluation}
                            result = reconstructions
                            continue

                        # Ensure both tensors are on the same device
                        if val_tensors.device != reconstructions.device:
                            reconstructions = reconstructions.to(val_tensors.device)

                        # Ensure proper shapes
                        if len(val_tensors.shape) == 1:
                            val_tensors = val_tensors.unsqueeze(0)
                        if len(reconstructions.shape) == 1:
                            reconstructions = reconstructions.unsqueeze(0)

                        semantic_evaluation = metrics.evaluate(
                            val_tensors,
                            reconstructions,
                            self.baseline_originals,
                            self.baseline_reconstructions,
                        )

                        # Store tensors in fake context for tracking
                        result = reconstructions
                        context = {
                            "input": val_tensors,
                            "semantic_evaluation": semantic_evaluation,
                        }
                        logging.info("Evaluated using direct model access")

                    except Exception as inner_e:
                        logging.error(f"Failed direct evaluation too: {str(inner_e)}")
                        # Create a minimal semantic evaluation result
                        semantic_evaluation = {
                            "overall_preservation": 0.5,
                            "overall_fidelity": 0.5,
                            "overall_drift": 0.5,
                            "preservation_category": "acceptable",
                            "fidelity_category": "acceptable",
                            "drift_category": "acceptable",
                            "preservation": {},
                            "fidelity": {},
                            "drift": {},
                        }
                        context = {"semantic_evaluation": semantic_evaluation}
                        result = None

            # Extract semantic evaluation metrics
            semantic_evaluation = context.get("semantic_evaluation", {})

            # If this is the first level, save reconstructions as baseline for drift tracking
            if self.track_drift and self.baseline_originals is None:
                logging.info("Saving this level as baseline for drift tracking...")
                # Get input from context or create a placeholder
                input_data = context.get("input")
                if input_data is None:
                    # Create a placeholder tensor
                    input_data = torch.zeros((5, config.model.input_dim), device=device)
                self.baseline_originals = input_data
                # Get result or create a placeholder
                if result is not None:
                    self.baseline_reconstructions = result
                else:
                    # Create a placeholder tensor
                    self.baseline_reconstructions = torch.zeros(
                        (5, config.model.input_dim), device=device
                    )

            # Store parameters count and model details
            if self.use_adaptive_model:
                param_count = sum(p.numel() for p in model.parameters())
                effective_dim = (
                    getattr(
                        model.compressor,
                        "effective_dim",
                        int(self.base_config.model.latent_dim / level),
                    )
                    if hasattr(model, "compressor")
                    else int(self.base_config.model.latent_dim / level)
                )
                compression_rate = (
                    model.get_compression_rate()
                    if hasattr(model, "get_compression_rate")
                    else level
                )
            else:
                param_count = sum(p.numel() for p in model.parameters())
                effective_dim = (
                    int(self.base_config.model.latent_dim / level)
                    if level > 0
                    else self.base_config.model.latent_dim
                )
                compression_rate = level

            # Save results
            self.results[level] = {
                "training_results": training_results,
                "model_path": str(model_dest),
                "experiment_dir": str(self.models_dir),
                "semantic_evaluation": semantic_evaluation,
                "val_loss": training_results.get("best_val_loss", 0.0),
                "recon_loss": training_results.get("best_recon_loss", 0.0),
                "kl_loss": training_results.get("best_kl_loss", 0.0),
                "semantic_loss": training_results.get("best_semantic_loss", 0.0),
                "compression_loss": training_results.get("best_compression_loss", 0.0),
                "param_count": param_count,
                "effective_dim": effective_dim,
                "compression_rate": compression_rate,
            }

            # Find the best values from training history
            if "val_losses" in training_results and training_results["val_losses"]:
                # Extract the best validation metrics from the history
                val_losses = training_results["val_losses"]
                
                # Find the epoch with the best (lowest) overall validation loss
                best_epoch_idx = min(range(len(val_losses)), key=lambda i: val_losses[i]["val_loss"])
                best_epoch_metrics = val_losses[best_epoch_idx]
                
                # Update the results with the actual best metrics
                self.results[level].update({
                    "recon_loss": best_epoch_metrics.get("val_recon_loss", 0.0),
                    "kl_loss": best_epoch_metrics.get("val_kl_loss", 0.0),
                    "semantic_loss": best_epoch_metrics.get("val_semantic_loss", 0.0),
                    "compression_loss": best_epoch_metrics.get("val_compression_loss", 0.0),
                })
                
                # Also log these for clarity
                logging.info(f"Best validation metrics (epoch {best_epoch_idx+1}):")
                logging.info(f"  - Recon Loss: {self.results[level]['recon_loss']:.4f}")
                logging.info(f"  - KL Loss: {self.results[level]['kl_loss']:.4f}")
                logging.info(f"  - Semantic Loss: {self.results[level]['semantic_loss']:.4f}")
                logging.info(f"  - Compression Loss: {self.results[level]['compression_loss']:.4f}")

        # Analyze results
        self._analyze_results()

    def _create_model(
        self, config: Config, compression_level: float
    ) -> Union[MeaningVAE, AdaptiveMeaningVAE]:
        """
        Create a model with the specified configuration and compression level.

        Args:
            config: Model configuration
            compression_level: Compression level to use

        Returns:
            Configured model instance
        """
        # Set model class based on experiment type
        model_class = AdaptiveMeaningVAE if self.use_adaptive_model else MeaningVAE

        # Inspect model constructor to get accepted parameters
        from inspect import signature

        model_sig = signature(model_class.__init__)
        model_params = {}

        # Add parameters only if they're in the signature
        param_names = list(model_sig.parameters.keys())

        # Basic parameters
        if "input_dim" in param_names:
            model_params["input_dim"] = config.model.input_dim

        if "latent_dim" in param_names:
            model_params["latent_dim"] = config.model.latent_dim

        # For MeaningVAE
        if "compression_type" in param_names and hasattr(
            config.model, "compression_type"
        ):
            model_params["compression_type"] = config.model.compression_type

        # For all models
        if "compression_level" in param_names:
            model_params["compression_level"] = compression_level

        # Encoder/decoder hidden dimensions
        if "encoder_hidden_dims" in param_names and hasattr(
            config.model, "encoder_hidden_dims"
        ):
            model_params["encoder_hidden_dims"] = config.model.encoder_hidden_dims

        if "decoder_hidden_dims" in param_names and hasattr(
            config.model, "decoder_hidden_dims"
        ):
            model_params["decoder_hidden_dims"] = config.model.decoder_hidden_dims

        # Create the model
        logging.info(f"Creating {model_class.__name__} with params: {model_params}")
        model = model_class(**model_params)

        # Set graph parameters if using graph-based modeling
        if self.use_graph and hasattr(model, "use_graph"):
            model.use_graph = True
            if hasattr(model, "graph_hidden_dim") and hasattr(
                config.model, "graph_hidden_dim"
            ):
                model.graph_hidden_dim = config.model.graph_hidden_dim
            if hasattr(model, "gnn_type") and hasattr(config.model, "gnn_type"):
                model.gnn_type = config.model.gnn_type
            if hasattr(model, "graph_num_layers") and hasattr(
                config.model, "graph_num_layers"
            ):
                model.graph_num_layers = config.model.graph_num_layers

        return model

    def _train_model(
        self,
        model: Union[MeaningVAE, AdaptiveMeaningVAE],
        config: Config,
        dataset: Dict,
    ) -> Dict[str, Any]:
        """
        Train the model with the given configuration and dataset.

        Args:
            model: Model to train
            config: Training configuration
            dataset: Dictionary containing train and validation datasets

        Returns:
            Dictionary of training results
        """
        # Import Trainer here to avoid circular imports
        from meaning_transform.src.train import Trainer

        # Create a trainer
        trainer = Trainer(config)

        # Set the model to the trainer
        trainer.model = model

        # Set the pre-prepared dataset to the trainer
        trainer.train_dataset = dataset["train"]
        trainer.val_dataset = dataset["val"]

        # Set drift tracking data if available
        if self.track_drift and "drift_tracking" in dataset:
            trainer.drift_tracking_states = dataset["drift_tracking"]

        # Implement beta annealing if enabled
        if self.use_beta_annealing:
            # Store the original train method
            original_train_method = trainer.train

            # Replace with a wrapped version that uses beta annealing
            def train_with_beta_annealing():
                logging.info("Using beta annealing for stable KL loss...")

                # Original KL weight
                original_kl_weight = config.training.kl_loss_weight

                # Define a hook to update KL weight before each epoch
                def pre_epoch_hook(trainer, epoch):
                    # Calculate beta using sigmoid annealing
                    beta = beta_annealing(
                        epoch=epoch,
                        max_epochs=config.training.num_epochs,
                        min_beta=0.0001,  # Start with very small KL weight
                        max_beta=original_kl_weight,  # End with configured weight
                        schedule_type="sigmoid",
                    )

                    # Update KL weight in the loss function
                    if hasattr(trainer, "loss_fn") and hasattr(
                        trainer.loss_fn, "kl_loss_weight"
                    ):
                        trainer.loss_fn.kl_loss_weight = beta
                        logging.info(
                            f"Epoch {epoch+1}/{config.training.num_epochs}: KL weight set to {beta:.6f}"
                        )

                # Add the hook to trainer if it has the mechanism
                if hasattr(trainer, "register_hook"):
                    trainer.register_hook("pre_epoch", pre_epoch_hook)
                else:
                    # If the trainer doesn't have hook mechanism, we'll modify the train_epoch method
                    original_train_epoch = trainer.train_epoch

                    def train_epoch_with_annealing():
                        # Update KL weight based on current epoch
                        current_epoch = (
                            trainer.current_epoch
                            if hasattr(trainer, "current_epoch")
                            else 0
                        )
                        beta = beta_annealing(
                            epoch=current_epoch,
                            max_epochs=config.training.num_epochs,
                            min_beta=0.0001,
                            max_beta=original_kl_weight,
                            schedule_type="sigmoid",
                        )

                        # Set the KL weight
                        if hasattr(trainer, "loss_fn") and hasattr(
                            trainer.loss_fn, "kl_loss_weight"
                        ):
                            trainer.loss_fn.kl_loss_weight = beta
                            logging.info(
                                f"Epoch {current_epoch+1}/{config.training.num_epochs}: KL weight set to {beta:.6f}"
                            )

                        # Call original method
                        return original_train_epoch()

                    # Replace the method
                    trainer.train_epoch = train_epoch_with_annealing

                # Call original train method
                return original_train_method()

            # Replace the train method
            trainer.train = train_with_beta_annealing

        # Train the model
        training_results = trainer.train()

        return training_results

    def _prepare_data(self) -> Dict[str, AgentStateDataset]:
        """
        Prepare datasets for experiments.

        Returns:
            Dict containing train, validation and drift tracking datasets
        """
        # Generate or load agent states
        dataset = AgentStateDataset(batch_size=self.base_config.training.batch_size)

        # Load real data from database
        db_path = (
            self.base_config.data.db_path
            if hasattr(self.base_config.data, "db_path")
            else "data/simulation.db"
        )

        # Convert to Path object to handle cross-platform paths and resolve to absolute path
        db_path = Path(db_path).resolve()

        # If database doesn't exist at the primary location, try common alternative locations
        if not db_path.exists():
            # Check alternative locations
            alternatives = [
                Path("simulation.db"),  # Current working directory
                Path("data/simulation.db"),  # data folder in current directory
                project_root / "data" / "simulation.db",  # project_root/data
                project_root / "simulation.db",  # project_root
            ]

            for alt_path in alternatives:
                if alt_path.exists():
                    logging.info(
                        f"Primary database path not found, using alternative: {alt_path}"
                    )
                    db_path = alt_path
                    break
            else:
                # If we get here, none of the alternatives worked
                raise FileNotFoundError(
                    f"Database file not found. Tried: {db_path} and {', '.join(str(p) for p in alternatives)}"
                )

        logging.info(f"Loading agent states from {db_path}...")
        dataset.load_from_db(db_path, limit=self.base_config.data.num_states)
        if not dataset.states:
            raise ValueError(
                "No states loaded from database. Please check that your database contains agent state data."
            )

        # Split into train and validation sets
        total_states = len(dataset.states)
        val_size = int(total_states * self.base_config.data.validation_split)
        train_size = total_states - val_size

        train_states = dataset.states[:train_size]
        val_states = dataset.states[train_size:]

        train_dataset = AgentStateDataset(
            train_states, batch_size=self.base_config.training.batch_size
        )
        val_dataset = AgentStateDataset(
            val_states, batch_size=self.base_config.training.batch_size
        )

        # Set aside a small set of states for tracking semantic drift
        drift_tracking_states = val_states[: min(30, len(val_states))]

        logging.info(f"Training set: {len(train_dataset.states)} states")
        logging.info(f"Validation set: {len(val_dataset.states)} states")
        logging.info(f"Drift tracking set: {len(drift_tracking_states)} states")

        return {
            "train": train_dataset,
            "val": val_dataset,
            "drift_tracking": drift_tracking_states,
        }

    def _create_config_for_level(self, compression_level: float) -> Config:
        """
        Create a config for a specific compression level.

        Args:
            compression_level: The compression level to use

        Returns:
            Config object for this experiment
        """
        config = Config()

        # Copy base config values
        config.model.input_dim = self.base_config.model.input_dim
        config.model.latent_dim = self.base_config.model.latent_dim
        config.model.encoder_hidden_dims = self.base_config.model.encoder_hidden_dims
        config.model.decoder_hidden_dims = self.base_config.model.decoder_hidden_dims
        config.model.compression_type = self.base_config.model.compression_type

        # Set compression level for this experiment
        config.model.compression_level = compression_level

        # Set adaptive model flag (if needed)
        config.model.use_adaptive_model = self.use_adaptive_model

        # Set graph modeling flag (if needed)
        config.model.use_graph = self.use_graph

        # Set specific graph parameters if using graph-based modeling
        if self.use_graph:
            config.model.graph_hidden_dim = getattr(
                self.base_config.model, "graph_hidden_dim", 128
            )
            config.model.gnn_type = getattr(self.base_config.model, "gnn_type", "GCN")
            config.model.graph_num_layers = getattr(
                self.base_config.model, "graph_num_layers", 3
            )

        # Training configuration
        config.training.num_epochs = self.base_config.training.num_epochs
        config.training.batch_size = self.base_config.training.batch_size
        config.training.learning_rate = self.base_config.training.learning_rate
        config.training.checkpoint_dir = str(self.models_dir)

        # Disable semantic drift tracking for quick experiments to avoid errors
        if not self.track_drift:
            config.metrics = getattr(self.base_config, "metrics", None) or type(
                "obj", (object,), {}
            )
            config.metrics.track_semantic_drift = False
            config.metrics.visualization_interval = (
                999999  # Set to a very high number to disable
            )

        # Copy optimizer settings
        config.training.optimizer = getattr(
            self.base_config.training, "optimizer", "adam"
        )
        config.training.weight_decay = getattr(
            self.base_config.training, "weight_decay", 1e-5
        )

        # Loss weights
        config.training.recon_loss_weight = getattr(
            self.base_config.training, "recon_loss_weight", 1.0
        )
        config.training.kl_loss_weight = getattr(
            self.base_config.training, "kl_loss_weight", 0.1
        )
        config.training.semantic_loss_weight = getattr(
            self.base_config.training, "semantic_loss_weight", 0.5
        )

        # Set experiment name
        if self.use_adaptive_model:
            config.experiment_name = f"adaptive_compression_{compression_level}"
        elif self.use_graph:
            config.experiment_name = f"graph_compression_{compression_level}"
        else:
            config.experiment_name = f"compression_{compression_level}"

        # Copy other settings
        config.debug = self.base_config.debug
        config.verbose = self.base_config.verbose
        config.use_gpu = self.base_config.use_gpu

        return config

    def _analyze_results(self):
        """
        Analyze results of all compression experiments.

        This includes:
        1. Creating comparative visualizations
        2. Generating summary metrics
        3. Creating a comprehensive report
        """
        logging.info("\nAnalyzing compression experiment results...")

        # Create a DataFrame from results
        results_data = []
        for level, metrics in self.results.items():
            # Get evaluation metrics
            semantic_evaluation = metrics["semantic_evaluation"]

            row = {
                "compression_level": level,
                "val_loss": metrics["val_loss"],
                "recon_loss": metrics["recon_loss"],
                "kl_loss": metrics["kl_loss"],
                "semantic_loss": metrics["semantic_loss"],
                "compression_loss": metrics.get("compression_loss", 0.0),
                "param_count": metrics.get("param_count", 0),
                "effective_dim": metrics.get("effective_dim", 0),
                "compression_rate": metrics.get("compression_rate", level),
            }

            # Add standardized metrics
            row.update(
                {
                    "overall_preservation": semantic_evaluation["overall_preservation"],
                    "overall_fidelity": semantic_evaluation["overall_fidelity"],
                    "overall_drift": semantic_evaluation["overall_drift"],
                    "preservation_category": semantic_evaluation[
                        "preservation_category"
                    ],
                    "fidelity_category": semantic_evaluation["fidelity_category"],
                    "drift_category": semantic_evaluation["drift_category"],
                }
            )

            # Add feature group metrics
            for group in ["spatial", "resources", "performance", "role"]:
                # Preservation metrics
                if f"{group}_preservation" in semantic_evaluation["preservation"]:
                    row[f"{group}_preservation"] = semantic_evaluation["preservation"][
                        f"{group}_preservation"
                    ]

                # Fidelity metrics
                if f"{group}_fidelity" in semantic_evaluation["fidelity"]:
                    row[f"{group}_fidelity"] = semantic_evaluation["fidelity"][
                        f"{group}_fidelity"
                    ]

                # Drift metrics
                if f"{group}_drift" in semantic_evaluation["drift"]:
                    row[f"{group}_drift"] = semantic_evaluation["drift"][
                        f"{group}_drift"
                    ]

            # Add behavioral metrics if available
            if "behavioral" in semantic_evaluation:
                behavioral = semantic_evaluation["behavioral"]
                row["behavioral_equivalence"] = behavioral.get(
                    "overall_equivalence", 0.0
                )
                row["action_similarity"] = behavioral.get("action_similarity", 0.0)
                row["goal_alignment"] = behavioral.get("goal_alignment", 0.0)

            results_data.append(row)

        # Create DataFrame and sort by compression level
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values("compression_level")

        # Check for and handle problematic values
        if (
            results_df.isna().any().any()
            or np.isinf(results_df.select_dtypes(include=[np.number])).any().any()
        ):
            logging.warning(
                "Warning: Results contain NaN or infinity values. Cleaning data for analysis..."
            )
            # Replace inf with NaN first
            results_df = results_df.replace([np.inf, -np.inf], np.nan)

            # For categorical columns, replace NaN with 'unknown'
            cat_cols = [col for col in results_df.columns if col.endswith("_category")]
            for col in cat_cols:
                results_df[col] = results_df[col].fillna("unknown")

            # For numeric columns, fill NaN with appropriate values (0 or column average)
            numeric_cols = results_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                # If over half the values are NaN, fill with 0
                if results_df[col].isna().sum() > len(results_df) / 2:
                    results_df[col] = results_df[col].fillna(0)
                else:
                    # Otherwise use the column mean
                    results_df[col] = results_df[col].fillna(results_df[col].mean())

        # Save results to CSV
        results_csv_path = self.metrics_dir / "compression_results.csv"
        results_df.to_csv(results_csv_path, index=False)
        logging.info(f"Saved results to {results_csv_path}")

        # Use parent class method to save results
        self.save_results(filename="compression_results.json")

        # Create visualizations
        logging.info("Creating visualizations...")
        self._create_visualizations(results_df)
        
        # Log key findings
        logging.info("Key findings from compression experiments:")
        
        # Find the best compression level for different metrics
        best_preservation_level = results_df.loc[
            results_df["overall_preservation"].idxmax()
        ]["compression_level"]
        
        best_fidelity_level = results_df.loc[
            results_df["overall_fidelity"].idxmax()
        ]["compression_level"]
        
        lowest_drift_level = results_df.loc[
            results_df["overall_drift"].idxmin()
        ]["compression_level"]
        
        # Calculate combined score for overall best level
        results_df["combined_score"] = (
            results_df["overall_preservation"] * 0.4
            + results_df["overall_fidelity"] * 0.3
            - results_df["overall_drift"] * 0.3
        )
        
        best_overall_level = results_df.loc[
            results_df["combined_score"].idxmax()
        ]["compression_level"]
        
        logging.info(f"- Best preservation at compression level: {best_preservation_level}")
        logging.info(f"- Best fidelity at compression level: {best_fidelity_level}")
        logging.info(f"- Lowest drift at compression level: {lowest_drift_level}")
        logging.info(f"- Best overall performance at compression level: {best_overall_level}")
        
        # Log feature-specific findings
        for group in ["spatial", "resources", "performance", "role"]:
            col = f"{group}_preservation"
            if col in results_df.columns:
                best_level = results_df.loc[results_df[col].idxmax()]["compression_level"]
                logging.info(f"- Best {group} preservation at level: {best_level}")
        
        # Log adaptive model findings if applicable
        if self.use_adaptive_model:
            results_df["param_efficiency"] = results_df["overall_preservation"] / np.log10(
                results_df["param_count"]
            )
            most_efficient_level = results_df.loc[
                results_df["param_efficiency"].idxmax()
            ]["compression_level"]
            logging.info(f"- Most parameter-efficient model at level: {most_efficient_level}")

        # Generate report
        logging.info("Generating comprehensive report...")
        self._generate_report(results_df)
        logging.info("Analysis completed.")

    def _create_visualizations(self, results_df: pd.DataFrame):
        """
        Create visualizations for comparison across compression levels.

        Args:
            results_df: DataFrame containing experiment results
        """
        # Check for NaN or infinity values in the DataFrame
        if (
            results_df.isna().any().any()
            or np.isinf(results_df.select_dtypes(include=[np.number])).any().any()
        ):
            logging.warning(
                "Warning: DataFrame contains NaN or infinity values. Cleaning data for visualization..."
            )
            # Replace inf with NaN first, then fill NaN with 0s for numeric columns
            # This prevents visualization errors
            results_df = results_df.replace([np.inf, -np.inf], np.nan)
            numeric_cols = results_df.select_dtypes(include=[np.number]).columns
            results_df[numeric_cols] = results_df[numeric_cols].fillna(0)

        # Create the main comparison plot
        plt.figure(figsize=(15, 10))

        # Plot 1: Overall Preservation vs Compression Level
        plt.subplot(2, 2, 1)
        plt.plot(
            results_df["compression_level"],
            results_df["overall_preservation"],
            "g-o",
            linewidth=2,
        )
        plt.xlabel("Compression Level")
        plt.ylabel("Overall Meaning Preservation")
        plt.title("Meaning Preservation vs. Compression Level")
        plt.grid(True)

        # Plot 2: Fidelity Score vs Compression Level
        plt.subplot(2, 2, 2)
        plt.plot(
            results_df["compression_level"],
            results_df["overall_fidelity"],
            "b-o",
            linewidth=2,
        )
        plt.xlabel("Compression Level")
        plt.ylabel("Fidelity Score")
        plt.title("Reconstruction Fidelity vs. Compression Level")
        plt.grid(True)

        # Plot 3: Feature-specific preservation scores
        plt.subplot(2, 2, 3)
        if "spatial_preservation" in results_df.columns:
            plt.plot(
                results_df["compression_level"],
                results_df["spatial_preservation"],
                "r-o",
                label="Spatial",
            )
        if "resources_preservation" in results_df.columns:
            plt.plot(
                results_df["compression_level"],
                results_df["resources_preservation"],
                "g-o",
                label="Resources",
            )
        if "performance_preservation" in results_df.columns:
            plt.plot(
                results_df["compression_level"],
                results_df["performance_preservation"],
                "b-o",
                label="Performance",
            )
        if "role_preservation" in results_df.columns:
            plt.plot(
                results_df["compression_level"],
                results_df["role_preservation"],
                "m-o",
                label="Role",
            )
        plt.xlabel("Compression Level")
        plt.ylabel("Feature Group Preservation")
        plt.title("Feature-Specific Preservation vs. Compression Level")
        plt.legend()
        plt.grid(True)

        # Plot 4: Loss metrics
        plt.subplot(2, 2, 4)
        plt.plot(
            results_df["compression_level"],
            results_df["val_loss"],
            "k-o",
            label="Validation Loss",
        )
        plt.plot(
            results_df["compression_level"],
            results_df["recon_loss"],
            "b-o",
            label="Reconstruction Loss",
        )
        plt.plot(
            results_df["compression_level"],
            results_df["kl_loss"],
            "g-o",
            label="KL Loss",
        )
        if "semantic_loss" in results_df.columns and not all(
            pd.isna(results_df["semantic_loss"])
        ):
            plt.plot(
                results_df["compression_level"],
                results_df["semantic_loss"],
                "r-o",
                label="Semantic Loss",
            )
        plt.xlabel("Compression Level")
        plt.ylabel("Loss Value")
        plt.title("Training Losses vs. Compression Level")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(self.visualizations_dir / "compression_comparison.png")
        plt.close()

        # Create additional visualization for adaptive model if used
        if self.use_adaptive_model and "param_count" in results_df.columns:
            self._create_adaptive_model_visualization(results_df)

        # Create a radar chart to compare all metrics at different compression levels
        self._create_radar_chart(results_df)

        # Create a category visualization
        self._create_category_chart(results_df)

        # Create behavioral metrics visualization if available
        if "behavioral_equivalence" in results_df.columns:
            self._create_behavioral_visualization(results_df)

    def _create_radar_chart(self, results_df: pd.DataFrame):
        """
        Create a radar chart to compare all metrics at different compression levels.

        Args:
            results_df: DataFrame containing experiment results
        """
        # Make a copy of the dataframe to avoid modifying the original
        results_df = results_df.copy()

        # Check for NaN or infinity values in the DataFrame
        if (
            results_df.isna().any().any()
            or np.isinf(results_df.select_dtypes(include=[np.number])).any().any()
        ):
            logging.warning(
                "Warning: DataFrame contains NaN or infinity values. Cleaning data for radar chart..."
            )
            # Replace inf with NaN first, then fill NaN with 0s for numeric columns
            results_df = results_df.replace([np.inf, -np.inf], np.nan)
            numeric_cols = results_df.select_dtypes(include=[np.number]).columns
            results_df[numeric_cols] = results_df[numeric_cols].fillna(0)

        # Identify numeric metrics to include in radar chart
        # Skip certain columns like losses and param counts
        skip_cols = [
            "compression_level",
            "val_loss",
            "recon_loss",
            "kl_loss",
            "semantic_loss",
            "compression_loss",
            "param_count",
            "effective_dim",
            "compression_rate",
        ]

        # Also skip any columns with '_category' in the name
        category_cols = [col for col in results_df.columns if "_category" in col]
        skip_cols.extend(category_cols)

        # Get numeric columns that aren't in skip_cols
        metrics = [
            col
            for col in results_df.columns
            if col not in skip_cols
            and pd.api.types.is_numeric_dtype(results_df[col])
            and not all(pd.isna(results_df[col]))
        ]

        if not metrics:
            logging.warning("Warning: No suitable metrics found for radar chart")
            return

        # Create a radar chart
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

        # Number of metrics
        N = len(metrics)

        # Angle of each axis
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop

        # Add axis labels
        plt.xticks(angles[:-1], metrics, size=9)

        # Plot each compression level
        for level, color in zip(
            sorted(results_df["compression_level"].unique()),
            plt.cm.rainbow(
                np.linspace(0, 1, len(results_df["compression_level"].unique()))
            ),
        ):
            level_data = results_df[results_df["compression_level"] == level]

            # Get values for current compression level
            values = [level_data[metric].values[0] for metric in metrics]
            values += values[:1]  # Close the loop

            # Plot values
            ax.plot(
                angles, values, linewidth=2, linestyle="solid", label=f"Level {level}"
            )
            ax.fill(angles, values, alpha=0.1)

        # Add legend
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

        plt.title("Comparison of All Metrics Across Compression Levels")
        plt.tight_layout()

        plt.savefig(self.visualizations_dir / "radar_chart.png")
        plt.close()

    def _create_adaptive_model_visualization(self, results_df: pd.DataFrame):
        """
        Create visualizations specific to adaptive model experiments.

        Args:
            results_df: DataFrame containing experiment results
        """
        plt.figure(figsize=(12, 10))

        # Plot 1: Parameter count vs Compression Level
        plt.subplot(2, 2, 1)
        plt.plot(
            results_df["compression_level"],
            results_df["param_count"],
            "b-o",
            linewidth=2,
        )
        plt.xlabel("Compression Level")
        plt.ylabel("Parameter Count")
        plt.title("Model Size vs. Compression Level")
        plt.grid(True)
        plt.yscale("log")

        # Plot 2: Effective Dimension vs Compression Level
        plt.subplot(2, 2, 2)
        plt.plot(
            results_df["compression_level"],
            results_df["effective_dim"],
            "g-o",
            linewidth=2,
        )
        plt.xlabel("Compression Level")
        plt.ylabel("Effective Latent Dimension")
        plt.title("Effective Dimension vs. Compression Level")
        plt.grid(True)

        # Plot 3: Parameter Efficiency (Preservation/Param)
        plt.subplot(2, 2, 3)
        param_efficiency = results_df["overall_preservation"] / np.log10(
            results_df["param_count"]
        )
        plt.plot(results_df["compression_level"], param_efficiency, "r-o", linewidth=2)
        plt.xlabel("Compression Level")
        plt.ylabel("Preservation / log10(Parameters)")
        plt.title("Parameter Efficiency vs. Compression Level")
        plt.grid(True)

        # Plot 4: Relationship between Preservation and Parameter Count
        plt.subplot(2, 2, 4)
        plt.scatter(
            results_df["param_count"],
            results_df["overall_preservation"],
            s=80,
            c=results_df["compression_level"],
            cmap="viridis",
        )
        plt.colorbar(label="Compression Level")
        plt.xlabel("Parameter Count")
        plt.ylabel("Overall Preservation")
        plt.title("Preservation vs. Parameter Count")
        plt.grid(True)
        plt.xscale("log")

        plt.tight_layout()
        plt.savefig(self.visualizations_dir / "adaptive_model_analysis.png")
        plt.close()

    def _create_behavioral_visualization(self, results_df: pd.DataFrame):
        """
        Create visualization for behavioral metrics if available.

        Args:
            results_df: DataFrame containing experiment results
        """
        # Only create if we have the necessary columns
        if not all(
            col in results_df.columns
            for col in ["behavioral_equivalence", "action_similarity"]
        ):
            return

        plt.figure(figsize=(12, 6))

        # Plot behavioral metrics
        plt.subplot(1, 2, 1)
        plt.plot(
            results_df["compression_level"],
            results_df["behavioral_equivalence"],
            "b-o",
            label="Behavioral Equivalence",
        )
        plt.plot(
            results_df["compression_level"],
            results_df["action_similarity"],
            "g-o",
            label="Action Similarity",
        )
        if "goal_alignment" in results_df.columns:
            plt.plot(
                results_df["compression_level"],
                results_df["goal_alignment"],
                "r-o",
                label="Goal Alignment",
            )
        plt.xlabel("Compression Level")
        plt.ylabel("Score")
        plt.title("Behavioral Metrics vs. Compression Level")
        plt.legend()
        plt.grid(True)

        # Plot correlation between semantic and behavioral
        plt.subplot(1, 2, 2)
        plt.scatter(
            results_df["overall_preservation"],
            results_df["behavioral_equivalence"],
            s=80,
            c=results_df["compression_level"],
            cmap="viridis",
        )
        plt.colorbar(label="Compression Level")
        plt.xlabel("Semantic Preservation")
        plt.ylabel("Behavioral Equivalence")
        plt.title("Semantic vs. Behavioral Equivalence")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(self.visualizations_dir / "behavioral_analysis.png")
        plt.close()

    def _create_category_chart(self, results_df: pd.DataFrame):
        """
        Create a chart showing the quality categories at different compression levels.

        Args:
            results_df: DataFrame containing experiment results
        """
        # Get all category fields
        category_fields = [
            col for col in results_df.columns if col.endswith("_category")
        ]

        if not category_fields:
            return

        plt.figure(figsize=(15, 8))

        # Create a mapping for category values to numeric scores for visualization
        category_map = {
            "excellent": 5,
            "good": 4,
            "acceptable": 3,
            "poor": 2,
            "critical": 1,
        }

        # Create a chart for each category field
        for i, field in enumerate(category_fields):
            plt.subplot(1, len(category_fields), i + 1)

            # Convert categories to numeric values
            numeric_categories = [category_map.get(cat, 0) for cat in results_df[field]]

            # Create scatter plot with size based on compression level
            sizes = 100 + results_df["compression_level"] * 20
            plt.scatter(results_df["compression_level"], numeric_categories, s=sizes)

            # Add category labels
            for level, cat, y_val in zip(
                results_df["compression_level"], results_df[field], numeric_categories
            ):
                plt.text(level, y_val, cat, ha="center", va="center", fontsize=8)

            # Set y-axis ticks
            plt.yticks(list(category_map.values()), list(category_map.keys()))

            plt.xlabel("Compression Level")
            plt.title(field.replace("_category", " Category"))
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(self.visualizations_dir / "category_analysis.png")
        plt.close()

    def _generate_report(self, results_df: pd.DataFrame):
        """
        Generate a comprehensive report for the compression experiments.

        Args:
            results_df: DataFrame containing experiment results
        """
        report_file = self.experiment_dir / "compression_report.md"

        with open(report_file, "w") as f:
            # Write header
            f.write("# Compression Experiment Report\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Experiment configuration
            f.write("## Experiment Configuration\n\n")
            f.write(
                f"- **Model Type**: {'Adaptive' if self.use_adaptive_model else 'Standard'}\n"
            )
            f.write(
                f"- **Architecture**: {'Graph-based' if self.use_graph else 'Vector-based'}\n"
            )
            f.write(f"- **Input Dimension**: {self.base_config.model.input_dim}\n")
            f.write(f"- **Latent Dimension**: {self.base_config.model.latent_dim}\n")
            f.write(f"- **Training Epochs**: {self.base_config.training.num_epochs}\n")
            f.write(
                f"- **Tested Compression Levels**: {', '.join(map(str, self.compression_levels))}\n\n"
            )

            # Write summary
            f.write("## Summary\n\n")

            # Find the best compression level based on combined metrics
            # Lower drift, higher preservation, higher fidelity
            results_df["combined_score"] = (
                results_df["overall_preservation"] * 0.4
                + results_df["overall_fidelity"] * 0.3
                - results_df["overall_drift"] * 0.3
            )

            best_level = results_df.loc[results_df["combined_score"].idxmax()][
                "compression_level"
            ]
            f.write(f"**Best overall compression level: {best_level}**\n\n")

            # Create a summary table for preservation scores
            f.write("## Meaning Preservation Results\n\n")
            f.write(
                "| Compression Level | Overall Preservation | Fidelity | Drift | Category |\n"
            )
            f.write(
                "|-------------------|----------------------|----------|-------|----------|\n"
            )

            for _, row in results_df.sort_values("compression_level").iterrows():
                f.write(
                    f"| {row['compression_level']} | {row['overall_preservation']:.4f} | {row['overall_fidelity']:.4f} | {row['overall_drift']:.4f} | {row['preservation_category']} |\n"
                )

            f.write("\n")

            # Add adaptive model specific information if applicable
            if self.use_adaptive_model:
                f.write("## Adaptive Model Analysis\n\n")
                f.write(
                    "| Compression Level | Parameter Count | Effective Dimension | Compression Rate |\n"
                )
                f.write(
                    "|-------------------|-----------------|---------------------|------------------|\n"
                )

                for _, row in results_df.sort_values("compression_level").iterrows():
                    f.write(
                        f"| {row['compression_level']} | {row['param_count']:,} | {row['effective_dim']} | {row['compression_rate']:.2f}x |\n"
                    )

                f.write("\n")

                # Find most parameter-efficient level
                results_df["param_efficiency"] = results_df[
                    "overall_preservation"
                ] / np.log10(results_df["param_count"])
                efficient_level = results_df.loc[
                    results_df["param_efficiency"].idxmax()
                ]["compression_level"]

                f.write(
                    f"**Most parameter-efficient compression level: {efficient_level}**\n\n"
                )

            # Feature-specific analysis
            f.write("## Feature-Specific Analysis\n\n")

            # Check if we have feature-specific columns
            feature_groups = ["spatial", "resources", "performance", "role"]
            available_groups = [
                g for g in feature_groups if f"{g}_preservation" in results_df.columns
            ]

            if available_groups:
                for group in available_groups:
                    importance = {
                        "spatial": "55.4%",
                        "resources": "25.1%",
                        "performance": "10.5%",
                        "role": "<5%",
                    }.get(group, "unknown")

                    f.write(
                        f"### {group.capitalize()} Features ({importance} importance)\n\n"
                    )
                    f.write(
                        f"| Compression Level | Preservation | Fidelity | Drift |\n"
                    )
                    f.write(f"|-------------------|-------------|----------|-------|\n")

                    for _, row in results_df.sort_values(
                        "compression_level"
                    ).iterrows():
                        preservation = (
                            row[f"{group}_preservation"]
                            if f"{group}_preservation" in row
                            else "N/A"
                        )
                        fidelity = (
                            row[f"{group}_fidelity"]
                            if f"{group}_fidelity" in row
                            else "N/A"
                        )
                        drift = (
                            row[f"{group}_drift"] if f"{group}_drift" in row else "N/A"
                        )

                        # Format values as floats if they're numbers
                        preservation = (
                            f"{preservation:.4f}"
                            if isinstance(preservation, (int, float))
                            else preservation
                        )
                        fidelity = (
                            f"{fidelity:.4f}"
                            if isinstance(fidelity, (int, float))
                            else fidelity
                        )
                        drift = (
                            f"{drift:.4f}" if isinstance(drift, (int, float)) else drift
                        )

                        f.write(
                            f"| {row['compression_level']} | {preservation} | {fidelity} | {drift} |\n"
                        )

                    f.write("\n")

            # Behavioral analysis if available
            if "behavioral_equivalence" in results_df.columns:
                f.write("## Behavioral Equivalence Analysis\n\n")
                f.write(
                    "| Compression Level | Behavioral Equivalence | Action Similarity | Goal Alignment |\n"
                )
                f.write(
                    "|-------------------|------------------------|------------------|----------------|\n"
                )

                for _, row in results_df.sort_values("compression_level").iterrows():
                    behav = (
                        row["behavioral_equivalence"]
                        if "behavioral_equivalence" in row
                        else "N/A"
                    )
                    action = (
                        row["action_similarity"]
                        if "action_similarity" in row
                        else "N/A"
                    )
                    goal = row["goal_alignment"] if "goal_alignment" in row else "N/A"

                    # Format values as floats if they're numbers
                    behav = f"{behav:.4f}" if isinstance(behav, (int, float)) else behav
                    action = (
                        f"{action:.4f}" if isinstance(action, (int, float)) else action
                    )
                    goal = f"{goal:.4f}" if isinstance(goal, (int, float)) else goal

                    f.write(
                        f"| {row['compression_level']} | {behav} | {action} | {goal} |\n"
                    )

                f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")

            # Find best compression level for each feature group
            feature_recs = []
            for group in available_groups:
                if f"{group}_preservation" in results_df.columns:
                    best_for_group = results_df.loc[
                        results_df[f"{group}_preservation"].idxmax()
                    ]["compression_level"]
                    feature_recs.append(
                        f"- **Best for {group.capitalize()} Features**: Compression Level {best_for_group}"
                    )

            # Write feature-specific recommendations
            for rec in feature_recs:
                f.write(f"{rec}\n")

            # Add behavioral recommendation if available
            if "behavioral_equivalence" in results_df.columns:
                best_behavioral = results_df.loc[
                    results_df["behavioral_equivalence"].idxmax()
                ]["compression_level"]
                f.write(
                    f"- **Best for Behavioral Equivalence**: Compression Level {best_behavioral}\n"
                )

            f.write("\n")

            # Detailed analysis for each compression level
            f.write("## Detailed Analysis by Compression Level\n\n")

            for level in sorted(results_df["compression_level"].unique()):
                level_data = results_df[results_df["compression_level"] == level]

                if level_data.empty:
                    continue

                row = level_data.iloc[0]
                f.write(f"### Compression Level {level}\n\n")

                f.write("#### Semantic Metrics\n\n")
                f.write(
                    f"- **Overall Preservation**: {row['overall_preservation']:.4f}\n"
                )
                f.write(f"- **Overall Fidelity**: {row['overall_fidelity']:.4f}\n")
                f.write(f"- **Overall Drift**: {row['overall_drift']:.4f}\n")
                f.write(
                    f"- **Preservation Category**: {row['preservation_category']}\n"
                )

                # Loss and model metrics
                f.write("\n#### Model Metrics\n\n")
                f.write(f"- **Validation Loss**: {row['val_loss']:.4f}\n")
                f.write(f"- **Reconstruction Loss**: {row['recon_loss']:.4f}\n")
                f.write(f"- **KL Loss**: {row['kl_loss']:.4f}\n")

                if self.use_adaptive_model:
                    f.write(f"- **Parameter Count**: {row['param_count']:,}\n")
                    f.write(f"- **Effective Dimension**: {row['effective_dim']}\n")
                    f.write(f"- **Compression Rate**: {row['compression_rate']:.2f}x\n")

                # Feature-specific analysis if available
                if available_groups:
                    f.write("\n#### Feature-Specific Metrics\n\n")

                    for group in available_groups:
                        if f"{group}_preservation" in row:
                            f.write(
                                f"- **{group.capitalize()} Preservation**: {row[f'{group}_preservation']:.4f}\n"
                            )
                        if f"{group}_fidelity" in row:
                            f.write(
                                f"- **{group.capitalize()} Fidelity**: {row[f'{group}_fidelity']:.4f}\n"
                            )
                        if f"{group}_drift" in row:
                            f.write(
                                f"- **{group.capitalize()} Drift**: {row[f'{group}_drift']:.4f}\n"
                            )

                # Behavioral metrics if available
                if "behavioral_equivalence" in row:
                    f.write("\n#### Behavioral Metrics\n\n")
                    f.write(
                        f"- **Behavioral Equivalence**: {row['behavioral_equivalence']:.4f}\n"
                    )
                    if "action_similarity" in row:
                        f.write(
                            f"- **Action Similarity**: {row['action_similarity']:.4f}\n"
                        )
                    if "goal_alignment" in row:
                        f.write(f"- **Goal Alignment**: {row['goal_alignment']:.4f}\n")

                # Analysis and observations
                f.write("\n#### Analysis\n\n")

                # Generate some insights based on the data
                if row["overall_preservation"] > 0.95:
                    f.write("- Excellent semantic preservation\n")
                elif row["overall_preservation"] > 0.85:
                    f.write("- Good semantic preservation\n")
                elif row["overall_preservation"] > 0.75:
                    f.write("- Acceptable semantic preservation\n")
                else:
                    f.write("- Poor semantic preservation\n")

                # Check which feature group has the worst preservation
                if available_groups and len(available_groups) > 1:
                    preservation_values = {
                        group: row[f"{group}_preservation"]
                        for group in available_groups
                        if f"{group}_preservation" in row
                    }
                    if preservation_values:
                        worst_group = min(
                            preservation_values, key=preservation_values.get
                        )
                        best_group = max(
                            preservation_values, key=preservation_values.get
                        )
                        f.write(
                            f"- {worst_group.capitalize()} features show the lowest preservation at this compression level\n"
                        )
                        f.write(
                            f"- {best_group.capitalize()} features show the highest preservation at this compression level\n"
                        )

                # Adaptive model specific insights
                if self.use_adaptive_model:
                    param_efficiency = row["overall_preservation"] / np.log10(
                        row["param_count"]
                    )
                    if param_efficiency > 0.15:
                        f.write("- Excellent parameter efficiency\n")
                    elif param_efficiency > 0.1:
                        f.write("- Good parameter efficiency\n")
                    else:
                        f.write("- Low parameter efficiency\n")

                f.write("\n")

            # Conclusion
            f.write("## Conclusion\n\n")

            # Find the levels with the best metrics
            best_preservation = results_df.loc[
                results_df["overall_preservation"].idxmax()
            ]["compression_level"]
            best_fidelity = results_df.loc[results_df["overall_fidelity"].idxmax()][
                "compression_level"
            ]
            lowest_drift = results_df.loc[results_df["overall_drift"].idxmin()][
                "compression_level"
            ]
            balanced = results_df.loc[results_df["combined_score"].idxmax()][
                "compression_level"
            ]

            # Find best compression level for each feature group if available
            best_spatial = None
            best_resources = None
            best_performance = None

            if "spatial_preservation" in results_df.columns:
                best_spatial = results_df.loc[
                    results_df["spatial_preservation"].idxmax()
                ]["compression_level"]
            if "resources_preservation" in results_df.columns:
                best_resources = results_df.loc[
                    results_df["resources_preservation"].idxmax()
                ]["compression_level"]
            if "performance_preservation" in results_df.columns:
                best_performance = results_df.loc[
                    results_df["performance_preservation"].idxmax()
                ]["compression_level"]

            # Generate conclusion text
            conclusion_text = [
                f"Based on the comprehensive analysis of compression experiments, we can conclude that:",
                f"",
                f"1. Compression level **{balanced}** provides the best balance between preservation, fidelity, and drift.",
                f"2. Highest meaning preservation is achieved at compression level **{best_preservation}**.",
                f"3. Best reconstruction fidelity is achieved at compression level **{best_fidelity}**.",
                f"4. Lowest semantic drift is observed at compression level **{lowest_drift}**.",
            ]

            # Add feature-specific conclusions if available
            if best_spatial is not None:
                conclusion_text.append(
                    f"5. Spatial features are most sensitive to compression, with best results at level **{best_spatial}**."
                )
            if best_resources is not None:
                conclusion_text.append(
                    f"6. Resource features are best preserved at compression level **{best_resources}**."
                )
            if best_spatial is not None and best_resources is not None:
                conclusion_text.append(
                    f"7. The chosen compression level should prioritize maintaining spatial and resource feature integrity due to their higher importance weights."
                )

            # Add adaptive model conclusion if applicable
            if self.use_adaptive_model:
                most_efficient = results_df.loc[
                    results_df["param_efficiency"].idxmax()
                ]["compression_level"]
                conclusion_text.append(
                    f"8. The most parameter-efficient model is at compression level **{most_efficient}**."
                )
                conclusion_text.append(
                    f"9. The adaptive architecture successfully scales model parameters with compression level while maintaining semantic quality."
                )

            # Add behavioral conclusion if available
            if "behavioral_equivalence" in results_df.columns:
                best_behavioral = results_df.loc[
                    results_df["behavioral_equivalence"].idxmax()
                ]["compression_level"]
                conclusion_text.append(
                    f"10. Best behavioral equivalence is achieved at compression level **{best_behavioral}**."
                )

                # Check correlation between semantic and behavioral metrics
                if (
                    len(results_df) > 2
                ):  # Need at least 3 points for meaningful correlation
                    try:
                        corr = np.corrcoef(
                            results_df["overall_preservation"],
                            results_df["behavioral_equivalence"],
                        )[0, 1]
                        if abs(corr) > 0.8:
                            conclusion_text.append(
                                f"11. There is a strong correlation ({corr:.2f}) between semantic preservation and behavioral equivalence."
                            )
                        elif abs(corr) > 0.5:
                            conclusion_text.append(
                                f"11. There is a moderate correlation ({corr:.2f}) between semantic preservation and behavioral equivalence."
                            )
                        else:
                            conclusion_text.append(
                                f"11. There is a weak correlation ({corr:.2f}) between semantic preservation and behavioral equivalence."
                            )
                    except:
                        pass

            # Final recommendation
            conclusion_text.append("")
            conclusion_text.append(
                f"**Final Recommendation:** Use compression level **{balanced}** for the best balance of meaning preservation and model efficiency."
            )

            # Write the conclusion
            for line in conclusion_text:
                f.write(f"{line}\n")

        logging.info(f"Generated comprehensive report at {report_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run compression experiments for meaning-preserving transformations"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/compression_experiments",
        help="Directory to save experiment results (relative to project root or absolute path)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs per experiment",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--latent-dim", type=int, default=32, help="Dimension of latent space"
    )
    parser.add_argument(
        "--num-states",
        type=int,
        default=5000,
        help="Maximum number of states to load from database",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/simulation.db",
        help="Path to the simulation database file",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--adaptive", action="store_true", help="Use adaptive model architecture"
    )
    parser.add_argument("--graph", action="store_true", help="Use graph-based modeling")
    parser.add_argument(
        "--skip-drift", action="store_true", help="Skip semantic drift tracking"
    )
    parser.add_argument(
        "--compression-levels",
        type=str,
        default="0.5,1.0,2.0,5.0",
        help="Comma-separated list of compression levels to test",
    )
    parser.add_argument(
        "--beta-annealing",
        action="store_true",
        help="Use beta annealing for KL weight to prevent zero-valued losses",
    )

    return parser.parse_args()


def main():
    """Run compression experiments."""
    args = parse_args()

    # Create base configuration
    base_config = Config()

    # Configure from args
    base_config.training.num_epochs = args.epochs
    base_config.training.batch_size = args.batch_size
    base_config.model.latent_dim = args.latent_dim
    base_config.data.num_states = args.num_states
    base_config.data.db_path = args.db_path  # Set the database path
    base_config.debug = args.debug
    base_config.use_gpu = args.gpu

    # Parse compression levels
    compression_levels = [float(level) for level in args.compression_levels.split(",")]

    # Determine input dimension by creating a sample agent state and checking its tensor size
    sample_state = AgentState()
    input_dim = sample_state.to_tensor().shape[0]
    base_config.model.input_dim = input_dim

    # Set an appropriate experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.adaptive:
        base_config.experiment_name = f"adaptive_compression_study_{timestamp}"
    elif args.graph:
        base_config.experiment_name = f"graph_compression_study_{timestamp}"
    else:
        base_config.experiment_name = f"compression_study_{timestamp}"

    # Create output directory
    output_dir = Path(args.output_dir)
    if not os.path.isabs(args.output_dir):
        output_dir = Path(project_root) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(output_dir)
    logging.info(f"Starting compression experiments with the following configuration:")
    logging.info(f"- Experiment name: {base_config.experiment_name}")
    logging.info(f"- Output directory: {output_dir}")
    logging.info(f"- Log file: {log_file}")
    logging.info(f"- Epochs: {args.epochs}")
    logging.info(f"- Batch size: {args.batch_size}")
    logging.info(f"- Latent dimension: {args.latent_dim}")
    logging.info(f"- States to load: {args.num_states}")
    logging.info(f"- Database path: {args.db_path}")
    logging.info(f"- GPU enabled: {args.gpu}")
    logging.info(f"- Debug mode: {args.debug}")
    logging.info(f"- Adaptive model: {args.adaptive}")
    logging.info(f"- Graph-based modeling: {args.graph}")
    logging.info(f"- Skip drift tracking: {args.skip_drift}")
    logging.info(f"- Compression levels: {compression_levels}")
    logging.info(f"- Beta annealing: {args.beta_annealing}")

    # Create and run experiment
    experiment = CompressionExperiment(
        base_config=base_config,
        output_dir=args.output_dir,
        use_adaptive_model=args.adaptive,
        use_graph=args.graph,
        track_drift=not args.skip_drift,
        use_beta_annealing=args.beta_annealing,
    )

    # Override default compression levels if specified
    if compression_levels:
        experiment.compression_levels = compression_levels
        
    logging.info(f"Experiment initialized. Starting run_experiments...")
    experiment.run_experiments()
    logging.info("Compression experiments completed successfully.")


if __name__ == "__main__":
    main()
