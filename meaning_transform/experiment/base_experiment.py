#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base Experiment Framework for Meaning-Preserving Transformations

This module provides a base class for experiments that handles common functionality:
- Directory structure creation
- Configuration management
- Result storage and tracking
- Common utility methods
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np


class BaseExperiment:
    """Base class for all meaning-preserving transformation experiments."""

    def __init__(
        self,
        base_config: Any = None,
        output_dir: str = None,
        experiment_name: str = None,
        project_root: Path = None,
    ):
        """
        Initialize base experiment.

        Args:
            base_config: Base configuration to use
            output_dir: Directory to save experiment results
            experiment_name: Name for this experiment (uses timestamp if not provided)
            project_root: Project root directory (auto-detected if not provided)
        """
        self.base_config = base_config

        # Create timestamp for experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"experiment_{timestamp}"

        # Determine project root if not provided
        if project_root is None:
            # Try to find project root by looking for setup.py or meaning_transform folder
            current_dir = Path(__file__).parent
            possible_root = current_dir.parent.parent
            if (possible_root / "setup.py").exists() or (possible_root / "meaning_transform").exists():
                project_root = possible_root
            else:
                # Default to two levels up from the current file
                project_root = current_dir.parent.parent

        self.project_root = project_root

        # Create output directory (relative to project root)
        if output_dir:
            # If absolute path is provided, use it as is
            if os.path.isabs(output_dir):
                self.output_dir = Path(output_dir)
            else:
                # Otherwise, make it relative to project root
                self.output_dir = project_root / output_dir
        else:
            # Default to results/{experiment_name} in project root
            self.output_dir = project_root / "results" / self.experiment_name

        self.experiment_dir = self.output_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create standard subdirectories
        self.models_dir = self.experiment_dir / "models"
        self.visualizations_dir = self.experiment_dir / "visualizations"
        self.metrics_dir = self.experiment_dir / "metrics"

        self.models_dir.mkdir(exist_ok=True)
        self.visualizations_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)

        # Initialize results dictionary
        self.results = {}

    def _save_config(self, config_dict: Dict[str, Any] = None, filename: str = "config.json"):
        """
        Save the configuration to a JSON file.

        Args:
            config_dict: Dictionary of configuration parameters (uses self.base_config if None)
            filename: Name of the file to save
        """
        if config_dict is None:
            # Convert the base config to a dictionary
            if hasattr(self.base_config, "__dict__"):
                config_dict = {}
                for section in ["model", "training", "data", "metrics"]:
                    if hasattr(self.base_config, section):
                        section_obj = getattr(self.base_config, section)
                        config_dict[section] = {
                            key: value
                            for key, value in vars(section_obj).items()
                            if not key.startswith("_")
                        }

                # Add top-level attributes
                for key, value in vars(self.base_config).items():
                    if not key.startswith("_") and key not in config_dict:
                        config_dict[key] = value
            elif isinstance(self.base_config, dict):
                config_dict = self.base_config
            else:
                config_dict = {"error": "Could not serialize config"}

        # Handle non-serializable types
        for section in config_dict:
            if isinstance(config_dict[section], dict):
                for key, value in config_dict[section].items():
                    if isinstance(value, tuple):
                        config_dict[section][key] = list(value)
                    elif isinstance(value, torch.Tensor):
                        config_dict[section][key] = value.tolist()
                    elif isinstance(value, np.ndarray):
                        config_dict[section][key] = value.tolist()

        config_path = self.experiment_dir / filename
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)

    def save_results(self, filename: str = "results.json"):
        """
        Save the experiment results to a JSON file.

        Args:
            filename: Name of the file to save
        """
        # Create a serializable copy of the results
        serializable_results = {}
        
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for inner_key, inner_value in value.items():
                    # Handle tensors, numpy arrays, and other non-serializable types
                    if isinstance(inner_value, torch.Tensor):
                        serializable_results[key][inner_key] = inner_value.tolist()
                    elif isinstance(inner_value, np.ndarray):
                        serializable_results[key][inner_key] = inner_value.tolist()
                    elif hasattr(inner_value, "tolist"):
                        serializable_results[key][inner_key] = inner_value.tolist()
                    elif hasattr(inner_value, "__dict__"):
                        serializable_results[key][inner_key] = str(inner_value)
                    else:
                        try:
                            # Try to serialize, use string representation if fails
                            json.dumps({inner_key: inner_value})
                            serializable_results[key][inner_key] = inner_value
                        except (TypeError, OverflowError):
                            serializable_results[key][inner_key] = str(inner_value)
            else:
                try:
                    # Try to serialize, use string representation if fails
                    json.dumps({key: value})
                    serializable_results[key] = value
                except (TypeError, OverflowError):
                    serializable_results[key] = str(value)

        # Save to file
        results_path = self.metrics_dir / filename
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=4)

    def _prepare_data(self):
        """
        Prepare data for experiments. This should be implemented by subclasses.
        
        Returns:
            Dictionary containing prepared datasets
        """
        raise NotImplementedError("Subclasses must implement _prepare_data")

    def run_experiments(self):
        """
        Run experiments. This should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement run_experiments")

    def _analyze_results(self):
        """
        Analyze results of all experiments. This should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _analyze_results") 