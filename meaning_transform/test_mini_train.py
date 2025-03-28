#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal training script to test metrics integration.
"""

import os
import torch
from src.config import Config
from src.train import Trainer

# Create a minimal configuration
config = Config()
config.experiment_name = "metrics_test"
config.training.num_epochs = 1
config.training.batch_size = 5
config.data.num_states = 20
config.debug = True
config.verbose = True

# Create trainer with explicit device setting
trainer = Trainer(config, device="cpu")

# Run training
print(f"Starting mini training run with {config.training.num_epochs} epochs...")
history = trainer.train()

# Check if the metrics directory and files were created
experiment_dir = history["experiment_dir"]
print(f"\nTraining completed. Results saved to: {experiment_dir}")

# Check for drift tracking files
drift_tracking_dir = os.path.join(experiment_dir, "drift_tracking")
if os.path.exists(drift_tracking_dir):
    print(f"Drift tracking directory created: {drift_tracking_dir}")
    drift_files = os.listdir(drift_tracking_dir)
    print(f"Found {len(drift_files)} drift tracking files")
else:
    print("Warning: Drift tracking directory not found")

# Check for visualization files
vis_dir = os.path.join(experiment_dir, "visualizations")
if os.path.exists(vis_dir):
    print(f"Visualizations directory created: {vis_dir}")
    vis_files = os.listdir(vis_dir)
    print(f"Found {len(vis_files)} visualization files")
else:
    print("Warning: Visualizations directory not found")

# Check if drift report was generated
drift_report = os.path.join(experiment_dir, "drift_report.md")
if os.path.exists(drift_report):
    print(f"Drift report generated: {drift_report}")
else:
    print("Warning: Drift report not found")

print("\nMini training test completed.") 