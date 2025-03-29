#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment Template Generator for Meaning-Preserving Transformations

This script generates .bat and .py files for new experiments based on templates.
It provides a structured way to create and manage experiments in a consistent manner.
"""

import argparse
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime

# Template for the Python script
PYTHON_TEMPLATE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
{exp_title} for Meaning-Preserving Transformations

{exp_description}
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.config import Config
from src.data import AgentState, AgentStateDataset
from src.metrics import SemanticMetrics, compute_feature_drift
from src.model import MeaningVAE
from src.train import Trainer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="{exp_title}")
    
    # Common arguments
    parser.add_argument("--output-dir", type=str, default="results/{exp_name}", 
                        help="Directory to save results")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--num-states", type=int, default=5000,
                        help="Number of agent states to use")
    parser.add_argument("--db-path", type=str, default="simulation.db",
                        help="Path to the simulation database")
    {custom_args}
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for training if available")
    
    return parser.parse_args()

def run_experiment(args):
    """Run the experiment with the given arguments."""
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the experiment configuration
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Set up the device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {{device}}")
    
    # TODO: Implement your experiment logic here
    {experiment_logic}
    
    # Save results
    print(f"Experiment completed! Results saved to {{args.output_dir}}")

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
'''

# Template for the batch script
BAT_TEMPLATE = '''@echo off
REM {exp_title} for Meaning-Preserving Transformations

echo Running {exp_name}...

REM Set Python path - update this if needed
set PYTHON=python

REM Create main results directory if it doesn't exist
if not exist "results" mkdir "results"

REM Create results directory if it doesn't exist
if not exist "results\\{results_dir}" mkdir "results\\{results_dir}"

REM Run {exp_name} with appropriate parameters
%PYTHON% meaning_transform/{script_path} ^
    --output-dir "results/{results_dir}" ^
    --epochs {epochs} ^
    --batch-size {batch_size} ^
    --num-states {num_states} ^
    --db-path "{db_path}" ^
{custom_params}    --gpu

echo {exp_title} completed! 
'''

# Template for the quick version batch script
QUICK_BAT_TEMPLATE = '''@echo off
REM Quick {exp_title} (reduced parameters for faster testing)

echo Running quick {exp_name}...

REM Set Python path - update this if needed
set PYTHON=python

REM Create results directory if it doesn't exist
if not exist "results\\{results_dir}" mkdir "results\\{results_dir}"

REM Run a smaller {exp_name} for quicker testing
%PYTHON% meaning_transform/{script_path} ^
    --output-dir "results/{results_dir}/quick_test" ^
    --epochs {quick_epochs} ^
    --batch-size {batch_size} ^
    --num-states {quick_num_states} ^
    --db-path "{db_path}" ^
{custom_params}    --gpu

echo Quick {exp_title} completed! 
'''

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create a new experiment template")
    parser.add_argument("name", type=str, help="Name of the experiment (snake_case)")
    parser.add_argument("--title", type=str, help="Title of the experiment")
    parser.add_argument("--description", type=str, default="This script runs experiments to analyze and evaluate the meaning-preserving transformation system.",
                        help="Description of the experiment")
    parser.add_argument("--custom-args", type=str, nargs="+", 
                        help="Custom arguments in format 'name:type:default:help'")
    parser.add_argument("--quick", action="store_true", 
                        help="Also generate a quick version batch file")
    return parser.parse_args()

def create_python_script(args, exp_name, output_dir):
    """Create the Python script for the experiment."""
    # Format custom arguments
    custom_args = ""
    if args.custom_args:
        for arg_str in args.custom_args:
            parts = arg_str.split(":")
            if len(parts) >= 3:
                name, type_name, default = parts[:3]
                help_text = parts[3] if len(parts) > 3 else f"{name.replace('_', ' ').title()}"
                custom_args += f'    parser.add_argument("--{name}", type={type_name}, default={default},\n'
                custom_args += f'                        help="{help_text}")\n'
    
    # Basic experiment logic placeholder
    experiment_logic = '''# Load data
    print(f"Loading {args.num_states} agent states from {args.db_path}")
    
    # Configure and train model
    config = Config()
    # Set configuration based on args
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train model
    training_history = trainer.train()
    
    # Analyze results
    # TODO: Add analysis code
    
    # Generate visualizations
    # TODO: Add visualization code'''
    
    # Fill the template
    script_content = PYTHON_TEMPLATE.format(
        exp_title=args.title,
        exp_description=args.description,
        exp_name=exp_name,
        custom_args=custom_args,
        experiment_logic=experiment_logic
    )
    
    # Write to file
    script_path = os.path.join(output_dir, f"run_{exp_name}.py")
    with open(script_path, "w") as f:
        f.write(script_content)
    
    return script_path

def create_batch_script(args, exp_name, output_dir, script_path, quick=False):
    """Create the batch script for the experiment."""
    # Format custom parameters
    custom_params = ""
    if args.custom_args:
        for arg_str in args.custom_args:
            parts = arg_str.split(":")
            if len(parts) >= 3:
                name, _, default = parts[:3]
                name_with_dashes = name.replace("_", "-")
                custom_params += f'    --{name_with_dashes} {default} ^\n'
    
    # Relative path to the Python script
    rel_script_path = os.path.relpath(script_path, start=os.path.dirname(output_dir))
    
    if quick:
        # Fill the quick template
        bat_content = QUICK_BAT_TEMPLATE.format(
            exp_title=args.title,
            exp_name=exp_name,
            results_dir=exp_name,
            script_path=rel_script_path,
            quick_epochs="10",
            batch_size="64",
            quick_num_states="1000",
            db_path="simulation.db",
            custom_params=custom_params
        )
        
        # Write to file
        bat_path = os.path.join(output_dir, f"run_quick_{exp_name}.bat")
    else:
        # Fill the template
        bat_content = BAT_TEMPLATE.format(
            exp_title=args.title,
            exp_name=exp_name,
            results_dir=exp_name,
            script_path=rel_script_path,
            epochs="30",
            batch_size="64",
            num_states="5000",
            db_path="simulation.db",
            custom_params=custom_params
        )
        
        # Write to file
        bat_path = os.path.join(output_dir, f"run_{exp_name}.bat")
    
    with open(bat_path, "w") as f:
        f.write(bat_content)
    
    return bat_path

def main():
    """Main function to create experiment templates."""
    args = parse_arguments()
    
    # Format the experiment name and title
    exp_name = args.name.lower().replace(" ", "_")
    if args.title:
        exp_title = args.title
    else:
        exp_title = " ".join(word.capitalize() for word in exp_name.split("_"))
    
    args.title = exp_title
    
    # Create the output directory
    output_dir = os.path.join("experiments")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create Python script
    script_path = create_python_script(args, exp_name, output_dir)
    print(f"Created Python script: {script_path}")
    
    # Create batch script
    bat_path = create_batch_script(args, exp_name, output_dir, script_path)
    print(f"Created batch script: {bat_path}")
    
    # Create quick batch script if requested
    if args.quick:
        quick_bat_path = create_batch_script(args, exp_name, output_dir, script_path, quick=True)
        print(f"Created quick batch script: {quick_bat_path}")
    
    print(f"\nExperiment '{exp_title}' templates created successfully!")
    print(f"To run the experiment, use: {bat_path}")
    if args.quick:
        print(f"For a quick test, use: {quick_bat_path}")

if __name__ == "__main__":
    main() 