# Experiment Management System

This directory contains tools for creating, managing, and running experiments for the Meaning-Preserving Transformation System.

## Overview

The experiment management system provides a consistent structure for:
- Creating new experiment templates
- Running experiments with standardized parameters
- Managing experiment results
- Creating summary reports

## Creating a New Experiment

Use the `create_experiment.py` script to create a new experiment template:

```bash
python experiments/create_experiment.py experiment_name [options]
```

Options:
- `--title "Experiment Title"` - Set the experiment title
- `--description "Description of the experiment"` - Set the experiment description
- `--custom-args "argument_name:type:default:help_text"` - Add custom command-line arguments
- `--quick` - Also generate a quick version batch file

Example:
```bash
python experiments/create_experiment.py semantic_preservation --title "Semantic Preservation Analysis" --description "This script analyzes how well meaning is preserved across transformations." --custom-args "preservation_threshold:float:0.85:Threshold for acceptable semantic preservation" "metrics:str:'drift,behavior,structure':Metrics to evaluate" --quick
```

This will create:
- `experiments/run_semantic_preservation.py` - Python script template for the experiment
- `experiments/run_semantic_preservation.bat` - Batch file to run the experiment
- `experiments/run_quick_semantic_preservation.bat` - Quick test version batch file

## Running Experiments

### Using Individual Batch Files

To run a single experiment, use its batch file:

```bash
experiments\run_semantic_preservation.bat
```

For a quick test with reduced parameters:

```bash
experiments\run_quick_semantic_preservation.bat
```

### Using the Experiment Manager (Batch)

The `run_experiments.bat` script provides a way to manage and run multiple experiments:

```bash
experiments\run_experiments.bat [options] [experiment_names...]
```

Options:
- `--list` - List all available experiments
- `--clean` - Clean the results directory
- `--report` - Generate a summary report
- `--quick` - Run experiments in quick mode
- `--results-dir DIR` - Specify a custom results directory
- `--help` - Show help message

Examples:
```bash
# List all available experiments
experiments\run_experiments.bat --list

# Run a single experiment in quick mode
experiments\run_experiments.bat --quick compression_experiments

# Run multiple experiments in sequence
experiments\run_experiments.bat compression_experiments feature_importance_analysis

# Clean the results directory
experiments\run_experiments.bat --clean
```

### Using the Experiment Manager (PowerShell)

For more advanced features, you can use the PowerShell script:

```powershell
.\experiments\run_experiments.ps1 [options]
```

Parameters:
- `-Experiments experiment1,experiment2` - Names of experiments to run
- `-QuickMode` - Run experiments in quick mode
- `-ListExperiments` - List all available experiments
- `-CleanResults` - Clean the results directory
- `-ResultsDir DIR` - Specify a custom results directory
- `-CreateReport` - Generate a summary report

Examples:
```powershell
# List all available experiments
.\experiments\run_experiments.ps1 -ListExperiments

# Run multiple experiments in quick mode
.\experiments\run_experiments.ps1 -Experiments compression_experiments,feature_importance_analysis -QuickMode

# Generate a summary report
.\experiments\run_experiments.ps1 -CreateReport
```

## Experiment Results

All experiment results are stored in the `results` directory at the project root by default, organized by experiment name. This ensures consistent storage regardless of where the experiment scripts are executed from. Each experiment directory typically contains:

- `config.json` - Experiment configuration
- Metrics files (e.g., `drift_metrics.json`, `performance_metrics.json`)
- Visualizations (PNG, SVG, or PDF files)
- Log files

### Path Handling

Experiment scripts follow these rules for output paths:
- Default path is always `<project_root>/results/<experiment_name>/`
- If a relative path is provided with `--output-dir`, it's resolved relative to the project root
- If an absolute path is provided, it's used as-is

This ensures consistency across all experiments and makes results easy to locate.

## Generating Reports

To generate a summary report of all experiment results:

```bash
experiments\run_experiments.bat --report
```

Or with PowerShell:

```powershell
.\experiments\run_experiments.ps1 -CreateReport
```

Reports are saved to `results/report/` and include summaries of experiment configurations, metrics, and visualizations.

## Best Practices

1. **Experiment Structure**:
   - Each experiment should focus on a specific aspect of the system
   - Use meaningful parameter names and sensible defaults
   - Include documentation in the script header

2. **Results Management**:
   - Save all results and configurations to the designated output directory
   - Include timestamps in filenames for versioning
   - Use standardized metrics formats for consistency

3. **Reproducibility**:
   - Save the full configuration with each experiment run
   - Include a seed for random operations if applicable
   - Document hardware/environment details in the results 