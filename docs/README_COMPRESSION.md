# Compression Experiments for Meaning-Preserving Transformations

This README provides instructions for running experiments to analyze how different compression levels affect semantic preservation in the meaning-preserving transformation system.

## Overview

The compression experiments script runs a series of model training sessions with varying compression levels (0.5, 1.0, 2.0, 5.0) and analyzes the results to determine:

1. How different compression rates affect semantic preservation
2. The relationship between compression level and model performance metrics
3. The optimal compression setting for balancing information density with meaning retention

## Prerequisites

- Python 3.7+
- PyTorch 1.9+
- Required dependencies (install via `pip install -r requirements.txt`)

## Running the Experiments

To run the compression experiments with default settings:

```bash
python run_compression_experiments.py
```

### Command-line Arguments

The script accepts the following command-line arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--output-dir` | Directory to save experiment results | "results/compression_experiments" |
| `--epochs` | Number of training epochs per experiment | 50 |
| `--batch-size` | Batch size for training | 64 |
| `--latent-dim` | Dimension of latent space | 32 |
| `--num-states` | Number of synthetic states to generate | 5000 |
| `--use-real-data` | Use real data from simulation.db instead of synthetic data | False |
| `--gpu` | Use GPU for training | False |
| `--debug` | Enable debug mode | False |

### Example Usage

Run experiments with 100 epochs using GPU acceleration:

```bash
python run_compression_experiments.py --epochs 100 --gpu
```

Run experiments with a smaller latent dimension (for faster training):

```bash
python run_compression_experiments.py --latent-dim 16 --num-states 3000
```

Use real data from simulation.db (if available):

```bash
python run_compression_experiments.py --use-real-data
```

## Experiment Results

After completion, the experiments will generate the following outputs in the specified output directory:

1. Trained models for each compression level
2. CSV file with all metrics for each compression level
3. Visualizations showing the relationship between compression and:
   - Validation loss 
   - Semantic drift
   - Model size
   - Normalized comparison of all metrics
4. A comprehensive analysis report in Markdown format

### Analysis Report

The generated analysis report includes:

- Identification of optimal compression level
- Comparative performance metrics for all compression levels
- Analysis of how compression affects semantic preservation
- Recommendations for different use cases

## Interpreting Results

When analyzing the results, consider:

1. **Semantic Drift vs. Compression Level**: Lower drift indicates better meaning preservation. The relationship may not be linear.

2. **Information Density**: Higher compression typically means less storage required, but potentially more meaning loss.

3. **Trade-offs**: The optimal compression setting balances semantic preservation with storage efficiency.

## Customizing Experiments

If you wish to test different compression levels than the default (0.5, 1.0, 2.0, 5.0), you'll need to modify the `compression_levels` list in the `CompressionExperiment` class initialization in the source code. 