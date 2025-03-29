# Meaning-Preserving Transformation: Compression Experiments Summary (March 28, 2025)

## Experiment Overview

We conducted a follow-up series of experiments testing different compression levels in our Meaning-Preserving Transformation system, evaluating how compression settings affect semantic preservation and model efficiency.

## Experimental Setup

### Model Configuration
- **Input Dimension**: 15
- **Latent Dimension**: 32
- **Hidden Dimensions**: [256, 128, 64] (encoder), [64, 128, 256] (decoder)
- **Compression Type**: Entropy bottleneck
- **Compression Levels Tested**: [0.5, 1.0, 2.0, 5.0]
- **VQ Embeddings**: 512
- **VQ Commitment Cost**: 0.25

### Training Configuration
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Weight Decay**: 1e-06
- **Number of Epochs**: 30
- **Loss Weights**: Reconstruction: 1.0, KL Divergence: 0.1, Semantic: 0.5
- **Optimizer**: Adam
- **Scheduler**: Cosine with step size 30, gamma 0.5

### Data Configuration
- **Number of States**: 5000 agent states from simulation database
- **Validation Split**: 20%
- **Test Split**: 10%

## Results

### Compression Performance

| Compression | Val Loss | Semantic Drift | Model Size (KB) |
|-------------|----------|----------------|-----------------|
| 0.5 | 30562.80 | 4.00 | 422.72 |
| 1.0 | 4305.50 | 1.50 | 422.72 |
| 2.0 | 43871.73 | 5.25 | 422.72 |
| 5.0 | 101700.88 | 7.47 | 422.72 |

### Semantic Drift Analysis

- Lowest semantic drift (1.50) was observed at compression level 1.0
- Highest semantic drift (7.47) was observed at compression level 5.0
- Clear trend showing semantic drift increases with higher compression levels
- Val loss follows a U-shaped curve with optimal point at 1.0

## Comparison with Previous Experiment

| Aspect | Previous Experiment | Current Experiment |
|--------|---------------------|-------------------|
| Optimal Compression | 0.5 | 1.0 |
| Training Epochs | 2 | 30 |
| Dataset Size | 50 synthetic states | 5000 real states |
| Best Semantic Drift | 4.49 | 1.50 |
| Worst Semantic Drift | 10.85 | 7.47 |
| Model Size | Constant (422.72 KB) | Constant (422.72 KB) |
| Val Loss Pattern | Monotonic increase | U-shaped curve |

Key observations:
- Extended training (30 vs 2 epochs) produced substantially better semantic preservation
- Using real data (5000 vs 50 states) yielded more nuanced compression behavior
- Mid-level compression (1.0) outperformed low compression (0.5) in the larger experiment
- Both experiments show constant model size regardless of compression level

## Interpretation

1. **Compression Sweet Spot**: Unlike the previous experiment where lowest compression was best, this experiment revealed an optimal mid-level compression point (1.0), suggesting a non-linear relationship between compression and semantic preservation.

2. **Training Duration Impact**: The 15× increase in training epochs dramatically improved the model's ability to preserve meaning at moderate compression.

3. **Data Quality Effect**: Using 100× more states from the simulation database rather than synthetic data produced more robust representations.

4. **Model Size Consistency**: The persistent issue of constant model size across compression levels indicates architectural limitations in the compression implementation.

## Next Steps

1. **Extended Training**: Further increase epochs (100+) to determine if representation quality continues to improve

2. **Ultra-Low Compression Test**: Evaluate compression levels between 0.5-1.0 (0.75, 0.85, 0.95) to find potential optimal points

3. **Architecture Revision**: Investigate why model size remains constant across compression levels

4. **Feature-Specific Analysis**: Examine how individual semantic properties respond to different compression levels

5. **Larger Dataset**: Scale to 20,000+ states to ensure sufficient training data for robust latent representations 