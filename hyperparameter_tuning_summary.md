# Meaning-Preserving Transformation: Hyperparameter Tuning Experiments Summary

## Experiment Overview

We conducted extensive hyperparameter tuning experiments for our Meaning-Preserving Transformation system, testing combinations of latent dimensions, compression levels, and semantic loss weights to find the optimal configuration for balancing semantic preservation with model efficiency.

## Experimental Setup

### Model Configuration
- **Input Dimension**: 15
- **Latent Dimensions Tested**: [16, 32, 64, 128]
- **Hidden Dimensions**: [256, 128, 64] (encoder), [64, 128, 256] (decoder)
- **Compression Type**: Entropy bottleneck
- **Compression Levels Tested**: [0.5, 1.0, 2.0]
- **VQ Embeddings**: 512
- **VQ Commitment Cost**: 0.25

### Training Configuration
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Weight Decay**: 1e-06
- **Number of Epochs**: 30
- **Semantic Loss Weights Tested**: [0.1, 0.5, 1.0, 2.0]
- **Optimizer**: Adam
- **Scheduler**: Cosine with step size 30, gamma 0.5

### Data Configuration
- **Number of States**: 5000 agent states from database
- **Validation Split**: 20%
- **Test Split**: 10%
- **Data Augmentation**: Enabled with noise level 0.05

## Results

### Top Performing Configurations

| Latent Dim | Compression | Semantic Weight | Val Loss | Semantic Drift | Model Size (KB) |
|------------|-------------|-----------------|----------|----------------|-----------------|
| 32 | 1.0 | 2.0 | 3858.78 | 1.35 | 423.92 |
| 16 | 1.0 | 0.5 | 4727.98 | 1.65 | 402.48 |
| 64 | 1.0 | 0.1 | 4244.66 | 1.40 | 484.80 |
| 128 | 1.0 | 2.0 | 3884.37 | 1.48 | 678.60 |

### Dimensional Analysis

- **Compression Level Effect**: Compression level 1.0 consistently outperformed both 0.5 and 2.0 across all latent dimensions, with semantic drift values in the 1.3-1.6 range compared to 3.6-7.6 for other levels.

- **Latent Dimension Effect**: Higher latent dimensions showed better average performance, but the optimal single configuration used latent dimension 32.

- **Semantic Weight Effect**: At compression level 1.0, higher semantic weights (1.0-2.0) generally improved performance, with the best result at 2.0 for the 32-dimensional model.

## Interpretation

Our hyperparameter tuning experiments revealed several key insights:

1. **Compression Level "Sweet Spot"**: A compression level of 1.0 represents an optimal balance point, with significantly better meaning preservation than either higher or lower compression levels.

2. **U-Shaped Performance Curve**: Compression demonstrates a U-shaped performance curve, where both under-compression (0.5) and over-compression (2.0) significantly degrade semantic preservation.

3. **Latent Dimension Trade-offs**: While larger latent dimensions (128) sometimes performed better, the 32-dimensional model with optimal compression and semantic weight achieved the best overall balance between model size and performance.

4. **Semantic Weight Effectiveness**: Higher semantic weights (especially 2.0) significantly improved meaning retention at the optimal compression level (1.0).

5. **Validation Loss Correlation**: Lower validation loss consistently correlated with better semantic preservation, with optimal configurations showing validation losses below 5000, compared to 24000-97000 for suboptimal configurations.

## Limitations

1. **Hyperparameter Resolution**: We tested discrete hyperparameter values with relatively large intervals, potentially missing finer optima between the tested values.

2. **Training Duration**: Each model was trained for a fixed number of epochs (30), whereas different hyperparameter configurations might benefit from different training durations.

3. **Architecture Invariance**: The same architecture was used across all experiments, limiting insights into how hyperparameters interact with different architectural choices.

4. **Semantic Feature Analysis**: The metrics don't break down semantic drift by specific features to identify which semantic properties are most affected by different hyperparameters.

## Next Steps

Based on our findings, we recommend the following next steps:

1. **Fine-tune Around Optimal Values**: Test narrower ranges around the optimal configuration (compression 0.8-1.2, semantic weights 1.5-2.5, latent dimensions 24-48).

2. **Adaptive Semantic Weight Scheduling**: Implement and test dynamic semantic weight adjustment during training.

3. **Architectural Modifications**: Explore how different encoder/decoder architectures interact with the optimal hyperparameters.

4. **Feature-Specific Analysis**: Analyze which semantic features benefit most from specific hyperparameter configurations.

5. **Extended Training**: Test if longer training durations with optimal hyperparameters can further improve semantic preservation.

## Conclusion

Our hyperparameter tuning experiments identified an optimal configuration (32D latent dimension, 1.0 compression, 2.0 semantic weight) that achieves excellent semantic preservation with reasonable model size. The clear U-shaped performance curve for compression levels highlights the importance of finding the right information density balance. The results demonstrate that proper hyperparameter selection can dramatically improve meaning preservation (semantic drift reduced from 7.6+ to 1.35) while maintaining computational efficiency. 