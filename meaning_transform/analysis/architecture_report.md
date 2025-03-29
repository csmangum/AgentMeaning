# Model Architecture Investigation Report

## 1. Problem Statement

Analysis of compression experiments revealed that the model size remains constant (422.7 KB) across all compression levels (0.5, 1.0, 2.0, 5.0). This is counterintuitive since higher compression should theoretically result in smaller model sizes. This investigation aims to:

1. Understand why model size remains constant despite varying compression levels
2. Develop alternative architectures that adapt their size to compression levels
3. Explore memory and storage optimization opportunities
4. Design feature-specific compression strategies based on feature importance findings

## 2. Current Architecture Analysis

### 2.1 Why Model Size Remains Constant

After examining the codebase, we found the following reasons why model size remains constant:

1. **Implementation of Compression Level**: The `compression_level` parameter in the `EntropyBottleneck` class only affects the forward pass behavior during training and inference. It scales the noise/quantization strength but doesn't change the architecture.

2. **Fixed Network Structure**: The model architecture (number of layers, neurons per layer) is fixed regardless of the compression level. The encoder, decoder, and compressor all have the same number of parameters.

3. **Parameter Count Determinants**: The parameter count is determined by the network dimensions (input_dim, latent_dim, hidden_dims) rather than the compression_level.

4. **Saved Model Size**: The size of saved models depends on the parameter count, not on how those parameters are used during inference.

5. **Compression Mechanism**: The entropy bottleneck performs compression during forward passes but maintains fixed parameter dimensions.

```python
# Current entropy bottleneck implementation
def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Project to get adaptive mu and log_scale
    projection = self.proj_compress(z)
    mu, log_scale = torch.chunk(projection, 2, dim=-1)
    
    # Apply base parameters with adaptive adjustments
    mu = mu + self.compress_mu
    log_scale = log_scale + self.compress_log_scale
    
    # Scale compression based on compression_level
    log_scale = log_scale - np.log(self.compression_level)
    # ...
```

The compression_level only affects the log_scale calculation, but not the network architecture or parameter count.

### 2.2 Compression vs Semantic Preservation

From the findings in compression experiments:

- Clear inverse relationship between compression level and semantic preservation
- Semantic drift increased from 4.49 at compression 0.5 to 10.85 at compression 5.0
- Validation loss increased from 82,021 at compression 0.5 to 598,654 at compression 5.0

## 3. Alternative Architecture Approaches

### 3.1 Adaptive Dimension Architecture

We propose an adaptive architecture that actually changes its structure based on compression level:

```python
class AdaptiveEntropyBottleneck(torch.nn.Module):
    def __init__(self, latent_dim: int, compression_level: float = 1.0):
        # ...
        # Calculate effective dimension based on compression level
        self.effective_dim = max(1, int(latent_dim / compression_level))
        
        # Learnable parameters sized to effective dimension
        self.compress_mu = torch.nn.Parameter(torch.zeros(self.effective_dim))
        self.compress_log_scale = torch.nn.Parameter(torch.zeros(self.effective_dim))
        
        # Projection layers with adaptive dimensions
        self.proj_down = torch.nn.Linear(latent_dim, self.effective_dim)
        self.nonlin = torch.nn.LeakyReLU()
        self.proj_up = torch.nn.Linear(self.effective_dim, latent_dim * 2)
```

Benefits:
- Model size varies with compression level (higher compression = fewer parameters)
- Memory usage scales proportionally to effective dimension
- Compression level has physical meaning in terms of network architecture
- More efficient for deployment at high compression settings

### 3.2 Pruning-Based Architecture

Another approach is to apply network pruning based on compression level:

1. Start with a full-sized model
2. Apply pruning to reduce parameter count based on compression level
3. Remove neurons/connections with least impact on semantic preservation
4. Fine-tune the pruned model to recover performance

Benefits:
- Can maintain semantic preservation of important connections
- Explicitly reduces parameter count and memory footprint
- More interpretable relationship between compression and model size

### 3.3 Feature-Specific Compression Architecture

Based on the feature importance findings (Step 12), we can design an architecture that applies different compression rates to different feature groups:

- Spatial features (55.4% importance): Apply low compression (0.5x)
- Resource features (25.1% importance): Apply medium compression (2.0x)
- Other features (<10% each): Apply high compression (5.0x)

Implementation options:
1. Separate encoder/decoder branches for each feature group
2. Multi-head bottleneck with different dimensions per group
3. Feature-weighted loss function

Benefits:
- Better semantic preservation of critical features
- More efficient use of parameter budget
- Improved overall meaning retention at the same total compression level

## 4. Memory and Storage Optimization

### 4.1 Current State

The current model has a fixed size of 422.7 KB across all compression levels. Parameter breakdown:
- Encoder: ~60%
- Decoder: ~35%
- Compressor: ~5%

### 4.2 Optimization Opportunities

1. **Dynamic Layer Sizing**: Adjust hidden layer dimensions based on compression level
2. **Quantization**: Use lower precision for weights (16-bit or 8-bit) at higher compression
3. **Sparse Representations**: Enforce sparsity in weights proportional to compression level
4. **Knowledge Distillation**: Train smaller models for high compression levels

## 5. Experimental Design

We've designed a comprehensive experimental plan to investigate these approaches:

1. **Baseline Architecture Analysis**: Detailed parameter count and memory analysis
2. **Adaptive Dimension Architecture**: Implementation and testing of the adaptively sized model
3. **Pruned Model Architecture**: Testing pruning techniques at different compression levels
4. **Feature-Grouped Architecture**: Testing feature-specific compression architectures
5. **Dynamic Dropout Architecture**: Testing variable dropout rates based on compression

Metrics to evaluate:
- Parameter count (total and by component)
- Model file size
- Semantic preservation (overall and by feature group)
- Training time and inference speed
- Memory usage during training and inference

## 6. Recommendations

Based on our investigation, we recommend:

1. **Implement the Adaptive Dimension Architecture**: This provides a direct relationship between compression level and model size while maintaining semantic preservation.

2. **Develop Feature-Specific Compression**: This approach optimizes compression based on feature importance, preserving critical semantic properties while reducing model size.

3. **Test Pruning Techniques**: For post-training optimization, implement pruning to further reduce model size at high compression levels.

4. **Extend Testing to Ultra-Low Compression**: Test compression levels below 0.5 (0.1, 0.25) to explore the lower bounds of the compression-preservation relationship.

## 7. Conclusion

The current architecture doesn't change model size with compression level due to its fixed parameter structure. By implementing an adaptive architecture that physically adjusts the network dimensions based on compression level, we can achieve more efficient models that truly reflect the compression-size relationship, while maintaining optimal semantic preservation for critical features.

The next steps involve implementing and evaluating these alternative architectures, focusing on both memory efficiency and semantic preservation quality. 