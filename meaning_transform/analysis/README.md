# Model Architecture Investigation

This directory contains scripts and documentation for Step 13 of the project implementation plan: Model Architecture Investigation.

## Contents

- `model_architecture_investigation.py` - Main analysis script that examines why model size remains constant despite varying compression levels
- `test_adaptive_model.py` - Test script that compares original and adaptive model architectures
- `architecture_report.md` - Detailed report documenting findings and recommendations
- `visualizations/` - Directory containing generated visualizations
- `results/` - Directory containing test results in JSON format

## Running the Investigation

Follow these steps to run the model architecture investigation:

1. **Analyze Current Architecture**

   ```
   python meaning_transform/analysis/model_architecture_investigation.py
   ```

   This script will:
   - Analyze why model size remains constant across compression levels
   - Print parameter counts for different components
   - Generate visualizations in the `visualizations/` directory

2. **Test Adaptive Architecture**

   ```
   python meaning_transform/analysis/test_adaptive_model.py
   ```

   This script will:
   - Compare model sizes between original and adaptive architectures
   - Test the feature-grouped model with different compression for different features
   - Evaluate reconstruction quality
   - Generate visualizations and save results

## Key Findings

1. **Current Architecture Analysis**
   - Model size remains constant because compression_level only affects forward pass behavior
   - The network dimensions stay fixed regardless of compression level
   - Compression is applied during inference but doesn't change the number of parameters

2. **Adaptive Architecture**
   - Implements a bottleneck that actually changes its size based on compression level
   - Higher compression = fewer parameters
   - Memory usage scales proportionally to compression level

3. **Feature-Specific Compression**
   - Applies different compression rates to different feature groups based on importance
   - Preserves critical features (spatial: 55.4%) with lower compression
   - Applies higher compression to less important features (status, role: <5% each)

## Visualization Outputs

The investigation generates several visualizations:
- Model size comparison between original and adaptive architectures
- Parameter count vs. compression level
- Effective dimension vs. compression level
- Reconstruction quality comparison
- Feature-specific reconstruction quality

## Next Steps

Based on the investigation, the following actions are recommended:

1. Implement the Adaptive Dimension Architecture in the main codebase
2. Develop Feature-Specific Compression for optimal semantic preservation
3. Test Pruning Techniques for post-training optimization
4. Extend Testing to Ultra-Low Compression levels 