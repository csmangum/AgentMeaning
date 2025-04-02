# Experiment Ideas

This document tracks potential experiments to run and specific questions to investigate in our meaning-preserving transformation system.

## Feature-Weighted Loss Follow-up Experiments

### Spatial Feature Preservation Investigation
- **Goal**: Understand why position features are poorly preserved despite receiving highest importance weight
- **Approach**: 
  - Compare different position encodings (raw coordinates, relative positions, normalized positions)
  - Try specialized position-only encoder with geometric constraints
  - Experiment with increased latent dimensions specifically for spatial features
  - Analyze gradient flow through the network for position features
- **Metrics**: Position feature preservation score, overall semantic similarity, visualizations of spatial encodings

### Binary Feature Degradation Analysis
- **Goal**: Understand why "is_alive" feature degrades with feature-weighted loss (-79% preservation)
- **Approach**:
  - Isolate is_alive feature and train model with only this feature weighted
  - Analyze potential interactions between is_alive and other features
  - Test different activation functions for binary features
  - Examine loss surface around binary feature representations
- **Metrics**: Binary feature accuracy, feature interaction measures, visualization of decision boundaries

### Progressive Weight Schedule Optimization
- **Goal**: Find optimal weight adjustment strategies for stable training 
- **Approach**:
  - Compare linear, exponential, step-wise, and custom schedules
  - Vary progression speed (epochs to reach target weights)
  - Test different initial/final weight ratios
  - Implement curriculum learning approach that introduces features in importance order
- **Metrics**: Training stability, convergence speed, final semantic preservation by feature

### Architectural Integration Experiments
- **Goal**: Combine feature-weighted loss with specialized architectures
- **Approach**:
  - Integrate with feature-grouped VAE
  - Test with graph neural networks for relational features
  - Implement residual connections for high-importance features
  - Create hybrid architecture with separate encoders for continuous vs binary features
- **Metrics**: Meaning preservation across feature types, model size efficiency, training time

## Advanced Experiment Ideas

### Adaptive Weight Dynamics
- **Goal**: Create weights that adjust automatically based on training dynamics
- **Approach**:
  - Implement meta-learning approach that optimizes feature weights
  - Design loss function that detects preservation challenges and adjusts weights
  - Create reinforcement learning system that optimizes weights based on semantic metrics
- **Metrics**: Semantic preservation improvement, weight adjustment patterns

### Cross-Feature Interaction Modeling
- **Goal**: Understand how feature interactions affect meaning preservation
- **Approach**:
  - Create correlation matrix between features
  - Design loss components that capture feature relationships
  - Test joint encoding of related features
- **Metrics**: Interaction strength measures, joint preservation metrics

### Extreme Compression Threshold Analysis
- **Goal**: Find breaking points where feature-weighted loss fails for each feature type
- **Approach**:
  - Push compression to extreme levels (10x, 20x, 50x)
  - Measure semantic preservation by feature type
  - Find minimum latent dimensions required for each feature
- **Metrics**: Feature preservation vs. compression level curves, threshold identification
