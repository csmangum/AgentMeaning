# Meaning Transform Library: Conceptual Glossary

## Core Concepts

### Agent State
A structured representation of an agent's properties at a point in time, including position, health, energy, inventory, role, goals, and other characteristics. AgentState serves as the fundamental unit being transformed and preserved through the system.

### Meaning Preservation
The core objective of maintaining semantic content through transformations, even as the structural representation changes. The system prioritizes preserving what an agent state "means" rather than just its raw data structure.

### Transformation Pipeline
The sequential process of converting an agent state through multiple representational forms: structured state → binary representation → latent encoding → compressed code → reconstructed state.

## Model Architecture

### MeaningVAE
The core neural model that learns to transform agent states into latent representations and back. It combines encoding, latent regularization, compression, and decoding to maintain semantic meaning across transformations.

### Encoder
Neural network component that maps agent states to a latent representation in a lower-dimensional space, capturing the essential semantic content of the state.

### Decoder
Neural network component that reconstructs agent states from their latent representations, attempting to preserve key semantic properties and relationships.

### Entropy Bottleneck
A compression mechanism that forces the latent representation to be more efficient by applying an entropy constraint, which helps distill the essential meaning from the state.

### Vector Quantizer
An alternative compression approach that maps continuous latent vectors to a discrete codebook, creating a more structured and compressed representation.

## Loss Functions

### Reconstruction Loss
Measures how accurately the model can reconstruct an agent state from its latent representation, using either mean squared error (MSE) or binary cross-entropy (BCE).

### KL Divergence Loss
Regularizes the latent space by encouraging the encoded representations to follow a normal distribution, balancing reconstruction accuracy with latent space structure.

### Semantic Loss
Evaluates preservation of higher-level semantic features between original and reconstructed states, focusing on meaningful properties rather than just structural similarity.

### Combined Loss
Integrates reconstruction, KL divergence, and semantic losses with configurable weights to balance different aspects of meaning preservation.

## Evaluation Metrics

### Semantic Metrics
Measurements that evaluate how well semantic properties are preserved in reconstructed states, including feature-specific similarities and overall semantic equivalence.

### Binary Feature Accuracy
Metrics for evaluating how well binary semantic properties (like "is_alive" or "threatened") are preserved across transformations.

### Numeric Feature Errors
Error measurements for continuous features like position, health, or energy, quantifying the degree of semantic drift in numeric properties.

### Role Accuracy
Specific metrics for evaluating how well agent roles (a key semantic property) are preserved during transformation and reconstruction.

## Drift Tracking

### Drift Tracker
A system for monitoring semantic drift over time or across different compression levels, helping identify thresholds where meaning begins to degrade.

### Latent Space Drift
Changes in the latent representation over time that may indicate semantic shift, even when reconstruction appears accurate.

### Semantic Drift
Gradual changes in the meaning of reconstructed states that accumulate over multiple transformations or compression cycles.

### Behavioral Drift
Changes in how an agent would behave given the reconstructed state versus the original state, a functional measure of meaning preservation.

## Taxonomy

### Transformation Types
Classification of different transformation approaches based on how they affect structure and semantics:
- Identity transformations (preserve both structure and meaning)
- Structural transformations (change structure while preserving meaning)
- Semantic transformations (preserve core meaning while allowing structural changes)
- Lossy transformations (lose some information while preserving essential meaning)

### Preservation Metrics
Formal measures for evaluating different aspects of preservation:
- Structural similarity
- Semantic equivalence
- Behavioral equivalence
- Information retention

## Visualization Tools

### T-SNE Visualization
Dimensionality reduction technique used to visualize high-dimensional latent spaces, helping identify clusters and relationships.

### Drift Visualization
Tools for tracking and visualizing semantic drift over time or compression levels, providing insight into meaning degradation patterns.

### Feature Importance Analysis
Methods for identifying which features contribute most to semantic preservation, helping prioritize aspects of meaning during transformation.

## Optimization Strategies

### Hyperparameter Tuning
Process of finding optimal model configurations that balance compression efficiency with semantic preservation.

### Compression Threshold Finder
Tool for identifying the maximum compression level that maintains semantic properties above a specified threshold.

### Feature-Based Optimization
Approach that weights different semantic features based on their importance for meaning preservation during training and evaluation. 