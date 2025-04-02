# Models Module

This module contains the core neural network components for the Meaning-Preserving Transformation System. These models work together to transform agent states across different representational forms while preserving semantic meaning.

## Components

### MeaningVAE (`meaning_vae.py`)

The central model of the system, implementing a Variational Autoencoder specifically designed for meaning preservation. Features:

- Support for both vector and graph-based representations of agent states
- Integration with multiple compression techniques (entropy bottleneck, adaptive entropy bottleneck, vector quantization)
- Ability to process knowledge graph representations using graph neural networks
- Interfaces for encoding, decoding, and measuring compression rates
- Reparameterization trick for sampling from latent distributions
- Configurable batch normalization and seed for reproducibility

### AdaptiveMeaningVAE (`adaptive_meaning_vae.py`)

An advanced VAE model that adaptively adjusts its bottleneck structure based on the compression level:
- Parameter count scales with compression level for efficiency
- Uses adaptive compression techniques to control information flow
- Provides simplified interface for dynamic compression control
- Reports effective dimension and compression rate metrics
- Compatible with standard encoder and decoder architectures

### FeatureGroupedVAE (`feature_grouped_vae.py`)

A specialized VAE that applies different compression rates to different feature groups:
- Allows preserving semantics of high-importance features while compressing others
- Supports configurable feature groups with individual compression levels
- Provides detailed analysis of per-group compression performance
- Balances overall compression with semantic preservation priorities
- Dynamically allocates latent space dimensions to feature groups based on importance

### Encoder (`encoder.py`)

Neural network module that maps agent states to latent representations:
- Configurable hidden layer architecture
- Outputs mean (μ) and log variance (log σ²) for the latent distribution
- Optional batch normalization for improved training stability
- Customizable activation functions (LeakyReLU by default)

### Decoder (`decoder.py`)

Neural network module that reconstructs agent states from latent representations:
- Mirror architecture of the encoder (in reverse)
- Configurable hidden layers
- Optional batch normalization
- Customizable activation functions (LeakyReLU by default)

### Compression Components

Multiple strategies for compressing the latent representation:

#### Vector Quantizer (`vector_quantizer.py`)

Implements Vector Quantized VAE (VQ-VAE) approach:
- Discrete latent representation through codebook lookup
- Learnable embedding space for discrete representations
- Measures codebook usage through perplexity metrics
- Uses straight-through estimator for gradient backpropagation
- Calculates compression rate based on codebook bit usage

#### Entropy Bottleneck (`entropy_bottleneck.py`)

Information-theoretic approach to latent compression:
- Adaptive compression levels through learnable parameters
- Stochastic during training, deterministic during inference
- Measures compression in terms of bits per dimension
- Allows control over compression-reconstruction trade-off
- Applies compression through projection and noise addition

#### AdaptiveEntropyBottleneck (`adaptive_entropy_bottleneck.py`)

Advanced entropy bottleneck that dynamically adjusts its structure based on compression level:
- Parameter count scales inversely with compression level for efficiency
- Projection-based dimensional reduction for effective compression
- Provides metrics for effective dimension and compression rate
- Supports deterministic compression during inference
- Efficiently compresses information through reduced dimensionality

## Utility Components

### Utils (`utils.py`)

Utility functions and base classes for the compression system:
- `BaseModelIO`: Common interface for model saving and loading
- `CompressionBase`: Base class for compression components with shared functionality
- `set_temp_seed`: Utility for temporarily setting random seed for reproducibility
- Various helper functions for data processing and metric calculation

## Usage

The models in this module form the core of the transformation system, allowing agent states to be encoded, compressed, and reconstructed while preserving their semantic meaning. The system supports both traditional vector-based representations and graph-structured data through integration with PyTorch Geometric.

The architecture is designed for modularity, allowing different compression techniques to be swapped in or combined based on specific requirements. The adaptive and feature-grouped approaches provide additional flexibility for balancing compression with semantic preservation. 