# Models Module

This module contains the core neural network components for the Meaning-Preserving Transformation System. These models work together to transform agent states across different representational forms while preserving semantic meaning.

## Components

### MeaningVAE (`meaning_vae.py`)

The central model of the system, implementing a Variational Autoencoder specifically designed for meaning preservation. Features:

- Support for both vector and graph-based representations of agent states
- Integration with multiple compression techniques
- Ability to process knowledge graph representations using graph neural networks
- Interfaces for encoding, decoding, and measuring compression rates

### Encoder (`encoder.py`)

Neural network module that maps agent states to latent representations:
- Configurable hidden layer architecture
- Outputs mean (μ) and log variance (log σ²) for the latent distribution
- Optional batch normalization for improved training stability

### Decoder (`decoder.py`)

Neural network module that reconstructs agent states from latent representations:
- Mirror architecture of the encoder (in reverse)
- Configurable hidden layers
- Optional batch normalization

### Compression Components

Two strategies for compressing the latent representation:

#### Vector Quantizer (`vector_quantizer.py`)

Implements Vector Quantized VAE (VQ-VAE) approach:
- Discrete latent representation through codebook lookup
- Learnable embedding space for discrete representations
- Measures codebook usage through perplexity metrics
- Uses straight-through estimator for gradient backpropagation

#### Entropy Bottleneck (`entropy_bottleneck.py`)

Information-theoretic approach to latent compression:
- Adaptive compression levels through learnable parameters
- Stochastic during training, deterministic during inference
- Measures compression in terms of bits per dimension
- Allows control over compression-reconstruction trade-off

## Usage

The models in this module form the core of the transformation system, allowing agent states to be encoded, compressed, and reconstructed while preserving their semantic meaning. The system supports both traditional vector-based representations and graph-structured data through integration with PyTorch Geometric. 