# Meaning-Preserving Transformation System: Model Architecture

This document describes the core model architecture for the Meaning-Preserving Transformation System.

## Overview

The model architecture implements a variational autoencoder (VAE) with different compression mechanisms to transform agent states through multiple representational forms while preserving semantic meaning.

The full pipeline is:
```
Agent State → Binary Serialization → Tensor Representation → Latent Encoding → Compression → Reconstruction
```

## Key Components

### 1. Encoder (Lungs)

The encoder maps agent states to latent representations. It consists of:
- Multiple fully-connected layers with LeakyReLU activations and batch normalization
- Output layers for mean (μ) and log-variance (log σ²) of the latent distribution

### 2. Decoder (Heart)

The decoder reconstructs agent states from latent representations. It consists of:
- Multiple fully-connected layers with LeakyReLU activations and batch normalization
- Final output layer that maps to the original agent state dimension

### 3. Compression Mechanisms (Liver)

Two compression mechanisms are implemented:

#### Entropy Bottleneck

- Uses learnable parameters to compress the latent representation
- Adds adaptive noise during training for stochastic compression
- Provides an entropy loss term to measure compression rate

#### Vector Quantization (VQ)

- Maintains a codebook of learnable embedding vectors
- Maps continuous latent vectors to the nearest discrete codes
- Uses straight-through gradient estimation for backpropagation
- Provides perplexity metric to measure codebook utilization

## Usage

### Creating a Model

```python
from meaning_transform.src.model import MeaningVAE

# Create a VAE with entropy bottleneck compression
model_entropy = MeaningVAE(
    input_dim=15,  # Dimension of agent state tensors
    latent_dim=8,  # Dimension of latent space
    compression_type="entropy",
    compression_level=1.0  # Higher values = more compression
)

# Create a VAE with vector quantization
model_vq = MeaningVAE(
    input_dim=15,
    latent_dim=8,
    compression_type="vq",
    vq_num_embeddings=512  # Size of the codebook
)
```

### Processing Agent States

```python
from meaning_transform.src.data import AgentState, generate_agent_states

# Generate or load agent states
agent_states = generate_agent_states(count=100)

# Convert to tensors
tensor_batch = torch.stack([state.to_tensor() for state in agent_states])

# Forward pass through the model
results = model.forward(tensor_batch)

# Accessing results
reconstructed_states = results["x_reconstructed"]
latent_vectors = results["z"]
compressed_vectors = results["z_compressed"]
kl_loss = results["kl_loss"]
compression_loss = results["compression_loss"]
```

### Encoding and Decoding

```python
# Set model to evaluation mode for inference
model.eval()

# Encode an agent state to compressed latent representation
with torch.no_grad():
    agent_tensor = agent_state.to_tensor().unsqueeze(0)  # Add batch dimension
    latent = model.encode(agent_tensor)
    
    # Decode back to agent state representation
    reconstructed = model.decode(latent)
```

## Loss Functions

The model uses multiple loss components:

1. **Reconstruction Loss**: Measures how well the original agent state is reconstructed
2. **KL Divergence**: Regularizes the latent space distribution
3. **Compression Loss**: 
   - For Entropy Bottleneck: Measures the entropy of the compressed representation
   - For VQ: Consists of codebook loss and commitment loss

## Testing

A test script is provided to demonstrate the model functionality:

```bash
python test_model.py
```

The test script:
1. Generates synthetic agent states
2. Creates models with different compression mechanisms
3. Processes agent states through the models
4. Computes reconstruction errors
5. Demonstrates the full pipeline with a single agent state 