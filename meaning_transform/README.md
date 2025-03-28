# Meaning-Preserving Transformation System

A system that can translate structured information across layers of form—without losing the meaning that makes it matter.

## Project Overview

This project explores the preservation of semantic meaning across multiple representational forms, including:
- Binary encoding
- Latent abstraction
- Compressed representation

The core architecture is based on a Variational Autoencoder (VAE) with additional components for semantic preservation and drift tracking.

## Key Components

- **Agent State Generation**: Creating synthetic agent states with meaningful properties
- **VAE Architecture**: Encoder-decoder with compression mechanisms
- **Semantic Loss**: Multi-layered loss functions that preserve meaning
- **Drift Tracking**: Monitoring semantic equivalence across transformations

## Installation

```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
cd meaning_transform
pip install -e .

# For development dependencies
pip install -e ".[dev]"
```

## Usage

```python
from meaning_transform.src.data import AgentState, generate_agent_states
from meaning_transform.src.model import MeaningVAE
from meaning_transform.src.config import Config

# Load configuration
config = Config()

# Generate synthetic agent states
states = generate_agent_states(100)

# Initialize model
model = MeaningVAE(
    input_dim=config.model.input_dim,
    latent_dim=config.model.latent_dim,
    compression_type=config.model.compression_type
)

# Training and evaluation code will be added in future iterations
```

## Project Structure

```
meaning_transform/
├── src/
│   ├── data.py                # Agent state ingestion & serialization
│   ├── model.py               # VAE encoder/decoder with compression
│   ├── loss.py                # Multi-layered loss functions
│   ├── train.py               # Training loop with drift tracking
│   ├── config.py              # Hyperparameters & runtime flags
│   ├── embedding.py           # Text-to-embedding transformer module
├── utils/
│   ├── visualization.py       # Core visualization classes
│   ├── visualize.py           # High-level visualization interface
│   ├── metrics.py             # Semantic extraction & loss computation
│   ├── drift.py               # Cosine/Euclidean drift tracking tools
│   └── audit.py               # SemanticAuditLogger
├── taxonomy/
│   └── taxonomy.yaml          # Schema of transformation types
├── notebooks/
│   ├── experiment_structural_semantic.ipynb
│   └── experiment_drift_analysis.ipynb
├── examples/
│   └── visualization_examples.py  # Examples of using visualization tools
├── results/
│   ├── visualizations/        # Visualization outputs
│   │   ├── latent_space/      # t-SNE/PCA visualizations
│   │   ├── loss_curves/       # Training dynamics visualization
│   │   ├── state_comparison/  # Original vs reconstructed comparisons
│   │   └── drift_tracking/    # Semantic drift tracking
│   ├── drift_logs/            # Semantic drift data
```

## Contributing

This is an exploratory research project. Contributions are welcome to help explore the boundaries of meaning preservation across transformations.

## License

[License information]

## Project Components

### Data Handling
- `src/data.py`: Implements agent state serialization, deserialization, and dataset management
  - `AgentState`: Class representing an agent's state with semantic properties
  - `AgentStateDataset`: Dataset for loading and batching agent states
  - Binary serialization/deserialization for efficient storage and transfer
  - Tensor conversion for neural network input
  - Integration with simulation database (SQLite)
  - Synthetic data generation for testing and experimentation

### How to Use
To load agent states from the simulation database:
```python
from meaning_transform.src.data import AgentStateDataset

# Load a dataset from the simulation
dataset = AgentStateDataset()
dataset.load_from_db("path/to/simulation.db", limit=1000)

# Get a batch of states as tensors
batch = dataset.get_batch()  # Returns a tensor of shape [batch_size, feature_size]
```

To convert agent states to binary format and back:
```python
from meaning_transform.src.data import AgentState, serialize_states, deserialize_states

# Create agent states
states = [AgentState(...), AgentState(...)]

# Serialize to binary
binary_data = serialize_states(states)

# Deserialize from binary
reconstructed_states = deserialize_states(binary_data)
```

### Visualization Tools
- `utils/visualization.py`: Core visualization classes for analyzing models and results
- `utils/visualize.py`: High-level interface for easier visualization

The visualization module provides several tools for understanding model behavior:

1. **Latent Space Visualization**
   - t-SNE and PCA projections to understand latent space structure
   - Latent interpolation to visualize semantic transitions

2. **Loss and Training Dynamics**
   - Loss curves for tracking convergence
   - Compression vs. reconstruction trade-off analysis

3. **State Comparison**
   - Feature-by-feature comparison between original and reconstructed states
   - Trajectory visualization in state space
   - Confusion matrices for categorical features

4. **Semantic Drift Tracking**
   - Visualize how meaning degrades over compression levels
   - Find optimal compression thresholds that preserve semantics

#### Example Usage
```python
from meaning_transform.utils import visualize

# Setup visualization directories
visualize.setup_visualization_dirs()

# Visualize latent space 
output_paths = visualize.visualize_latent_space(
    latent_vectors=encoded_states,
    labels=state_labels
)

# Visualize semantic drift over compression levels
output_path = visualize.visualize_semantic_drift(
    iterations=compression_levels,
    semantic_scores=scores,
    compression_levels=compression_levels
)
```

For more detailed usage examples, see `examples/visualization_examples.py` and `utils/README_VISUALIZATION.md`. 