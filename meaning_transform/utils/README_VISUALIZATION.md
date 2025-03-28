# Visualization Tools

This directory contains visualization tools for the meaning-preserving transformation system. These tools help visualize latent space, training dynamics, state comparisons, and semantic drift.

## Overview

The visualization module provides four main types of visualizations:

1. **Latent Space Visualization**
   - t-SNE and PCA projections of latent space
   - Latent space interpolation between states

2. **Loss Curves and Training Dynamics**
   - Training/validation loss curves
   - Compression vs. reconstruction trade-off plots

3. **State Comparison**
   - Feature comparison between original and reconstructed states
   - State trajectories visualization
   - Confusion matrices for categorical features

4. **Semantic Drift Tracking**
   - Semantic drift over iterations or compression levels
   - Compression threshold finder

## Usage

### Quick Start

The simplest way to use these visualization tools is through the high-level API in `visualize.py`:

```python
from utils import visualize

# Setup visualization directories
visualize.setup_visualization_dirs()

# Visualize latent space
output_paths = visualize.visualize_latent_space(
    latent_vectors=encoded_states,
    labels=state_labels
)

# Visualize loss curves
output_path = visualize.visualize_loss_curves(
    loss_history=history
)

# Visualize state comparison
output_path = visualize.visualize_state_comparison(
    original_features=orig_features,
    reconstructed_features=recon_features
)

# Visualize semantic drift
output_path = visualize.visualize_semantic_drift(
    iterations=iterations,
    semantic_scores=scores
)
```

See `examples/visualization_examples.py` for complete examples of all visualization types.

### Advanced Usage

For more customized visualizations, you can use the underlying visualization classes directly:

```python
from utils.visualization import LatentSpaceVisualizer, LossVisualizer

# Create a latent space visualizer
latent_viz = LatentSpaceVisualizer(output_dir="custom/output/path")

# Create a custom t-SNE visualization
fig = latent_viz.visualize_tsne(
    latent_vectors=encoded_states,
    labels=state_labels,
    metadata={"custom": "metadata"}
)

# Customize the figure further
fig.suptitle("My Custom Title")
plt.savefig("custom_path.png")
```

## Visualization Classes

### `LatentSpaceVisualizer`

Visualizes the latent space of the model.

Methods:
- `visualize_tsne()`: Create t-SNE visualization of the latent space
- `visualize_pca()`: Create PCA visualization of the latent space
- `visualize_latent_interpolation()`: Visualize interpolation between two states in latent space

### `LossVisualizer`

Visualizes loss curves and training dynamics.

Methods:
- `update()`: Update loss history with new values
- `save_history()`: Save loss history to a JSON file
- `load_history()`: Load loss history from a JSON file
- `plot_losses()`: Plot loss curves
- `plot_compression_vs_reconstruction()`: Plot compression level vs. reconstruction error

### `StateComparisonVisualizer`

Visualizes comparisons between original and reconstructed states.

Methods:
- `plot_feature_comparison()`: Plot comparison of features between original and reconstructed states
- `plot_state_trajectories()`: Plot trajectories of original and reconstructed states
- `plot_confusion_matrices()`: Plot confusion matrices for categorical features

### `DriftVisualizer`

Visualizes semantic drift over training or compression levels.

Methods:
- `plot_semantic_drift()`: Plot semantic drift over training iterations or compression levels
- `plot_threshold_finder()`: Plot compression threshold finder results

## Integration with Training

To integrate these visualizations with your training loop, you can do the following:

```python
from utils import visualize

# Initialize loss visualizer
loss_viz = visualize.LossVisualizer()

# In your training loop
for epoch in range(num_epochs):
    # ... training code ...
    
    # Update loss history
    losses = {
        "total_loss": total_loss.item(),
        "reconstruction_loss": recon_loss.item(),
        "kl_loss": kl_loss.item(),
        "semantic_loss": semantic_loss.item()
    }
    loss_viz.update(epoch, losses)
    
    # Periodically generate visualizations
    if epoch % 10 == 0:
        # Visualize loss curves
        loss_viz.plot_losses(output_file=f"losses_epoch_{epoch}.png")
        
        # Visualize latent space
        with torch.no_grad():
            encoded_states = encoder(test_data)
        visualize.visualize_latent_space(
            latent_vectors=encoded_states,
            output_file=f"latent_epoch_{epoch}.png"
        )
        
        # Visualize reconstruction quality
        with torch.no_grad():
            reconstructed = model(test_data)
        visualize.visualize_state_comparison(
            original_features=extract_features(test_data),
            reconstructed_features=extract_features(reconstructed),
            output_file=f"comparison_epoch_{epoch}.png"
        )
```

## Output Directory Structure

By default, visualizations are saved in the following directory structure:

```
results/
└── visualizations/
    ├── latent_space/
    │   ├── latent_tsne.png
    │   ├── latent_pca.png
    │   └── latent_interpolation.png
    ├── loss_curves/
    │   ├── loss_curves.png
    │   ├── compression_vs_reconstruction.png
    │   └── loss_history.json
    ├── state_comparison/
    │   ├── feature_comparison.png
    │   ├── state_trajectories.png
    │   └── confusion_matrices.png
    └── drift_tracking/
        ├── semantic_drift.png
        └── threshold_finder.png
```

You can customize the output directories by passing the `output_dir` parameter to the visualization functions.

## Examples

See the `examples/visualization_examples.py` script for complete examples of all visualization types. 