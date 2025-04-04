# Loss Function Debugging and Fixes

This document explains the changes made to fix the zero-valued loss issues in the Meaning-Preserving Transformation System.

## Issue Summary

The original implementation occasionally resulted in zero-valued reconstruction and KL divergence losses, which caused training to stall. The main issues identified were:

1. Lack of explicit validation to prevent or detect zero-valued losses
2. Numerical instability in KL divergence calculation
3. Missing device handling for tensors
4. No gradient monitoring during training

## Implemented Fixes

### 1. Loss Function Validation

Both the `ReconstructionLoss` and `KLDivergenceLoss` have been updated to include:
- Detection of NaN and Inf values in inputs
- Minimum thresholds to prevent zero losses (1e-8)
- Detailed debugging information when issues are detected
- Component-wise analysis of loss calculations

### 2. Combined Loss Improvements

The `CombinedLoss` and `FeatureWeightedLoss` classes now:
- Validate all inputs and intermediate tensors
- Print detailed debugging information for each loss component
- Track individual feature contributions in the weighted loss
- Enforce minimum total loss values (1e-6)

### 3. Beta Annealing

A new `beta_annealing` function has been added to gradually increase the KL weight during training, which helps avoid posterior collapse and stabilizes training. This is particularly useful for VAE models where balancing reconstruction and KL losses is crucial.

## How to Use Beta Annealing

### Basic Usage

```python
from meaning_transform.src.loss import beta_annealing, CombinedLoss

# In your training loop
for epoch in range(epochs):
    # Calculate beta for current epoch
    beta = beta_annealing(
        epoch=epoch,
        max_epochs=epochs,
        min_beta=0.0001,
        max_beta=1.0,
        schedule_type="sigmoid"
    )
    
    # Create loss function with current beta
    loss_fn = CombinedLoss(
        recon_loss_weight=1.0,
        kl_loss_weight=beta,
        semantic_loss_weight=0.5,
    )
    
    # Use loss_fn in your training step
    # ...
```

### Available Annealing Schedules

- **linear**: Linear progression from min_beta to max_beta
- **sigmoid**: Smooth sigmoid progression (recommended for most cases)
- **cyclical**: Cycles between min_beta and max_beta multiple times during training

## Debug Experiments

Two debug experiments have been added to help diagnose and understand loss issues:

1. `debug_loss.py`: Basic experiment to test loss functions with different weight configurations
2. `debug_loss_extended.py`: Advanced debugging with detailed component analysis and gradient tracking
3. `run_beta_annealing_debug.py`: Demonstration of beta annealing with visualization

To run these experiments:

```
python meaning_transform/experiment/debug_loss.py
python meaning_transform/experiment/debug_loss_extended.py
python meaning_transform/experiment/run_beta_annealing_debug.py
```

## Example: Modifying an Existing Experiment

To update an existing experiment to use beta annealing:

```python
# Add import
from meaning_transform.src.loss import beta_annealing

# In the training loop
def train(self, epochs=50, batch_size=64):
    for epoch in range(epochs):
        # Calculate beta
        beta = beta_annealing(epoch, epochs, min_beta=0.0001, max_beta=1.0)
        
        # Update loss function KL weight
        self.loss_fn.kl_loss_weight = beta
        
        # Continue with standard training
        # ...
```

## Troubleshooting

If you still encounter zero-valued losses:

1. Check model initialization (Xavier/Kaiming initialization may help)
2. Verify input data normalization (values in a reasonable range)
3. Try gradient clipping to avoid numerical instability
4. Use a smaller learning rate to avoid premature convergence
5. Increase the batch size to get more stable gradients
6. Check for issues in the encoder/decoder architecture

## Contact

If you encounter issues or have questions, please contact the development team. 