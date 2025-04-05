import pytest
import torch
import numpy as np

from meaning_transform.src.models.entropy_bottleneck import EntropyBottleneck
from meaning_transform.src.models.meaning_vae import MeaningVAE


def test_entropy_bottleneck_scaling_behavior():
    """
    Test that EntropyBottleneck correctly scales latent representation differently
    in training vs inference mode.
    
    This test specifically validates the fix for the April 1st regression
    where mu was scaled during training, causing poor reconstructions.
    """
    # Test parameters
    latent_dim = 10
    compression_level = 2.0  # Use non-1.0 to detect scaling issues
    batch_size = 16
    seed = 42
    
    # Create test data and model
    torch.manual_seed(seed)
    z = torch.randn(batch_size, latent_dim)
    bottleneck = EntropyBottleneck(latent_dim=latent_dim, compression_level=compression_level, seed=seed)
    
    # Get scaling behavior in training mode
    bottleneck.train()
    z_train, _ = bottleneck(z)
    
    # Get scaling behavior in eval mode
    bottleneck.eval()
    z_eval, _ = bottleneck(z)
    
    # The key expectation: eval mode outputs should be more compressed (smaller values)
    # compared to training mode outputs
    train_mean_abs = torch.abs(z_train).mean().item()
    eval_mean_abs = torch.abs(z_eval).mean().item()
    
    # In eval mode, values should be approximately compression_level times smaller
    # We use a range to account for rounding effects
    scaling_ratio = train_mean_abs / eval_mean_abs
    
    print(f"Training mode mean abs: {train_mean_abs:.6f}")
    print(f"Eval mode mean abs: {eval_mean_abs:.6f}")
    print(f"Scaling ratio: {scaling_ratio:.6f} (expected ~{compression_level:.1f})")
    
    # The ratio should be approximately the compression level
    # Allow some tolerance due to stochastic factors
    assert 0.7 * compression_level < scaling_ratio < 1.3 * compression_level, \
        f"Expected scaling ratio near {compression_level}, got {scaling_ratio}"


def test_entropy_bottleneck_in_vae_context():
    """
    Test the EntropyBottleneck in the context of a full MeaningVAE model,
    ensuring reconstruction loss stays within expected ranges.
    """
    # Test parameters
    input_dim = 20
    latent_dim = 10
    compression_level = 2.0
    batch_size = 16
    seed = 42
    
    # Create model and test data
    torch.manual_seed(seed)
    model = MeaningVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        compression_type="entropy",
        compression_level=compression_level,
        seed=seed
    )
    
    # Test data
    x = torch.randn(batch_size, input_dim)
    
    # Test in training mode
    model.train()
    output_train = model(x)
    recon_train = output_train["reconstruction"]
    recon_loss_train = torch.nn.functional.mse_loss(recon_train, x).item()
    
    # Test in eval mode
    model.eval()
    output_eval = model(x)
    recon_eval = output_eval["reconstruction"]
    recon_loss_eval = torch.nn.functional.mse_loss(recon_eval, x).item()
    
    print(f"Training mode reconstruction loss: {recon_loss_train:.6f}")
    print(f"Eval mode reconstruction loss: {recon_loss_eval:.6f}")
    
    # Define acceptable loss range based on empirical observations
    # The exact values may need adjustment based on your model characteristics
    max_acceptable_loss = 15.0  # Adjust based on what's reasonably expected
    
    # We expect reconstruction loss to be reasonable (not extremely high)
    assert recon_loss_train < max_acceptable_loss, \
        f"Training reconstruction loss too high: {recon_loss_train}"
    assert recon_loss_eval < max_acceptable_loss, \
        f"Eval reconstruction loss too high: {recon_loss_eval}"
    
    # The ratio between train and eval reconstruction loss should be reasonable
    # In this specific model, eval loss should be higher due to quantization
    assert 0.5 < recon_loss_eval / recon_loss_train < 5.0, \
        f"Unexpected ratio between eval and train losses: {recon_loss_eval / recon_loss_train}"


def test_bottleneck_output_stability():
    """
    Test that the EntropyBottleneck produces stable outputs across
    repeated calls with the same input.
    """
    latent_dim = 10
    compression_level = 1.0
    batch_size = 16
    seed = 42
    
    # Create test data and model
    torch.manual_seed(seed)
    z = torch.randn(batch_size, latent_dim)
    bottleneck = EntropyBottleneck(latent_dim=latent_dim, compression_level=compression_level, seed=seed)
    
    # Get outputs in eval mode
    bottleneck.eval()
    z_eval1, _ = bottleneck(z)
    z_eval2, _ = bottleneck(z)
    
    # Outputs should be identical in eval mode (deterministic)
    assert torch.allclose(z_eval1, z_eval2), "EntropyBottleneck outputs not deterministic in eval mode"
    
    # Get outputs in train mode
    bottleneck.train()
    with torch.no_grad():  # Avoid accumulating gradients
        z_train1, _ = bottleneck(z)
        
        # Reset the seed to ensure repeatability
        torch.manual_seed(seed)
        bottleneck.seed = seed
        z_train2, _ = bottleneck(z)
    
    # With the same seed, outputs should be similar in train mode
    # (not identical due to randomness, but statistically similar)
    train_mean_diff = torch.abs(z_train1 - z_train2).mean().item()
    assert train_mean_diff < 0.1, f"Train mode outputs too different even with same seed: {train_mean_diff}" 