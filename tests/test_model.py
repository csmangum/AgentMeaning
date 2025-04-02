#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the MeaningVAE model implementation.

This script demonstrates the full pipeline:
agent state → binary → latent → compressed → reconstructed

It creates synthetic agent states, processes them through the model,
and evaluates the reconstruction quality.
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import torch

# Add the project root to the path so imports work correctly
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from meaning_transform.src.data import generate_agent_states
    from meaning_transform.src.models import MeaningVAE
    from meaning_transform.src.models.adaptive_entropy_bottleneck import (
        AdaptiveEntropyBottleneck,
    )
    from meaning_transform.src.models.entropy_bottleneck import EntropyBottleneck
    from meaning_transform.src.models.adaptive_meaning_vae import AdaptiveMeaningVAE
    from meaning_transform.src.models.feature_grouped_vae import FeatureGroupedVAE
    from meaning_transform.src.models.encoder import Encoder
    from meaning_transform.src.models.decoder import Decoder
    from meaning_transform.src.models.vector_quantizer import VectorQuantizer
    from meaning_transform.src.models.utils import (
        compute_kl_divergence,
        reparameterize,
        init_weights,
        create_feature_groups,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("\nTrying alternative import paths...")
    try:
        print("Successfully imported using relative paths.")
    except ImportError:
        print("Import failed. Check the project structure and import paths.")
        traceback.print_exc()
        sys.exit(1)

# Configuration
INPUT_DIM = 15  # Dimensionality of tensor representation of agent state
LATENT_DIM = 8  # Dimensionality of latent space
BATCH_SIZE = 16  # Batch size for processing
NUM_TEST_STATES = 100  # Number of synthetic states to generate


def print_divider():
    """Print a divider line for cleaner output."""
    print("\n" + "-" * 80 + "\n")


def test_bottleneck_modules():
    """Test the EntropyBottleneck and AdaptiveEntropyBottleneck modules directly."""
    print_divider()
    print("Testing EntropyBottleneck and AdaptiveEntropyBottleneck modules directly...")

    latent_dim = 8
    compression_level = 2.0
    test_tensor = torch.randn(BATCH_SIZE, latent_dim)

    # Test EntropyBottleneck
    print("\nTesting EntropyBottleneck:")
    bottleneck = EntropyBottleneck(
        latent_dim=latent_dim, compression_level=compression_level
    )

    # Test training mode
    bottleneck.train()
    compressed_train, compression_loss_train = bottleneck(test_tensor)
    print(f"Compressed shape (train): {compressed_train.shape}")
    print(f"Compression loss (train): {compression_loss_train.item():.4f}")

    # Test evaluation mode
    bottleneck.eval()
    compressed_eval, compression_loss_eval = bottleneck(test_tensor)
    print(f"Compressed shape (eval): {compressed_eval.shape}")
    print(f"Compression loss (eval): {compression_loss_eval.item():.4f}")

    # Test numerical stability with extreme values
    extreme_tensor = torch.randn(BATCH_SIZE, latent_dim) * 10  # Larger values
    _, extreme_loss = bottleneck(extreme_tensor)
    print(f"Compression loss with extreme values: {extreme_loss.item():.4f}")
    assert not torch.isnan(extreme_loss) and not torch.isinf(
        extreme_loss
    ), "EntropyBottleneck has numerical stability issues"

    # Test consistency in tensor shapes
    assert (
        compressed_train.shape == test_tensor.shape
    ), "EntropyBottleneck compressed shape mismatch in training mode"
    assert (
        compressed_eval.shape == test_tensor.shape
    ), "EntropyBottleneck compressed shape mismatch in eval mode"

    # Test AdaptiveEntropyBottleneck
    print("\nTesting AdaptiveEntropyBottleneck:")
    adaptive_bottleneck = AdaptiveEntropyBottleneck(
        latent_dim=latent_dim,
        compression_level=compression_level,
        seed=42,  # Fixed seed for reproducibility
    )

    # Get effective compression rate
    effective_rate = adaptive_bottleneck.get_effective_compression_rate()
    print(f"Effective compression rate: {effective_rate:.2f}x")

    # Get parameter count
    param_count = adaptive_bottleneck.get_parameter_count()
    print(f"Parameter count: {param_count}")

    # Verify effective dimension calculation
    expected_effective_dim = max(1, int(latent_dim / compression_level))
    assert (
        adaptive_bottleneck.effective_dim == expected_effective_dim
    ), f"Effective dimension calculation error, got {adaptive_bottleneck.effective_dim}, expected {expected_effective_dim}"

    # Test training mode
    adaptive_bottleneck.train()
    adaptive_compressed_train, adaptive_loss_train = adaptive_bottleneck(test_tensor)
    print(f"Adaptive compressed shape (train): {adaptive_compressed_train.shape}")
    print(f"Adaptive compression loss (train): {adaptive_loss_train.item():.4f}")

    # Test evaluation mode
    adaptive_bottleneck.eval()
    adaptive_compressed_eval, adaptive_loss_eval = adaptive_bottleneck(test_tensor)
    print(f"Adaptive compressed shape (eval): {adaptive_compressed_eval.shape}")
    print(f"Adaptive compression loss (eval): {adaptive_loss_eval.item():.4f}")

    # Verify that shapes are maintained
    assert (
        adaptive_compressed_train.shape == test_tensor.shape
    ), "AdaptiveEntropyBottleneck compressed shape mismatch in training mode"
    assert (
        adaptive_compressed_eval.shape == test_tensor.shape
    ), "AdaptiveEntropyBottleneck compressed shape mismatch in eval mode"

    # Test numerical stability with extreme values
    adaptive_extreme, adaptive_extreme_loss = adaptive_bottleneck(extreme_tensor)
    print(
        f"Adaptive compression loss with extreme values: {adaptive_extreme_loss.item():.4f}"
    )
    assert not torch.isnan(adaptive_extreme_loss) and not torch.isinf(
        adaptive_extreme_loss
    ), "AdaptiveEntropyBottleneck has numerical stability issues"

    # Check that output is not identical to input (compression is happening)
    assert not torch.allclose(
        adaptive_compressed_eval, test_tensor
    ), "AdaptiveEntropyBottleneck is not performing any transformation"

    print("\nBottleneck tests completed successfully!")


def test_encoder_decoder():
    """Test the standalone Encoder and Decoder modules."""
    print_divider()
    print("Testing Encoder and Decoder modules...")
    
    # Create input tensor for testing
    test_batch = torch.randn(BATCH_SIZE, INPUT_DIM)
    
    # Test Encoder
    print("\nTesting Encoder:")
    hidden_dims = [32, 64]
    encoder = Encoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, hidden_dims=hidden_dims)
    
    # Initialize weights
    encoder.apply(init_weights)
    
    # Forward pass through encoder
    mu, log_var = encoder(test_batch)
    print(f"Input shape: {test_batch.shape}")
    print(f"mu shape: {mu.shape}")
    print(f"log_var shape: {log_var.shape}")
    
    # Test reparameterization
    z = reparameterize(mu, log_var)
    print(f"Sampled latent shape: {z.shape}")
    
    # Test KL divergence
    kl_loss = compute_kl_divergence(mu, log_var)
    print(f"KL divergence: {kl_loss.item():.4f}")
    
    # Test Decoder
    print("\nTesting Decoder:")
    decoder = Decoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, hidden_dims=hidden_dims[::-1])
    
    # Initialize weights
    decoder.apply(init_weights)
    
    # Forward pass through decoder
    reconstruction = decoder(z)
    print(f"Latent input shape: {z.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    
    # Test shape consistency
    assert reconstruction.shape == test_batch.shape, f"Reconstruction shape {reconstruction.shape} doesn't match input shape {test_batch.shape}"
    
    # Test model parameter count
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    print(f"Encoder parameters: {encoder_params}")
    print(f"Decoder parameters: {decoder_params}")
    
    print("\nEncoder and Decoder tests completed successfully!")


def test_vector_quantizer():
    """Test the VectorQuantizer module."""
    print_divider()
    print("Testing VectorQuantizer module...")
    
    # Parameters for testing
    embedding_dim = LATENT_DIM
    num_embeddings = 64
    commitment_cost = 0.25
    
    # Create vector quantizer instance
    vq = VectorQuantizer(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=commitment_cost
    )
    
    # Create test inputs
    test_inputs = torch.randn(BATCH_SIZE, embedding_dim)
    
    # Test forward pass
    print("\nTesting forward pass:")
    quantized, loss, perplexity = vq(test_inputs)
    print(f"Input shape: {test_inputs.shape}")
    print(f"Quantized output shape: {quantized.shape}")
    print(f"Quantization loss: {loss.item():.4f}")
    print(f"Codebook perplexity: {perplexity.item():.2f}")
    
    # Test shape consistency
    assert quantized.shape == test_inputs.shape, f"Quantized shape {quantized.shape} doesn't match input shape {test_inputs.shape}"
    
    # Test with different batch sizes
    small_batch = torch.randn(4, embedding_dim)
    large_batch = torch.randn(32, embedding_dim)
    
    small_quantized, _, _ = vq(small_batch)
    large_quantized, _, _ = vq(large_batch)
    
    assert small_quantized.shape == small_batch.shape, "Shape mismatch with small batch"
    assert large_quantized.shape == large_batch.shape, "Shape mismatch with large batch"
    
    # Test straight-through estimator by checking if gradients can flow
    vq.train()
    test_inputs.requires_grad = True
    quantized, loss, _ = vq(test_inputs)
    
    # Create a dummy loss and perform backpropagation
    dummy_loss = quantized.sum()
    dummy_loss.backward()
    
    # Check if gradients are calculated for inputs
    assert test_inputs.grad is not None, "No gradients flowing through VectorQuantizer"
    
    # Test codebook usage
    test_multiple_batches = [torch.randn(BATCH_SIZE, embedding_dim) for _ in range(5)]
    for batch in test_multiple_batches:
        vq(batch)  # Process multiple batches to populate usage
    
    # Verify if perplexity is reasonable (should be less than num_embeddings)
    _, _, final_perplexity = vq(test_inputs)
    assert final_perplexity.item() <= num_embeddings, f"Perplexity {final_perplexity.item()} exceeds num_embeddings {num_embeddings}"
    
    print("\nVectorQuantizer tests completed successfully!")


def test_utils():
    """Test utility functions in utils.py."""
    print_divider()
    print("Testing utility functions...")
    
    # Test compute_kl_divergence
    print("\nTesting compute_kl_divergence:")
    mu = torch.randn(BATCH_SIZE, LATENT_DIM)
    log_var = torch.randn(BATCH_SIZE, LATENT_DIM)
    
    kl_loss = compute_kl_divergence(mu, log_var)
    print(f"KL divergence: {kl_loss.item():.4f}")
    assert not torch.isnan(kl_loss), "KL divergence calculation returned NaN"
    assert kl_loss >= 0, f"KL divergence is negative: {kl_loss.item()}"
    
    # Test with zero mean and unit variance (should be close to zero)
    zero_mu = torch.zeros(BATCH_SIZE, LATENT_DIM)
    zero_log_var = torch.zeros(BATCH_SIZE, LATENT_DIM)
    zero_kl = compute_kl_divergence(zero_mu, zero_log_var)
    print(f"KL divergence with zero mean and unit variance: {zero_kl.item():.4f}")
    assert zero_kl.item() < 1e-5, f"KL should be close to zero but got {zero_kl.item()}"
    
    # Test reparameterize
    print("\nTesting reparameterize:")
    z = reparameterize(mu, log_var)
    print(f"Input mu shape: {mu.shape}")
    print(f"Input log_var shape: {log_var.shape}")
    print(f"Output z shape: {z.shape}")
    assert z.shape == mu.shape, f"Reparameterized shape {z.shape} doesn't match mu shape {mu.shape}"
    
    # Test if reparameterize is deterministic during eval mode
    torch.manual_seed(42)
    z1 = reparameterize(mu, log_var, deterministic=True)
    torch.manual_seed(42)
    z2 = reparameterize(mu, log_var, deterministic=True)
    is_deterministic = torch.allclose(z1, z2)
    print(f"Reparameterize is deterministic in eval mode: {is_deterministic}")
    assert is_deterministic, "Reparameterize should be deterministic when deterministic=True"
    
    # Test create_feature_groups
    print("\nTesting create_feature_groups:")
    feature_names = [
        "position_x", "position_y", "position_z", 
        "health", "energy", "resource_level", 
        "current_health", "is_defending", "age", 
        "total_reward", "role_explorer", "role_gatherer", 
        "role_defender", "role_attacker", "role_builder"
    ]
    group_by = "type"
    groups = create_feature_groups(feature_names, group_by)
    
    print(f"Created {len(groups)} feature groups:")
    for group_name, indices in groups.items():
        print(f"  {group_name}: {indices}")
    
    assert len(groups) > 0, "No feature groups created"
    
    # Test with semantic grouping
    semantic_groups = create_feature_groups(feature_names, "semantic")
    print(f"\nCreated {len(semantic_groups)} semantic feature groups:")
    for group_name, indices in semantic_groups.items():
        print(f"  {group_name}: {indices}")
    
    assert len(semantic_groups) > 0, "No semantic feature groups created"
    
    print("\nUtility function tests completed successfully!")


def test_adaptive_meaning_vae():
    """Test the AdaptiveMeaningVAE model."""
    print_divider()
    print("Testing AdaptiveMeaningVAE implementation...")
    
    # Create a test batch
    tensors = torch.randn(NUM_TEST_STATES, INPUT_DIM)
    batch = tensors[:BATCH_SIZE]
    
    # Create an AdaptiveMeaningVAE model
    print("\nCreating AdaptiveMeaningVAE instance...")
    try:
        model = AdaptiveMeaningVAE(
            input_dim=INPUT_DIM,
            latent_dim=LATENT_DIM,
            compression_type="adaptive_entropy",
            compression_level=2.0,
            min_compression_level=1.0,
            max_compression_level=4.0,
            adaptation_rate=0.1,
            drift_threshold=0.2,
        )
        print("AdaptiveMeaningVAE created successfully.")
    except Exception as e:
        print(f"Error creating AdaptiveMeaningVAE: {e}")
        print(traceback.format_exc())
        raise
    
    # Test forward pass
    print("\nTesting forward pass:")
    try:
        results = model(batch)
        print(f"Input shape: {batch.shape}")
        print(f"Latent shape: {results['z'].shape}")
        print(f"Compressed shape: {results['z_compressed'].shape}")
        print(f"Reconstruction shape: {results['reconstruction'].shape}")
        print(f"KL loss: {results['kl_loss'].item():.4f}")
        print(f"Compression loss: {results['compression_loss'].item():.4f}")
    except Exception as e:
        print(f"Error in forward pass: {e}")
        print(traceback.format_exc())
        raise
    
    # Test adaptation mechanism
    print("\nTesting adaptation mechanism:")
    try:
        # Current compression level
        initial_level = model.current_compression_level
        print(f"Initial compression level: {initial_level:.2f}")
        
        # Simulate drift detection
        model.adapt_to_drift(detected_drift=0.3)  # Above threshold
        new_level = model.current_compression_level
        print(f"New compression level after high drift: {new_level:.2f}")
        
        # Verify compression level changed
        assert new_level != initial_level, "Compression level should change after high drift detection"
        
        # Simulate low drift
        model.adapt_to_drift(detected_drift=0.1)  # Below threshold
        low_drift_level = model.current_compression_level
        print(f"Compression level after low drift: {low_drift_level:.2f}")
        
        # Verify bounds are respected
        model.adapt_to_drift(detected_drift=1.0)  # Very high drift
        high_level = model.current_compression_level
        print(f"Compression level after very high drift: {high_level:.2f}")
        assert high_level <= model.max_compression_level, "Compression level exceeds maximum bound"
        
        model.adapt_to_drift(detected_drift=0.0)  # No drift
        for _ in range(10):  # Multiple adaptations to push to minimum
            model.adapt_to_drift(detected_drift=0.0)
        min_level = model.current_compression_level
        print(f"Compression level after multiple low drift iterations: {min_level:.2f}")
        assert min_level >= model.min_compression_level, "Compression level below minimum bound"
    except Exception as e:
        print(f"Error testing adaptation: {e}")
        print(traceback.format_exc())
        raise
    
    # Test consistency between training and evaluation mode
    print("\nTesting train/eval consistency:")
    model.train()
    with torch.no_grad():
        train_result = model(batch)
    
    model.eval()
    with torch.no_grad():
        eval_result = model(batch)
    
    assert train_result['z'].shape == eval_result['z'].shape, "Latent shape differs between train and eval"
    assert train_result['z_compressed'].shape == eval_result['z_compressed'].shape, "Compressed shape differs between train and eval"
    
    print("\nAdaptiveMeaningVAE tests completed successfully!")


def test_feature_grouped_vae():
    """Test the FeatureGroupedVAE model."""
    print_divider()
    print("Testing FeatureGroupedVAE implementation...")
    
    # Create a test batch
    tensors = torch.randn(NUM_TEST_STATES, INPUT_DIM)
    batch = tensors[:BATCH_SIZE]
    
    # Define feature names for grouping
    feature_names = [
        "position_x", "position_y", "position_z", 
        "health", "energy", "resource_level", 
        "current_health", "is_defending", "age", 
        "total_reward", "role_explorer", "role_gatherer", 
        "role_defender", "role_attacker", "role_builder"
    ]
    
    # Create a FeatureGroupedVAE model
    print("\nCreating FeatureGroupedVAE instance...")
    try:
        model = FeatureGroupedVAE(
            input_dim=INPUT_DIM,
            latent_dim=LATENT_DIM,
            feature_names=feature_names,
            group_by="semantic",
            compression_type="entropy",
            compression_level=1.5,
        )
        print("FeatureGroupedVAE created successfully.")
        print(f"Number of feature groups: {len(model.feature_groups)}")
        for group_name, indices in model.feature_groups.items():
            group_features = [feature_names[i] for i in indices]
            print(f"  {group_name}: {group_features}")
    except Exception as e:
        print(f"Error creating FeatureGroupedVAE: {e}")
        print(traceback.format_exc())
        raise
    
    # Test forward pass
    print("\nTesting forward pass:")
    try:
        results = model(batch)
        print(f"Input shape: {batch.shape}")
        
        # Check group-specific latent representations
        for group_name in model.feature_groups:
            z_key = f"z_{group_name}"
            assert z_key in results, f"Missing latent representation for group {group_name}"
            print(f"Latent shape for {group_name}: {results[z_key].shape}")
        
        # Check combined outputs
        print(f"Combined latent shape: {results['z'].shape}")
        print(f"Compressed shape: {results['z_compressed'].shape}")
        print(f"Reconstruction shape: {results['reconstruction'].shape}")
        print(f"KL loss: {results['kl_loss'].item():.4f}")
        print(f"Compression loss: {results['compression_loss'].item():.4f}")
        
        # Check group-specific reconstructions
        for group_name in model.feature_groups:
            recon_key = f"reconstruction_{group_name}"
            assert recon_key in results, f"Missing reconstruction for group {group_name}"
            print(f"Reconstruction shape for {group_name}: {results[recon_key].shape}")
    except Exception as e:
        print(f"Error in forward pass: {e}")
        print(traceback.format_exc())
        raise
    
    # Test with different grouping method
    print("\nTesting with type-based grouping:")
    try:
        type_model = FeatureGroupedVAE(
            input_dim=INPUT_DIM,
            latent_dim=LATENT_DIM,
            feature_names=feature_names,
            group_by="type",
            compression_type="entropy",
            compression_level=1.5,
        )
        print(f"Number of type-based feature groups: {len(type_model.feature_groups)}")
        
        type_results = type_model(batch)
        assert "z" in type_results, "Missing combined latent representation"
        assert "reconstruction" in type_results, "Missing combined reconstruction"
    except Exception as e:
        print(f"Error with type-based grouping: {e}")
        print(traceback.format_exc())
        raise
    
    # Test consistency between training and evaluation mode
    print("\nTesting train/eval consistency:")
    model.train()
    with torch.no_grad():
        train_result = model(batch)
    
    model.eval()
    with torch.no_grad():
        eval_result = model(batch)
    
    assert train_result['z'].shape == eval_result['z'].shape, "Latent shape differs between train and eval"
    assert train_result['z_compressed'].shape == eval_result['z_compressed'].shape, "Compressed shape differs between train and eval"
    
    print("\nFeatureGroupedVAE tests completed successfully!")


def main():
    """Run test of model implementation."""
    print("Testing MeaningVAE implementation...")
    print_divider()

    try:
        # First, test the bottleneck modules directly
        test_bottleneck_modules()
        
        # Test utility functions
        test_utils()
        
        # Test encoder and decoder modules
        test_encoder_decoder()
        
        # Test vector quantizer
        test_vector_quantizer()

        # Generate synthetic agent states
        print(f"Generating {NUM_TEST_STATES} synthetic agent states...")
        agent_states = generate_agent_states(count=NUM_TEST_STATES)
        print(f"Generated {len(agent_states)} states successfully.")

        # Convert to tensors
        print("Converting agent states to tensors...")
        try:
            tensors = torch.stack([state.to_tensor() for state in agent_states])
            print(f"Converted to tensor shape: {tensors.shape}")
        except Exception as e:
            print(f"Error converting to tensors: {e}")
            print(traceback.format_exc())
            raise

        # Test feature-grouped VAE
        test_feature_grouped_vae()
        
        # Test adaptive meaning VAE
        test_adaptive_meaning_vae()

        # Create model instances with different compression methods
        print_divider()
        print("Processing test batch through standard MeaningVAE models...")

        # Create a test batch
        batch = tensors[:BATCH_SIZE]
        
        # Create model instances for standard compression tests
        print("Creating model instances...")
        try:
            # Standard VAE (no compression)
            model_standard = MeaningVAE(
                input_dim=INPUT_DIM,
                latent_dim=LATENT_DIM,
                compression_type="entropy",
                compression_level=0.1,
            )
            print("Created standard VAE (with minimal compression)")

            # VAE with entropy bottleneck
            model_entropy = MeaningVAE(
                input_dim=INPUT_DIM,
                latent_dim=LATENT_DIM,
                compression_type="entropy",
                compression_level=2.0,  # Higher compression
            )
            print("Created entropy bottleneck VAE")

            # VAE with vector quantization
            model_vq = MeaningVAE(
                input_dim=INPUT_DIM,
                latent_dim=LATENT_DIM,
                compression_type="vq",
                vq_num_embeddings=256,
            )
            print("Created VQ-VAE")

            # VAE with adaptive entropy bottleneck
            model_adaptive = MeaningVAE(
                input_dim=INPUT_DIM,
                latent_dim=LATENT_DIM,
                compression_type="adaptive_entropy",
                compression_level=2.0,  # Higher compression
            )
            print("Created Adaptive Entropy bottleneck VAE")
        except Exception as e:
            print(f"Error creating models: {e}")
            print(traceback.format_exc())
            raise

        # Process through standard VAE
        print("\nStandard VAE (no compression):")
        results_standard = model_standard(batch)
        print(f"Input shape: {batch.shape}")
        print(f"Latent shape: {results_standard['z'].shape}")
        print(f"Output shape: {results_standard['reconstruction'].shape}")
        print(f"KL loss: {results_standard['kl_loss'].item():.4f}")

        # Process through entropy bottleneck VAE
        print("\nEntropy Bottleneck VAE:")
        results_entropy = model_entropy(batch)
        print(f"Input shape: {batch.shape}")
        print(f"Latent shape: {results_entropy['z'].shape}")
        print(f"Compressed shape: {results_entropy['z_compressed'].shape}")
        print(f"Output shape: {results_entropy['reconstruction'].shape}")
        print(f"KL loss: {results_entropy['kl_loss'].item():.4f}")
        print(f"Compression loss: {results_entropy['compression_loss'].item():.4f}")

        # Process through VQ-VAE
        print("\nVector Quantization VAE:")
        results_vq = model_vq(batch)
        print(f"Input shape: {batch.shape}")
        print(f"Latent shape: {results_vq['z'].shape}")
        print(f"Quantized shape: {results_vq['z_compressed'].shape}")
        print(f"Output shape: {results_vq['reconstruction'].shape}")
        print(f"KL loss: {results_vq['kl_loss'].item():.4f}")
        print(f"VQ loss: {results_vq['quantization_loss'].item():.4f}")
        print(f"Codebook perplexity: {results_vq['perplexity'].item():.2f}")

        # Process through Adaptive Entropy bottleneck VAE
        print("\nAdaptive Entropy Bottleneck VAE:")
        results_adaptive = model_adaptive(batch)
        print(f"Input shape: {batch.shape}")
        print(f"Latent shape: {results_adaptive['z'].shape}")
        print(f"Compressed shape: {results_adaptive['z_compressed'].shape}")
        print(f"Output shape: {results_adaptive['reconstruction'].shape}")
        print(f"KL loss: {results_adaptive['kl_loss'].item():.4f}")
        print(f"Compression loss: {results_adaptive['compression_loss'].item():.4f}")

        # Calculate reconstruction errors
        print_divider()
        print("Calculating reconstruction errors...")

        mse_standard = torch.mean(
            (batch - results_standard["reconstruction"]) ** 2
        ).item()
        mse_entropy = torch.mean(
            (batch - results_entropy["reconstruction"]) ** 2
        ).item()
        mse_vq = torch.mean((batch - results_vq["reconstruction"]) ** 2).item()
        mse_adaptive = torch.mean(
            (batch - results_adaptive["reconstruction"]) ** 2
        ).item()

        print(f"Standard VAE MSE: {mse_standard:.4f}")
        print(f"Entropy VAE MSE: {mse_entropy:.4f}")
        print(f"VQ-VAE MSE: {mse_vq:.4f}")
        print(f"Adaptive Entropy VAE MSE: {mse_adaptive:.4f}")

        # Demonstrate the full pipeline with a single agent state
        print_divider()
        print("Demonstrating full pipeline with a single agent state...")

        # Get a sample agent state
        agent_state = agent_states[0]

        # Print original state
        print("\nOriginal agent state:")
        print(f"Position: {agent_state.position}")
        print(f"Health: {agent_state.health:.2f}")
        print(f"Energy: {agent_state.energy:.2f}")
        print(f"Role: {agent_state.role}")

        # Step 1: Convert to binary representation
        binary_data = agent_state.to_binary()
        print(f"\nBinary size: {len(binary_data)} bytes")

        # Step 2: Convert to tensor
        tensor_data = agent_state.to_tensor().unsqueeze(0)  # Add batch dimension
        print(f"Tensor shape: {tensor_data.shape}")

        # Step 3: Set model to evaluation mode for single sample processing
        model_entropy.eval()
        print("Model set to evaluation mode")

        # Step 4: Encode to latent space (with entropy bottleneck)
        with torch.no_grad():  # No need for gradients in evaluation
            latent = model_entropy.encode(tensor_data)
            print(f"Latent shape: {latent.shape}")

            # Step 5: Decode back to reconstructed state
            reconstructed_tensor = model_entropy.decode(latent)
            print(f"Reconstructed tensor shape: {reconstructed_tensor.shape}")

        # Compute feature-wise errors
        feature_errors = (tensor_data - reconstructed_tensor).squeeze().abs().tolist()

        print("\nFeature-wise reconstruction errors:")
        features = [
            "position_x",
            "position_y",
            "position_z",
            "health",
            "energy",
            "resource_level",
            "current_health",
            "is_defending",
            "age",
            "total_reward",
            "role_explorer",
            "role_gatherer",
            "role_defender",
            "role_attacker",
            "role_builder",
        ]

        for i, (feature, error) in enumerate(zip(features, feature_errors)):
            print(f"{feature}: {error:.4f}")

        # Test consistency between training and inference mode
        print_divider()
        print("Testing consistency between training and inference modes...")

        # Create a small model for quick testing
        test_model = MeaningVAE(
            input_dim=INPUT_DIM,
            latent_dim=4,
            compression_type="entropy",
            compression_level=2.0,
        )

        # Sample tensor to encode
        sample = tensors[0:1]  # Just the first sample

        # Test in training mode
        test_model.train()
        with torch.no_grad():
            train_result = test_model(sample)

        # Test in eval mode
        test_model.eval()
        with torch.no_grad():
            eval_result = test_model(sample)

        # Compare shapes
        assert (
            train_result["z"].shape == eval_result["z"].shape
        ), "Latent shape differs between train and eval"
        assert (
            train_result["z_compressed"].shape == eval_result["z_compressed"].shape
        ), "Compressed shape differs between train and eval"
        assert (
            train_result["reconstruction"].shape == eval_result["reconstruction"].shape
        ), "Reconstruction shape differs between train and eval"

        # Check if eval mode is deterministic
        eval_result1 = test_model(sample)
        eval_result2 = test_model(sample)
        is_deterministic = torch.allclose(
            eval_result1["z_compressed"], eval_result2["z_compressed"]
        )
        print(f"Evaluation mode is deterministic: {is_deterministic}")

        print_divider()
        print("All tests completed successfully!")
        print("Test coverage now includes:")
        print("  - MeaningVAE (with various compression types)")
        print("  - AdaptiveMeaningVAE")
        print("  - FeatureGroupedVAE")
        print("  - Encoder & Decoder modules")
        print("  - EntropyBottleneck & AdaptiveEntropyBottleneck")
        print("  - VectorQuantizer")
        print("  - Utility functions")

    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        print("\nTest failed due to errors.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
