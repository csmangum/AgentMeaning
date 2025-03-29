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
import torch
import numpy as np
import traceback
from pathlib import Path

# Add the project root to the path so imports work correctly
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from meaning_transform.src.data import AgentState, generate_agent_states
    from meaning_transform.src.model import MeaningVAE
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("\nTrying alternative import paths...")
    try:
        from src.data import AgentState, generate_agent_states
        from src.model import MeaningVAE
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


def main():
    """Run test of model implementation."""
    print("Testing MeaningVAE implementation...")
    print_divider()
    
    try:
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
        
        # Create model instances with different compression methods
        print("Creating model instances...")
        try:
            # Standard VAE (no compression)
            model_standard = MeaningVAE(
                input_dim=INPUT_DIM,
                latent_dim=LATENT_DIM,
                compression_type="entropy",
                compression_level=0.1
            )
            print("Created standard VAE (with minimal compression)")
            
            # VAE with entropy bottleneck
            model_entropy = MeaningVAE(
                input_dim=INPUT_DIM,
                latent_dim=LATENT_DIM,
                compression_type="entropy",
                compression_level=2.0  # Higher compression
            )
            print("Created entropy bottleneck VAE")
            
            # VAE with vector quantization
            model_vq = MeaningVAE(
                input_dim=INPUT_DIM,
                latent_dim=LATENT_DIM,
                compression_type="vq",
                vq_num_embeddings=256
            )
            print("Created VQ-VAE")
        except Exception as e:
            print(f"Error creating models: {e}")
            print(traceback.format_exc())
            raise
        
        # Process batch through each model
        print_divider()
        print("Processing test batch through models...")
        
        # Create a test batch
        batch = tensors[:BATCH_SIZE]
        
        # Process through standard VAE
        print("\nStandard VAE (no compression):")
        results_standard = model_standard(batch)
        print(f"Input shape: {batch.shape}")
        print(f"Latent shape: {results_standard['z'].shape}")
        print(f"Output shape: {results_standard['x_reconstructed'].shape}")
        print(f"KL loss: {results_standard['kl_loss'].item():.4f}")
        
        # Process through entropy bottleneck VAE
        print("\nEntropy Bottleneck VAE:")
        results_entropy = model_entropy(batch)
        print(f"Input shape: {batch.shape}")
        print(f"Latent shape: {results_entropy['z'].shape}")
        print(f"Compressed shape: {results_entropy['z_compressed'].shape}")
        print(f"Output shape: {results_entropy['x_reconstructed'].shape}")
        print(f"KL loss: {results_entropy['kl_loss'].item():.4f}")
        print(f"Compression loss: {results_entropy['compression_loss'].item():.4f}")
        
        # Process through VQ-VAE
        print("\nVector Quantization VAE:")
        results_vq = model_vq(batch)
        print(f"Input shape: {batch.shape}")
        print(f"Latent shape: {results_vq['z'].shape}")
        print(f"Quantized shape: {results_vq['z_compressed'].shape}")
        print(f"Output shape: {results_vq['x_reconstructed'].shape}")
        print(f"KL loss: {results_vq['kl_loss'].item():.4f}")
        print(f"VQ loss: {results_vq['vq_loss'].item():.4f}")
        print(f"Codebook perplexity: {results_vq['perplexity'].item():.2f}")
        
        # Calculate reconstruction errors
        print_divider()
        print("Calculating reconstruction errors...")
        
        mse_standard = torch.mean((batch - results_standard['x_reconstructed'])**2).item()
        mse_entropy = torch.mean((batch - results_entropy['x_reconstructed'])**2).item()
        mse_vq = torch.mean((batch - results_vq['x_reconstructed'])**2).item()
        
        print(f"Standard VAE MSE: {mse_standard:.4f}")
        print(f"Entropy VAE MSE: {mse_entropy:.4f}")
        print(f"VQ-VAE MSE: {mse_vq:.4f}")
        
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
            "position_x", "position_y", "position_z", 
            "health", "energy", "resource_level", 
            "current_health", "is_defending", "age", "total_reward",
            "role_explorer", "role_gatherer", "role_defender", 
            "role_attacker", "role_builder"
        ]
        
        for i, (feature, error) in enumerate(zip(features, feature_errors)):
            print(f"{feature}: {error:.4f}")
        
        print_divider()
        print("Test completed successfully!")
    
    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        print("\nTest failed due to errors.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 