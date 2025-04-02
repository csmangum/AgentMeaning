#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the models module implementation.

This script tests the functionality of the models module, including:
- Encoder network
- Decoder network
- Vector Quantizer
- Entropy Bottleneck
- MeaningVAE model
"""

import sys
import os
from pathlib import Path
import pytest
import torch
import numpy as np

# Add the project root to the path so imports work correctly
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Skip all tests if necessary imports are not available
pytorch_available = True
try:
    import torch
    import torch.nn as nn
except ImportError:
    pytorch_available = False

pytorch_geometric_available = True
try:
    from torch_geometric.data import Data, Batch
except ImportError:
    pytorch_geometric_available = False

models_available = True
try:
    from meaning_transform.src.models.encoder import Encoder
    from meaning_transform.src.models.decoder import Decoder
    from meaning_transform.src.models.vector_quantizer import VectorQuantizer
    from meaning_transform.src.models.entropy_bottleneck import EntropyBottleneck
    from meaning_transform.src.models.meaning_vae import MeaningVAE
    from meaning_transform.src.data import AgentState, generate_agent_states
    from meaning_transform.src.knowledge_graph import AgentStateToGraph
except ImportError:
    models_available = False

# Decorator to skip tests if modules not available
skip_if_dependencies_missing = pytest.mark.skipif(
    not (pytorch_available and models_available),
    reason="PyTorch or models module not available"
)

skip_if_geometric_missing = pytest.mark.skipif(
    not (pytorch_available and models_available and pytorch_geometric_available),
    reason="PyTorch Geometric not available"
)


@skip_if_dependencies_missing
class TestEncoder:
    """Tests for the Encoder network."""

    def test_init(self):
        """Test initialization of Encoder."""
        input_dim = 15
        latent_dim = 8
        
        # Test with default parameters
        encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim)
        assert encoder.input_dim == input_dim
        assert encoder.latent_dim == latent_dim
        assert encoder.hidden_dims == [256, 128, 64]
        assert encoder.use_batch_norm is True
        
        # Test with custom parameters
        hidden_dims = [32, 16]
        encoder2 = Encoder(
            input_dim=input_dim, 
            latent_dim=latent_dim, 
            hidden_dims=hidden_dims,
            use_batch_norm=False
        )
        assert encoder2.hidden_dims == hidden_dims
        assert encoder2.use_batch_norm is False

    def test_forward(self):
        """Test forward pass of Encoder."""
        input_dim = 15
        latent_dim = 8
        batch_size = 4
        
        encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim)
        
        # Create random input
        x = torch.randn(batch_size, input_dim)
        
        # Forward pass
        mu, log_var = encoder(x)
        
        # Verify output shapes
        assert mu.shape == (batch_size, latent_dim)
        assert log_var.shape == (batch_size, latent_dim)
        
        # Verify different outputs for different inputs
        x2 = torch.randn(batch_size, input_dim)
        mu2, log_var2 = encoder(x2)
        
        assert not torch.allclose(mu, mu2)
        assert not torch.allclose(log_var, log_var2)


@skip_if_dependencies_missing
class TestDecoder:
    """Tests for the Decoder network."""

    def test_init(self):
        """Test initialization of Decoder."""
        latent_dim = 8
        output_dim = 15
        
        # Test with default parameters
        decoder = Decoder(latent_dim=latent_dim, output_dim=output_dim)
        assert decoder.latent_dim == latent_dim
        assert decoder.output_dim == output_dim
        assert decoder.hidden_dims == [64, 128, 256]
        assert decoder.use_batch_norm is True
        
        # Test with custom parameters
        hidden_dims = [16, 32]
        decoder2 = Decoder(
            latent_dim=latent_dim, 
            output_dim=output_dim, 
            hidden_dims=hidden_dims,
            use_batch_norm=False
        )
        assert decoder2.hidden_dims == hidden_dims
        assert decoder2.use_batch_norm is False

    def test_forward(self):
        """Test forward pass of Decoder."""
        latent_dim = 8
        output_dim = 15
        batch_size = 4
        
        decoder = Decoder(latent_dim=latent_dim, output_dim=output_dim)
        
        # Create random latent vectors
        z = torch.randn(batch_size, latent_dim)
        
        # Forward pass
        output = decoder(z)
        
        # Verify output shape
        assert output.shape == (batch_size, output_dim)
        
        # Verify different outputs for different inputs
        z2 = torch.randn(batch_size, latent_dim)
        output2 = decoder(z2)
        
        assert not torch.allclose(output, output2)


@skip_if_dependencies_missing
class TestVectorQuantizer:
    """Tests for the Vector Quantizer."""

    def test_init(self):
        """Test initialization of Vector Quantizer."""
        latent_dim = 8
        num_embeddings = 512
        
        # Test with default parameters
        vq = VectorQuantizer(latent_dim=latent_dim, num_embeddings=num_embeddings)
        assert vq.latent_dim == latent_dim
        assert vq.num_embeddings == num_embeddings
        assert vq.commitment_cost == 0.25
        
        # Test with custom parameters
        commitment_cost = 0.5
        vq2 = VectorQuantizer(
            latent_dim=latent_dim, 
            num_embeddings=num_embeddings, 
            commitment_cost=commitment_cost
        )
        assert vq2.commitment_cost == commitment_cost

    def test_forward(self):
        """Test forward pass of Vector Quantizer."""
        latent_dim = 8
        num_embeddings = 512
        batch_size = 4
        
        vq = VectorQuantizer(latent_dim=latent_dim, num_embeddings=num_embeddings)
        
        # Create random latent vectors
        z = torch.randn(batch_size, latent_dim)
        
        # Forward pass
        z_quantized, vq_loss, perplexity = vq(z)
        
        # Verify output shapes
        assert z_quantized.shape == (batch_size, latent_dim)
        assert isinstance(vq_loss, torch.Tensor) and vq_loss.numel() == 1
        assert isinstance(perplexity, torch.Tensor) and perplexity.numel() == 1
        
        # Check perplexity is in expected range (1 <= perplexity <= num_embeddings)
        assert 1 <= perplexity.item() <= num_embeddings
        
        # Check that straight-through estimator preserves gradients
        # by ensuring quantized vectors have requires_grad=True if inputs do
        z = torch.randn(batch_size, latent_dim, requires_grad=True)
        z_quantized, _, _ = vq(z)
        assert z_quantized.requires_grad


@skip_if_dependencies_missing
class TestEntropyBottleneck:
    """Tests for the Entropy Bottleneck."""

    def test_init(self):
        """Test initialization of Entropy Bottleneck."""
        latent_dim = 8
        
        # Test with default parameters
        eb = EntropyBottleneck(latent_dim=latent_dim)
        assert eb.latent_dim == latent_dim
        assert eb.compression_level == 1.0
        
        # Test with custom parameters
        compression_level = 2.0
        eb2 = EntropyBottleneck(
            latent_dim=latent_dim, 
            compression_level=compression_level
        )
        assert eb2.compression_level == compression_level

    def test_forward_training(self):
        """Test forward pass of Entropy Bottleneck during training."""
        latent_dim = 8
        batch_size = 4
        
        eb = EntropyBottleneck(latent_dim=latent_dim)
        eb.train()
        
        # Create random latent vectors
        z = torch.randn(batch_size, latent_dim)
        
        # Forward pass
        z_compressed, compression_loss = eb(z)
        
        # Verify output shapes
        assert z_compressed.shape == (batch_size, latent_dim)
        assert isinstance(compression_loss, torch.Tensor) and compression_loss.numel() == 1
        
        # Different random inputs should produce different compressed outputs
        z2 = torch.randn(batch_size, latent_dim)
        z_compressed2, _ = eb(z2)
        
        assert not torch.allclose(z_compressed, z_compressed2)

    def test_forward_inference(self):
        """Test forward pass of Entropy Bottleneck during inference."""
        latent_dim = 8
        batch_size = 4
        
        eb = EntropyBottleneck(latent_dim=latent_dim)
        eb.eval()
        
        # Create random latent vectors
        z = torch.randn(batch_size, latent_dim)
        
        # Forward pass
        z_compressed, compression_loss = eb(z)
        
        # Verify output shapes
        assert z_compressed.shape == (batch_size, latent_dim)
        assert isinstance(compression_loss, torch.Tensor) and compression_loss.numel() == 1

        # In evaluation mode, we should get rounded values divided by compression_level
        # So running twice with the same input should give the same output
        z_compressed2, _ = eb(z)
        assert torch.allclose(z_compressed, z_compressed2)


@skip_if_dependencies_missing
class TestMeaningVAE:
    """Tests for the MeaningVAE model."""

    def test_init(self):
        """Test initialization of MeaningVAE."""
        input_dim = 15
        latent_dim = 8
        
        # Test with default parameters
        vae = MeaningVAE(input_dim=input_dim, latent_dim=latent_dim)
        assert vae.input_dim == input_dim
        assert vae.latent_dim == latent_dim
        assert vae.compression_type == "entropy"
        assert vae.compression_level == 1.0
        assert vae.use_batch_norm is True
        assert vae.use_graph is False
        
        # Test with custom parameters
        vae2 = MeaningVAE(
            input_dim=input_dim, 
            latent_dim=latent_dim,
            compression_type="vq",
            vq_num_embeddings=256,
            use_batch_norm=False,
            use_graph=True,
            graph_hidden_dim=64,
            gnn_type="GAT",
            graph_num_layers=2
        )
        assert vae2.compression_type == "vq"
        assert vae2.use_batch_norm is False
        assert vae2.use_graph is True

    def test_reparameterize(self):
        """Test reparameterization trick."""
        input_dim = 15
        latent_dim = 8
        batch_size = 4
        
        vae = MeaningVAE(input_dim=input_dim, latent_dim=latent_dim)
        
        # Create random mu and log_var
        mu = torch.randn(batch_size, latent_dim)
        log_var = torch.randn(batch_size, latent_dim)
        
        # Sample latent vectors
        z = vae.reparameterize(mu, log_var)
        
        # Verify output shape
        assert z.shape == (batch_size, latent_dim)
        
        # Verify sampling is stochastic
        z2 = vae.reparameterize(mu, log_var)
        assert not torch.allclose(z, z2)

    def test_forward_tensor_entropy(self):
        """Test forward pass with tensor input and entropy bottleneck."""
        input_dim = 15
        latent_dim = 8
        batch_size = 4
        
        vae = MeaningVAE(
            input_dim=input_dim, 
            latent_dim=latent_dim,
            compression_type="entropy"
        )
        
        # Create random input tensor
        x = torch.randn(batch_size, input_dim)
        
        # Forward pass
        results = vae(x)
        
        # Verify output dictionary
        assert 'mu' in results
        assert 'log_var' in results
        assert 'z' in results
        assert 'z_compressed' in results
        assert 'reconstruction' in results
        assert 'kl_loss' in results
        assert 'compression_loss' in results
        
        # Verify shapes
        assert results['mu'].shape == (batch_size, latent_dim)
        assert results['log_var'].shape == (batch_size, latent_dim)
        assert results['z'].shape == (batch_size, latent_dim)
        assert results['z_compressed'].shape == (batch_size, latent_dim)
        assert results['reconstruction'].shape == (batch_size, input_dim)
        assert results['kl_loss'].numel() == 1
        assert results['compression_loss'].numel() == 1

    def test_forward_tensor_vq(self):
        """Test forward pass with tensor input and vector quantization."""
        input_dim = 15
        latent_dim = 8
        batch_size = 4
        
        vae = MeaningVAE(
            input_dim=input_dim, 
            latent_dim=latent_dim,
            compression_type="vq",
            vq_num_embeddings=256
        )
        
        # Create random input tensor
        x = torch.randn(batch_size, input_dim)
        
        # Forward pass
        results = vae(x)
        
        # Verify output dictionary
        assert 'mu' in results
        assert 'log_var' in results
        assert 'z' in results
        assert 'z_compressed' in results
        assert 'reconstruction' in results
        assert 'kl_loss' in results
        assert 'quantization_loss' in results
        assert 'perplexity' in results
        
        # Verify shapes
        assert results['mu'].shape == (batch_size, latent_dim)
        assert results['log_var'].shape == (batch_size, latent_dim)
        assert results['z'].shape == (batch_size, latent_dim)
        assert results['z_compressed'].shape == (batch_size, latent_dim)
        assert results['reconstruction'].shape == (batch_size, input_dim)
        assert results['kl_loss'].numel() == 1
        assert results['quantization_loss'].numel() == 1
        assert results['perplexity'].numel() == 1

    @skip_if_geometric_missing
    def test_encode_decode_tensor(self):
        """Test encode and decode methods with tensor input."""
        input_dim = 15
        latent_dim = 8
        batch_size = 4
        
        vae = MeaningVAE(input_dim=input_dim, latent_dim=latent_dim)
        
        # Create random input tensor
        x = torch.randn(batch_size, input_dim)
        
        # Test encode
        z = vae.encode(x)
        assert z.shape == (batch_size, latent_dim)
        
        # Test decode
        reconstruction = vae.decode(z)
        assert reconstruction.shape == (batch_size, input_dim)

    @skip_if_geometric_missing
    def test_convert_agent_to_graph(self):
        """Test converting AgentState to graph."""
        input_dim = 15
        latent_dim = 8
        
        vae = MeaningVAE(
            input_dim=input_dim, 
            latent_dim=latent_dim,
            use_graph=True
        )
        
        # Create agent state
        agent = AgentState(
            position=(1.0, 2.0, 3.0),
            health=0.8,
            energy=0.7,
            role="explorer"
        )
        
        # Convert to graph
        graph_data = vae.convert_agent_to_graph(agent)
        
        # Verify output
        assert isinstance(graph_data, Data)
        assert hasattr(graph_data, 'x')
        assert hasattr(graph_data, 'edge_index')
        assert graph_data.x.shape[0] > 0  # Should have nodes
        assert len(graph_data.x.shape) == 2  # Node features should be 2D

    @skip_if_geometric_missing
    def test_convert_agents_to_graph(self):
        """Test converting multiple AgentStates to graph."""
        input_dim = 15
        latent_dim = 8
        num_agents = 3
        
        vae = MeaningVAE(
            input_dim=input_dim, 
            latent_dim=latent_dim,
            use_graph=True
        )
        
        # Create agent states
        agents = generate_agent_states(count=num_agents, random_seed=42)
        
        # Convert to graph
        graph_data = vae.convert_agents_to_graph(agents)
        
        # Verify output
        assert isinstance(graph_data, Data)
        assert hasattr(graph_data, 'x')
        assert hasattr(graph_data, 'edge_index')
        assert graph_data.x.shape[0] > num_agents  # Should have more nodes than agents

    def test_save_load(self, tmp_path):
        """Test saving and loading model."""
        input_dim = 15
        latent_dim = 8
        
        vae = MeaningVAE(input_dim=input_dim, latent_dim=latent_dim)
        
        # Save model
        filepath = os.path.join(tmp_path, "meaning_vae.pt")
        vae.save(filepath)
        
        # Load model
        vae2 = MeaningVAE(input_dim=input_dim, latent_dim=latent_dim)
        vae2.load(filepath)
        
        # Verify model parameters are the same
        for p1, p2 in zip(vae.parameters(), vae2.parameters()):
            assert torch.allclose(p1, p2)

    def test_compression_rate(self):
        """Test getting compression rate."""
        input_dim = 15
        latent_dim = 8
        
        # Test with entropy bottleneck
        vae = MeaningVAE(
            input_dim=input_dim, 
            latent_dim=latent_dim,
            compression_type="entropy",
            compression_level=2.0
        )
        rate = vae.get_compression_rate()
        assert rate > 0
        
        # Test with vector quantization
        vae2 = MeaningVAE(
            input_dim=input_dim, 
            latent_dim=latent_dim,
            compression_type="vq",
            vq_num_embeddings=256
        )
        rate2 = vae2.get_compression_rate()
        assert rate2 > 0


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 