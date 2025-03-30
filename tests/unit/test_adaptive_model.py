#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for adaptive model implementations in meaning_transform.

Tests the following classes:
- AdaptiveEntropyBottleneck
- AdaptiveMeaningVAE
- FeatureGroupedVAE
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Add the project root to the path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from meaning_transform.src.adaptive_model import (
    AdaptiveEntropyBottleneck,
    AdaptiveMeaningVAE,
    FeatureGroupedVAE,
)


class TestAdaptiveEntropyBottleneck:
    """Test class for AdaptiveEntropyBottleneck."""

    @pytest.fixture
    def bottleneck_params(self):
        """Return common parameters for bottleneck tests."""
        return {"latent_dim": 32, "batch_size": 16}

    def test_initialization(self, bottleneck_params):
        """Test that bottleneck initializes with different compression levels."""
        for compression in [0.5, 1.0, 2.0, 5.0]:
            bottleneck = AdaptiveEntropyBottleneck(
                latent_dim=bottleneck_params["latent_dim"],
                compression_level=compression,
            )

            # Check effective dimension calculation
            expected_dim = max(1, int(bottleneck_params["latent_dim"] / compression))
            assert (
                bottleneck.effective_dim == expected_dim
            ), f"Expected effective_dim {expected_dim}, got {bottleneck.effective_dim}"

            # Check that parameters match expected shape
            assert bottleneck.compress_mu.shape == (
                bottleneck.effective_dim,
            ), "Incorrect mu shape"
            assert bottleneck.compress_log_scale.shape == (
                bottleneck.effective_dim,
            ), "Incorrect log scale shape"

            # Check projection layers
            assert bottleneck.proj_down.in_features == bottleneck_params["latent_dim"]
            assert bottleneck.proj_down.out_features == bottleneck.effective_dim
            assert bottleneck.proj_up.in_features == bottleneck.effective_dim
            assert (
                bottleneck.proj_up.out_features == bottleneck_params["latent_dim"] * 2
            )

    def test_forward(self, bottleneck_params):
        """Test forward pass of bottleneck."""
        compression = 2.0
        bottleneck = AdaptiveEntropyBottleneck(
            latent_dim=bottleneck_params["latent_dim"], compression_level=compression
        )

        # Create random input tensor
        z = torch.randn(
            bottleneck_params["batch_size"], bottleneck_params["latent_dim"]
        )

        # Test in training mode
        bottleneck.train()
        z_compressed, loss = bottleneck(z)

        # Check shapes
        assert (
            z_compressed.shape == z.shape
        ), f"Expected shape {z.shape}, got {z_compressed.shape}"
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.ndim == 0, "Loss should be a scalar"

        # Test in eval mode
        bottleneck.eval()
        z_compressed, loss = bottleneck(z)

        # Check shapes again
        assert (
            z_compressed.shape == z.shape
        ), f"Expected shape {z.shape}, got {z_compressed.shape}"

    def test_compression_rate(self, bottleneck_params):
        """Test compression rate calculation."""
        for compression in [0.5, 1.0, 2.0, 5.0]:
            bottleneck = AdaptiveEntropyBottleneck(
                latent_dim=bottleneck_params["latent_dim"],
                compression_level=compression,
            )

            rate = bottleneck.get_effective_compression_rate()
            expected_rate = float(bottleneck_params["latent_dim"]) / float(
                bottleneck.effective_dim
            )

            assert (
                abs(rate - expected_rate) < 1e-5
            ), f"Expected rate {expected_rate}, got {rate}"

    def test_parameter_count(self, bottleneck_params):
        """Test that parameter count decreases with compression level."""
        compression_levels = [0.5, 1.0, 2.0, 5.0, 10.0]
        param_counts = []

        for compression in compression_levels:
            bottleneck = AdaptiveEntropyBottleneck(
                latent_dim=bottleneck_params["latent_dim"],
                compression_level=compression,
            )
            param_count = sum(p.numel() for p in bottleneck.parameters())
            param_counts.append(param_count)

        # Check that parameter count decreases as compression increases
        for i in range(1, len(param_counts)):
            assert param_counts[i] <= param_counts[i - 1], (
                f"Parameter count should decrease with higher compression. "
                f"Got {param_counts[i-1]} at level {compression_levels[i-1]} and "
                f"{param_counts[i]} at level {compression_levels[i]}"
            )


class TestAdaptiveMeaningVAE:
    """Test class for AdaptiveMeaningVAE."""

    @pytest.fixture
    def vae_params(self):
        """Return common parameters for VAE tests."""
        return {"input_dim": 64, "latent_dim": 16, "batch_size": 16}

    def test_initialization(self, vae_params):
        """Test that VAE initializes with different compression levels."""
        for compression in [0.5, 1.0, 2.0, 5.0]:
            vae = AdaptiveMeaningVAE(
                input_dim=vae_params["input_dim"],
                latent_dim=vae_params["latent_dim"],
                compression_level=compression,
            )

            # Check that components are initialized properly
            assert hasattr(vae, "encoder"), "Encoder not initialized"
            assert hasattr(vae, "decoder"), "Decoder not initialized"
            assert hasattr(vae, "compressor"), "Compressor not initialized"

            # Check that compressor has right compression level
            assert vae.compressor.compression_level == compression

            # Check input/output dimensions
            assert vae.encoder.input_dim == vae_params["input_dim"]
            assert vae.decoder.output_dim == vae_params["input_dim"]

    def test_forward(self, vae_params):
        """Test forward pass of VAE."""
        vae = AdaptiveMeaningVAE(
            input_dim=vae_params["input_dim"],
            latent_dim=vae_params["latent_dim"],
            compression_level=2.0,
        )

        # Create random input tensor
        x = torch.randn(vae_params["batch_size"], vae_params["input_dim"])

        # Test forward pass
        output = vae(x)

        # Check that all expected outputs are present
        expected_keys = [
            "reconstruction",
            "mu",
            "log_var",
            "z",
            "z_compressed",
            "kl_loss",
            "compression_loss",
            "effective_dim",
        ]
        for key in expected_keys:
            assert key in output, f"Missing output: {key}"

        # Check shapes
        assert output["reconstruction"].shape == x.shape
        assert output["mu"].shape == (
            vae_params["batch_size"],
            vae_params["latent_dim"],
        )
        assert output["log_var"].shape == (
            vae_params["batch_size"],
            vae_params["latent_dim"],
        )
        assert output["z"].shape == (vae_params["batch_size"], vae_params["latent_dim"])
        assert output["z_compressed"].shape == (
            vae_params["batch_size"],
            vae_params["latent_dim"],
        )

        # Check scalar losses
        assert output["kl_loss"].ndim == 0, "KL loss should be a scalar"
        assert (
            output["compression_loss"].ndim == 0
        ), "Compression loss should be a scalar"

    def test_encode_decode(self, vae_params):
        """Test encoding and decoding functions."""
        vae = AdaptiveMeaningVAE(
            input_dim=vae_params["input_dim"],
            latent_dim=vae_params["latent_dim"],
            compression_level=2.0,
        )

        # Create random input tensor
        x = torch.randn(vae_params["batch_size"], vae_params["input_dim"])

        # Test encode
        z = vae.encode(x)
        assert z.shape == (vae_params["batch_size"], vae_params["latent_dim"])

        # Test decode
        x_reconstructed = vae.decode(z)
        assert x_reconstructed.shape == x.shape

        # Test encode-decode pipeline
        x_reconstructed_pipeline = vae.decode(vae.encode(x))
        assert x_reconstructed_pipeline.shape == x.shape

    def test_save_load(self, vae_params):
        """Test saving and loading the model."""
        # Use a fixed seed for reproducibility
        fixed_seed = 42
        
        vae = AdaptiveMeaningVAE(
            input_dim=vae_params["input_dim"],
            latent_dim=vae_params["latent_dim"],
            compression_level=2.0,
            seed=fixed_seed
        )
        
        # Set model to eval mode for deterministic behavior
        vae.eval()

        # Create random input for comparison
        torch.manual_seed(fixed_seed)  # Set seed for input data
        x = torch.randn(vae_params["batch_size"], vae_params["input_dim"])
        before_save = vae(x)["reconstruction"]

        # Save and load model
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            filepath = tmp.name
            vae.save(filepath)

            # Create a new model with the same parameters
            new_vae = AdaptiveMeaningVAE(
                input_dim=vae_params["input_dim"],
                latent_dim=vae_params["latent_dim"],
                compression_level=2.0,
                seed=fixed_seed
            )
            
            # Set model to eval mode for deterministic behavior
            new_vae.eval()

            # Load saved parameters
            new_vae.load(filepath)

        # Clean up temporary file
        try:
            os.unlink(filepath)
        except:
            pass

        # Test that the loaded model produces the same output
        after_load = new_vae(x)["reconstruction"]
        assert torch.allclose(before_save, after_load, rtol=1e-4, atol=1e-4)

    def test_compression_rate(self, vae_params):
        """Test compression rate calculation."""
        for compression in [0.5, 1.0, 2.0, 5.0]:
            vae = AdaptiveMeaningVAE(
                input_dim=vae_params["input_dim"],
                latent_dim=vae_params["latent_dim"],
                compression_level=compression,
            )

            rate = vae.get_compression_rate()
            expected_rate = vae_params["latent_dim"] / vae.compressor.effective_dim

            assert (
                abs(rate - expected_rate) < 1e-5
            ), f"Expected rate {expected_rate}, got {rate}"


class TestFeatureGroupedVAE:
    """Test class for FeatureGroupedVAE."""

    @pytest.fixture
    def vae_params(self):
        """Return common parameters for VAE tests."""
        return {"input_dim": 90, "latent_dim": 30, "batch_size": 16}

    @pytest.fixture
    def feature_groups(self, vae_params):
        """Return feature groups for testing."""
        return {
            "important": (0, 30, 0.5),  # High importance - low compression
            "medium": (30, 60, 2.0),  # Medium importance - medium compression
            "less_important": (60, 90, 5.0),  # Low importance - high compression
        }

    def test_initialization(self, vae_params, feature_groups):
        """Test that grouped VAE initializes properly."""
        vae = FeatureGroupedVAE(
            input_dim=vae_params["input_dim"],
            latent_dim=vae_params["latent_dim"],
            feature_groups=feature_groups,
            base_compression_level=1.0,
        )

        # Check that components are initialized properly
        assert hasattr(vae, "encoder"), "Encoder not initialized"
        assert hasattr(vae, "decoder"), "Decoder not initialized"
        assert hasattr(vae, "bottlenecks"), "Bottlenecks not initialized"

        # Check that all feature groups have a bottleneck
        for group in feature_groups:
            assert group in vae.bottlenecks, f"No bottleneck for group {group}"

        # Check that latent dimensions are allocated
        total_latent_dim = 0
        for group, (start, end) in vae.group_latent_dims.items():
            group_dim = end - start
            total_latent_dim += group_dim

            # Check that bottleneck effective dim matches compression
            bottleneck = vae.bottlenecks[group]
            start_idx, end_idx, compression = feature_groups[group]
            feature_count = end_idx - start_idx

            assert bottleneck.latent_dim == group_dim
            assert (
                bottleneck.compression_level == compression * vae.base_compression_level
            )

        # Check total latent dim
        assert (
            total_latent_dim == vae_params["latent_dim"]
        ), f"Total latent dims should sum to {vae_params['latent_dim']}, got {total_latent_dim}"

    def test_default_feature_groups(self, vae_params):
        """Test that default feature groups are created when none are provided."""
        vae = FeatureGroupedVAE(
            input_dim=vae_params["input_dim"],
            latent_dim=vae_params["latent_dim"],
            feature_groups=None,
            base_compression_level=1.0,
        )

        # Should create 3 equal groups
        assert (
            len(vae.bottlenecks) == 3
        ), f"Should create 3 default groups, got {len(vae.bottlenecks)}"

    def test_forward(self, vae_params, feature_groups):
        """Test forward pass of grouped VAE."""
        vae = FeatureGroupedVAE(
            input_dim=vae_params["input_dim"],
            latent_dim=vae_params["latent_dim"],
            feature_groups=feature_groups,
            base_compression_level=1.0,
        )

        # Create random input tensor
        x = torch.randn(vae_params["batch_size"], vae_params["input_dim"])

        # Test forward pass
        output = vae(x)

        # Check that all expected outputs are present
        expected_keys = [
            "reconstruction",
            "mu",
            "log_var",
            "z",
            "z_compressed",
            "kl_loss",
            "compression_loss",
            "feature_group_dims",
        ]
        for key in expected_keys:
            assert key in output, f"Missing output: {key}"

        # Check shapes
        assert output["reconstruction"].shape == x.shape
        assert output["mu"].shape == (
            vae_params["batch_size"],
            vae_params["latent_dim"],
        )
        assert output["log_var"].shape == (
            vae_params["batch_size"],
            vae_params["latent_dim"],
        )
        assert output["z"].shape == (vae_params["batch_size"], vae_params["latent_dim"])
        assert output["z_compressed"].shape == (
            vae_params["batch_size"],
            vae_params["latent_dim"],
        )

        # Check scalar losses
        assert output["kl_loss"].ndim == 0, "KL loss should be a scalar"
        assert (
            output["compression_loss"].ndim == 0
        ), "Compression loss should be a scalar"

    def test_encode_decode(self, vae_params, feature_groups):
        """Test encoding and decoding functions."""
        vae = FeatureGroupedVAE(
            input_dim=vae_params["input_dim"],
            latent_dim=vae_params["latent_dim"],
            feature_groups=feature_groups,
            base_compression_level=1.0,
        )

        # Create random input tensor
        x = torch.randn(vae_params["batch_size"], vae_params["input_dim"])

        # Test encode
        z = vae.encode(x)
        assert z.shape == (vae_params["batch_size"], vae_params["latent_dim"])

        # Test decode
        x_reconstructed = vae.decode(z)
        assert x_reconstructed.shape == x.shape

        # Test encode-decode pipeline
        x_reconstructed_pipeline = vae.decode(vae.encode(x))
        assert x_reconstructed_pipeline.shape == x.shape

    def test_save_load(self, vae_params, feature_groups):
        """Test saving and loading the model."""
        # Use a fixed seed for reproducibility
        fixed_seed = 42
        
        vae = FeatureGroupedVAE(
            input_dim=vae_params["input_dim"],
            latent_dim=vae_params["latent_dim"],
            feature_groups=feature_groups,
            base_compression_level=1.0,
            seed=fixed_seed
        )
        
        # Set model to eval mode for deterministic behavior
        vae.eval()

        # Create random input for comparison
        torch.manual_seed(fixed_seed)  # Set seed for input data
        x = torch.randn(vae_params["batch_size"], vae_params["input_dim"])
        before_save = vae(x)["reconstruction"]

        # Save and load model
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            filepath = tmp.name
            vae.save(filepath)

            # Create a new model with the same parameters
            new_vae = FeatureGroupedVAE(
                input_dim=vae_params["input_dim"],
                latent_dim=vae_params["latent_dim"],
                feature_groups=feature_groups,
                base_compression_level=1.0,
                seed=fixed_seed
            )
            
            # Set model to eval mode for deterministic behavior
            new_vae.eval()

            # Load saved parameters
            new_vae.load(filepath)

        # Clean up temporary file
        try:
            os.unlink(filepath)
        except:
            pass

        # Test that the loaded model produces the same output
        after_load = new_vae(x)["reconstruction"]
        assert torch.allclose(before_save, after_load, rtol=1e-4, atol=1e-4)

    def test_compression_rate(self, vae_params, feature_groups):
        """Test compression rate calculation."""
        vae = FeatureGroupedVAE(
            input_dim=vae_params["input_dim"],
            latent_dim=vae_params["latent_dim"],
            feature_groups=feature_groups,
            base_compression_level=1.0,
        )

        rates = vae.get_compression_rate()

        # Check each group rate
        for group in feature_groups:
            assert group in rates, f"Missing compression rate for group {group}"

            bottleneck = vae.bottlenecks[group]
            start, end = vae.group_latent_dims[group]
            group_latent_dim = end - start

            expected_rate = group_latent_dim / bottleneck.effective_dim
            assert (
                abs(rates[group] - expected_rate) < 1e-5
            ), f"For group {group}, expected rate {expected_rate}, got {rates[group]}"

        # Check overall rate
        assert "overall" in rates, "Missing overall compression rate"

    def test_feature_group_analysis(self, vae_params, feature_groups):
        """Test feature group analysis."""
        vae = FeatureGroupedVAE(
            input_dim=vae_params["input_dim"],
            latent_dim=vae_params["latent_dim"],
            feature_groups=feature_groups,
            base_compression_level=2.0,
        )

        analysis = vae.get_feature_group_analysis()

        # Check analysis for each group
        for group, (start_idx, end_idx, compression) in feature_groups.items():
            assert group in analysis, f"Missing analysis for group {group}"
            group_analysis = analysis[group]

            assert group_analysis["feature_range"] == (start_idx, end_idx)
            assert group_analysis["feature_count"] == end_idx - start_idx
            assert group_analysis["compression"] == compression
            assert group_analysis["base_compression"] == 2.0
            assert group_analysis["overall_compression"] == compression * 2.0


if __name__ == "__main__":
    # Run tests manually if file is executed directly
    pytest.main(["-v", __file__])
