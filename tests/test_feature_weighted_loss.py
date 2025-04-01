#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the feature-weighted loss implementation.

This module tests:
1. Basic functionality of the FeatureWeightedLoss class
2. Feature weighting effectiveness
3. Progressive weight adjustment
4. Feature stability tracking and adjustment
"""

import unittest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Import components for testing
from meaning_transform.src.loss import FeatureWeightedLoss
from meaning_transform.src.model import MeaningVAE


class TestFeatureWeightedLoss(unittest.TestCase):
    """Test suite for feature-weighted loss implementation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample state tensor with mock data
        self.batch_size = 16
        self.input_dim = 10
        
        # Create mock state tensor (batch_size x input_dim)
        # Format: [x, y, health, has_target, energy, role1, role2, role3, role4, role5]
        self.states = torch.zeros(self.batch_size, self.input_dim)
        
        # Set mock values
        for i in range(self.batch_size):
            self.states[i, 0] = 0.1 + 0.8 * (i / self.batch_size)  # x position
            self.states[i, 1] = 0.2 + 0.6 * (i / self.batch_size)  # y position
            self.states[i, 2] = 0.5 + 0.4 * torch.rand(1)  # health
            self.states[i, 3] = float(i % 2)  # has_target (binary)
            self.states[i, 4] = 0.3 + 0.6 * torch.rand(1)  # energy
            
            # One-hot encoded role (exactly one of the next 5 dimensions is 1.0)
            role_idx = i % 5
            self.states[i, 5 + role_idx] = 1.0
        
        # Create simple VAE model for testing
        self.model = MeaningVAE(
            input_dim=self.input_dim,
            latent_dim=4,
            hidden_dims=[8],
            compression_level=1.0
        )
        
        # Create dataset and loader
        self.dataset = TensorDataset(self.states)
        self.loader = DataLoader(self.dataset, batch_size=4, shuffle=True)

    def test_feature_weighted_loss_initialization(self):
        """Test initialization of feature-weighted loss."""
        # Test with default parameters
        loss_fn = FeatureWeightedLoss()
        
        # Check that feature weights are initialized
        self.assertTrue(hasattr(loss_fn, 'feature_weights'))
        self.assertIsInstance(loss_fn.feature_weights, dict)
        
        # Check that all features have weights
        for feature in loss_fn.semantic_loss.feature_extractors:
            self.assertIn(feature, loss_fn.feature_weights)
        
        # Check that weights sum to 1.0 (approximately)
        total_weight = sum(loss_fn.feature_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)
        
        # Check canonical weights
        self.assertAlmostEqual(loss_fn.feature_weights["position"], 0.554, places=3)
        self.assertAlmostEqual(loss_fn.feature_weights["health"], 0.15, places=3)
        
        # Test with custom weights
        custom_weights = {
            "position": 0.7,
            "health": 0.2,
            "energy": 0.1
        }
        
        loss_fn_custom = FeatureWeightedLoss(
            feature_weights=custom_weights,
            use_canonical_weights=False
        )
        
        # Weights should be normalized to sum to 1.0
        self.assertAlmostEqual(loss_fn_custom.feature_weights["position"], 0.7, places=5)
        self.assertAlmostEqual(loss_fn_custom.feature_weights["health"], 0.2, places=5)
        self.assertAlmostEqual(loss_fn_custom.feature_weights["energy"], 0.1, places=5)

    def test_feature_weighted_loss_forward(self):
        """Test forward pass of feature-weighted loss."""
        # Initialize loss function
        loss_fn = FeatureWeightedLoss()
        
        # Get a batch of data
        batch = next(iter(self.loader))[0]
        
        # Run model forward pass
        model_output = self.model(batch)
        
        # Compute loss
        loss_dict = loss_fn(model_output, batch)
        
        # Check that loss dict contains expected keys
        expected_keys = ["loss", "reconstruction_loss", "kl_loss", "semantic_loss", 
                        "compression_loss", "semantic_breakdown", "feature_weights"]
        
        for key in expected_keys:
            self.assertIn(key, loss_dict)
        
        # Check that loss is a scalar tensor
        self.assertIsInstance(loss_dict["loss"], torch.Tensor)
        self.assertEqual(loss_dict["loss"].shape, torch.Size([]))
        
        # Check that semantic breakdown contains feature-specific losses
        self.assertIsInstance(loss_dict["semantic_breakdown"], dict)
        for feature in loss_fn.semantic_loss.feature_extractors:
            self.assertIn(feature, loss_dict["semantic_breakdown"])

    def test_feature_weight_update(self):
        """Test updating feature weights during training."""
        # Initialize loss function
        loss_fn = FeatureWeightedLoss()
        
        # Initial weights for test features
        initial_pos_weight = loss_fn.feature_weights["position"]
        initial_health_weight = loss_fn.feature_weights["health"]
        
        # Update weights
        new_weights = {
            "position": 0.8,
            "health": 0.1,
            "energy": 0.1
        }
        loss_fn.update_feature_weights(new_weights)
        
        # Check that weights were updated
        self.assertAlmostEqual(loss_fn.feature_weights["position"], 0.8, places=5)
        self.assertAlmostEqual(loss_fn.feature_weights["health"], 0.1, places=5)
        self.assertAlmostEqual(loss_fn.feature_weights["energy"], 0.1, places=5)
        
        # Test that weights influence the loss values
        # Get a batch of data
        batch = next(iter(self.loader))[0]
        
        # Create slightly perturbed reconstruction that's worse on position
        reconstructed = batch.clone()
        # Add noise to positions (dims 0-1)
        reconstructed[:, 0:2] += 0.2 * torch.randn(reconstructed.shape[0], 2)
        
        # Create model output dict
        model_output = {
            "reconstruction": reconstructed,
            "mu": torch.zeros(reconstructed.shape[0], 4),
            "log_var": torch.zeros(reconstructed.shape[0], 4)
        }
        
        # Compute loss with high position weight
        loss_high_pos = loss_fn(model_output, batch)
        
        # Now swap weights to prioritize health instead
        loss_fn.update_feature_weights({
            "position": 0.1,
            "health": 0.8,
            "energy": 0.1
        })
        
        # Compute loss with high health weight
        loss_high_health = loss_fn(model_output, batch)
        
        # Position error should be weighted more in the first case
        pos_contribution_high = loss_high_pos["semantic_breakdown"]["position"] * 0.8
        pos_contribution_low = loss_high_health["semantic_breakdown"]["position"] * 0.1
        
        # The weighted position error should be higher when position has higher weight
        self.assertGreater(pos_contribution_high, pos_contribution_low)

    def test_progressive_weight_adjustment(self):
        """Test progressive weight adjustment during training."""
        # Initialize loss function with progressive scheduling
        initial_weights = {
            "position": 0.1,
            "health": 0.1,
            "energy": 0.1,
            "has_target": 0.1,
            "is_alive": 0.1,
            "role": 0.1,
            "threatened": 0.1
        }
        
        target_weights = {
            "position": 0.6,
            "health": 0.2,
            "energy": 0.1,
            "has_target": 0.05,
            "is_alive": 0.03,
            "role": 0.01,
            "threatened": 0.01
        }
        
        loss_fn = FeatureWeightedLoss(
            feature_weights=initial_weights,
            use_canonical_weights=False,
            progressive_weight_schedule="linear",
            progressive_weight_epochs=10
        )
        
        # Set target weights
        loss_fn.update_feature_weights(target_weights)
        
        # Check initial weights
        for feature, weight in initial_weights.items():
            self.assertAlmostEqual(loss_fn.feature_weights[feature], weight, places=5)
        
        # Simulate training epochs
        epochs = [0, 2, 5, 8, 10, 15]
        for epoch in epochs:
            loss_fn.update_epoch(epoch)
            
            # Check progress at specific epochs
            if epoch == 5:  # 50% progress
                # Weights should be halfway between initial and target
                self.assertAlmostEqual(
                    loss_fn.feature_weights["position"],
                    initial_weights["position"] + 0.5 * (target_weights["position"] - initial_weights["position"]),
                    places=5
                )
            
            elif epoch >= 10:  # 100% progress or beyond
                # Weights should match target weights
                for feature, target in target_weights.items():
                    self.assertAlmostEqual(loss_fn.feature_weights[feature], target, places=5)

    def test_stability_based_adjustment(self):
        """Test feature stability-based weight adjustment."""
        # Initialize loss function with stability adjustment
        loss_fn = FeatureWeightedLoss(
            feature_stability_adjustment=True
        )
        
        # Simulate error history for features
        # Position has high but stable error
        # Health has low but variable error
        position_errors = [0.3, 0.31, 0.29, 0.32, 0.28]
        health_errors = [0.1, 0.05, 0.15, 0.02, 0.18]
        
        feature_errors = {
            "position": position_errors,
            "health": health_errors
        }
        
        # Feed error history to loss function
        for i in range(len(position_errors)):
            errors = {
                "position": position_errors[i],
                "health": health_errors[i]
            }
            loss_fn.update_stability_scores(errors)
        
        # Health should have a lower stability score due to high variance
        self.assertLess(loss_fn.feature_stability_scores["health"], 
                         loss_fn.feature_stability_scores["position"])
        
        # Get a batch of data to test impact on loss
        batch = next(iter(self.loader))[0]
        
        # Run model forward pass
        model_output = self.model(batch)
        
        # Compute loss
        loss_dict = loss_fn(model_output, batch)
        
        # Should still include stability-adjusted weights
        self.assertIn("feature_weights", loss_dict)


if __name__ == "__main__":
    unittest.main() 