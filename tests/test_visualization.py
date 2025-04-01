# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Test module for visualization tools.

# This module tests the visualization capabilities implemented in utils/visualization.py
# and utils/visualize.py.
# """

# import os
# import sys
# import unittest
# import torch
# import numpy as np
# import shutil
# import matplotlib
# from pathlib import Path

# # Add the project root to the path
# project_root = str(Path(__file__).resolve().parent.parent)
# if project_root not in sys.path:
#     sys.path.append(project_root)

# matplotlib.use('Agg')  # Use non-interactive backend for testing
# import matplotlib.pyplot as plt
# from collections import defaultdict

# # Change the imports to use the correct paths
# from meaning_transform.src.visualization import (
#     LatentSpaceVisualizer,
#     LossVisualizer,
#     StateComparisonVisualizer,
#     DriftVisualizer
# )
# from meaning_transform.src.visualize import setup_visualization_dirs


# class TestVisualization(unittest.TestCase):
#     """Test case for visualization tools."""
    
#     def setUp(self):
#         """Set up test fixtures."""
#         self.test_output_dir = "test_results/visualization_test"
#         os.makedirs(self.test_output_dir, exist_ok=True)
        
#         # Create synthetic test data
#         self.latent_dim = 16
#         self.num_samples = 50
#         self.latent_vectors = torch.randn(self.num_samples, self.latent_dim)
#         self.labels = torch.randint(0, 5, (self.num_samples,))
        
#         # Set random seed for reproducibility
#         np.random.seed(42)
#         torch.manual_seed(42)
    
#     def tearDown(self):
#         """Tear down test fixtures."""
#         # On Windows, sometimes files are still in use when trying to remove the directory
#         # We'll use a more robust approach with retries and error handling
#         if os.path.exists(self.test_output_dir):
#             try:
#                 shutil.rmtree(self.test_output_dir)
#             except PermissionError:
#                 # On Windows, sometimes we can't delete because files are still in use
#                 # We'll just print a warning and continue
#                 print(f"Warning: Could not delete test directory {self.test_output_dir}. It may be in use.")
#                 pass
    
#     def test_setup_visualization_dirs(self):
#         """Test setup_visualization_dirs function."""
#         # Test with default directory
#         dirs = setup_visualization_dirs(self.test_output_dir)
        
#         # Check that all expected directories exist
#         for dir_name, dir_path in dirs.items():
#             self.assertTrue(os.path.exists(dir_path))
#             self.assertTrue(os.path.isdir(dir_path))
    
#     def test_latent_space_visualization(self):
#         """Test latent space visualization functions."""
#         # Test t-SNE visualization
#         latent_viz = LatentSpaceVisualizer(output_dir=self.test_output_dir)
#         tsne_fig = latent_viz.visualize_tsne(
#             latent_vectors=self.latent_vectors,
#             labels=self.labels,
#             metadata={"test": "metadata"},
#             output_file="test_tsne.png"
#         )
        
#         # Check that figure was created
#         self.assertIsInstance(tsne_fig, plt.Figure)
        
#         # Check that output file was created
#         output_path = os.path.join(self.test_output_dir, "test_tsne.png")
#         self.assertTrue(os.path.exists(output_path))
        
#         # Test PCA visualization
#         pca_fig = latent_viz.visualize_pca(
#             latent_vectors=self.latent_vectors,
#             labels=self.labels,
#             metadata={"test": "metadata"},
#             output_file="test_pca.png"
#         )
        
#         # Check that figure was created
#         self.assertIsInstance(pca_fig, plt.Figure)
        
#         # Check that output file was created
#         output_path = os.path.join(self.test_output_dir, "test_pca.png")
#         self.assertTrue(os.path.exists(output_path))
    
#     def test_latent_interpolation(self):
#         """Test latent interpolation visualization."""
#         # Create dummy encoder and decoder functions
#         encoder = lambda x: torch.randn(self.latent_dim)
#         decoder_matrix = torch.randn(self.latent_dim, 10)
#         decoder = lambda z: torch.matmul(z, decoder_matrix)
        
#         # Create test states
#         state_a = torch.randn(10)
#         state_b = torch.randn(10)
        
#         # Test latent interpolation
#         latent_viz = LatentSpaceVisualizer(output_dir=self.test_output_dir)
#         interp_fig = latent_viz.visualize_latent_interpolation(
#             decode_fn=decoder,
#             state_a=state_a,
#             state_b=state_b,
#             encoder=encoder,
#             steps=5,
#             output_file="test_interpolation.png"
#         )
        
#         # Check that figure was created
#         self.assertIsInstance(interp_fig, plt.Figure)
        
#         # Check that output file was created
#         output_path = os.path.join(self.test_output_dir, "test_interpolation.png")
#         self.assertTrue(os.path.exists(output_path))
    
#     def test_loss_visualization(self):
#         """Test loss visualization functions."""
#         # Create synthetic loss history
#         loss_viz = LossVisualizer(output_dir=self.test_output_dir)
        
#         # Add loss values
#         for epoch in range(10):
#             loss_viz.update(epoch, {
#                 "total_loss": 1.0 - 0.05 * epoch,
#                 "reconstruction_loss": 0.7 - 0.03 * epoch,
#                 "kl_loss": 0.3 - 0.02 * epoch
#             })
        
#         # Test loss curve plotting
#         loss_fig = loss_viz.plot_losses(
#             output_file="test_loss_curves.png"
#         )
        
#         # Check that figure was created
#         self.assertIsInstance(loss_fig, plt.Figure)
        
#         # Check that output file was created
#         output_path = os.path.join(self.test_output_dir, "test_loss_curves.png")
#         self.assertTrue(os.path.exists(output_path))
        
#         # Test saving and loading history
#         loss_viz.save_history(filename="test_history.json")
#         history_path = os.path.join(self.test_output_dir, "test_history.json")
#         self.assertTrue(os.path.exists(history_path))
        
#         # Create new visualizer and load history
#         new_viz = LossVisualizer(output_dir=self.test_output_dir)
#         new_viz.load_history(filename="test_history.json")
        
#         # Check that loaded history matches original
#         for loss_name in ["total_loss", "reconstruction_loss", "kl_loss"]:
#             self.assertEqual(
#                 len(loss_viz.loss_history[loss_name]),
#                 len(new_viz.loss_history[loss_name])
#             )
    
#     def test_compression_vs_reconstruction(self):
#         """Test compression vs. reconstruction visualization."""
#         # Create synthetic data
#         compression_levels = np.linspace(1, 32, 8)
#         reconstruction_errors = [0.8 * np.exp(-x / 10) + 0.1 for x in compression_levels]
#         semantic_losses = [0.7 * np.exp(-x / 8) + 0.05 for x in compression_levels]
        
#         # Test visualization
#         loss_viz = LossVisualizer(output_dir=self.test_output_dir)
#         comp_fig = loss_viz.plot_compression_vs_reconstruction(
#             compression_levels=compression_levels,
#             reconstruction_errors=reconstruction_errors,
#             semantic_losses=semantic_losses,
#             output_file="test_compression.png"
#         )
        
#         # Check that figure was created
#         self.assertIsInstance(comp_fig, plt.Figure)
        
#         # Check that output file was created
#         output_path = os.path.join(self.test_output_dir, "test_compression.png")
#         self.assertTrue(os.path.exists(output_path))
    
#     def test_state_comparison(self):
#         """Test state comparison visualization."""
#         # Create synthetic feature data
#         num_samples = 5
#         original_features = {
#             "position": np.random.rand(num_samples, 2),
#             "health": np.random.rand(num_samples) * 100,
#             "energy": np.random.rand(num_samples) * 50,
#             "is_alive": np.random.randint(0, 2, num_samples)
#         }
        
#         # Create reconstructed features with noise
#         reconstructed_features = {}
#         for feature, values in original_features.items():
#             if feature == "is_alive":
#                 reconstructed = values.copy()
#                 flip_idx = np.random.choice(num_samples, size=1)
#                 reconstructed[flip_idx] = 1 - reconstructed[flip_idx]
#                 reconstructed_features[feature] = reconstructed
#             else:
#                 noise = np.random.normal(0, 0.05 * np.std(values), values.shape)
#                 reconstructed_features[feature] = values + noise
        
#         # Test feature comparison
#         state_viz = StateComparisonVisualizer(output_dir=self.test_output_dir)
#         feat_fig = state_viz.plot_feature_comparison(
#             original_features=original_features,
#             reconstructed_features=reconstructed_features,
#             example_indices=[0, 1],
#             output_file="test_feature_comparison.png"
#         )
        
#         # Check that figure was created
#         self.assertIsInstance(feat_fig, plt.Figure)
        
#         # Check that output file was created
#         output_path = os.path.join(self.test_output_dir, "test_feature_comparison.png")
#         self.assertTrue(os.path.exists(output_path))
    
#     def test_state_trajectories(self):
#         """Test state trajectory visualization."""
#         # Create synthetic trajectories
#         num_steps = 10
#         original_states = [torch.tensor([float(i)/num_steps, np.sin(i * np.pi / num_steps)]) for i in range(num_steps)]
#         reconstructed_states = [state + torch.randn_like(state) * 0.05 for state in original_states]
        
#         # Test trajectory visualization
#         state_viz = StateComparisonVisualizer(output_dir=self.test_output_dir)
#         traj_fig = state_viz.plot_state_trajectories(
#             original_states=original_states,
#             reconstructed_states=reconstructed_states,
#             output_file="test_trajectories.png"
#         )
        
#         # Check that figure was created
#         self.assertIsInstance(traj_fig, plt.Figure)
        
#         # Check that output file was created
#         output_path = os.path.join(self.test_output_dir, "test_trajectories.png")
#         self.assertTrue(os.path.exists(output_path))
    
#     def test_confusion_matrices(self):
#         """Test confusion matrix visualization."""
#         # Create synthetic confusion matrices
#         confusion_matrices = {
#             "role": np.array([
#                 [8, 2, 0],
#                 [1, 6, 3],
#                 [0, 2, 9]
#             ]),
#             "is_alive": np.array([
#                 [15, 5],
#                 [3, 17]
#             ])
#         }
        
#         # Test confusion matrix visualization
#         state_viz = StateComparisonVisualizer(output_dir=self.test_output_dir)
#         cm_fig = state_viz.plot_confusion_matrices(
#             confusion_matrices=confusion_matrices,
#             output_file="test_confusion_matrices.png"
#         )
        
#         # Check that figure was created
#         self.assertIsInstance(cm_fig, plt.Figure)
        
#         # Check that output file was created
#         output_path = os.path.join(self.test_output_dir, "test_confusion_matrices.png")
#         self.assertTrue(os.path.exists(output_path))
    
#     def test_semantic_drift(self):
#         """Test semantic drift visualization."""
#         # Create synthetic drift data
#         iterations = list(range(0, 50, 5))
#         features = ["position", "health", "overall"]
        
#         semantic_scores = {}
#         for feature in features:
#             if feature == "overall":
#                 base = 0.95
#                 decay = 0.002
#                 scores = [base - decay * i for i in iterations]
#             else:
#                 base = 0.90
#                 decay = 0.003
#                 noise = 0.03
#                 scores = [max(0, min(1, base - decay * i + np.random.normal(0, noise))) for i in iterations]
            
#             semantic_scores[feature] = scores
        
#         # Test drift visualization
#         drift_viz = DriftVisualizer(output_dir=self.test_output_dir)
#         drift_fig = drift_viz.plot_semantic_drift(
#             iterations=iterations,
#             semantic_scores=semantic_scores,
#             output_file="test_semantic_drift.png"
#         )
        
#         # Check that figure was created
#         self.assertIsInstance(drift_fig, plt.Figure)
        
#         # Check that output file was created
#         output_path = os.path.join(self.test_output_dir, "test_semantic_drift.png")
#         self.assertTrue(os.path.exists(output_path))
    
#     def test_threshold_finder(self):
#         """Test threshold finder visualization."""
#         # Create synthetic threshold data
#         compression_levels = np.linspace(1, 32, 8)
#         semantic_scores = [min(1.0, 0.7 + 0.3 * (1 - np.exp(-x / 8))) for x in compression_levels]
#         reconstruction_errors = [0.8 * np.exp(-x / 10) + 0.1 for x in compression_levels]
        
#         # Test threshold finder visualization
#         drift_viz = DriftVisualizer(output_dir=self.test_output_dir)
#         thresh_result = drift_viz.plot_threshold_finder(
#             compression_levels=compression_levels,
#             semantic_scores=semantic_scores,
#             reconstruction_errors=reconstruction_errors,
#             threshold=0.9,
#             output_file="test_threshold_finder.png"
#         )
        
#         # Check that figure and optimal compression were returned
#         self.assertIsInstance(thresh_result[0], plt.Figure)
#         self.assertIsInstance(thresh_result[1], float)
        
#         # Check that output file was created
#         output_path = os.path.join(self.test_output_dir, "test_threshold_finder.png")
#         self.assertTrue(os.path.exists(output_path))
    
#     def test_high_level_interface(self):
#         """Test high-level interface functions."""
#         # Create synthetic data
#         compression_levels = np.linspace(1, 32, 8)
#         semantic_scores = {
#             "overall": [min(1.0, 0.7 + 0.3 * (1 - np.exp(-x / 8))) for x in compression_levels],
#             "position": [min(1.0, 0.6 + 0.3 * (1 - np.exp(-x / 10))) for x in compression_levels]
#         }
#         reconstruction_errors = [0.8 * np.exp(-x / 10) + 0.1 for x in compression_levels]
        
#         # Test high-level drift visualization
#         output_path = visualize.visualize_semantic_drift(
#             iterations=list(compression_levels),
#             semantic_scores=semantic_scores,
#             compression_levels=compression_levels,
#             output_dir=self.test_output_dir
#         )
        
#         # Check that output file was created
#         self.assertTrue(os.path.exists(output_path))
        
#         # Test high-level latent space visualization
#         output_paths = visualize.visualize_latent_space(
#             latent_vectors=self.latent_vectors,
#             labels=self.labels,
#             output_dir=self.test_output_dir
#         )
        
#         # Check that output files were created
#         for path in output_paths.values():
#             self.assertTrue(os.path.exists(path))


# if __name__ == "__main__":
#     unittest.main() 