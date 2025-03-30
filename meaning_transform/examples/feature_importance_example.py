#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script demonstrating feature importance analysis with standardized metrics.

This script shows how to:
1. Analyze feature importance for agent states
2. Compute feature group weights based on importance
3. Visualize feature importance
4. Use standardized metrics with feature importance-based weighting
"""

import os
import sqlite3
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from meaning_transform.src.data import AgentStateDataset
from meaning_transform.src.feature_importance import (
    FeatureImportanceAnalyzer,
    analyze_feature_importance,
)
from meaning_transform.src.model import MeaningVAE
from meaning_transform.src.standardized_metrics import StandardizedMetrics


def load_real_data_from_db(db_path, limit=5000):
    """
    Load real agent state data from SQLite database.

    Args:
        db_path: Path to SQLite database
        limit: Maximum number of samples to load

    Returns:
        agent_states: Tensor of agent states
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)

    # Query agent states based on the correct schema from models.py
    query = f"""
    SELECT 
        position_x, position_y, position_z, 
        resource_level, current_health, is_defending,
        total_reward, age, step_number
    FROM agent_states 
    ORDER BY step_number DESC
    LIMIT {limit}
    """

    # Load into DataFrame
    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"Loaded {len(df)} agent states from database")

    # Convert boolean to float
    if "is_defending" in df.columns:
        df["is_defending"] = df["is_defending"].astype(float)

    # Print column info for debugging
    print("\nFeatures loaded from database:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")

    # Convert to numpy array and then to tensor
    states_array = df.to_numpy().astype(np.float32)
    agent_states = torch.tensor(states_array)

    # Normalize features to [0,1] range for compatibility with the model
    # Position
    if agent_states.shape[1] >= 3:
        pos_max = torch.max(torch.abs(agent_states[:, 0:3]))
        if pos_max > 0:
            agent_states[:, 0:3] = (agent_states[:, 0:3] / pos_max + 1) / 2

    # Resources and health (assuming values are in a reasonable range)
    if agent_states.shape[1] >= 5:
        # Normalize resource_level
        resource_max = torch.max(agent_states[:, 3])
        if resource_max > 0:
            agent_states[:, 3] = agent_states[:, 3] / resource_max

        # Normalize health (assuming 0-100 range)
        health_max = torch.max(agent_states[:, 4])
        if health_max > 0:
            agent_states[:, 4] = agent_states[:, 4] / health_max

    # Age and step number - normalize by max value
    if agent_states.shape[1] >= 8:
        age_max = torch.max(agent_states[:, 7])
        if age_max > 0:
            agent_states[:, 7] = agent_states[:, 7] / age_max

    if agent_states.shape[1] >= 9:
        step_max = torch.max(agent_states[:, 8])
        if step_max > 0:
            agent_states[:, 8] = agent_states[:, 8] / step_max

    # Final clamp to ensure all values are in [0,1]
    agent_states = torch.clamp(agent_states, 0.0, 1.0)

    # Create a fake "role" feature for demonstration (one-hot encoded with 5 roles)
    # This simulates the expected format for SemanticMetrics
    n_samples = agent_states.shape[0]
    n_roles = 5
    roles = torch.zeros((n_samples, n_roles))

    # Assign random roles based on health level
    role_indices = torch.clamp(
        torch.floor(agent_states[:, 4] * n_roles).long(), 0, n_roles - 1
    )
    roles[torch.arange(n_samples), role_indices] = 1.0

    # Add a has_target feature based on is_defending
    has_target = agent_states[:, 5:6]  # is_defending can be used as has_target

    # Construct the final tensor in the expected format for SemanticMetrics
    # [position_x, position_y, health, has_target, energy, role1, role2, role3, role4, role5]
    final_tensor = torch.zeros((n_samples, 10))
    final_tensor[:, 0:2] = agent_states[:, 0:2]  # position x,y
    final_tensor[:, 2] = agent_states[:, 4]  # health
    final_tensor[:, 3] = has_target.squeeze()  # has_target
    final_tensor[:, 4] = agent_states[:, 3]  # energy (resource_level)
    final_tensor[:, 5:10] = roles  # role one-hot encoding

    print(f"\nConstructed tensor with shape: {final_tensor.shape}")
    print(
        f"Feature ranges: min={torch.min(final_tensor).item():.4f}, max={torch.max(final_tensor).item():.4f}"
    )

    return final_tensor


def main():
    """
    Run the feature importance analysis example.
    """
    print("Feature Importance Analysis Example")
    print("===================================")

    # Load real data from SQLite database
    print("\nLoading real agent states from database...")
    db_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../data/simulation.db")
    )
    original_states = load_real_data_from_db(db_path, limit=2000)

    n_samples = original_states.shape[0]
    print(
        f"Loaded and processed {n_samples} agent states with {original_states.shape[1]} features"
    )

    # Create simple behavior vectors for importance analysis (using reward and movement)
    print("\nCreating behavior vectors from agent states...")
    behavior_vectors = np.random.rand(n_samples, 4)  # Placeholder for actual behavior

    # Create binary outcome for importance analysis based on health
    print("\nCreating binary outcome features...")
    survival_outcome = (original_states[:, 2] > 0.5).cpu().numpy()  # Based on health
    print(
        f"Created {len(survival_outcome)} outcome values (survival rate: {np.mean(survival_outcome):.2f})"
    )

    # Create a model for compression/reconstruction
    print("\nCreating model and running inference...")
    input_dim = original_states.shape[1]
    latent_dim = 16
    model = MeaningVAE(input_dim=input_dim, latent_dim=latent_dim)

    # Encode and reconstruct
    with torch.no_grad():
        # The model might return different values than expected, so handle accordingly
        model_output = model(original_states)

        # Check what kind of output the model returns
        if isinstance(model_output, tuple) and len(model_output) == 3:
            # Expected original output format: reconstructed, mu, logvar
            reconstructed_states, _, _ = model_output
        elif isinstance(model_output, torch.Tensor):
            # Direct tensor output
            reconstructed_states = model_output
        elif isinstance(model_output, dict) and "reconstruction" in model_output:
            # Dictionary output with named fields
            reconstructed_states = model_output["reconstruction"]
        else:
            # Default handling if unsure about format
            print("Warning: Unknown model output format. Using first element.")
            if isinstance(model_output, tuple):
                reconstructed_states = model_output[0]
            else:
                reconstructed_states = model_output

    # Basic error calculation for context
    mse = ((original_states - reconstructed_states) ** 2).mean().item()
    print(f"Mean Squared Error: {mse:.6f}")

    # Run feature importance analysis
    print("\nAnalyzing feature importance...")
    importance_results = analyze_feature_importance(
        original_states=original_states,
        reconstructed_states=reconstructed_states,
        behavior_vectors=behavior_vectors,
        outcome_values=survival_outcome,
        outcome_type="binary",
        create_visualizations=True,
    )

    # Display feature importance scores
    print("\nFeature Importance Scores:")
    for feature, score in sorted(
        importance_results["feature_importance"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"  {feature}: {score:.4f}")

    # Display group weights
    print("\nFeature Group Weights:")
    for group, weight in sorted(
        importance_results["group_weights"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {group}: {weight:.4f}")

    # Create directory for outputs
    os.makedirs("results", exist_ok=True)

    # Save visualizations
    print("\nSaving visualizations...")
    importance_results["feature_importance_figure"].savefig(
        "results/feature_importance.png", dpi=300, bbox_inches="tight"
    )
    importance_results["group_importance_figure"].savefig(
        "results/group_importance.png", dpi=300, bbox_inches="tight"
    )

    # Use standardized metrics with custom feature weights
    print("\nEvaluating with standardized metrics...")

    # Create metrics with canonical weights
    canonical_metrics = StandardizedMetrics(normalize_scores=True)
    canonical_results = canonical_metrics.evaluate(
        original_states, reconstructed_states
    )

    # Create custom metrics with the computed weights
    custom_weights = importance_results["group_weights"]

    # Create a new StandardizedMetrics instance
    custom_metrics = StandardizedMetrics(normalize_scores=True)

    # Override the standard weights with our computed weights
    custom_metrics.FEATURE_GROUP_WEIGHTS = custom_weights
    custom_results = custom_metrics.evaluate(original_states, reconstructed_states)

    # Compare results
    print("\nCanonical Weights Evaluation:")
    print(f"  Overall Preservation: {canonical_results['overall_preservation']:.4f}")
    print(f"  Overall Fidelity: {canonical_results['overall_fidelity']:.4f}")
    print(f"  Preservation Category: {canonical_results['preservation_category']}")
    print(f"  Fidelity Category: {canonical_results['fidelity_category']}")

    print("\nCustom Weights Evaluation:")
    print(f"  Overall Preservation: {custom_results['overall_preservation']:.4f}")
    print(f"  Overall Fidelity: {custom_results['overall_fidelity']:.4f}")
    print(f"  Preservation Category: {custom_results['preservation_category']}")
    print(f"  Fidelity Category: {custom_results['fidelity_category']}")

    # Show differences in group-level metrics
    print("\nGroup-Level Preservation Metrics:")
    for group in ["spatial", "resources", "performance", "role"]:
        canonical = canonical_results["preservation"].get(f"{group}_preservation", 0.0)
        custom = custom_results["preservation"].get(f"{group}_preservation", 0.0)
        diff = custom - canonical
        print(
            f"  {group}: {canonical:.4f} (canonical) vs {custom:.4f} (custom), diff: {diff:.4f}"
        )

    print("\nExample completed. See 'results/' directory for visualizations.")

    # Advanced usage: drift analysis with different weights
    print("\nDemonstrating drift analysis with different weights...")

    # Simulate a drift scenario by making small changes to the model
    # (in practice, this would be two different training runs or compression levels)
    baseline_reconstructed = reconstructed_states.clone()

    # Introduce some drift (more error in spatial features)
    drift_reconstructed = reconstructed_states.clone()
    drift_reconstructed[:, 0:2] += torch.randn_like(drift_reconstructed[:, 0:2]) * 0.05

    # Measure drift with canonical weights
    canonical_drift = canonical_metrics.measure_drift(
        original_states, drift_reconstructed, original_states, baseline_reconstructed
    )

    # Measure drift with custom weights
    custom_drift = custom_metrics.measure_drift(
        original_states, drift_reconstructed, original_states, baseline_reconstructed
    )

    # Show drift metrics
    print("\nDrift Metrics (Canonical Weights):")
    for key, value in sorted(canonical_drift.items()):
        if key == "drift_category" or key.endswith("_drift"):
            print(f"  {key}: {value if isinstance(value, str) else f'{value:.4f}'}")

    print("\nDrift Metrics (Custom Weights):")
    for key, value in sorted(custom_drift.items()):
        if key == "drift_category" or key.endswith("_drift"):
            print(f"  {key}: {value if isinstance(value, str) else f'{value:.4f}'}")

    print("\nFeature importance analysis example completed!")


if __name__ == "__main__":
    main()
