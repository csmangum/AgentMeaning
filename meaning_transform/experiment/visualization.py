#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization Utilities for Meaning-Preserving Transformation Experiments

This module provides standard visualization functions for consistent experiment reporting:
- Radar charts for comparing metrics
- Category visualizations for quality assessment
- Comparison charts for parameter studies
- Behavioral analysis visualizations
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_radar_chart(
    results_df: pd.DataFrame,
    output_path: Union[str, Path],
    level_column: str = "compression_level",
    skip_cols: List[str] = None,
    title: str = "Comparison of All Metrics Across Levels",
):
    """
    Create a radar chart to compare all metrics at different experiment levels.

    Args:
        results_df: DataFrame containing experiment results
        output_path: Path to save the visualization
        level_column: Column name for the level/parameter being varied
        skip_cols: Columns to skip in the radar chart
        title: Chart title
    """
    if skip_cols is None:
        skip_cols = [level_column]
    else:
        skip_cols = list(skip_cols) + [level_column]
    
    # Also skip any columns with '_category' in the name
    category_cols = [col for col in results_df.columns if "_category" in col]
    skip_cols.extend(category_cols)

    # Get numeric columns that aren't in skip_cols
    metrics = [
        col
        for col in results_df.columns
        if col not in skip_cols
        and pd.api.types.is_numeric_dtype(results_df[col])
        and not all(pd.isna(results_df[col]))
    ]

    if not metrics:
        print("Warning: No suitable metrics found for radar chart")
        return

    # Create a radar chart
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

    # Number of metrics
    N = len(metrics)

    # Angle of each axis
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Add axis labels
    plt.xticks(angles[:-1], metrics, size=9)

    # Plot each level
    for level, color in zip(
        sorted(results_df[level_column].unique()),
        plt.cm.rainbow(
            np.linspace(0, 1, len(results_df[level_column].unique()))
        ),
    ):
        level_data = results_df[results_df[level_column] == level]

        # Get values for current level
        values = [level_data[metric].values[0] for metric in metrics]
        values += values[:1]  # Close the loop

        # Plot values
        ax.plot(
            angles, values, linewidth=2, linestyle="solid", label=f"Level {level}"
        )
        ax.fill(angles, values, alpha=0.1)

    # Add legend
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    plt.title(title)
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()


def create_category_chart(
    results_df: pd.DataFrame,
    output_path: Union[str, Path],
    level_column: str = "compression_level",
):
    """
    Create a chart showing quality categories at different levels.

    Args:
        results_df: DataFrame containing experiment results
        output_path: Path to save the visualization
        level_column: Column name for the level/parameter being varied
    """
    # Get all category fields
    category_fields = [col for col in results_df.columns if col.endswith("_category")]

    if not category_fields:
        print("Warning: No category fields found for category chart")
        return

    plt.figure(figsize=(15, 8))

    # Create a mapping for category values to numeric scores for visualization
    category_map = {
        "excellent": 5,
        "good": 4,
        "acceptable": 3,
        "poor": 2,
        "critical": 1,
    }

    # Create a chart for each category field
    for i, field in enumerate(category_fields):
        plt.subplot(1, len(category_fields), i + 1)

        # Convert categories to numeric values for plotting
        numeric_categories = [category_map.get(cat, 0) for cat in results_df[field]]

        # Create scatter plot with size based on level
        sizes = 100 + results_df[level_column] * 20
        plt.scatter(results_df[level_column], numeric_categories, s=sizes)

        # Add category labels
        for level, cat, y_val in zip(
            results_df[level_column], results_df[field], numeric_categories
        ):
            plt.text(level, y_val, cat, ha="center", va="center", fontsize=8)

        # Set y-axis ticks
        plt.yticks(list(category_map.values()), list(category_map.keys()))

        plt.xlabel(level_column.replace("_", " ").title())
        plt.title(field.replace("_category", " Category"))
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_comparison_plot(
    results_df: pd.DataFrame,
    output_path: Union[str, Path],
    level_column: str = "compression_level",
    metrics: List[str] = None,
    subplot_layout: Tuple[int, int] = (2, 2),
    figsize: Tuple[int, int] = (15, 10),
    titles: List[str] = None,
    ylabels: List[str] = None,
):
    """
    Create a multi-panel comparison plot for experiment metrics.

    Args:
        results_df: DataFrame containing experiment results
        output_path: Path to save the visualization
        level_column: Column name for the level/parameter being varied
        metrics: List of metrics to include in each subplot (if None, will use common metrics)
        subplot_layout: Layout of subplots as (rows, cols) 
        figsize: Figure size as (width, height)
        titles: List of subplot titles (should match number of subplots)
        ylabels: List of y-axis labels (should match number of subplots)
    """
    # Default metrics if none provided
    if metrics is None:
        # Standard preservation metrics
        if "overall_preservation" in results_df.columns:
            metrics = [
                ["overall_preservation"],
                ["overall_fidelity"],
                # Feature-specific metrics
                ["spatial_preservation", "resources_preservation", 
                 "performance_preservation", "role_preservation"],
                # Loss metrics
                ["val_loss", "recon_loss", "kl_loss", "semantic_loss"]
            ]
        else:
            # Default to all numeric columns, up to 4 groups
            numeric_cols = [
                col for col in results_df.columns
                if col != level_column and pd.api.types.is_numeric_dtype(results_df[col])
            ]
            
            # Split into groups of at most 4
            n = min(4, len(numeric_cols))
            metrics = [[] for _ in range(n)]
            for i, col in enumerate(numeric_cols):
                metrics[i % n].append(col)

    # Default titles
    if titles is None:
        titles = ["Metric Comparison"] * len(metrics)
    
    # Default ylabels
    if ylabels is None:
        ylabels = ["Value"] * len(metrics)

    # Create the figure
    plt.figure(figsize=figsize)

    # Create each subplot
    for i, metric_group in enumerate(metrics):
        plt.subplot(subplot_layout[0], subplot_layout[1], i + 1)
        
        for metric in metric_group:
            if metric in results_df.columns:
                plt.plot(
                    results_df[level_column],
                    results_df[metric],
                    marker="o",
                    linewidth=2,
                    label=metric.replace("_", " ").title()
                )
        
        plt.xlabel(level_column.replace("_", " ").title())
        plt.ylabel(ylabels[i])
        plt.title(titles[i])
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_behavioral_visualization(
    results_df: pd.DataFrame,
    output_path: Union[str, Path],
    level_column: str = "compression_level",
):
    """
    Create visualization for behavioral metrics.

    Args:
        results_df: DataFrame containing experiment results
        output_path: Path to save the visualization
        level_column: Column name for the level/parameter being varied
    """
    # Only create if we have the necessary columns
    if not all(
        col in results_df.columns
        for col in ["behavioral_equivalence", "action_similarity"]
    ):
        print("Warning: Missing behavioral metrics for visualization")
        return

    plt.figure(figsize=(12, 6))

    # Plot behavioral metrics
    plt.subplot(1, 2, 1)
    plt.plot(
        results_df[level_column],
        results_df["behavioral_equivalence"],
        "b-o",
        label="Behavioral Equivalence",
    )
    plt.plot(
        results_df[level_column],
        results_df["action_similarity"],
        "g-o",
        label="Action Similarity",
    )
    if "goal_alignment" in results_df.columns:
        plt.plot(
            results_df[level_column],
            results_df["goal_alignment"],
            "r-o",
            label="Goal Alignment",
        )
    plt.xlabel(level_column.replace("_", " ").title())
    plt.ylabel("Score")
    plt.title("Behavioral Metrics vs. Level")
    plt.legend()
    plt.grid(True)

    # Plot correlation between semantic and behavioral
    plt.subplot(1, 2, 2)
    plt.scatter(
        results_df["overall_preservation"],
        results_df["behavioral_equivalence"],
        s=80,
        c=results_df[level_column],
        cmap="viridis",
    )
    plt.colorbar(label=level_column.replace("_", " ").title())
    plt.xlabel("Semantic Preservation")
    plt.ylabel("Behavioral Equivalence")
    plt.title("Semantic vs. Behavioral Equivalence")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_parameter_efficiency_visualization(
    results_df: pd.DataFrame,
    output_path: Union[str, Path],
    level_column: str = "compression_level",
    param_column: str = "param_count",
    score_column: str = "overall_preservation",
):
    """
    Create visualization for parameter efficiency.

    Args:
        results_df: DataFrame containing experiment results
        output_path: Path to save the visualization
        level_column: Column name for the level/parameter being varied
        param_column: Column name for parameter count
        score_column: Column name for the score/metric to use
    """
    if param_column not in results_df.columns or score_column not in results_df.columns:
        print(f"Warning: Missing columns {param_column} or {score_column} for parameter efficiency visualization")
        return

    plt.figure(figsize=(12, 10))

    # Plot 1: Parameter count vs Level
    plt.subplot(2, 2, 1)
    plt.plot(
        results_df[level_column],
        results_df[param_column],
        "b-o",
        linewidth=2,
    )
    plt.xlabel(level_column.replace("_", " ").title())
    plt.ylabel("Parameter Count")
    plt.title("Model Size vs. Level")
    plt.grid(True)
    plt.yscale("log")

    # Plot 2: Effective Dimension vs Level (if available)
    if "effective_dim" in results_df.columns:
        plt.subplot(2, 2, 2)
        plt.plot(
            results_df[level_column],
            results_df["effective_dim"],
            "g-o",
            linewidth=2,
        )
        plt.xlabel(level_column.replace("_", " ").title())
        plt.ylabel("Effective Latent Dimension")
        plt.title("Effective Dimension vs. Level")
        plt.grid(True)
    else:
        # Alternative plot if effective_dim not available
        plt.subplot(2, 2, 2)
        compression_rate = results_df[level_column] if "compression_rate" not in results_df.columns else results_df["compression_rate"]
        plt.plot(
            results_df[level_column],
            compression_rate,
            "g-o",
            linewidth=2,
        )
        plt.xlabel(level_column.replace("_", " ").title())
        plt.ylabel("Compression Rate")
        plt.title("Compression Rate vs. Level")
        plt.grid(True)

    # Plot 3: Parameter Efficiency
    plt.subplot(2, 2, 3)
    param_efficiency = results_df[score_column] / np.log10(results_df[param_column])
    plt.plot(results_df[level_column], param_efficiency, "r-o", linewidth=2)
    plt.xlabel(level_column.replace("_", " ").title())
    plt.ylabel(f"{score_column.replace('_', ' ').title()} / log10(Parameters)")
    plt.title("Parameter Efficiency vs. Level")
    plt.grid(True)

    # Plot 4: Relationship between Score and Parameter Count
    plt.subplot(2, 2, 4)
    plt.scatter(
        results_df[param_column],
        results_df[score_column],
        s=80,
        c=results_df[level_column],
        cmap="viridis",
    )
    plt.colorbar(label=level_column.replace("_", " ").title())
    plt.xlabel("Parameter Count")
    plt.ylabel(score_column.replace("_", " ").title())
    plt.title(f"{score_column.replace('_', ' ').title()} vs. Parameter Count")
    plt.grid(True)
    plt.xscale("log")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close() 