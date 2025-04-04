#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Importance Hierarchy Robustness Analysis Script

Performs feature importance analysis with cross-validation and generates comprehensive visualizations.
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from pathlib import Path

# Add project root to path
sys.path.append('.')

from meaning_transform.src.data import AgentStateDataset
from meaning_transform.src.feature_importance import FeatureImportanceAnalyzer

def calculate_permutation_importance(X, y, feature_names):
    """Calculate permutation importance using RandomForest."""
    # Train a RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Calculate permutation importance
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    
    # Convert to dictionary
    importance_dict = {}
    for i, feature in enumerate(feature_names):
        importance_dict[feature] = result.importances_mean[i]
    
    # Normalize to sum to 1
    total = sum(importance_dict.values())
    if total > 0:
        for feature in importance_dict:
            importance_dict[feature] /= total
    
    return importance_dict

def main():
    # Setup output directory
    output_dir = 'findings/feature_importance_robustness'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define feature extractors to analyze
    feature_extractors = [
        'position', 'health', 'energy', 'is_alive', 
        'has_target', 'threatened', 'role'
    ]
    
    # Load data
    print('Loading data from simulation.db...')
    dataset = AgentStateDataset()
    dataset.load_from_db('data/simulation.db', limit=5000)
    print(f'Loaded {len(dataset.states)} agent states from database')
    
    # Convert states to tensors
    print('Converting to tensors...')
    states_tensor = torch.stack([state.to_tensor() for state in dataset.states])
    
    # Create feature matrices for importance analysis
    analyzer = FeatureImportanceAnalyzer(feature_extractors)
    feature_matrices, combined_matrix = analyzer.extract_feature_matrix(states_tensor)
    
    # Create a synthetic target variable (combination of features)
    # Using a linear combination with different weights ensures differences in importance
    print('Creating synthetic target for importance analysis...')
    np.random.seed(42)
    true_weights = {
        'position': 0.55,  # Spatial is most important
        'health': 0.15,    # Resources are second
        'energy': 0.10,
        'is_alive': 0.05,  # Performance features
        'has_target': 0.03,
        'threatened': 0.02,
        'role': 0.10       # Role has moderate importance
    }
    
    # Create target using true weights with some noise
    y = np.zeros(len(states_tensor))
    start_idx = 0
    for feature_name, matrix in feature_matrices.items():
        # Add weighted contribution of this feature (average of its columns)
        feature_contribution = np.mean(matrix, axis=1) * true_weights[feature_name]
        y += feature_contribution
    
    # Add some noise
    y += np.random.normal(0, 0.1, size=len(y))
    
    # Basic importance analysis
    print('Analyzing feature importance...')
    
    # Calculate permutation importance
    importance = calculate_permutation_importance(combined_matrix, y, feature_extractors)
    
    # Cross-validation analysis
    print('Analyzing with cross-validation...')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = {f: [] for f in feature_extractors}
    
    for i, (train_idx, test_idx) in enumerate(kf.split(combined_matrix)):
        print(f'Processing fold {i+1}/5')
        X_train = combined_matrix[train_idx]
        y_train = y[train_idx]
        X_test = combined_matrix[test_idx]
        y_test = y[test_idx]
        
        # Calculate importance for this fold
        fold_importance = calculate_permutation_importance(X_train, y_train, feature_extractors)
        
        # Store results
        for f, score in fold_importance.items():
            fold_scores[f].append(score)
    
    # Calculate statistics
    mean_scores = {f: np.mean(fold_scores[f]) for f in feature_extractors}
    std_scores = {f: np.std(fold_scores[f]) for f in feature_extractors}
    
    # Calculate group weights
    group_weights = {
        'spatial': mean_scores['position'],
        'resources': mean_scores['health'] + mean_scores['energy'],
        'performance': mean_scores['is_alive'] + mean_scores['has_target'] + mean_scores['threatened'],
        'role': mean_scores['role']
    }
    
    # Save results
    print('Saving results...')
    results = {
        'importance': mean_scores,
        'std_dev': std_scores,
        'group_weights': group_weights,
        'fold_scores': fold_scores,
        'true_weights': true_weights
    }
    
    with open(f'{output_dir}/importance_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            'importance': {k: float(v) for k, v in results['importance'].items()},
            'std_dev': {k: float(v) for k, v in results['std_dev'].items()},
            'group_weights': {k: float(v) for k, v in results['group_weights'].items()},
            'fold_scores': {k: [float(x) for x in v] for k, v in results['fold_scores'].items()},
            'true_weights': results['true_weights']
        }
        json.dump(serializable_results, f, indent=2)
    
    # Create visualizations
    
    # 1. Feature importance bar chart
    plt.figure(figsize=(12, 6))
    sorted_features = sorted(mean_scores.keys(), key=lambda x: mean_scores[x], reverse=True)
    sorted_values = [mean_scores[f] for f in sorted_features]
    
    ax = sns.barplot(x=sorted_features, y=sorted_values)
    plt.xticks(rotation=45)
    plt.title('Feature Importance Scores')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png')
    
    # 2. Box plot showing distribution across folds
    plt.figure(figsize=(10, 6))
    fold_df = pd.DataFrame(fold_scores)
    # Reorder columns by importance
    fold_df = fold_df[sorted_features]
    ax = sns.boxplot(data=fold_df)
    plt.xticks(rotation=45)
    plt.title('Feature Importance Distribution Across Folds')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fold_distribution.png')
    
    # 3. Feature group pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(list(group_weights.values()), labels=list(group_weights.keys()), 
            autopct='%1.1f%%', startangle=90, explode=[0.1, 0, 0, 0])
    plt.title('Feature Group Importance Distribution')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/group_importance_pie.png')
    
    # 4. Feature stability (inverse of coefficient of variation)
    stability = {}
    for f in feature_extractors:
        mean = np.mean(fold_scores[f])
        std = np.std(fold_scores[f])
        if mean > 0:
            stability[f] = 1.0 - (std / mean)  # Higher value = more stable
        else:
            stability[f] = 0.0
    
    plt.figure(figsize=(12, 6))
    sorted_by_stability = sorted(stability.keys(), key=lambda x: stability[x], reverse=True)
    ax = sns.barplot(x=sorted_by_stability, y=[stability[f] for f in sorted_by_stability])
    plt.xticks(rotation=45)
    plt.title('Feature Importance Stability Across Folds')
    plt.ylabel('Stability Score (higher = more stable)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/importance_stability.png')
    
    # 5. Comparison with super-quick analysis (if available)
    super_quick_results_path = 'findings/feature_importance_robustness_super_quick/importance_results.json'
    if os.path.exists(super_quick_results_path):
        try:
            with open(super_quick_results_path, 'r') as f:
                quick_results = json.load(f)
                
            quick_importance = quick_results.get('importance', {})
            if quick_importance:
                # Create comparison
                plt.figure(figsize=(14, 8))
                comparison_data = []
                
                for feature in feature_extractors:
                    if feature in quick_importance and feature in mean_scores:
                        comparison_data.append({
                            'Feature': feature,
                            'Full Analysis': mean_scores[feature],
                            'Quick Analysis': quick_importance[feature]
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df = pd.melt(comparison_df, id_vars=['Feature'], 
                                       value_vars=['Full Analysis', 'Quick Analysis'],
                                       var_name='Analysis Type', value_name='Importance')
                
                sns.barplot(data=comparison_df, x='Feature', y='Importance', hue='Analysis Type')
                plt.xticks(rotation=45)
                plt.title('Feature Importance: Full vs. Quick Analysis')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/full_vs_quick_comparison.png')
                
                print("Created comparison with super-quick analysis")
        except Exception as e:
            print(f"Could not create comparison with super-quick analysis: {e}")
    
    # 6. True weights vs. discovered importance
    plt.figure(figsize=(14, 8))
    comparison_data = []
    for feature in feature_extractors:
        comparison_data.append({
            'Feature': feature,
            'Discovered Importance': mean_scores[feature],
            'True Weight': true_weights[feature]
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = pd.melt(comparison_df, id_vars=['Feature'], 
                           value_vars=['Discovered Importance', 'True Weight'],
                           var_name='Type', value_name='Value')
    
    sns.barplot(data=comparison_df, x='Feature', y='Value', hue='Type')
    plt.xticks(rotation=45)
    plt.title('True Weights vs. Discovered Importance')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/true_vs_discovered.png')
    
    # Print results
    print('Feature importance:', mean_scores)
    print('Group weights:', group_weights)
    print('Analysis completed!')

if __name__ == "__main__":
    main() 