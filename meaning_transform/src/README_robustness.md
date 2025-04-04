# Feature Importance Hierarchy Robustness Analysis

This module implements step 18 of the project roadmap, creating a comprehensive framework for validating and testing the robustness of feature importance hierarchies in agent state representations.

## Key Components

### 1. Cross-Validation Framework
- Implements k-fold cross-validation (default: 5 folds) to test stability of feature importance rankings
- Calculates importance scores and ranks for each feature across multiple data splits
- Provides statistical analysis of importance score distribution and rank stability

### 2. Multi-Method Importance Analysis
- Compares different importance measurement techniques:
  - Permutation importance (baseline method)
  - Random Forest feature importance
  - SHAP (SHapley Additive exPlanations) values
- Analyzes correlation between different importance measurement methods
- Identifies which features show consistent importance across methods

### 3. Context Robustness Testing
- Tests importance hierarchy across different simulation contexts:
  - Standard test environment
  - Combat-focused environment (high threat, low health)
  - Resource gathering environment (low threat, high health)
  - Exploration environment (medium threat, wider position range)
- Compares feature rankings across contexts to identify universally important features

### 4. Feature Extraction Sensitivity Analysis
- Tests stability of importance rankings with varying feature extraction parameters
- Analyzes how different extraction methods affect feature importance:
  - Position normalization variations
  - Role encoding variations (default vs. one-hot)
  - Binary threshold variations
- Identifies features whose importance is sensitive to extraction methodology

### 5. Visualization Dashboard
- Creates comprehensive visualization suite:
  - Box plots of importance distribution across folds
  - Heatmaps of importance scores by fold
  - Stability analysis scatter plots
  - Context comparison bar charts
  - Method comparison visualizations
  - HTML dashboard combining all visualizations

## Usage

Run the complete analysis using the provided experiment script:

```bash
python experiments/run_feature_importance_robustness.py --num_states 5000 --n_folds 5 --output_dir findings/feature_importance_robustness
```

Parameters:
- `--dataset_path`: Path to existing dataset (optional)
- `--num_states`: Number of synthetic states to generate if no dataset provided (default: 5000)
- `--model_path`: Path to pretrained model (optional)
- `--n_folds`: Number of folds for cross-validation (default: 5)
- `--random_seed`: Random seed for reproducibility (default: 42)
- `--output_dir`: Directory to save results (default: findings/feature_importance_robustness)

## Requirements

Additional Python packages required:
- numpy>=1.20.0
- torch>=1.9.0
- pandas>=1.3.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- scikit-learn>=0.24.0
- shap>=0.39.0
- tqdm>=4.60.0

Install with:
```bash
pip install -r experiments/feature_importance_requirements.txt
```

## Output

The analysis generates:
1. Detailed statistical reports on feature importance robustness
2. Visualizations of importance stability across multiple dimensions
3. CSV files with raw data for further analysis
4. HTML dashboard summarizing all findings in one view

## Integration with Prior Results

This analysis builds on previous feature importance findings (step 12) by:
1. Validating the original importance hierarchy (spatial: 55.4%, resources: 25.1%, performance: 10.5%, etc.)
2. Testing robustness across different datasets, contexts, and measurement methods
3. Identifying which features maintain stable rankings across different conditions
4. Quantifying confidence intervals for feature importance measurements 