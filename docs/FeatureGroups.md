# Standardized Feature Groups

This document explains the standardized feature groups used in the meaning-preserving transformation system, their importance weights, and how they inform evaluation.

## Feature Importance Analysis

The system organizes agent state features into standardized groups based on rigorous feature importance analysis. This analysis was conducted by:

1. **Reconstruction analysis**: Measuring how errors in each feature affect reconstruction quality
2. **Behavioral prediction**: Assessing each feature's importance for predicting agent behavior
3. **Outcome prediction**: Evaluating how features contribute to predicting simulation outcomes

Through this multi-faceted analysis, we determined the relative importance of different features to an agent's functional meaning.

## Standard Feature Groups

Our analysis revealed four primary feature groups with distinct importance levels:

### 1. Spatial Features (55.4% importance)

**Features included:**
- Position (x, y coordinates)
- Movement vectors (when available)

**Why important:** Spatial positioning is fundamentally tied to an agent's role in the environment, determining its sphere of influence, access to resources, and relationship to other agents. Position errors often have the most direct impact on agent behavior.

### 2. Resource Features (25.1% importance)

**Features included:**
- Health (normalized 0-1)
- Energy (normalized 0-1)
- Other resources (domain-specific)

**Why important:** Resource levels directly constrain an agent's capabilities and determine its ability to perform actions. They represent the "power budget" that shapes decision-making and survival.

### 3. Performance Features (10.5% importance)

**Features included:**
- Is alive (binary)
- Has target (binary)
- Threatened (binary)
- Other performance indicators

**Why important:** These features capture the agent's operational state and immediate goals/threats. While less important than position or resources in isolation, they provide critical context for interpreting other features.

### 4. Role Features (<5% importance)

**Features included:**
- Agent role/class
- Specialization
- Other role indicators

**Why important:** Role features provide categorical context but generally have less direct impact on moment-to-moment behavior. The meaning of an agent's role is often implicit in its position, resources, and performance.

## Canonical Importance Weights

Based on our analysis, we've established the following canonical weights for feature groups:

```python
CANONICAL_IMPORTANCE_WEIGHTS = {
    "spatial": 0.554,    # Position and movement
    "resources": 0.251,  # Health, energy, other resources
    "performance": 0.105, # Status indicators
    "role": 0.050,       # Agent type and specialization
}
```

These weights are used in standardized metrics to calculate weighted preservation scores that accurately reflect the importance of different features to agent meaning.

## Dynamic Weight Calculation

While canonical weights provide a useful baseline, the `FeatureImportanceAnalyzer` can also compute custom weights based on specific datasets or simulation contexts:

```python
from meaning_transform.src.feature_importance import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer(canonical_weights=False)
custom_weights = analyzer.analyze_importance_for_behavior(agent_states, behavior_vectors)
group_weights = analyzer.compute_importance_weights(custom_weights)
```

This allows for domain-specific tuning where the importance of features may differ from the canonical distribution.

## Visualization

The feature importance analysis provides visualizations to help understand the relative importance of features:

![Feature Importance](./images/feature_importance.png)

*Example visualization showing relative importance of individual features*

![Group Importance](./images/group_importance.png)

*Example visualization showing feature group importance distribution*

## Integration with Standardized Metrics

The standardized feature groups and their importance weights are directly integrated into the `StandardizedMetrics` class:

```python
from meaning_transform.src.standardized_metrics import StandardizedMetrics

# Use canonical weights
metrics = StandardizedMetrics()

# Or with custom weights
custom_metrics = StandardizedMetrics()
custom_metrics.FEATURE_GROUP_WEIGHTS = custom_weights

# Evaluate with weighted importance
results = metrics.evaluate(original_states, reconstructed_states)
```

This ensures that metrics for preservation, fidelity, and drift properly account for the relative importance of different aspects of agent state.

## Applications

Standardized feature groups are used throughout the system:

1. **Weighted semantic loss** during training to focus on preserving important features
2. **Preservation scoring** that prioritizes features most critical to meaning
3. **Drift tracking** that weights changes by feature importance
4. **Visualization dashboards** that highlight the most significant aspects of preservation
5. **Experiment comparison** using consistent feature group definitions

## Customization

While the canonical feature groups provide a solid foundation, they can be customized for specific domains:

1. Add domain-specific features to the appropriate existing groups
2. Create new feature groups for unique domains
3. Recompute importance weights based on domain-specific behavior or outcomes
4. Update the FEATURE_GROUPS and FEATURE_GROUP_WEIGHTS constants in your metrics instance

## Benchmarks and Thresholds

Different feature groups have different error tolerance thresholds:

| Feature Group | Excellent | Good | Acceptable | Poor | Critical |
|---------------|-----------|------|------------|------|----------|
| Spatial       | ≥ 0.95    | ≥ 0.90 | ≥ 0.85     | ≥ 0.70 | < 0.70   |
| Resources     | ≥ 0.95    | ≥ 0.90 | ≥ 0.85     | ≥ 0.70 | < 0.70   |
| Performance   | ≥ 0.97    | ≥ 0.95 | ≥ 0.90     | ≥ 0.80 | < 0.80   |
| Role          | ≥ 0.99    | ≥ 0.97 | ≥ 0.95     | ≥ 0.90 | < 0.90   |

These thresholds reflect the differential impact of errors in different feature groups, with higher standards for the binary and categorical features in the performance and role groups. 