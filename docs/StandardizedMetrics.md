# Standardized Metrics: Operational Definitions

This document provides precise operational definitions for the standardized metrics framework used to evaluate meaning preservation in agent state transformations.

## Core Concepts

### Meaning Preservation

**Definition**: The degree to which the essential semantic properties of an agent state are maintained across transformations, weighted by the importance of each property to agent behavior and function.

**Mathematical formulation**:
- Let $O$ be the original agent state tensor
- Let $R$ be the reconstructed agent state tensor
- Let $f_i(X)$ be the extraction function for semantic feature $i$
- Let $w_i$ be the importance weight for feature $i$

Preservation score for feature $i$:
$P_i(O, R) = \exp(-L_i(f_i(O), f_i(R)))$

Where $L_i$ is an appropriate loss function for feature type $i$.

Overall preservation:
$P_{overall}(O, R) = \frac{\sum_i w_i \cdot P_i(O, R)}{\sum_i w_i}$

### Fidelity

**Definition**: The accuracy of reconstruction at the raw feature level, without importance weighting. Fidelity measures how precisely the exact values are reconstructed, even for less important features.

**Mathematical formulation**:
For binary features:
$F_{binary}(O, R) = \text{Accuracy}(f_{binary}(O), f_{binary}(R))$

For categorical features:
$F_{categorical}(O, R) = \text{Accuracy}(f_{categorical}(O), f_{categorical}(R))$

For continuous features:
$F_{continuous}(O, R) = \exp(-\text{RMSE}(f_{continuous}(O), f_{continuous}(R)))$

Overall fidelity:
$F_{overall}(O, R) = \frac{1}{n} \sum_i F_i(O, R)$

### Semantic Drift

**Definition**: The degradation of meaning preservation over time or across compression levels compared to a baseline. Positive drift values indicate worse performance (more meaning loss).

**Mathematical formulation**:
- Let $O_b, R_b$ be the baseline original and reconstructed states
- Let $O_c, R_c$ be the current original and reconstructed states

Drift for feature $i$:
$D_i(O_b, R_b, O_c, R_c) = \max(0, P_i(O_b, R_b) - P_i(O_c, R_c))$

Overall drift:
$D_{overall} = \frac{\sum_i w_i \cdot D_i}{\sum_i w_i}$

## Standardized Feature Groups

Based on feature importance analysis, we've established the following standard feature groups:

### Spatial Features (55.4% importance)
- Position (x, y coordinates)
- Movement vectors (where applicable)

### Resource Features (25.1% importance)
- Health (normalized 0-1)
- Energy (normalized 0-1)
- Other resources (where applicable)

### Performance Features (10.5% importance)
- Is alive (binary)
- Has target (binary)
- Threatened (binary)
- Other performance indicators

### Role Features (<5% importance)
- Agent role/class
- Specialization
- Other role indicators

## Standardization Procedures

### Score Normalization

All metrics are normalized to a [0,1] range for consistent comparison:
- 1.0 represents perfect preservation/fidelity
- 0.0 represents complete loss of meaning/structure

For error-based metrics, we apply the transformation:
$\text{Normalized score} = \exp(-\text{error})$

### Feature Weighting

Standard weights based on feature importance analysis:
- Spatial features: 0.554
- Resource features: 0.251
- Performance features: 0.105
- Role features: 0.050

These weights are used in weighted averages to calculate overall preservation and drift scores.

## Qualitative Performance Categories

Each metric is mapped to a qualitative category based on established thresholds:

| Category | Preservation/Fidelity | Drift |
|----------|------------------------|-------|
| Excellent | ≥ 0.95 | ≤ 0.05 |
| Good | ≥ 0.90 | ≤ 0.10 |
| Acceptable | ≥ 0.85 | ≤ 0.15 |
| Poor | ≥ 0.70 | ≤ 0.30 |
| Critical | < 0.70 | > 0.30 |

Note: For drift, lower values are better (less meaning loss).

## Usage Guidelines

### When to Use Each Metric

- **Preservation**: When evaluating how well the essential meaning is maintained across transformations, especially for behavioral equivalence.
- **Fidelity**: When evaluating raw reconstruction quality, particularly useful for debugging encoding/decoding issues.
- **Drift**: When comparing performance across time, compression levels, or different model architectures.

### Comparison Across Experiments

When comparing different experiments:
1. Always use the standardized metrics
2. Use the same feature groups and weights across experiments
3. Report both preservation and fidelity scores
4. Include drift measurements when comparing to a baseline

## Implementation Reference

The standardized metrics are implemented in `meaning_transform/src/standardized_metrics.py`:

```python
from meaning_transform.src.standardized_metrics import StandardizedMetrics

# Create metrics instance
metrics = StandardizedMetrics()

# Evaluate with standardized metrics
results = metrics.evaluate(original_states, reconstructed_states)

# Access specific metrics
preservation = results["overall_preservation"]
fidelity = results["overall_fidelity"]
category = results["preservation_category"]  # e.g., "good", "acceptable"
``` 