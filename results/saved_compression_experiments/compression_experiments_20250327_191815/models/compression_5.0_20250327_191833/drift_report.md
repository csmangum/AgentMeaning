# Semantic Drift Analysis Report

## Summary Statistics

| Metric | Mean | Min | Max | Std Dev |
|--------|------|-----|-----|--------|
| overall | 0.3595 | 0.3424 | 0.3765 | 0.0241 |
| position | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| health | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| energy | 0.7333 | 0.7185 | 0.7480 | 0.0209 |
| is_alive | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| role | 0.7276 | 0.6811 | 0.7741 | 0.0658 |
| has_target_accuracy | 0.4000 | 0.4000 | 0.4000 | 0.0000 |
| is_alive_accuracy | 0.4000 | 0.4000 | 0.4000 | 0.0000 |
| role_accuracy | 0.5000 | 0.4000 | 0.6000 | 0.1414 |

## Compression Analysis

Relationship between compression level and semantic preservation:

### Feature Degradation Order (fastest to slowest):

- **position**: degrades at 0.0000 per step
- **is_alive**: degrades at 0.0000 per step
- **threatened**: degrades at 0.0000 per step
- **health**: degrades at -0.0000 per step
- **energy**: degrades at -0.0148 per step
- **role**: degrades at -0.0465 per step
