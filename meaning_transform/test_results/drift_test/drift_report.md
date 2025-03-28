# Semantic Drift Analysis Report

## Summary Statistics

| Metric | Mean | Min | Max | Std Dev |
|--------|------|-----|-----|--------|
| overall | 0.9891 | 0.9786 | 0.9982 | 0.0081 |
| position | 0.9864 | 0.9749 | 0.9984 | 0.0121 |
| health | 0.9848 | 0.9691 | 0.9952 | 0.0112 |
| energy | 0.9898 | 0.9775 | 0.9983 | 0.0090 |
| is_alive | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| role | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| has_target_accuracy | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| is_alive_accuracy | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| role_accuracy | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## Compression Analysis

Relationship between compression level and semantic preservation:

- **Minimum acceptable compression level**: 2.50 bits per dimension
  (maintains at least 90% semantic preservation)

### Feature Degradation Order (fastest to slowest):

- **health**: degrades at 0.0065 per step
- **position**: degrades at 0.0059 per step
- **energy**: degrades at 0.0052 per step
- **is_alive**: degrades at 0.0000 per step
- **role**: degrades at 0.0000 per step
- **threatened**: degrades at 0.0000 per step
