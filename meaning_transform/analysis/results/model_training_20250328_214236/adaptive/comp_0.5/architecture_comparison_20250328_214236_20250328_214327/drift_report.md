# Semantic Drift Analysis Report

## Summary Statistics

| Metric | Mean | Min | Max | Std Dev |
|--------|------|-----|-----|--------|
| overall | 0.6208 | 0.6135 | 0.6425 | 0.0072 |
| position | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| health | 0.7941 | 0.7323 | 0.9646 | 0.0592 |
| energy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| is_alive | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| role | 0.9887 | 0.9782 | 0.9980 | 0.0045 |
| has_target_accuracy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| is_alive_accuracy | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| role_accuracy | 0.7267 | 0.4500 | 0.9500 | 0.1193 |

## Compression Analysis

Relationship between compression level and semantic preservation:

### Feature Degradation Order (fastest to slowest):

- **health**: degrades at 0.0113 per step
- **role**: degrades at 0.0004 per step
- **energy**: degrades at 0.0000 per step
- **position**: degrades at 0.0000 per step
- **is_alive**: degrades at 0.0000 per step
- **threatened**: degrades at 0.0000 per step
