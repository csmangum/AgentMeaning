# Semantic Drift Analysis Report

## Summary Statistics

| Metric | Mean | Min | Max | Std Dev |
|--------|------|-----|-----|--------|
| overall | 0.4518 | 0.4515 | 0.4522 | 0.0005 |
| position | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| health | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| energy | 0.9122 | 0.9063 | 0.9181 | 0.0083 |
| is_alive | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| role | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| has_target_accuracy | 0.3000 | 0.3000 | 0.3000 | 0.0000 |
| is_alive_accuracy | 0.4500 | 0.3000 | 0.6000 | 0.2121 |
| role_accuracy | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## Compression Analysis

Relationship between compression level and semantic preservation:

### Feature Degradation Order (fastest to slowest):

- **energy**: degrades at 0.0059 per step
- **position**: degrades at 0.0000 per step
- **role**: degrades at 0.0000 per step
- **threatened**: degrades at 0.0000 per step
- **health**: degrades at -0.0000 per step
- **is_alive**: degrades at -0.0000 per step
