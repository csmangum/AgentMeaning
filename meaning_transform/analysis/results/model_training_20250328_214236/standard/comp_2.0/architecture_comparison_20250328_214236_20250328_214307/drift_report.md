# Semantic Drift Analysis Report

## Summary Statistics

| Metric | Mean | Min | Max | Std Dev |
|--------|------|-----|-----|--------|
| overall | 0.6070 | 0.3840 | 0.6399 | 0.0621 |
| position | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| health | 0.9330 | 0.9080 | 0.9978 | 0.0256 |
| energy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| is_alive | 0.9338 | 0.0067 | 1.0000 | 0.2565 |
| role | 0.9058 | 0.8746 | 0.9608 | 0.0279 |
| has_target_accuracy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| is_alive_accuracy | 0.9967 | 0.9500 | 1.0000 | 0.0129 |
| role_accuracy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Compression Analysis

Relationship between compression level and semantic preservation:

### Feature Degradation Order (fastest to slowest):

- **health**: degrades at 0.0056 per step
- **role**: degrades at 0.0041 per step
- **position**: degrades at 0.0000 per step
- **is_alive**: degrades at 0.0000 per step
- **threatened**: degrades at 0.0000 per step
- **energy**: degrades at -0.0000 per step
