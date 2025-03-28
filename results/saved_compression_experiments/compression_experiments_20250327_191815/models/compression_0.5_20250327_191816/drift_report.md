# Semantic Drift Analysis Report

## Summary Statistics

| Metric | Mean | Min | Max | Std Dev |
|--------|------|-----|-----|--------|
| overall | 0.3589 | 0.3386 | 0.3792 | 0.0287 |
| position | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| health | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| energy | 0.5509 | 0.3783 | 0.7235 | 0.2441 |
| is_alive | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| role | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| has_target_accuracy | 0.2500 | 0.2000 | 0.3000 | 0.0707 |
| is_alive_accuracy | 0.3500 | 0.3000 | 0.4000 | 0.0707 |
| role_accuracy | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## Compression Analysis

Relationship between compression level and semantic preservation:

### Feature Degradation Order (fastest to slowest):

- **is_alive**: degrades at 0.0000 per step
- **role**: degrades at 0.0000 per step
- **threatened**: degrades at 0.0000 per step
- **position**: degrades at -0.0000 per step
- **health**: degrades at -0.0000 per step
- **energy**: degrades at -0.1726 per step
