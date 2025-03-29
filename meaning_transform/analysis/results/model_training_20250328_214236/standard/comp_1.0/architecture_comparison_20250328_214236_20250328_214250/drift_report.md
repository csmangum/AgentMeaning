# Semantic Drift Analysis Report

## Summary Statistics

| Metric | Mean | Min | Max | Std Dev |
|--------|------|-----|-----|--------|
| overall | 0.6122 | 0.3940 | 0.6406 | 0.0630 |
| position | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| health | 0.8921 | 0.8485 | 0.9980 | 0.0412 |
| energy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| is_alive | 0.9291 | 0.0067 | 1.0000 | 0.2655 |
| role | 0.9688 | 0.9512 | 0.9802 | 0.0085 |
| has_target_accuracy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| is_alive_accuracy | 0.9964 | 0.9500 | 1.0000 | 0.0134 |
| role_accuracy | 0.2750 | 0.1000 | 0.5000 | 0.1411 |

## Compression Analysis

Relationship between compression level and semantic preservation:

### Feature Degradation Order (fastest to slowest):

- **health**: degrades at 0.0092 per step
- **energy**: degrades at 0.0000 per step
- **position**: degrades at 0.0000 per step
- **is_alive**: degrades at 0.0000 per step
- **threatened**: degrades at 0.0000 per step
- **role**: degrades at -0.0010 per step
