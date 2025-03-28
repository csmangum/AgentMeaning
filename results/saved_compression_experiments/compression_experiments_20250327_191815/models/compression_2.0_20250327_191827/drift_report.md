# Semantic Drift Analysis Report

## Summary Statistics

| Metric | Mean | Min | Max | Std Dev |
|--------|------|-----|-----|--------|
| overall | 0.3992 | 0.3824 | 0.4161 | 0.0239 |
| position | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| health | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| energy | 0.7659 | 0.7493 | 0.7826 | 0.0235 |
| is_alive | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| role | 0.8751 | 0.8122 | 0.9380 | 0.0890 |
| has_target_accuracy | 0.3000 | 0.3000 | 0.3000 | 0.0000 |
| is_alive_accuracy | 0.6500 | 0.6000 | 0.7000 | 0.0707 |
| role_accuracy | 0.7500 | 0.6000 | 0.9000 | 0.2121 |

## Compression Analysis

Relationship between compression level and semantic preservation:

### Feature Degradation Order (fastest to slowest):

- **is_alive**: degrades at 0.0000 per step
- **health**: degrades at 0.0000 per step
- **position**: degrades at 0.0000 per step
- **threatened**: degrades at 0.0000 per step
- **energy**: degrades at -0.0166 per step
- **role**: degrades at -0.0629 per step
