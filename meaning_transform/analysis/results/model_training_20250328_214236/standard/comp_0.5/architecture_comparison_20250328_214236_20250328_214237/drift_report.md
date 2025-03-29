# Semantic Drift Analysis Report

## Summary Statistics

| Metric | Mean | Min | Max | Std Dev |
|--------|------|-----|-----|--------|
| overall | 0.3955 | 0.3930 | 0.4016 | 0.0023 |
| position | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| health | 0.8863 | 0.8556 | 0.9634 | 0.0293 |
| energy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| is_alive | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| role | 0.9836 | 0.9666 | 0.9900 | 0.0064 |
| has_target_accuracy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| is_alive_accuracy | 0.0136 | 0.0000 | 0.0500 | 0.0234 |
| role_accuracy | 0.5864 | 0.1500 | 0.7500 | 0.1645 |

## Compression Analysis

Relationship between compression level and semantic preservation:

### Feature Degradation Order (fastest to slowest):

- **health**: degrades at 0.0077 per step
- **energy**: degrades at 0.0000 per step
- **position**: degrades at 0.0000 per step
- **threatened**: degrades at 0.0000 per step
- **is_alive**: degrades at -0.0000 per step
- **role**: degrades at -0.0016 per step
