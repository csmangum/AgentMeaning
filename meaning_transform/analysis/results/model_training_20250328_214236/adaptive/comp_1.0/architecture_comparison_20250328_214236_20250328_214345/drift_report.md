# Semantic Drift Analysis Report

## Summary Statistics

| Metric | Mean | Min | Max | Std Dev |
|--------|------|-----|-----|--------|
| overall | 0.6137 | 0.3923 | 0.6385 | 0.0614 |
| position | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| health | 0.9068 | 0.8720 | 0.9864 | 0.0304 |
| energy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| is_alive | 0.9338 | 0.0067 | 1.0000 | 0.2565 |
| role | 0.9616 | 0.9608 | 0.9627 | 0.0010 |
| has_target_accuracy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| is_alive_accuracy | 0.9967 | 0.9500 | 1.0000 | 0.0129 |
| role_accuracy | 0.0200 | 0.0000 | 0.0500 | 0.0254 |

## Compression Analysis

Relationship between compression level and semantic preservation:

### Feature Degradation Order (fastest to slowest):

- **health**: degrades at 0.0060 per step
- **position**: degrades at 0.0000 per step
- **is_alive**: degrades at 0.0000 per step
- **threatened**: degrades at 0.0000 per step
- **energy**: degrades at -0.0000 per step
- **role**: degrades at -0.0001 per step
