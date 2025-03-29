# Semantic Drift Analysis Report

## Summary Statistics

| Metric | Mean | Min | Max | Std Dev |
|--------|------|-----|-----|--------|
| overall | 0.6069 | 0.3923 | 0.6636 | 0.0873 |
| position | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| health | 0.9600 | 0.9510 | 0.9884 | 0.0095 |
| energy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| is_alive | 0.8676 | 0.0067 | 1.0000 | 0.3495 |
| role | 0.9242 | 0.8728 | 0.9608 | 0.0250 |
| has_target_accuracy | 0.0667 | 0.0000 | 0.2000 | 0.0673 |
| is_alive_accuracy | 0.9933 | 0.9500 | 1.0000 | 0.0176 |
| role_accuracy | 0.0033 | 0.0000 | 0.0500 | 0.0129 |

## Compression Analysis

Relationship between compression level and semantic preservation:

### Feature Degradation Order (fastest to slowest):

- **role**: degrades at 0.0033 per step
- **health**: degrades at 0.0023 per step
- **position**: degrades at 0.0000 per step
- **is_alive**: degrades at 0.0000 per step
- **threatened**: degrades at 0.0000 per step
- **energy**: degrades at -0.0000 per step
