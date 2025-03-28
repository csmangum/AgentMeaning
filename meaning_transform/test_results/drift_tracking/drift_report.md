# Semantic Drift Analysis Report

## Summary Statistics

| Metric | Mean | Min | Max | Std Dev |
|--------|------|-----|-----|--------|
| overall | 0.6932 | 0.4594 | 1.0000 | 0.2486 |
| position | 0.9206 | 0.8492 | 1.0000 | 0.0540 |
| health | 0.9085 | 0.8329 | 1.0000 | 0.0645 |
| energy | 0.9203 | 0.8499 | 1.0000 | 0.0616 |
| is_alive | 0.3755 | 0.0000 | 1.0000 | 0.5171 |
| role | 0.9644 | 0.9094 | 1.0000 | 0.0369 |
| has_target_accuracy | 0.8633 | 0.6875 | 1.0000 | 0.1169 |
| is_alive_accuracy | 0.9219 | 0.7812 | 1.0000 | 0.0852 |
| role_accuracy | 0.7930 | 0.4688 | 1.0000 | 0.2126 |

## Compression Analysis

Relationship between compression level and semantic preservation:

- **Minimum acceptable compression level**: 3.00 bits per dimension
  (maintains at least 90% semantic preservation)

### Feature Degradation Order (fastest to slowest):

- **is_alive**: degrades at 0.1250 per step
- **threatened**: degrades at 0.1250 per step
- **health**: degrades at 0.0209 per step
- **position**: degrades at 0.0188 per step
- **energy**: degrades at 0.0177 per step
- **role**: degrades at 0.0113 per step
