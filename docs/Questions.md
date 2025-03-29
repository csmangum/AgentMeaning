# Meaning-Preserving Transformation: Research Questions

This document tracks all research questions from our experiments and their answers when available.

## Open Questions

### Compression Mechanisms and Strategies
1. How does the entropy bottleneck affect different semantic properties differently? *(Initial Proof of Concept)*
2. How do different compression mechanisms (entropy vs. VQ) affect meaning preservation? *(Compression Level)*
3. Are there additional compression levels between 0.5-1.0 and 1.0-2.0 that might perform even better? *(Extended Compression)*
4. What is the relationship between feature importance and semantic drift at varying compression levels? *(Feature Importance)*
5. How can we leverage feature importance findings to create an adaptive compression strategy? *(Feature Importance)*

### Feature Preservation and Representation
1. How can we better preserve hierarchical or relational semantic properties? *(Compression Level)*
2. Would importance scores change with different latent space dimensions or compression levels? *(Feature Importance)*
3. How might architectural modifications that prioritize spatial features affect overall meaning preservation? *(Feature Importance)*

### Model Architecture and Scalability
1. What adaptive architecture would allow model size to vary with compression level? *(Extended Compression)*
2. What is the exact relationship between latent dimension and semantic preservation? *(Hyperparameter Tuning)*
3. Why does the 32D latent dimension outperform both smaller and larger dimensions with optimal parameters? *(Hyperparameter Tuning)*
4. Is there an optimal ratio between input dimension and latent dimension for meaning preservation? *(Hyperparameter Tuning)*

### Training Dynamics and Optimization
1. Would integrating semantic loss earlier in training provide better results than constant weighting? *(Hyperparameter Tuning)*
2. How do the optimal hyperparameters change when scaling to much larger datasets? *(Hyperparameter Tuning)*

### Context and Environment Dependencies
1. Is there a fundamental difference in how synthetic vs. real agent states respond to compression? *(Extended Compression)*
2. How do importance rankings vary across different simulation contexts or environments? *(Feature Importance)*

### Feature-Grouped Compression
1. Why do resource and performance features show high reconstruction errors despite moderate compression?
2. What explains the exceptional reconstruction quality of status features despite aggressive compression?
3. How does expansion rather than compression of spatial features affect downstream task performance?
4. What is the optimal balance between feature-specific compression rates for overall meaning preservation?
5. How would explicit semantic loss functions further improve group-specific compression strategies?
6. Is there a correlation between a feature's intrinsic complexity and its optimal compression strategy?
7. How might multi-stage compression approaches with progressive feature prioritization improve results?

## Answered Questions

1. **What is the optimal balance between reconstruction accuracy and semantic preservation?**
   - Initial finding: The optimal balance appears at the lowest tested compression level (0.5), with semantic drift of 4.49 and validation loss of 82,021.
   - Revised finding: Optimal balance appears at medium compression (1.0) with semantic drift of 1.50 and validation loss of 4,306, which is 2.67× better in preservation than 0.5 compression (drift: 4.49).
   - Specific validation loss pattern: 30,563 at 0.5, 4,306 at 1.0, 101,701 at 5.0 compression.
   - Further refined: Hyperparameter tuning confirms optimal balance at compression 1.0 with proper semantic weight (2.0), achieving semantic drift of 1.35 and validation loss of 3,859.

2. **What is the fundamental relationship between compression and meaning?**
   - Initial finding: Strong linear correlation between compression level and semantic drift (R² ≈ 0.94). Clear inverse relationship observed: higher compression consistently leads to greater semantic drift (4.49 at 0.5 to 10.85 at 5.0).
   - Later discovery: Non-linear relationship confirmed by U-shaped performance curve centered around 1.0 compression, with drift variation from 1.50 at 1.0 to 7.47 at 5.0 compression.
   - High compression disproportionately impacts quality: increase from 2.0→5.0 compression (150% increase) only adds 6% more drift.
   - Comprehensive confirmation: Hyperparameter tuning across multiple dimensions and semantic weights consistently shows U-shaped relationship with optimal performance at compression level 1.0.

3. **Can we identify a compression threshold beyond which semantic drift becomes unacceptable?**
   - While we don't have an exact threshold, semantic drift becomes severe (>10.0) at compression levels ≥2.0.
   - First experiment showed consistent drift increase: 4.49 at 0.5, 7.72 at 1.0, 10.22 at 2.0, and 10.85 at 5.0.
   - Validation loss increased by ~7× from lowest to highest compression level (82,021 → 598,654).
   - Refined with extended experiments: With proper training and hyperparameters, acceptable semantic drift (1.3-1.6) is achievable at compression level 1.0, while drift increases to 3.6-7.6 range at levels 0.5 or 2.0.

4. **How does training duration affect semantic preservation?**
   - Substantial improvement (3× better drift metrics) with 15× more training epochs (from 2 to 30 epochs).
   - Dataset quality significantly impacts results: expansion from 50 synthetic to 5,000 real agent states fundamentally changed performance characteristics.
   - Full implementation of all loss components in conjunction with extended training produced non-linear relationship between compression and preservation.

5. **Why does model size remain constant despite varying compression levels?**
   - The model size remains constant at 422.72KB across all compression levels because the architectural parameters (like layer dimensions) stay the same.
   - The compression level parameter affects the behavior of the model during training (likely through regularization techniques like KL divergence weight), not the actual architecture size.
   - Hyperparameter tuning confirmed that model size varies with latent dimension but not with compression level or semantic weight changes.

6. **Is there a non-linear relationship between compression and semantic preservation?**
   - Yes, data shows a clear non-linear relationship. The semantic drift values (1.50, 4.00, 5.25, 7.47) don't increase linearly with compression levels (0.5, 1.0, 2.0, 5.0).
   - Compression level 1.0 has the lowest semantic drift (1.50), performing significantly better than both lower (0.5) and higher compression levels.
   - Hyperparameter tuning across 48 different configurations confirmed this U-shaped relationship, with optimal performance consistently at compression level 1.0.

7. **Why does mid-level compression (1.0) outperform lower compression (0.5) with real data?**
   - This suggests an optimal "sweet spot" where enough compression forces the model to learn meaningful representations without discarding critical information.
   - At 0.5, the model might be preserving too much noise or irrelevant details, while at 1.0 it's discarding noise while keeping essential semantic information.
   - The results demonstrate a "Goldilocks zone" around compression level 1.0 that optimally balances information density and semantic preservation.

8. **How would compression levels below 0.5 affect semantic preservation?**
   - Based on the U-shaped curve observed, compression levels below 0.5 would likely result in higher semantic drift and validation loss.
   - They would likely preserve even more noise and irrelevant features, potentially causing overfitting and compromising semantic preservation.

9. **Why do discrete features show better preservation than continuous ones?**
   - Based on the drift reports, discrete features (position, health, is_alive) consistently show better preservation (0.0000 drift) across all compression levels.
   - Continuous and categorical features (energy, role) show higher drift, especially at higher compression levels.
   - This suggests that the model prioritizes preserving binary/discrete state information first, while continuous values are more susceptible to compression-induced degradation.
   - The difference is likely due to simpler encoding/decoding requirements for discrete features (binary/categorical decisions) compared to continuous values that require exact reconstruction.

10. **How do different feature types have different optimal compression thresholds?**
    - Role features show perfect preservation (1.0000 accuracy) at compression levels 0.5 and 1.0, then degrade at 2.0 (0.7500) and 5.0 (0.5000).
    - Energy features show variable preservation that doesn't strictly correlate with compression level (0.5509 at 0.5, 0.9122 at 1.0, 0.7659 at 2.0, 0.7333 at 5.0).
    - Binary features like is_alive maintain stable preservation across compression levels.
    - This indicates that categorical features (role) have a sharper threshold (between 1.0-2.0) beyond which they degrade rapidly, while continuous features (energy) degrade more gradually and unevenly.
    - The optimal threshold appears to be feature-dependent: discrete features tolerate higher compression, while continuous and relational features require lower compression levels.
    - Feature importance analysis confirms heterogeneous importance: spatial features (55.4%) are most critical, followed by resource features (25.1%), performance (10.5%), with status and role features contributing minimally (<5% each).

11. **Is there an optimal latent space dimensionality for balancing compression and meaning?**
    - Yes, hyperparameter tuning experiments identified 32 dimensions as providing the best balance of model size and semantic preservation.
    - With optimal compression (1.0) and semantic weight (2.0), the 32D model achieved the lowest semantic drift (1.35).
    - While larger dimensions (64D, 128D) sometimes performed better on average, they came with significant model size increases (16D: 402KB, 32D: 424KB, 64D: 485KB, 128D: 679KB).
    - For the input dimension of 15, a latent dimension of 32 (approximately 2x input dimension) provided the optimal balance.

12. **What architectural changes might allow higher compression while maintaining semantic integrity?**
    - Increasing the semantic loss weight from the default 0.5 to 2.0 significantly improves semantic preservation at all compression levels.
    - At compression level 1.0, changing semantic weight from 0.5 to 2.0 reduced semantic drift from 1.54 to 1.35, a 12% improvement.
    - This suggests that properly weighting the semantic component of the loss function can counteract some negative effects of compression.
    - The optimal configuration (32D, 1.0 compression, 2.0 semantic weight) achieved 5.6x better semantic preservation than the worst configuration tested.
    - Feature importance analysis suggests architecture could be optimized to give special attention to spatial and resource features, which account for over 80% of meaning preservation.

13. **How do different hyperparameters interact in their effect on meaning preservation?**
    - Strong interdependence exists between latent dimension, compression level, and semantic weight.
    - At the optimal compression level (1.0), all latent dimensions performed reasonably well (drift 1.35-1.67).
    - Away from optimal compression, larger latent dimensions became more beneficial for preserving meaning.
    - Higher semantic weights were most effective at the optimal compression level (1.0) and with the optimal latent dimension (32).
    - Model size varied only with latent dimension, not with compression level or semantic weight changes.

14. **What is the relationship between latent space organization and semantic preservation?**
    - Feature importance analysis reveals that spatial features (55.4%) and resource features (25.1%) account for over 80% of semantic importance.
    - This suggests that latent space is primarily organized around the representation of position and resource attributes.
    - The low importance of role features (4.2%) despite their complexity (one-hot encoding with 5 dimensions) indicates the latent space prioritizes semantically important features rather than simply reflecting input dimensionality.
    - Status features show low stability scores (0.0), suggesting inconsistent latent space representation across different contexts.
    - Performance features show the highest stability (0.499), indicating consistent representation in latent space despite moderate importance (10.5%).
