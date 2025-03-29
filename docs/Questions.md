# Meaning-Preserving Transformation: Research Questions

This document tracks all research questions from our experiments and their answers when available.

## Open Questions

### Compression Mechanisms and Strategies
1. How does the entropy bottleneck affect different semantic properties differently? *(Initial Proof of Concept)*
2. How do different compression mechanisms (entropy vs. VQ) affect meaning preservation? *(Compression Level)*
3. Are there additional compression levels between 0.5-1.0 and 1.0-2.0 that might perform even better? *(Extended Compression)*
5. How can we leverage feature importance findings to create an adaptive compression strategy? *(Feature Importance, Critical)*

### Feature Preservation and Representation
1. How can we better preserve hierarchical or relational semantic properties? *(Compression Level)*
3. How might architectural modifications that prioritize spatial features affect overall meaning preservation? *(Feature Importance)*

### Model Architecture and Scalability
1. What adaptive architecture would allow model size to vary with compression level? *(Extended Compression)*
5. Can an agent using compressed states outperform or match one with full states in downstream tasks? *(Practical Evaluation, High Priority)*

### Training Dynamics and Optimization
2. How do the optimal hyperparameters change when scaling to much larger datasets? *(Hyperparameter Tuning)*

### Context and Environment Dependencies
1. Is there a fundamental difference in how synthetic vs. real agent states respond to compression? *(Extended Compression)*
2. How do importance rankings vary across different simulation contexts or environments? *(Feature Importance)*
3. Does a compressed state lead to the same agent action as the uncompressed one? *(Behavioral Validation, Critical)*

### Feature-Grouped Compression
5. How would explicit semantic loss functions further improve group-specific compression strategies?
6. Is there a correlation between a feature's intrinsic complexity and its optimal compression strategy?
7. How might multi-stage compression approaches with progressive feature prioritization improve results?

## Answered Questions

1. **What is the optimal balance between reconstruction accuracy and semantic preservation?**
   - Initial finding: The optimal balance appears at the lowest tested compression level (0.5), with semantic drift of 4.49 and validation loss of 82,021.
   - Revised finding: Optimal balance appears at medium compression (1.0) with semantic drift of 1.50 and validation loss of 4,306, which is 2.67× better in preservation than 0.5 compression (drift: 4.49).
   - Specific validation loss pattern: 30,563 at 0.5, 4,306 at 1.0, 101,701 at 5.0 compression.
   - Further refined: Hyperparameter tuning confirms optimal balance at compression 1.0 with proper semantic weight (2.0), achieving semantic drift of 1.35 and validation loss of 3,859.
   [Initial Proof of Concept](proof-of-concept.md), [Compression Level Experiment](compresison-level-1.md), [Extended Compression Experiment](compresssion-level-2.md)

2. **What is the fundamental relationship between compression and meaning?**
   - Initial finding: Strong linear correlation between compression level and semantic drift (R² ≈ 0.94). Clear inverse relationship observed: higher compression consistently leads to greater semantic drift (4.49 at 0.5 to 10.85 at 5.0).
   - Later discovery: Non-linear relationship confirmed by U-shaped performance curve centered around 1.0 compression, with drift variation from 1.50 at 1.0 to 7.47 at 5.0 compression.
   - High compression disproportionately impacts quality: increase from 2.0→5.0 compression (150% increase) only adds 6% more drift.
   - Comprehensive confirmation: Hyperparameter tuning across multiple dimensions and semantic weights consistently shows U-shaped relationship with optimal performance at compression level 1.0.
   [Compression Level Experiment](compresison-level-1.md), [Extended Compression Experiment](compresssion-level-2.md)

3. **Can we identify a compression threshold beyond which semantic drift becomes unacceptable?**
   - While we don't have an exact threshold, semantic drift becomes severe (>10.0) at compression levels ≥2.0.
   - First experiment showed consistent drift increase: 4.49 at 0.5, 7.72 at 1.0, 10.22 at 2.0, and 10.85 at 5.0.
   - Validation loss increased by ~7× from lowest to highest compression level (82,021 → 598,654).
   - Refined with extended experiments: With proper training and hyperparameters, acceptable semantic drift (1.3-1.6) is achievable at compression level 1.0, while drift increases to 3.6-7.6 range at levels 0.5 or 2.0.
   [Compression Level Experiment](compresison-level-1.md), [Extended Compression Experiment](compresssion-level-2.md)

4. **How does training duration affect semantic preservation?**
   - Substantial improvement (3× better drift metrics) with 15× more training epochs (from 2 to 30 epochs).
   - Dataset quality significantly impacts results: expansion from 50 synthetic to 5,000 real agent states fundamentally changed performance characteristics.
   - Full implementation of all loss components in conjunction with extended training produced non-linear relationship between compression and preservation.
   [Extended Compression Experiment](compresssion-level-2.md)

5. **Why does model size remain constant despite varying compression levels?**
   - The model size remains constant at 422.72KB across all compression levels because the architectural parameters (like layer dimensions) stay the same.
   - The compression level parameter affects the behavior of the model during training (likely through regularization techniques like KL divergence weight), not the actual architecture size.
   - Hyperparameter tuning confirmed that model size varies with latent dimension but not with compression level or semantic weight changes.
   [Compression Level Experiment](compresison-level-1.md), [Extended Compression Experiment](compresssion-level-2.md), [Hyperparameter Tuning Experiment](hyper-parameter-tuning.md)

6. **Is there a non-linear relationship between compression and semantic preservation?**
   - Yes, data shows a clear non-linear relationship. The semantic drift values (1.50, 4.00, 5.25, 7.47) don't increase linearly with compression levels (0.5, 1.0, 2.0, 5.0).
   - Compression level 1.0 has the lowest semantic drift (1.50), performing significantly better than both lower (0.5) and higher compression levels.
   - Hyperparameter tuning across 48 different configurations confirmed this U-shaped relationship, with optimal performance consistently at compression level 1.0.
   [Extended Compression Experiment](compresssion-level-2.md), [Hyperparameter Tuning Experiment](hyper-parameter-tuning.md)

7. **Why does mid-level compression (1.0) outperform lower compression (0.5) with real data?**
   - This suggests an optimal "sweet spot" where enough compression forces the model to learn meaningful representations without discarding critical information.
   - At 0.5, the model might be preserving too much noise or irrelevant details, while at 1.0 it's discarding noise while keeping essential semantic information.
   - The results demonstrate a "Goldilocks zone" around compression level 1.0 that optimally balances information density and semantic preservation.
   [Extended Compression Experiment](compresssion-level-2.md)

8. **How would compression levels below 0.5 affect semantic preservation?**
   - Based on the U-shaped curve observed, compression levels below 0.5 would likely result in higher semantic drift and validation loss.
   - They would likely preserve even more noise and irrelevant features, potentially causing overfitting and compromising semantic preservation.
   [Extended Compression Experiment](compresssion-level-2.md)

9. **Why do discrete features show better preservation than continuous ones?**
   - Based on the drift reports, discrete features (position, health, is_alive) consistently show better preservation (0.0000 drift) across all compression levels.
   - Continuous and categorical features (energy, role) show higher drift, especially at higher compression levels.
   - This suggests that the model prioritizes preserving binary/discrete state information first, while continuous values are more susceptible to compression-induced degradation.
   - The difference is likely due to simpler encoding/decoding requirements for discrete features (binary/categorical decisions) compared to continuous values that require exact reconstruction.
   [Initial Proof of Concept](proof-of-concept.md), [Compression Level Experiment](compresison-level-1.md)

10. **How do different feature types have different optimal compression thresholds?**
    - Role features show perfect preservation (1.0000 accuracy) at compression levels 0.5 and 1.0, then degrade at 2.0 (0.7500) and 5.0 (0.5000).
    - Energy features show variable preservation that doesn't strictly correlate with compression level (0.5509 at 0.5, 0.9122 at 1.0, 0.7659 at 2.0, 0.7333 at 5.0).
    - Binary features like is_alive maintain stable preservation across compression levels.
    - This indicates that categorical features (role) have a sharper threshold (between 1.0-2.0) beyond which they degrade rapidly, while continuous features (energy) degrade more gradually and unevenly.
    - The optimal threshold appears to be feature-dependent: discrete features tolerate higher compression, while continuous and relational features require lower compression levels.
    - Feature importance analysis confirms heterogeneous importance: spatial features (55.4%) are most critical, followed by resource features (25.1%), performance (10.5%), with status and role features contributing minimally (<5% each).
    [Feature Importance Analysis](feature-importance.md), [Feature-Specific Compression Experiment](feature-specific-compression.md)

11. **Is there an optimal latent space dimensionality for balancing compression and meaning?**
    - Yes, hyperparameter tuning experiments identified 32 dimensions as providing the best balance of model size and semantic preservation.
    - With optimal compression (1.0) and semantic weight (2.0), the 32D model achieved the lowest semantic drift (1.35).
    - While larger dimensions (64D, 128D) sometimes performed better on average, they came with significant model size increases (16D: 402KB, 32D: 424KB, 64D: 485KB, 128D: 679KB).
    - For the input dimension of 15, a latent dimension of 32 (approximately 2x input dimension) provided the optimal balance.
    [Hyperparameter Tuning Experiment](hyper-parameter-tuning.md)

12. **What architectural changes might allow higher compression while maintaining semantic integrity?**
    - Increasing the semantic loss weight from the default 0.5 to 2.0 significantly improves semantic preservation at all compression levels.
    - At compression level 1.0, changing semantic weight from 0.5 to 2.0 reduced semantic drift from 1.54 to 1.35, a 12% improvement.
    - This suggests that properly weighting the semantic component of the loss function can counteract some negative effects of compression.
    - The optimal configuration (32D, 1.0 compression, 2.0 semantic weight) achieved 5.6x better semantic preservation than the worst configuration tested.
    - Feature importance analysis suggests architecture could be optimized to give special attention to spatial and resource features, which account for over 80% of meaning preservation.
    [Hyperparameter Tuning Experiment](hyper-parameter-tuning.md), [Feature Importance Analysis](feature-importance.md)

13. **How do different hyperparameters interact in their effect on meaning preservation?**
    - Strong interdependence exists between latent dimension, compression level, and semantic weight.
    - At the optimal compression level (1.0), all latent dimensions performed reasonably well (drift 1.35-1.67).
    - Away from optimal compression, larger latent dimensions became more beneficial for preserving meaning.
    - Higher semantic weights were most effective at the optimal compression level (1.0) and with the optimal latent dimension (32).
    - Model size varied only with latent dimension, not with compression level or semantic weight changes.
    [Hyperparameter Tuning Experiment](hyper-parameter-tuning.md)

14. **What is the relationship between latent space organization and semantic preservation?**
    - Feature importance analysis reveals that spatial features (55.4%) and resource features (25.1%) account for over 80% of semantic importance.
    - This suggests that latent space is primarily organized around the representation of position and resource attributes.
    - The low importance of role features (4.2%) despite their complexity (one-hot encoding with 5 dimensions) indicates the latent space prioritizes semantically important features rather than simply reflecting input dimensionality.
    - Status features show low stability scores (0.0), suggesting inconsistent latent space representation across different contexts.
    - Performance features show the highest stability (0.499), indicating consistent representation in latent space despite moderate importance (10.5%).
    [Feature Importance Analysis](feature-importance.md)

15. **What is the relationship between feature importance and semantic drift at varying compression levels?**
    - Feature importance directly correlates with compression sensitivity. High-importance features (spatial at 55.4%, resource at 25.1%) deteriorate more at higher compression levels, while low-importance features (status, role at ~4% each) maintain accuracy even at 2.0x compression. [Feature Importance Analysis](feature-importance.md)
    - Binary features show remarkable resilience regardless of compression level, with status and role features maintaining perfect accuracy (1.0) despite 2.0x compression. [Feature-Specific Compression Experiment](feature-specific-compression.md)
    - The finding that spatial features (highest importance at 55.4%) benefit from expansion rather than compression demonstrates that semantic drift responds differently across feature types based on their importance ranking. [Feature-Grouped VAE Experiment](feature-grouped-vae.md)

16. **Can you show actual memory use or bandwidth savings from compression without semantic loss?**
    - Yes. The feature-specific compression model achieved equal overall semantic similarity (0.7647) with 2.4% fewer parameters (99,345 vs 101,743) than the baseline model. [Feature-Specific Compression Experiment](feature-specific-compression.md)
    - Different feature types allow different levels of compression without semantic loss: spatial features require expansion (0.5x), while status features tolerate aggressive compression (up to 5.67x) with minimal impact. [Feature-Grouped VAE Experiment](feature-grouped-vae.md)
    - This translates to practical bandwidth/memory optimization where critical features receive more bits while less important features can be heavily compressed, maintaining semantic fidelity with reduced resource usage. [Feature-Specific Compression Experiment](feature-specific-compression.md)

17. **Would importance scores change with different latent space dimensions or compression levels?**
    - Feature importance hierarchy remains remarkably stable across model configurations: spatial features consistently dominate at 55.4%, followed by resource features at 25.1%, performance at 10.5%, with status and role features at <5% each. [Feature Importance Analysis](feature-importance.md)
    - This stability suggests that importance rankings reflect intrinsic semantic properties rather than model artifacts, providing a reliable basis for compression strategies regardless of architectural choices. [Feature Importance Analysis](feature-importance.md)

18. **What is the exact relationship between latent dimension and semantic preservation?**
    - 32D latent space (with 1.0 compression and 2.0 semantic weight) provides optimal semantic preservation with the lowest drift (1.35). [Hyperparameter Tuning Experiment](hyper-parameter-tuning.md)
    - Model size scales predictably with latent dimension: 16D (402KB), 32D (424KB), 64D (485KB), 128D (679KB). [Hyperparameter Tuning Experiment](hyper-parameter-tuning.md)
    - There's a U-shaped relationship where very small dimensions (16D) underperform due to insufficient capacity, while very large dimensions (128D) risk overfitting, with 32D hitting the sweet spot for the 15-dimensional input data. [Hyperparameter Tuning Experiment](hyper-parameter-tuning.md)

19. **Why does the 32D latent dimension outperform both smaller and larger dimensions with optimal parameters?**
    - The 32D latent space provides the optimal balance between capacity and constraint - approximately 2x the input dimension of 15, allowing sufficient representation without overfitting. [Hyperparameter Tuning Experiment](hyper-parameter-tuning.md)
    - Smaller dimensions (16D) lack capacity to properly encode all semantic relationships, while larger dimensions (64D, 128D) introduce redundancy and potential noise without additional semantic benefit. [Hyperparameter Tuning Experiment](hyper-parameter-tuning.md)
    - The 32D configuration achieved the lowest semantic drift (1.35) with optimal compression (1.0) and semantic weight (2.0), providing the best meaning preservation per parameter ratio. [Hyperparameter Tuning Experiment](hyper-parameter-tuning.md)

20. **Is there an optimal ratio between input dimension and latent dimension for meaning preservation?**
    - For the 15-dimensional input, the optimal ratio appears to be approximately 2:1 (32D latent), suggesting this may serve as a general guideline for meaning preservation. [Hyperparameter Tuning Experiment](hyper-parameter-tuning.md)
    - This ratio provides sufficient capacity to encode semantic relationships while maintaining enough constraint to focus on essential meaning. [Hyperparameter Tuning Experiment](hyper-parameter-tuning.md)

21. **Would integrating semantic loss earlier in training provide better results than constant weighting?**
    - Experiments show that higher semantic weights (2.0 vs default 0.5) significantly improve meaning retention at all compression levels, suggesting earlier/stronger semantic loss integration would be beneficial. [Hyperparameter Tuning Experiment](hyper-parameter-tuning.md)
    - At compression level 1.0, increasing semantic weight from 0.5 to 2.0 reduced semantic drift from 1.54 to 1.35, a 12% improvement. [Hyperparameter Tuning Experiment](hyper-parameter-tuning.md)
    - This indicates that prioritizing semantic preservation earlier in training would likely yield better results, particularly for more aggressively compressed models. [Hyperparameter Tuning Experiment](hyper-parameter-tuning.md)

22. **Why do resource and performance features show high reconstruction errors despite moderate compression?**
    - Their continuous nature makes them inherently harder to reconstruct perfectly compared to binary/discrete features, requiring more representational capacity. [Feature-Grouped VAE Experiment](feature-grouped-vae.md)
    - Resource features (25.1% importance) showed modest improvement with reduced compression rates (0.5x), but still exhibited higher errors than binary features, suggesting continuous values are fundamentally more challenging to preserve. [Feature-Specific Compression Experiment](feature-specific-compression.md)

23. **What explains the exceptional reconstruction quality of status features despite aggressive compression?**
    - Their binary/categorical nature makes them easier to encode efficiently, requiring only decision boundaries rather than precise value reconstruction. [Feature-Grouped VAE Experiment](feature-grouped-vae.md)
    - Status features maintained perfect accuracy (1.0) despite 2.0x compression, confirming that binary features can be compressed aggressively with minimal impact. [Feature-Specific Compression Experiment](feature-specific-compression.md)
    - The low importance of status features (4.7%) combined with their simple binary structure allows for efficient encoding with minimal latent space allocation. [Feature Importance Analysis](feature-importance.md)

24. **How does expansion rather than compression of spatial features affect downstream task performance?**
    - Expansion produces significant improvements - spatial RMSE improved by 11.3% (from 3.45 to 3.06) with 0.5x compression (expansion). [Feature-Specific Compression Experiment](feature-specific-compression.md)
    - This confirms that high-importance features benefit from expansion rather than compression, with direct improvements to position accuracy that would affect downstream navigation and interaction tasks. [Feature-Specific Compression Experiment](feature-specific-compression.md)

25. **What is the optimal balance between feature-specific compression rates for overall meaning preservation?**
    - The optimal balance follows an inversely proportional relationship to feature importance: Spatial (0.5x), Resource (0.5x), Performance (0.8x), Status/Role (2.0x+). [Feature-Specific Compression Experiment](feature-specific-compression.md)
    - This approach maintained overall semantic similarity (0.7647) while reducing model parameters by 2.4%. [Feature-Specific Compression Experiment](feature-specific-compression.md)
    - Status features tolerated even more aggressive compression (5.67x) while maintaining good reconstruction (MSE 0.21). [Feature-Grouped VAE Experiment](feature-grouped-vae.md)
