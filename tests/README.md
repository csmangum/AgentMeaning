# Unit Tests for Meaning Transform

This directory contains unit tests for the `meaning_transform` package.

## Running Tests

You can run all tests using the provided runner script:

```bash
python -m meaning_transform.tests.run_tests
```

Or run individual test files:

```bash
python -m pytest meaning_transform/tests/test_adaptive_model.py -v
```

## Test Structure

- `test_adaptive_bottleneck.py`: Tests for the adaptive bottleneck implementations with visualization
- `test_adaptive_model.py`: Comprehensive unit tests for all adaptive model classes
- `test_combined_loss.py`: Tests for the combined loss functions
- `test_data_loading.py`: Tests for data loading and preprocessing
- `test_drift_tracking.py`: Tests for semantic drift tracking over time
- `test_evaluation.py`: Tests for model evaluation metrics
- `test_explainability.py`: Tests for knowledge graph visualization, latent space visualization, and model explainability components
- `test_loss.py`: Tests for various loss functions
- `test_metrics.py`: Tests for semantic metrics calculation
- `test_mini_train.py`: Tests for the mini-training loop
- `test_model.py`: Tests for the MeaningVAE model implementations
- `test_training.py`: Tests for the training process
- `test_visualization.py`: Tests for result visualization
- `run_tests.py`: Script to run all tests in the directory

## Adding New Tests

When adding new tests:

1. Create a new file named `test_*.py`
2. Use the pytest framework for writing tests
3. Add test classes and methods following the existing pattern
4. Ensure test methods start with `test_`
5. Use fixtures for common setup code

## Test Results

Test results and visualizations are saved in the `results` directory, which is created automatically when running tests. 