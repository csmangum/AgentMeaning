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