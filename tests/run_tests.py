#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test runner script for the meaning_transform package.

This script runs all unit tests in the tests directory.
Execute with: python -m meaning_transform.tests.run_tests
"""

import os
import sys
import pytest
from pathlib import Path

# Add the project root to the path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)


def run_tests():
    """Run all unit tests in the tests directory."""
    # Get the directory of this file
    test_dir = Path(__file__).resolve().parent
    
    # Create a results directory if it doesn't exist
    results_dir = test_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all test files
    test_files = [str(f) for f in test_dir.glob("test_*.py")]
    print(f"Found {len(test_files)} test files")
    
    # Run tests with pytest
    return pytest.main(["-v"] + test_files)


if __name__ == "__main__":
    sys.exit(run_tests()) 