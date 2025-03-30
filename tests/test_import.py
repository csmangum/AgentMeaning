#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple test to verify that imports are working correctly.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

def test_imports():
    """Test that imports are working correctly."""
    print("Testing imports...")
    
    # Test essential imports
    try:
        from meaning_transform.src.config import Config
        print("✓ Config imported successfully")
        
        from meaning_transform.src.model import MeaningVAE
        print("✓ MeaningVAE imported successfully")
        
        from meaning_transform.src.loss import CombinedLoss
        print("✓ CombinedLoss imported successfully")
        
        from meaning_transform.src.metrics import SemanticMetrics
        print("✓ SemanticMetrics imported successfully")
        
        from meaning_transform.src.train import Trainer
        print("✓ Trainer imported successfully")
        
        from meaning_transform.src.data import AgentStateDataset
        print("✓ AgentStateDataset imported successfully")
        
        print("\nAll imports successful!")
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

if __name__ == "__main__":
    test_imports() 