#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization utilities for the meaning-preserving transformation system.

This module provides high-level functions for setting up visualization directories
and creating common visualizations used throughout the system.
"""

import os
from typing import Dict, List, Optional, Union, Any
import matplotlib.pyplot as plt
import numpy as np
import torch


def setup_visualization_dirs(base_dir: str = "results") -> Dict[str, str]:
    """
    Create directories for storing visualization outputs.
    
    Args:
        base_dir: Base directory for visualizations
        
    Returns:
        Dictionary mapping directory names to paths
    """
    # Create main directories
    dirs = {
        "base": base_dir,
        "latent_space": os.path.join(base_dir, "latent_space"),
        "reconstructions": os.path.join(base_dir, "reconstructions"),
        "training": os.path.join(base_dir, "training"),
        "drift": os.path.join(base_dir, "drift"),
        "features": os.path.join(base_dir, "features"),
        "graphs": os.path.join(base_dir, "graphs"),
    }
    
    # Create all directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs 