#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="meaning_transform",
    version="0.1.0",
    description="Meaning-Preserving Transformation System",
    author="Agent Meaning Project",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.3.0",
        "transformers>=4.9.0",
        "sentence-transformers>=2.0.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "jupyter>=1.0.0",
        "ipywidgets>=7.6.0",
        "tensorboard>=2.6.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.5b0",
            "isort>=5.9.0",
        ],
    },
    python_requires=">=3.8",
) 