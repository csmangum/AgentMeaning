# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.11]

### Added

#### Standardized Metrics Framework
- Implemented `StandardizedMetrics` class extending `SemanticMetrics` with consistent evaluation methods
- Defined clear operational definitions for preservation, fidelity, and drift metrics
- Created feature importance-weighted metrics based on previous analysis (spatial: 55.4%, resources: 25.1%, etc.)
- Added qualitative performance categories with thresholds (excellent, good, acceptable, poor, critical)
- Implemented standardized feature groups for consistent evaluation across experiments
- Enhanced drift tracking with baseline comparison capabilities
- Updated `Trainer` class to use standardized metrics throughout
- Modified compression experiments to leverage the standardized metrics system
- Created comprehensive documentation with mathematical formulations
- Removed legacy compatibility layers in favor of standardized evaluation

## [0.1.10]

### Added

#### Feature Importance Analysis System
- Implemented `FeatureImportanceAnalyzer` class for evaluating feature contributions to agent state meaning
- Added support for analyzing importance based on behavior vectors and outcomes
- Created visualization tools for feature importance with detailed group-level breakdown
- Integrated importance analysis with standardized metrics for weighted evaluation
- Added drift analysis capabilities with importance-weighted metrics
- Implemented feature group weighting calculation based on importance scores
- Created examples demonstrating feature importance analysis workflow
- Added database integration for loading and analyzing real agent state data
- Enhanced visualization outputs with detailed importance breakdowns
- Implemented customizable feature group categorization
- Created comprehensive documentation for the feature importance API

## [0.1.9]

### Added

#### Graph-Based Agent State Representation Integration
- Enhanced `MeaningVAE` to support both vector-based and graph-based inputs
- Added graph conversion methods to `AgentState` class for transforming states to knowledge graphs
- Updated `AgentStateDataset` with graph batching functionality and multi-agent graph creation
- Modified training pipeline in `Trainer` class to handle graph data processing
- Implemented specialized semantic drift tracking for graph representations
- Added graph-specific visualization tools for latent space exploration
- Integrated PyTorch Geometric support for efficient graph neural network operations
- Created seamless transition between vector and graph representations
- Added configuration options for controlling graph model parameters
- Enhanced checkpointing and visualization for graph-based models

## [0.1.8]

### Added

#### Knowledge Graph & Graph Neural Network Implementation
- Implemented `AgentStateToGraph` converter for transforming agent states into knowledge graphs
- Created `GraphEncoder` and `GraphDecoder` for processing graph-structured agent data
- Added `VGAE` (Variational Graph Autoencoder) for graph-based compression
- Implemented `GraphCompressionModel` combining VGAE with semantic preservation techniques
- Created specialized loss functions with `GraphVAELoss` and `GraphSemanticLoss`
- Added PyTorch Geometric integration for efficient graph neural network operations
- Implemented NetworkX integration for knowledge graph construction and manipulation
- Created `KnowledgeGraphDataset` for handling collections of agent state graphs

#### Interactive Visualization & Explainability
- Implemented `AgentStateDashboard` for interactive exploration of agent states
- Created `LatentSpaceExplorer` for visualizing and traversing latent representations
- Added support for interactive graph visualization with Dash and Cytoscape
- Implemented `GraphVisualizer` and `LatentSpaceVisualizer` for advanced data exploration
- Created `ModelExplainer` for interpreting model decisions and important features
- Added support for visualizing feature importance and node attributions in agent graphs
- Implemented dashboard for real-time monitoring of agent state compression

#### State Detail Experimentation
- Added infrastructure for systematic experimentation with agent state compositions
- Implemented tooling for analyzing impact of state detail granularity on meaning preservation
- Created framework for testing different weighting strategies across property combinations
- Added support for analyzing categorical vs. continuous property distributions
- Implemented utilities for measuring knowledge graph density effects on semantic preservation
- Created examples showing usage of new graph-based agent state representation

#### Package Configuration
- Expanded package imports in `__init__.py` with new components
- Updated project documentation with enhanced module descriptions
- Added new dependencies including networkx, torch-geometric, rdflib, dash, and XAI libraries
- Created example scripts demonstrating knowledge graph and dashboard functionality

## [0.1.7]

### Added

#### Adaptive Architecture Implementation
- Implemented `AdaptiveEntropyBottleneck` class that physically changes parameter count based on compression level
- Created `FeatureGroupedVAE` with separate bottlenecks for different feature types
- Added feature importance-based compression allocation system
- Implemented dynamic parameter scaling that reduces model size at higher compression levels
- Added group-specific latent dimension calculation based on feature count and importance
- Created specialized encoder/decoder pathways for different feature groups
- Implemented test suite in `test_adaptive_architecture.py` to verify parameter count reduction
- Added visualization tools for parameter count vs. compression level analysis
- Created example script demonstrating the adaptive architecture benefits
- Fixed dimension compatibility issues between grouped features
- Added detailed documentation on adaptive architecture usage

## [0.1.6]

### Added

#### Comprehensive Visualization Tools Implementation
- Created modular visualization architecture with four core classes:
  - `LatentSpaceVisualizer` for inspecting latent space structure
  - `LossVisualizer` for tracking training dynamics
  - `StateComparisonVisualizer` for comparing original vs. reconstructed states
  - `DriftVisualizer` for semantic drift monitoring
- Implemented high-level API in `visualize.py` for simplified access to all visualization tools
- Added 7 specialized visualization types:
  - t-SNE and PCA for latent space exploration
  - Latent interpolation visualization to examine continuous semantic transitions
  - Loss curves tracking with history saving/loading
  - Compression vs. reconstruction trade-off analysis
  - Feature-by-feature comparison between original and reconstructed states
  - State trajectory visualization for tracking agent movement
  - Semantic drift tracking to monitor meaning preservation over compression levels
- Created detailed interpretation guide (`VISUALIZATION_GUIDE.md`) for analyzing visualization outputs
- Implemented comprehensive testing framework in `test_visualization.py`
- Added example script (`visualization_examples.py`) with synthetic data for demonstration
- Integrated with existing metrics system for semantic interpretation of results
- Fixed visualization compatibility issues for different matplotlib versions
- Added multi-dimensional feature handling for feature comparison plots

## [0.1.5]

### Added

#### Metrics and Evaluation Implementation
- Created comprehensive `SemanticMetrics` class for evaluating semantic preservation
- Implemented `DriftTracker` for monitoring semantic drift across compression levels
- Added `CompressionThresholdFinder` for identifying optimal compression thresholds
- Implemented latent space metrics including cluster quality measurements
- Created t-SNE visualization for latent space exploration
- Added detailed feature-specific metrics:
  - Binary feature accuracy, precision, recall, and F1 score
  - Role classification accuracy with confusion matrices
  - Numeric feature error metrics (MAE, RMSE, MAPE)
- Generated comprehensive drift analysis reports in markdown format
- Added visualization tools for drift tracking and latent space mapping
- Integrated semantic metrics with trainer for automatic drift tracking
- Created example scripts for evaluating trained models
- Added test scripts for validating all metrics components
- Fixed t-SNE visualization for small sample sizes

## [0.1.4]

### Added

#### Training Infrastructure Implementation
- Created comprehensive `Trainer` class with full training pipeline
- Implemented epoch-based training loop with metrics tracking
- Added model checkpointing and resumption capabilities
- Implemented semantic drift tracking to monitor meaning preservation
- Created visualization tools for training curves and drift analysis
- Added early stopping based on validation metrics
- Integrated optimizers (Adam, SGD) and learning rate schedulers
- Implemented detailed metrics logging and JSON configuration serialization
- Created example training script with command-line interface
- Developed unit tests for trainer initialization, training, and checkpointing
- Fixed dimension compatibility issues between model and data
- Added visualization of feature-specific semantic loss breakdown
- Created detailed documentation on training infrastructure usage

## [0.1.3]

### Added

#### Loss Functions Implementation
- Enhanced `SemanticLoss` class with comprehensive semantic feature extraction
- Implemented specialized loss functions for different feature types:
  - Position and continuous values using MSE
  - Binary features (has_target, is_alive) using BCE
  - Role and derived features with appropriate metrics
- Added `detailed_breakdown` method for analyzing semantic preservation by feature type
- Enhanced `CombinedLoss` with weighted combination of reconstruction, KL-divergence, and semantic losses
- Implemented "threatened" state detection as a derived semantic feature
- Created test scripts for validating loss components independently and together
- Fixed tensor detachment issues for proper loss tracking
- Added semantic breakdown logging for monitoring meaning preservation during training

## [0.1.2]

### Added

#### Core Model Architecture Implementation
- Completed implementation of VAE architecture in `model.py`
- Implemented `EntropyBottleneck` compression mechanism for continuous latent space
- Added `VectorQuantizer` for discrete latent representation with codebook learning
- Enhanced `MeaningVAE` with full pipeline functionality
- Implemented `save` and `load` methods for model serialization
- Added `get_compression_rate` utility for calculating compression efficiency
- Created `test_model.py` for validating the complete transformation pipeline
- Added documentation in `MODEL_README.md` explaining architecture and usage
- Fixed batch normalization issues for single-sample inference

## [0.1.1]

### Added

#### Data Handling Implementation
- Enhanced `AgentState` class with comprehensive state representation
- Implemented binary serialization and deserialization with tuple handling
- Added database integration to load agent states from simulation.db
- Created tensor conversion for neural network input
- Added batch processing functionality in `AgentStateDataset`
- Implemented synthetic data generation for testing
- Developed `test_data_loading.py` for validating data handling features
- Updated project README with data handling usage examples

## [0.1.0]

### Added

#### Project Structure
- Created complete directory structure based on the project plan
- Set up all necessary subdirectories for source, utilities, results, and notebooks

#### Core Modules
- Created `data.py` with `AgentState` class and dataset handling infrastructure
- Implemented `model.py` with VAE architecture including:
  - Encoder and Decoder classes
  - Compression mechanisms (EntropyBottleneck and VectorQuantizer)
  - Complete MeaningVAE pipeline
- Added `loss.py` with multi-layered loss functions:
  - ReconstructionLoss
  - KLDivergenceLoss
  - SemanticLoss
  - CombinedLoss for weighted combination
- Developed `config.py` with comprehensive configuration options using dataclasses

#### Project Configuration
- Created `requirements.txt` with all necessary dependencies
- Added `setup.py` for package installation
- Implemented proper `__init__.py` files for all package directories

#### Documentation
- Added comprehensive README.md with project overview, installation and usage instructions
- Created taxonomy schema in YAML format for classifying transformation types
- Added Jupyter notebook template for structural vs. semantic experiments

#### Deployment
- Set up results directories for storing:
  - Loss curves
  - Drift logs
  - Reconstruction examples
  - Latent space visualizations

### Notes
- All implementations include proper type hints, docstrings, and placeholder methods for future development
- Core functionality is in place but actual implementation of methods marked with TODO comments will be addressed in future versions 