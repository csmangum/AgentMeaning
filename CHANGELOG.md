# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.27]

### Fixed

#### Critical Loss Function Fixes
- Fixed zero reconstruction loss issue that emerged after models module refactoring
- Added comprehensive validation checks in CombinedLoss to prevent zero reconstruction loss:
  - Added strict input validation to ensure model_output contains required keys
  - Added shape validation to ensure reconstruction and original tensors match
  - Implemented error detection and detailed debugging for near-zero reconstruction loss
  - Added detailed sample value comparison for easier debugging
- Enhanced training loop with robust error checks:
  - Added safety checks to detect and immediately fail on zero reconstruction loss
  - Updated loss parameter access to correctly identify reconstruction_loss vs recon_loss key
  - Added warnings for suspiciously low KL divergence loss
  - Improved error messages with detailed diagnostics
- Added comprehensive logging to detect and report loss calculation issues

## [0.1.26]

### Added

#### Model Improvements
- Added `adapt_config` parameter to BaseModelIO.load() method to control adaptation of model configuration
- Made AdaptiveEntropyBottleneck truly idempotent by detecting already compressed values
- Added comprehensive NaN/infinity validation to all model components:
  - Added strict validation in Encoder forward method that raises ValueError for NaN/infinity
  - Added strict validation in Decoder forward method that raises ValueError for NaN/infinity
  - Added validation in FeatureGroupedVAE's forward, encode, and decode methods
- Added use_batch_norm to adaptable parameters list in BaseModelIO
- Enhanced FeatureGroupedVAE with minimum dimension control:
  - Added `min_group_dim` parameter to control minimum allowed dimensions per feature group
  - Implemented comprehensive checks to prevent groups from falling below minimum threshold
  - Added dynamic latent dimension adjustment when all groups are at minimum
  - Improved dimension reduction logic with better group selection for reductions
  - Added detailed logging for dimension allocation decisions

### Fixed

#### Compression Behavior Improvements
- Improved AdaptiveEntropyBottleneck to detect quantized values using fractional part analysis
- Fixed double compression behavior in AdaptiveEntropyBottleneck by adding early return for already compressed inputs
- Added epsilon threshold buffer to reliably detect quantized values

## [0.1.25]

### Fixed

#### Comprehensive Models Module Improvements
- Fixed Vector Quantizer perplexity calculation with improved numerical stability approach
- Fixed inconsistent latent space handling in MeaningVAE to ensure reconstruction always uses compressed representation
- Removed redundant compression in AdaptiveEntropyBottleneck to prevent excessive information loss
- Improved seed handling in FeatureGroupedVAE for consistent reproducibility
- Added proper input validation to FeatureGroupedVAE's forward method
- Enhanced BaseModelIO.load() method with intelligent configuration adaptation:
  - Added critical vs adaptable parameter distinction for safer model loading
  - Implemented clear error raising for incompatible dimensions
  - Added model type compatibility checking with VAE family recognition
  - Created detailed adaptation logging with clear warnings
  - Added override points for customizing compatibility logic in subclasses

#### Input Validation Enhancements
- Added comprehensive input validation in Encoder forward method
- Added comprehensive input validation in Decoder forward method 
- Fixed and enhanced validation in BaseModelIO.load() for more reliable model loading
- Added additional validation in AdaptiveEntropyBottleneck initialization
- Enhanced graph-specific validation in MeaningVAE for proper graph attribute checking
- Implemented standardized input validation across all model components
- Added detailed error messages to improve debugging experience

## [0.1.24]

### Fixed

#### MeaningVAE Improvements
- Fixed inconsistent latent space handling in graph data path to properly decode from compressed representation
- Added missing torch_geometric import to resolve type annotation error
- Ensured consistent behavior between tensor and graph paths when using compression
- Improved consistency of reconstruction outputs across all model configurations

## [0.1.23]

### Added

#### Enhanced Test Coverage
- Added comprehensive test suite for AdaptiveMeaningVAE including drift adaptation mechanism
- Implemented dedicated tests for FeatureGroupedVAE with semantic and type-based grouping
- Added standalone tests for Encoder and Decoder components
- Created detailed VectorQuantizer tests including gradient flow verification
- Implemented utility function tests for KL divergence, reparameterization, and feature grouping
- Enhanced test output with detailed metrics and component-specific validation
- Standardized test structure across all model components
- Added train/eval mode consistency verification for all models

## [0.1.22]

### Added

#### Modular Utilities Framework
- Implemented `CompressionBase` class for standardized behavior across compression methods
- Created `BaseModelIO` class to standardize model serialization and loading
- Added `set_temp_seed` context manager for consistent random seed handling

### Fixed

#### Comprehensive Architecture Improvements
- Standardized behavior between training and inference modes across all models
- Implemented consistent input validation with proper error messages
- Fixed seed handling to preserve random state consistently across all operations
- Enhanced numerical stability in compression loss calculations
- Made the reparameterization method consistent across all model variants

#### EntropyBottleneck Fixes
- Fixed inconsistencies in quantization behavior during inference
- Improved numerical stability in all bottleneck implementations
- Enhanced deterministic rounding for better consistency

#### AdaptiveEntropyBottleneck Refinements
- Fixed inconsistent quantization between training and inference
- Standardized behavior with other bottleneck implementations
- Improved noise handling during training for better consistency

#### MeaningVAE and AdaptiveMeaningVAE Enhancements
- Standardized encoding/decoding behavior across all model variants
- Implemented consistent graph data handling
- Added proper parameter validation throughout

## [0.1.21]

### Fixed

#### VectorQuantizer Improvements
- Replaced the inefficient distance calculation with `torch.cdist` for better performance and memory efficiency
- Added proper handling for unused codebook entries in the perplexity calculation to prevent potential division by zero
- Improved numerical stability of the perplexity calculation

## [0.1.20]

### Fixed

#### AdaptiveMeaningVAE Fixes
- Fixed inconsistency in encode method to use different behavior between training and inference modes
- Enhanced forward method to properly compress mu during inference rather than sampled z for consistency
- Improved load method with compatibility checks for model architecture and configuration
- Added detailed warnings for mismatched configuration parameters during model loading
- Ensured consistent behavior between encode, forward, and reparameterize methods

## [0.1.19]

### Fixed

#### FeatureGroupedVAE Improvements
- Fixed feature group unpacking in `get_feature_group_analysis` method to properly handle dictionary access
- Improved latent dimension allocation with a two-pass approach to ensure exact match with specified latent_dim
- Enhanced overall compression rate calculation to use a weighted average based on feature counts for more meaningful metrics
- Optimized dimension distribution logic to prioritize groups with lower compression values
- Added more robust feature group handling throughout the model

## [0.1.18]

### Fixed

#### Comprehensive Model Fixes
- Added proper support for the "adaptive_entropy" compression type in MeaningVAE
- Fixed inconsistency in encode method between training and inference modes
- Improved numerical stability across all bottleneck implementations
- Standardized compression behavior between EntropyBottleneck and AdaptiveEntropyBottleneck
- Enhanced deterministic rounding in EntropyBottleneck for consistent quantization
- Fixed scaling issues in EntropyBottleneck by using torch.log instead of numpy.log for device compatibility
- Optimized AdaptiveEntropyBottleneck seed handling to prevent global random state modification
- Made KL loss calculation consistent between graph and non-graph paths

## [0.1.17]

### Fixed

#### AdaptiveEntropyBottleneck Refinements
- Improved seed handling in AdaptiveEntropyBottleneck to use a generator-based approach for better reproducibility
- Fixed deterministic quantization during inference to ensure proper rounding of latent representations
- Enhanced consistency between training and inference modes for more reliable state reconstruction
- Improved device compatibility with conditional handling for MPS devices
- Optimized training-inference pipeline for more consistent reconstruction quality

## [0.1.16]

### Fixed

#### Entropy Bottleneck Improvements
- Fixed scaling inconsistency between training and inference modes in EntropyBottleneck
- Improved numerical stability in compression loss calculation for both EntropyBottleneck and AdaptiveEntropyBottleneck
- Fixed redundant seed handling in AdaptiveEntropyBottleneck that affected global random state
- Enhanced quantization behavior in AdaptiveEntropyBottleneck for more consistent behavior

#### Test Suite Enhancements
- Added dedicated tests for EntropyBottleneck and AdaptiveEntropyBottleneck classes
- Implemented numerical stability tests with extreme values
- Added tests for consistency between training and inference modes
- Enhanced test coverage for AdaptiveEntropyBottleneck with parameter count verification
- Added deterministic behavior verification in evaluation mode

## [0.1.15]

### Added

#### Models Module Refinement
- Restructured core neural networks into a dedicated models module with modular components
- Implemented MeaningVAE as central orchestration model with support for both vector and graph inputs
- Created specialized Encoder and Decoder modules with configurable architectures
- Implemented two compression strategies:
  - EntropyBottleneck providing information-theoretic compression with adaptive levels
  - VectorQuantizer implementing discrete latent representation with codebook learning
- Added integrated support for PyTorch Geometric for graph-based representations
- Enhanced models with comprehensive documentation in README.md
- Improved model architecture to align with the project's conceptual framework
- Optimized parameter initialization for stable training across compression levels
- Added flexible configuration options for all model components

## [0.1.14]

### Added

#### Modular Pipeline Architecture
- Implemented flexible `Pipeline` framework for chainable data transformations
- Created base `PipelineComponent` abstract class for consistent component interface
- Implemented specialized components for encoding, compression, decoding, and graph conversion
- Added conditional processing with `ConditionalComponent` for type-specific handling
- Implemented parallel processing with `BranchComponent` for feature-specific transformations
- Created `PipelineFactory` with common pipeline configurations
- Added comprehensive documentation and example usage
- Implemented test suite for verifying pipeline components
- Added features for pipeline composition, modification, and extension
- Created example demonstrating feature-specific processing pipelines

## [0.1.13]

### Added

#### Enhanced Compression Experiments Framework
- Updated compression experiment system to support adaptive models and graph-based representations
- Integrated `AdaptiveMeaningVAE` with parameter count tracking and efficiency analysis
- Added visualization improvements including radar charts, parameter efficiency plots, and category analysis
- Implemented comprehensive semantic evaluation with standardized metrics framework
- Added behavioral metrics integration for semantic-behavioral correlation analysis
- Enhanced report generation with adaptive model analysis and feature-specific preservation metrics
- Implemented detailed feature-group analysis with preservation, fidelity, and drift metrics
- Added command-line arguments for adaptive modeling, graph-based experiments, and custom compression levels
- Created enhanced visualizations for comparing metrics across compression levels 
- Optimized evaluation pipeline with improved baseline tracking for drift measurement
- Generated more informative experiment reports with clear recommendations based on comprehensive metrics

## [0.1.12]

### Added

#### Feature-Weighted Loss Implementation
- Implemented `FeatureWeightedLoss` class that prioritizes features based on importance scores
- Added progressive weight adjustment system with linear scheduling during training
- Created feature-specific loss components using canonical weights from feature importance analysis
- Implemented comprehensive testing across multiple compression levels (0.5, 1.0, 2.0, 5.0)
- Discovered significant performance improvements at lower compression levels (+4.77% at 0.5x)
- Identified degradation patterns at higher compression levels (-10.82% at 5.0x)
- Added stability tracking for feature-specific losses during training progression
- Documented compression level thresholds where feature-weighted approach becomes ineffective
- Created specialized visualization tools for weight progression and feature-specific preservation
- Identified binary vs. continuous feature preservation divergence patterns
- Added detailed experiment reports with comprehensive analysis
- Generated findings document capturing key insights and architectural implications

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