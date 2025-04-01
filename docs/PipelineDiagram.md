# Meaning Preservation Pipeline Diagram

This document visualizes the complete pipeline for transforming agent states while preserving meaning, showing example data at each stage.

```
MEANING PRESERVATION PIPELINE
=============================

[1. AGENT STATE (Dict)]
{
  "agent_id": "agent_42",
  "position": [3.5, 2.1, 0.0],
  "health": 85.0,
  "energy": 72.3,
  "role": "explorer",
  "resource_level": 45.6,
  "is_defending": False,
  "age": 247,
  "total_reward": 1250.5
}
        |
        | serialize
        ▼
[2. BINARY REPRESENTATION (Tensor)]
tensor([3.5, 2.1, 0.0, 85.0, 72.3, 1.0, 0.0, 0.0, 0.0, 0.0, 45.6, 85.0, 0.0, 247.0, 1250.5])
        |
        | transform to graph (optional path)
        ▼
[2a. KNOWLEDGE GRAPH]
Nodes:
- agent_42 (type: agent)
- position_agent_42 (type: property, value: [3.5, 2.1, 0.0])
- health_agent_42 (type: property, value: 85.0)
- role_agent_42 (type: property, value: "explorer")
- ...

Edges:
- (agent_42) --[has_position]--> (position_agent_42)
- (agent_42) --[has_health]--> (health_agent_42)
- (agent_42) --[has_role]--> (role_agent_42)
- ...
        |
        | graph encoder (GNN)
        ▼
[2b. GRAPH EMBEDDINGS]
Node embeddings: tensor([[-0.24, 0.51, ..., 0.13], ...])
Graph embedding: tensor([-0.12, 0.35, 0.67, ..., 0.21])
        |
        | VAE encoder
        ▼
[3. LATENT SPACE]
μ (mean): tensor([-0.21, 0.44, 0.02, -0.57, 0.38, ...]) (32-dim vector)
σ (std dev): tensor([0.12, 0.08, 0.15, 0.07, 0.10, ...]) (32-dim vector)
z (sampled): tensor([-0.18, 0.46, -0.01, -0.62, 0.42, ...]) (32-dim vector)
        |
        | compression
        ▼
[4a. ENTROPY BOTTLENECK COMPRESSION]
Adapted μ: tensor([-0.25, 0.50, 0.00, -0.60, 0.40, ...])
Log scale: tensor([-1.2, -1.5, -1.0, -1.8, -1.3, ...])
Compressed z: tensor([-0.2, 0.5, 0.0, -0.6, 0.4, ...]) (rounded values)
        |
   OR   |
        ▼
[4b. VECTOR QUANTIZATION]
Codebook indices: tensor([42, 17, 63, 8, 29, ...]) (discrete indices)
Quantized z: tensor([-0.19, 0.47, -0.02, -0.61, 0.41, ...]) (from codebook)
        |
        | VAE decoder
        ▼
[5. RECONSTRUCTED STATE (Tensor)]
tensor([3.48, 2.08, 0.02, 84.7, 71.9, 0.98, 0.01, 0.01, 0.00, 0.00, 45.4, 84.7, 0.01, 246.3, 1248.9])
        |
        | deserialize
        ▼
[6. RECONSTRUCTED AGENT STATE (Dict)]
{
  "agent_id": "agent_42",
  "position": [3.48, 2.08, 0.02],
  "health": 84.7,
  "energy": 71.9,
  "role": "explorer",
  "resource_level": 45.4,
  "is_defending": False,
  "age": 246,
  "total_reward": 1248.9
}
        |
        | semantic evaluation
        ▼
[7. MEANING PRESERVATION METRICS]
Reconstruction L2 Loss: 2.31
Behavioral Similarity: 0.97
Semantic Preservation Score: 0.93
Knowledge Graph Structural Consistency: 0.95
```

## Explanation

The pipeline shows the transformation of agent states through multiple representational forms:

1. **Agent State**: Initial dictionary representation with agent properties
2. **Binary Representation**: Serialized tensor form of the agent state
3. **Knowledge Graph** (optional path): Structured graph with nodes for agents and properties
4. **Graph Embeddings** (if using graph path): Node and graph-level embeddings from GNN
5. **Latent Space**: Compressed representation with mean and variance 
6. **Compression Methods**:
   - **Entropy Bottleneck**: Information-theoretic compression
   - **Vector Quantization**: Discrete codebook-based compression
7. **Reconstructed State**: Decoded tensor and dictionary representation
8. **Meaning Preservation Metrics**: Evaluation of semantic preservation

The system is designed to preserve semantic meaning through all transformations, not just structural accuracy. 