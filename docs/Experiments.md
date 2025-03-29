# **Experiments**

To evaluate the capacity of our system to transform information while preserving semantic integrity, we design a series of experiments centered on structured agent state data. These experiments aim to measure not only reconstruction accuracy and compression effectiveness, but also the **functional and behavioral consistency** of the original and reconstructed states.

The central hypothesis is this:  
> A system trained with the proper constraints and architecture can transform agent states through multiple representations—binary, latent, compressed—and recover them with their **meaning intact**, even when structure and form vary.

---

### **5.1 Dataset**

We use a collection of **structured agent states**, formatted as Python dictionaries. Each state reflects the condition of an agent in a simulation or controlled environment. A state may include:

- Numerical values (e.g. position, velocity, energy)
- Categorical variables (e.g. agent role or team)
- Boolean flags (e.g. is_alive, has_target)
- Nested or grouped values (e.g. inventory, allies)

#### **Data Generation Options**
- **Synthetic Simulation Logs:** Generate controlled sequences of agent states from a simulator (e.g. grid world, swarm env, particle physics).
- **RL-trained Agents:** Export logged states from reinforcement learning runs (e.g. Gym environments).
- **Handcrafted State Variations:** To probe the system’s sensitivity to minor or semantic-preserving differences.

All states are **serialized** into fixed-length binary vectors using a custom flattening and encoding process (Section 3.2.1).

---

### **5.2 Evaluation Metrics**

#### **5.2.1 Reconstruction Fidelity**
- Binary reconstruction loss (MSE or BCE)
- Bitwise match rate (exact reconstruction percentage)

#### **5.2.2 Compression Effectiveness**
- Latent dimensionality vs. input size
- Estimated entropy of latent space
- Optional: compress using arithmetic coding and compute bits per state

#### **5.2.3 Semantic Equivalence**
- **Behavioral Equivalence:**  
  - Run both original and reconstructed agent states through a fixed simulation environment or policy model.
  - Measure outcome differences (e.g., action taken, reward obtained, final state).
- **Structural Equivalence:**  
  - Compare key features (e.g., health, location) for deltas.
- **Function-Preserving Distance:**  
  - A proxy metric based on downstream task impact.

#### **5.2.4 Latent Space Analysis**
- PCA or t-SNE plots of latent vectors
- Cluster analysis to see if semantically similar states group together

---

### **5.3 Baselines**

To contextualize results, we compare against several baselines:

| Model / Method             | Description                                         |
|----------------------------|-----------------------------------------------------|
| Raw zlib Compression       | Standard compression of serialized state            |
| Vanilla Autoencoder        | AE without KL loss or compression constraints       |
| Random Latent Mapping      | Control test with no structure learning             |
| VQ-VAE                     | Discrete latent codebook model                      |
| Oracle Semantic Model      | Fixed symbolic model known to preserve meaning      |

---

### **5.4 Experiment 1: Structural Fidelity Under Compression**

**Goal:** Evaluate how compression ratio affects the ability to reconstruct the original binary state.

- Vary the size of the latent space (bottleneck dimension)
- Track reconstruction error and compression ratio
- Observe phase transitions—at what point does meaning collapse?

---

### **5.5 Experiment 2: Semantic Preservation**

**Goal:** Determine whether semantic properties of the state persist through transformation.

- Define key semantic features (e.g., "agent is in danger", "goal is reached")
- Compare these features before and after compression → reconstruction
- Include functional tests (e.g., pass through a fixed policy or environment)

---

### **5.6 Experiment 3: Latent Geometry of Meaning**

**Goal:** Understand how meaning is organized in latent space.

- Select sets of agent states with known semantic similarity
- Visualize latent codes using t-SNE or PCA
- Measure cluster coherence and separation

---

### **5.7 Implementation Details**

- **Framework:** PyTorch with custom serialization module
- **Training:** Adam optimizer, learning rate scheduler
- **Hardware:** GPU acceleration (1–2 consumer GPUs sufficient)
- **Data size:** Start with 10k–100k agent states for initial training

---

### **5.8 Hypotheses**

1. **Compression with Meaning:**  
   The system will be able to reduce agent states into low-dimensional latent codes while preserving semantic outcomes.

2. **Latent Regularity:**  
   States with similar meaning will cluster in latent space—even if their binary forms differ.

3. **Fidelity Threshold:**  
   There will be a clear threshold of compression beyond which semantic meaning is lost, which can be visualized and measured.