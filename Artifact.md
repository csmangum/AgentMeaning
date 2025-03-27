# **Artifact Plan (v1.0)**

### **Structure**
```
meaning-transform/
├── README.md
├── notebooks/
│   └── experiment1_structural_semantic.ipynb
├── src/
│   ├── model.py          # VAE model + encoder/decoder
│   ├── data.py           # synthetic state gen + serialization
│   ├── loss.py           # reconstruction + semantic loss
│   └── train.py          # training loop
├── results/
│   ├── loss_curves.png
│   ├── semantic_comparison_examples.png
│   └── latent_space_plot.png
└── utils/
    └── viz.py            # for plotting + visual output
```

### **Goal**
A clean notebook that runs **Experiment 1**, logs:
- Latent size vs. reconstruction loss
- Latent size vs. semantic loss
- Plots of original vs. reconstructed examples
- A few visuals of latent space

---

## **Step-by-Step Build Outline**

### **Step 1 – `data.py`: State + Serializer**
- Generate synthetic agent states
- Serialize to float vector
- Deserialize back for inspection

### **Step 2 – `model.py`: VAE**
- Standard encoder/decoder
- Reparameterization
- Latent dim as hyperparameter

### **Step 3 – `loss.py`: Loss Functions**
- `vae_loss_with_semantic()`
- `semantic_loss()` and `extract_semantic_features()`

### **Step 4 – `train.py`: Training Loop**
- Sweep latent dims
- Log recon + semantic loss per epoch
- Save model checkpoints if needed

### **Step 5 – `viz.py`: Plots**
- Plot:
  - Loss curves
  - Semantic loss breakdown
  - Latent t-SNE (optional)
  - Original vs. recon examples