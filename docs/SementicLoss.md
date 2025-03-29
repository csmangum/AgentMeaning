# **Semantic Loss**

## **Step 1: Define Core Semantic Features**

We'll expand beyond position and alive status to include:

| Feature         | Purpose                                      |
|------------------|----------------------------------------------|
| `x`, `y`         | Spatial location                             |
| `velocity`       | Derived from change in position (if temporal)
| `health`         | Indicates survivability                      |
| `has_target`     | Intentionality or engagement state           |
| `energy`         | Capacity to act or move                      |
| `role`           | Task or context-relevant identity            |
| `priority_state` | Optional derived metric (e.g. “threatened”)  |

Since you’re working with synthetic or simulation agent states, you can define as many as are meaningful. For now, let’s expand to **six semantic components**.

---

## **Step 2: Updated Semantic Extractor**

```python
def extract_semantic_features(state_tensor):
    """
    Extracts higher-level semantic info from serialized state tensor.
    Assumes 10D input vector.
    """
    x = state_tensor[:, 0]
    y = state_tensor[:, 1]
    health = state_tensor[:, 2] * 100.0  # denormalized
    is_alive = (health > 10).float()
    has_target = state_tensor[:, 3]
    energy = state_tensor[:, 4] * 100.0  # denormalized
    role_idx = torch.argmax(state_tensor[:, 5:], dim=1)

    # Example of derived semantic condition: "threatened"
    # Agent is alive, has a target, and low health
    threatened = ((has_target == 1.0) & (health < 30)).float()

    # Stack core semantic features
    sem = torch.stack([
        x, y, health / 100.0,
        has_target, energy / 100.0,
        is_alive,
        role_idx.float() / 5.0,  # normalized for loss
        threatened
    ], dim=1)

    return sem
```

---

## **Step 3: Enhanced Semantic Loss Function**

```python
def semantic_loss(original, reconstructed):
    """
    Compares full semantic representation between original and reconstructed states.
    """
    sem_orig = extract_semantic_features(original)
    sem_recon = extract_semantic_features(reconstructed)

    # Use MSE across all features except where special
    position_loss = F.mse_loss(sem_orig[:, :2], sem_recon[:, :2])
    health_loss = F.mse_loss(sem_orig[:, 2], sem_recon[:, 2])
    has_target_loss = F.binary_cross_entropy(sem_recon[:, 3], sem_orig[:, 3])
    energy_loss = F.mse_loss(sem_orig[:, 4], sem_recon[:, 4])
    alive_loss = F.binary_cross_entropy(sem_recon[:, 5], sem_orig[:, 5])
    role_loss = F.mse_loss(sem_orig[:, 6], sem_recon[:, 6])  # role index
    threat_loss = F.binary_cross_entropy(sem_recon[:, 7], sem_orig[:, 7])

    total = (
        position_loss +
        health_loss +
        has_target_loss +
        energy_loss +
        alive_loss +
        role_loss +
        threat_loss
    )

    return total
```

---

## **Step 4: Logging Semantic Breakdown (Optional)**

If you want more detailed analysis per epoch:

```python
def detailed_semantic_loss_breakdown(original, reconstructed):
    sem_orig = extract_semantic_features(original)
    sem_recon = extract_semantic_features(reconstructed)

    losses = {
        "position": F.mse_loss(sem_orig[:, :2], sem_recon[:, :2]).item(),
        "health": F.mse_loss(sem_orig[:, 2], sem_recon[:, 2]).item(),
        "has_target": F.binary_cross_entropy(sem_recon[:, 3], sem_orig[:, 3]).item(),
        "energy": F.mse_loss(sem_orig[:, 4], sem_recon[:, 4]).item(),
        "is_alive": F.binary_cross_entropy(sem_recon[:, 5], sem_orig[:, 5]).item(),
        "role": F.mse_loss(sem_orig[:, 6], sem_recon[:, 6]).item(),
        "threatened": F.binary_cross_entropy(sem_recon[:, 7], sem_orig[:, 7]).item()
    }
    return losses
```

You can log and visualize these to see **what aspects of meaning degrade first** under compression.

---

## **Result:**

This metric becomes a **rich measure of semantic degradation**, offering:
- Finer-grained supervision
- Clearer insights into which meanings are hardest to preserve
- More reliable signal than structure-level loss alone