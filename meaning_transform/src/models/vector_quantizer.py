from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector quantization layer for discrete latent representation."""

    def __init__(
        self, latent_dim: int, num_embeddings: int, commitment_cost: float = 0.25
    ):
        """
        Initialize vector quantizer.

        Args:
            latent_dim: Dimension of latent vectors
            num_embeddings: Number of embedding vectors
            commitment_cost: Weight for commitment loss
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Embedding table
        self.embedding = nn.Embedding(num_embeddings, latent_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """
        Quantize latent vectors.

        Args:
            z: Latent vectors [B, D]

        Returns:
            quantized: Quantized latent vectors
            vq_loss: Vector quantization loss
            perplexity: Perplexity of the codebook usage
        """
        # Reshape z -> (batch, latent_dim)
        input_shape = z.shape
        flat_z = z.reshape(-1, self.latent_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_z**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_z, self.embedding.weight.t())
        )

        # Get the indices of the closest embedding vectors
        encoding_indices = torch.argmin(distances, dim=1)

        # Quantize the latent vectors
        quantized = self.embedding(encoding_indices).reshape(input_shape)

        # Compute the VQ loss
        q_latent_loss = F.mse_loss(quantized.detach(), z)
        commitment_loss = F.mse_loss(quantized, z.detach())
        vq_loss = q_latent_loss + self.commitment_cost * commitment_loss

        # Compute the perplexity (measure of codebook usage)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Straight-through estimator
        # Pass the gradient from quantized to input z
        quantized = z + (quantized - z).detach()

        return quantized, vq_loss, perplexity
