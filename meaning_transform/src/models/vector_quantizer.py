from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from meaning_transform.src.models.utils import CompressionBase, set_temp_seed


class VectorQuantizer(CompressionBase):
    """Vector quantization layer for discrete latent representation."""

    def __init__(
        self,
        latent_dim: int,
        num_embeddings: int,
        commitment_cost: float = 0.25,
        seed: int = None,
    ):
        """
        Initialize vector quantizer.

        Args:
            latent_dim: Dimension of latent vectors
            num_embeddings: Number of embedding vectors
            commitment_cost: Weight for commitment loss
            seed: Random seed for reproducibility
        """
        # Initialize with compression level 1.0, it's not used directly in VQ
        super().__init__(latent_dim, compression_level=1.0)

        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.seed = seed

        # For VQ, effective dimension is set to latent_dim since we don't reduce dimensions
        self.effective_dim = latent_dim

        # Initialize embedding table with deterministic seed if provided
        with set_temp_seed(seed):
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
        # Validate input
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(z)}")
        if z.dim() != 2 or z.size(1) != self.latent_dim:
            raise ValueError(
                f"Expected shape (batch_size, {self.latent_dim}), got {z.shape}"
            )

        # Reshape z -> (batch, latent_dim)
        input_shape = z.shape
        flat_z = z.reshape(-1, self.latent_dim)

        # Calculate distances using cdist for better efficiency
        distances = torch.cdist(flat_z, self.embedding.weight, p=2).pow(2)

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

        # Add mask to handle unused codebook entries with improved numerical stability
        mask = avg_probs > 0
        if torch.any(mask):
            # Calculate perplexity only for used codebook entries with numerical stability
            masked_probs = avg_probs[mask]

            # Clamp probabilities for numerical stability
            log_probs = torch.log(torch.clamp(masked_probs, min=1e-10))
            perplexity = torch.exp(-torch.sum(masked_probs * log_probs))
        else:
            # If no codebook entries are used (unlikely but possible)
            perplexity = torch.tensor(1.0, device=z.device)

        # Straight-through estimator
        # Pass the gradient from quantized to input z
        quantized = z + (quantized - z).detach()

        return quantized, vq_loss, perplexity

    def get_compression_rate(self) -> float:
        """
        Calculate effective compression rate for VQ.

        For VQ, the compression rate is estimated based on number of bits needed
        to represent the codebook index vs. full precision latent.

        Returns:
            Compression rate estimate
        """
        # Full precision latent uses 32 bits per dimension
        bits_full_precision = self.latent_dim * 32

        # Codebook index uses log2(num_embeddings) bits
        bits_codebook_index = torch.log2(torch.tensor(self.num_embeddings)).item()

        # Return ratio of full precision bits to codebook bits
        return bits_full_precision / bits_codebook_index
