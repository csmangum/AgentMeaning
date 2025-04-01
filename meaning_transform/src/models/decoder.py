from typing import List

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Decoder network that maps latent representations back to agent states."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        use_batch_norm: bool = True,
    ):
        """
        Initialize decoder network.

        Args:
            latent_dim: Dimension of latent space
            output_dim: Dimension of output agent state
            hidden_dims: List of hidden layer dimensions (in reverse order from encoder)
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [64, 128, 256]
        self.use_batch_norm = use_batch_norm

        # Build decoder architecture
        modules = []
        in_dim = latent_dim

        for h_dim in self.hidden_dims:
            if self.use_batch_norm:
                modules.append(
                    nn.Sequential(
                        nn.Linear(in_dim, h_dim), nn.BatchNorm1d(h_dim), nn.LeakyReLU()
                    )
                )
            else:
                modules.append(nn.Sequential(nn.Linear(in_dim, h_dim), nn.LeakyReLU()))
            in_dim = h_dim

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Linear(self.hidden_dims[-1], output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            z: Latent representation

        Returns:
            x_reconstructed: Reconstructed agent state
        """
        x = self.decoder(z)
        x_reconstructed = self.final_layer(x)

        return x_reconstructed
