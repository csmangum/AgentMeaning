from typing import List, Tuple

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encoder network that maps agent states to latent representations."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = None,
        use_batch_norm: bool = True,
    ):
        """
        Initialize encoder network.

        Args:
            input_dim: Dimension of input agent state
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.use_batch_norm = use_batch_norm

        # Build encoder architecture
        modules = []
        in_dim = input_dim

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

        self.encoder = nn.Sequential(*modules)
        self.mu = nn.Linear(self.hidden_dims[-1], latent_dim)
        self.log_var = nn.Linear(self.hidden_dims[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.

        Args:
            x: Input tensor of agent state

        Returns:
            mu: Mean of latent representation
            log_var: Log variance of latent representation
        """
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)

        return mu, log_var
