import contextlib
import warnings
from typing import Any, Dict, Union

import torch
import torch.nn as nn


@contextlib.contextmanager
def set_temp_seed(seed=None):
    """
    Context manager for temporarily setting a random seed.
    Restores the previous random state after execution.

    Args:
        seed: Optional random seed to set
    """
    if seed is None:
        yield
        return

    state = torch.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(state)


class BaseModelIO:
    """Base class with standardized save/load methods for models."""

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration as a dictionary."""
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "compression_level": getattr(self, "compression_level", 1.0),
            "compression_type": getattr(self, "compression_type", None),
            "seed": getattr(self, "seed", None),
            "model_type": self.__class__.__name__,
            "version": "1.0",
        }

    def save(self, filepath: str) -> None:
        """Standard save method for model persistence."""
        model_data = {
            "state_dict": self.state_dict(),
            "config": self.get_config(),
        }
        torch.save(model_data, filepath)

    def load(self, filepath: str) -> None:
        """Standard load method with compatibility checks."""
        model_data = torch.load(filepath)

        # Compatibility check
        if "config" in model_data:
            config = model_data["config"]
            # Check model type
            if (
                "model_type" in config
                and config["model_type"] != self.__class__.__name__
            ):
                warnings.warn(
                    f"Loading {config['model_type']} into {self.__class__.__name__}"
                )

            # Check critical parameters
            for param in ["input_dim", "latent_dim"]:
                if param in config and getattr(self, param) != config[param]:
                    warnings.warn(f"Model parameter mismatch: {param}")

        # Load state dict
        self.load_state_dict(model_data.get("state_dict", model_data))


class CompressionBase(nn.Module):
    """Base class for all compression methods."""

    def __init__(self, latent_dim: int, compression_level: float = 1.0):
        """
        Initialize compression base.

        Args:
            latent_dim: Dimension of latent space
            compression_level: Level of compression (higher = more compression)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.compression_level = compression_level
        self.effective_dim = self._calculate_effective_dim()

    def _calculate_effective_dim(self) -> int:
        """Calculate effective dimension based on compression level."""
        return max(1, int(self.latent_dim / self.compression_level))

    def get_compression_rate(self) -> float:
        """Return actual compression rate."""
        return float(self.latent_dim) / float(self.effective_dim)

    def forward(self, z: torch.Tensor) -> Union[torch.Tensor, tuple]:
        """Apply compression to input tensor."""
        raise NotImplementedError("Subclasses must implement forward method")
