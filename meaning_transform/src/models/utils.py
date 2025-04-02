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

    def load(self, filepath: str, adapt_config: bool = False) -> None:
        """
        Load method with intelligent configuration adaptation.
        
        This method attempts to adapt the model to the loaded configuration
        when possible, but raises errors for critical incompatibilities.
        
        Args:
            filepath: Path to the saved model file
            adapt_config: Whether to adapt adaptable parameters to match the loaded config
            
        Raises:
            ValueError: If critical parameters are incompatible
            TypeError: If model types are fundamentally incompatible
            RuntimeError: If state dict loading fails
        """
        model_data = torch.load(filepath)
        
        # Early return if there's no config (legacy model)
        if "config" not in model_data:
            try:
                self.load_state_dict(model_data)
                warnings.warn("Loaded legacy model without configuration data")
                return
            except Exception as e:
                raise RuntimeError(f"Failed to load legacy model: {str(e)}")
                
        config = model_data["config"]
        adaptation_log = []
        
        # Check model type compatibility
        if "model_type" in config:
            loaded_type = config["model_type"]
            current_type = self.__class__.__name__
            
            # Check if model types are compatible
            if loaded_type != current_type:
                # Critical model architecture incompatibility
                if not self._are_model_types_compatible(loaded_type, current_type):
                    raise TypeError(
                        f"Cannot load {loaded_type} model into {current_type} instance. "
                        f"Models have incompatible architectures."
                    )
                adaptation_log.append(f"Adapting {loaded_type} model to {current_type}")
        
        # Critical parameters that must match exactly
        critical_params = ["input_dim", "latent_dim"]
        for param in critical_params:
            if param in config and hasattr(self, param):
                loaded_value = config[param]
                current_value = getattr(self, param)
                
                if loaded_value != current_value:
                    raise ValueError(
                        f"Critical parameter mismatch: {param} "
                        f"(loaded: {loaded_value}, current: {current_value})"
                    )
        
        # Adaptable parameters
        adaptable_params = ["compression_level", "compression_type", "seed", "use_batch_norm"]
        for param in adaptable_params:
            if param in config and hasattr(self, param):
                loaded_value = config[param]
                current_value = getattr(self, param)
                
                if loaded_value != current_value:
                    # Adapt the parameter if requested and possible
                    if adapt_config and self._can_adapt_parameter(param, loaded_value, current_value):
                        setattr(self, param, loaded_value)
                        adaptation_log.append(
                            f"Adapted {param}: {current_value} → {loaded_value}"
                        )
                    else:
                        # Parameter is different but we'll try to load anyway
                        adaptation_log.append(
                            f"Parameter mismatch: {param} "
                            f"(loaded: {loaded_value}, using: {current_value})"
                        )
        
        # Load state dict
        try:
            self.load_state_dict(model_data.get("state_dict", model_data))
        except Exception as e:
            # Create a detailed error message with adaptation log
            error_msg = f"Failed to load state dict: {str(e)}\n"
            if adaptation_log:
                error_msg += "Adaptation log:\n" + "\n".join(adaptation_log)
            raise RuntimeError(error_msg)
                
        # Log successful adaptations
        if adaptation_log:
            warnings.warn("Model loaded with adaptations:\n" + "\n".join(adaptation_log))
            
    def _are_model_types_compatible(self, loaded_type: str, current_type: str) -> bool:
        """
        Check if two model types are compatible for loading.
        
        Override this in subclasses to provide custom compatibility logic.
        """
        # Default compatibility groups
        vae_family = ["MeaningVAE", "AdaptiveMeaningVAE", "FeatureGroupedVAE"]
        
        # Same family models are compatible
        if loaded_type in vae_family and current_type in vae_family:
            return True
            
        # By default, only identical types are compatible
        return loaded_type == current_type
        
    def _can_adapt_parameter(self, param_name: str, loaded_value: Any, current_value: Any) -> bool:
        """
        Check if a parameter can be adapted from loaded to current value.
        
        Override this in subclasses to provide custom adaptation logic.
        """
        # By default, we can adapt non-structural parameters
        if param_name in ["compression_level", "seed"]:
            return True
            
        # For compression_type, only adapt if the bottleneck mechanism is similar
        if param_name == "compression_type":
            entropy_types = ["entropy", "adaptive_entropy"]
            if loaded_value in entropy_types and current_value in entropy_types:
                return True
                
        # Allow use_batch_norm to be adapted if we're loading with adaptation
        if param_name == "use_batch_norm":
            return True
            
        return False


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
        
        # Validate parameters
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if compression_level <= 0:
            raise ValueError(f"compression_level must be positive, got {compression_level}")
            
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
