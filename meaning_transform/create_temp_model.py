#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a temporary model for testing the evaluate_semantic_preservation.py script.
"""

import torch
from src.model import MeaningVAE

# Create a model
model = MeaningVAE(input_dim=50, latent_dim=32)

# Save the model
torch.save({'model_state_dict': model.state_dict()}, 'temp_model.pt')

print('Temporary model saved to temp_model.pt') 