"""
Neural Network Compression Framework - Base Module
Core functionality for loading models and basic operations
"""

import torch
import torch.nn as nn
import numpy as np
import copy
import warnings
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')


class BaseCompressionFramework:
    """Base class for neural network compression with model loading capabilities."""
    
    def __init__(self, model_path: str, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model_path = model_path
        self.device = device
        self.original_model = self._load_model()
        self.results = {}

    def _create_generic_model(self, state_dict):
        """Create a generic model container that can hold any state dict"""
        
        class GenericModel(nn.Module):
            def __init__(self, state_dict):
                super().__init__()
                
                # Create parameters and buffers to match the state dict
                for key, value in state_dict.items():
                    if isinstance(value, torch.Tensor):
                        # Determine if it's a parameter or buffer
                        if value.requires_grad or 'weight' in key or 'bias' in key:
                            self.register_parameter(key.replace('.', '_'), nn.Parameter(value))
                        else:
                            self.register_buffer(key.replace('.', '_'), value)
                
                # Store the original state dict structure for later use
                self._original_keys = list(state_dict.keys())
                self._key_mapping = {k.replace('.', '_'): k for k in state_dict.keys()}
                
            def forward(self, x):
                # Generic forward pass - just return input
                # This is a placeholder since we don't know the actual architecture
                return x
                
            def state_dict(self, *args, **kwargs):
                # Override state_dict to return with original keys
                current_state = super().state_dict(*args, **kwargs)
                original_state = {}
                
                for new_key, value in current_state.items():
                    if new_key in self._key_mapping:
                        original_key = self._key_mapping[new_key]
                        original_state[original_key] = value
                    else:
                        original_state[new_key] = value
                        
                return original_state
                
            def load_state_dict(self, state_dict, strict=True):
                # Convert keys for loading
                converted_state = {}
                for key, value in state_dict.items():
                    new_key = key.replace('.', '_')
                    converted_state[new_key] = value
                    
                super().load_state_dict(converted_state, strict=strict)
        
        # Create model and load the state dict
        model = GenericModel(state_dict)
        return model

    def _load_model(self):
        """Load model from .pth file"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Check if it's already a complete model
            if isinstance(checkpoint, nn.Module):
                return checkpoint
            
            # If it's a state dict, we need to create a generic container
            if isinstance(checkpoint, dict):
                # Extract state dict if nested
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Create a generic model container
                model = self._create_generic_model(state_dict)
                return model
            else:
                # Try to treat it as a model
                return checkpoint
                
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

    def _infer_input_shape(self, model):
        """Infer the input shape expected by the model"""
        # For GenericModel with PINN architecture
        if hasattr(model, '_original_keys'):
            # Look for Fourier layer B matrix
            for key in model._original_keys:
                if 'fourier' in key.lower() and 'B' in key:
                    param = getattr(model, key.replace('.', '_'), None)
                    if param is not None:
                        # B matrix shape is [input_dim, fourier_features/2]
                        input_dim = param.shape[0]
                        return (input_dim,)
            
            # Look for input layer hints
            for key in model._original_keys:
                if 'input' in key.lower() and 'weight' in key:
                    param = getattr(model, key.replace('.', '_'), None)
                    if param is not None and len(param.shape) == 2:
                        # For PINN, the input layer comes after Fourier transform
                        # So we need to check the Fourier input dimension
                        fourier_output_dim = param.shape[1]
                        # Typical PINN has 2D or 3D input (spatial coordinates)
                        if fourier_output_dim == 256:  # Common Fourier size
                            return (2,)  # 2D coordinates
                        elif fourier_output_dim == 512:
                            return (3,)  # 3D coordinates
        
        # Try to find the first linear layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # This might be after Fourier transform, so check size
                if module.in_features > 100:  # Likely Fourier features
                    return (2,)  # Default to 2D coordinates
                else:
                    return (module.in_features,)
        
        # Default fallback
        return (2,)

    def _is_pinn_architecture(self, model):
        """Check if the model is a PINN-like architecture"""
        # Check for Fourier layer or specific PINN characteristics
        if hasattr(model, '_original_keys'):
            return any('fourier' in key.lower() for key in model._original_keys)
        
        for name, module in model.named_modules():
            if 'fourier' in name.lower():
                return True
        
        # Check if first layer expects small input (typical for PINN with coordinates)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if module.in_features <= 10:  # Small input dimension suggests coordinates
                    return True
                break
        
        return False

    def _analyze_architecture(self, model):
        """Analyze model architecture"""
        conv_layers = []
        fc_layers = []
        
        # For GenericModel, analyze parameters directly
        if hasattr(model, '_original_keys'):
            for key in model._original_keys:
                param = getattr(model, key.replace('.', '_'), None)
                if param is not None and isinstance(param, nn.Parameter):
                    shape = param.shape
                    if len(shape) == 4:  # Conv weight
                        conv_layers.append(('conv', {
                            'out_channels': shape[0],
                            'in_channels': shape[1],
                            'kernel_size': (shape[2], shape[3])
                        }))
                    elif len(shape) == 2 and shape[0] > 1 and shape[1] > 1:  # Linear weight
                        fc_layers.append(('fc', {
                            'out_features': shape[0],
                            'in_features': shape[1]
                        }))
        else:
            # Original logic for standard models
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    conv_layers.append(('conv', {
                        'out_channels': module.out_channels,
                        'in_channels': module.in_channels,
                        'kernel_size': module.kernel_size
                    }))
                elif isinstance(module, nn.Linear):
                    fc_layers.append(('fc', {
                        'out_features': module.out_features,
                        'in_features': module.in_features
                    }))
        
        # If no conv layers found but we have fc layers, assume it's a fully connected network
        if not conv_layers and fc_layers:
            # Add a dummy conv layer for compatibility with student model creation
            conv_layers = [('conv', {'out_channels': 64, 'in_channels': 3, 'kernel_size': (3, 3)})]
        
        return {'conv_layers': conv_layers, 'fc_layers': fc_layers}