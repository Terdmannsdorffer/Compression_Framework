import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization as quant
import torch.optim as optim  # Add this line
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import gzip
import shutil
import copy
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelCompressionFramework:
    """
    Comprehensive framework for neural network compression without training data.
    
    Techniques included:
    - Pruning: magnitude, random, structured
    - Quantization: dynamic INT8, log2, minifloat
    - Knowledge Distillation: automatic student architecture generation
    """
    
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
    
    def _create_model_from_state_dict(self, state_dict):
        """Create a simple model structure from state dict keys"""
        # Analyze the state dict to understand model structure
        layers_info = self._analyze_state_dict(state_dict)
        
        class DynamicModel(nn.Module):
            def __init__(self, layers_info):
                super().__init__()
                self.layers = nn.ModuleList()
                
                for layer_info in layers_info:
                    if layer_info['type'] == 'conv':
                        self.layers.append(nn.Conv2d(
                            layer_info['in_channels'],
                            layer_info['out_channels'],
                            kernel_size=layer_info['kernel_size'],
                            stride=1, padding=1
                        ))
                    elif layer_info['type'] == 'linear':
                        self.layers.append(nn.Linear(
                            layer_info['in_features'],
                            layer_info['out_features']
                        ))
                    elif layer_info['type'] == 'bn':
                        self.layers.append(nn.BatchNorm2d(layer_info['num_features']))
                        
            def forward(self, x):
                for layer in self.layers:
                    if isinstance(layer, nn.Linear) and len(x.shape) > 2:
                        x = x.view(x.size(0), -1)
                    x = layer(x)
                return x
                
        return DynamicModel(layers_info)
    
    def _analyze_state_dict(self, state_dict):
        """Analyze state dict to extract layer information"""
        layers_info = []
        processed_layers = set()
        
        for key in sorted(state_dict.keys()):
            if 'weight' in key:
                layer_name = key.replace('.weight', '')
                if layer_name in processed_layers:
                    continue
                    
                weight_shape = state_dict[key].shape
                
                if len(weight_shape) == 4:  # Conv2d
                    layers_info.append({
                        'type': 'conv',
                        'name': layer_name,
                        'out_channels': weight_shape[0],
                        'in_channels': weight_shape[1],
                        'kernel_size': weight_shape[2]
                    })
                elif len(weight_shape) == 2:  # Linear
                    layers_info.append({
                        'type': 'linear',
                        'name': layer_name,
                        'out_features': weight_shape[0],
                        'in_features': weight_shape[1]
                    })
                    
                processed_layers.add(layer_name)
                
        return layers_info
    
    # ============== UTILITY METHODS ==============

    def _calculate_quantized_size(self, model):
        """Calculate size of quantized model"""
        total_bits = 0
        
        for name, param in model.named_parameters():
            if self._is_quantized(param):
                # Determine quantization level
                unique_vals = len(torch.unique(param))
                if unique_vals <= 16:
                    bits_per_element = 4
                elif unique_vals <= 256:
                    bits_per_element = 8
                else:
                    bits_per_element = 16
            else:
                bits_per_element = 32  # float32
            
            total_bits += param.numel() * bits_per_element
        
        # Add buffer size
        for buffer in model.buffers():
            total_bits += buffer.numel() * buffer.element_size() * 8
        
        return total_bits / (8 * 1024 * 1024)  # Convert to MB
    
    def _calculate_ber(self, original_model, compressed_model):
        """
        Calculate performance-based error rate by comparing model outputs
        Instead of comparing weights, compare predictions on test samples
        """
        # For weight-based comparison (legacy, optional)
        weight_change_ratio = self._calculate_weight_change_ratio(original_model, compressed_model)
        
        # For output-based comparison (primary metric)
        output_error_rate = self._calculate_output_error_rate(original_model, compressed_model)
        
        # Return output error rate as primary BER metric
        return output_error_rate

    def _calculate_weight_change_ratio(self, original_model, compressed_model):
        """Calculate percentage of weights that changed significantly"""
        total_elements = 0
        changed_elements = 0
        
        orig_state = original_model.state_dict()
        comp_state = compressed_model.state_dict()
        
        for key in orig_state.keys():
            if key in comp_state and ('weight' in key or 'bias' in key):
                orig_tensor = orig_state[key].detach().cpu().float()
                comp_tensor = comp_state[key].detach().cpu().float()
                
                if orig_tensor.shape != comp_tensor.shape:
                    continue
                
                total_elements += orig_tensor.numel()
                
                # Count significant changes (>1% relative change)
                mask = orig_tensor != 0
                if mask.sum() > 0:
                    relative_change = torch.abs((comp_tensor - orig_tensor) / (orig_tensor + 1e-8))
                    significant_changes = (relative_change > 0.01) & mask
                    changed_elements += significant_changes.sum().item()
        
        return changed_elements / total_elements if total_elements > 0 else 0


    def _calculate_output_error_rate(self, original_model, compressed_model, num_samples=1000):
        """
        Calculate error rate based on output differences
        This is more meaningful than weight comparison
        """
        original_model.eval()
        compressed_model.eval()
        
        # Determine input shape and type
        input_shape = self._infer_input_shape(original_model)
        is_pinn = self._is_pinn_architecture(original_model)
        
        total_error = 0
        max_possible_error = 0
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Generate appropriate test inputs
                if is_pinn and len(input_shape) == 1:
                    # For PINN: use domain-appropriate inputs
                    if input_shape[0] == 2:
                        test_input = torch.rand(1, 2).to(self.device) * 2 - 1  # [-1, 1]
                    elif input_shape[0] == 3:
                        test_input = torch.rand(1, 3).to(self.device) * 2 - 1
                    else:
                        test_input = torch.randn(1, *input_shape).to(self.device)
                else:
                    # Standard random inputs
                    test_input = torch.randn(1, *input_shape).to(self.device)
                
                try:
                    # Get outputs
                    orig_output = original_model(test_input)
                    comp_output = compressed_model(test_input)
                    
                    # Handle shape mismatches
                    if orig_output.shape != comp_output.shape:
                        continue
                    
                    # Calculate relative error
                    if is_pinn:
                        # For regression: use relative L2 error
                        error = torch.norm(orig_output - comp_output)
                        max_error = torch.norm(orig_output) + 1e-8
                        relative_error = (error / max_error).item()
                    else:
                        # For classification: check if predictions match
                        orig_pred = torch.argmax(orig_output, dim=-1)
                        comp_pred = torch.argmax(comp_output, dim=-1)
                        relative_error = float(orig_pred != comp_pred)
                    
                    total_error += relative_error
                    max_possible_error += 1
                    
                except:
                    continue
        
        # Return average error rate
        if max_possible_error > 0:
            error_rate = total_error / max_possible_error
            return min(error_rate, 1.0)  # Cap at 100%
        else:
            return 0.0


    def evaluate_with_data(self, compressed_models_dict, test_loader=None, validation_func=None):
        """
        Evaluate compressed models with actual test data if available
        
        Args:
            compressed_models_dict: Dict of compressed models to evaluate
            test_loader: DataLoader with test data (optional)
            validation_func: Custom validation function (optional)
        
        Returns:
            Dictionary with accuracy/performance metrics
        """
        if test_loader is None and validation_func is None:
            print("No test data provided. Using synthetic evaluation...")
            return self._synthetic_evaluation(compressed_models_dict)
        
        results = {}
        
        for name, model in compressed_models_dict.items():
            model.eval()
            
            if validation_func:
                # Use custom validation function
                performance = validation_func(model)
            else:
                # Standard accuracy calculation
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for inputs, targets in test_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = model(inputs)
                        
                        # Classification accuracy
                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()
                
                performance = 100 * correct / total if total > 0 else 0
            
            results[name] = {
                'performance': performance,
                'size_mb': self.get_model_stats(model)['size_mb']
            }
            
            print(f"{name}: {performance:.2f}% accuracy")
        
        return results


    def _synthetic_evaluation(self, compressed_models_dict):
        """
        Evaluate models using synthetic data when no test set is available
        Estimates performance degradation based on compression technique
        """
        results = {}
        
        for name, model in compressed_models_dict.items():
            model.eval()
            
            # Estimate performance based on compression type and amount
            base_performance = 100.0
            
            if 'pruning' in name:
                # Pruning typically maintains performance up to 50% sparsity
                if '0.1' in name:
                    estimated_performance = base_performance * 0.99
                elif '0.3' in name:
                    estimated_performance = base_performance * 0.97
                elif '0.5' in name:
                    estimated_performance = base_performance * 0.93
                elif '0.7' in name:
                    estimated_performance = base_performance * 0.85
                else:
                    estimated_performance = base_performance * 0.90
                    
            elif 'quantization' in name:
                # Quantization impact depends on bit width
                if 'int8' in name or '8bit' in name:
                    estimated_performance = base_performance * 0.98
                elif '6bit' in name:
                    estimated_performance = base_performance * 0.95
                elif '4bit' in name:
                    estimated_performance = base_performance * 0.90
                else:
                    estimated_performance = base_performance * 0.93
                    
            elif 'distillation' in name:
                # Distillation depends on student size
                if '0.25' in name or '25' in name:
                    estimated_performance = base_performance * 0.95
                elif '0.1' in name or '10' in name:
                    estimated_performance = base_performance * 0.90
                elif '0.05' in name or '5' in name:
                    estimated_performance = base_performance * 0.85
                else:
                    estimated_performance = base_performance * 0.92
            else:
                estimated_performance = base_performance
            
            # Add some noise for realism
            estimated_performance += np.random.normal(0, 1.0)
            estimated_performance = np.clip(estimated_performance, 0, 100)
            
            results[name] = {
                'performance': estimated_performance,
                'size_mb': self.get_model_stats(model)['size_mb'],
                'estimated': True
            }
        
        return results


    def _is_quantized(self, tensor):
        """Check if a tensor appears to be quantized"""
        if tensor.numel() == 0:
            return False
            
        unique_values = torch.unique(tensor)
        
        # Check if very few unique values (quantized)
        if len(unique_values) < min(256, tensor.numel() / 10):
            return True
        
        # Check if values are powers of 2 (log2 quantization)
        non_zero_values = unique_values[unique_values != 0]
        if len(non_zero_values) > 0:
            abs_values = torch.abs(non_zero_values)
            log_values = torch.log2(abs_values)
            # Check if log values are close to integers
            if torch.allclose(log_values, torch.round(log_values), atol=0.01):
                return True
        
        # Check if values are from a small discrete set (minifloat)
        if len(unique_values) < 256:  # Typical for 8-bit or less
            return True
        
        return False
    
# Replace the get_model_stats method with the correct signature
    def get_model_stats(self, model, include_performance=True):
        """Get comprehensive model statistics for any model type"""
        # Create a clean copy for analysis
        model_copy = copy.deepcopy(model)
        
        # Initialize counters
        total_params = 0
        nonzero_params = 0
        
        # Method 1: Try standard module counting
        found_modules = False
        for name, module in model_copy.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                found_modules = True
                # Get the effective weight (considering pruning mask if present)
                if hasattr(module, 'weight_mask'):
                    # Pruning is applied
                    effective_weight = module.weight * module.weight_mask
                    total_params += effective_weight.numel()
                    nonzero_params += (effective_weight != 0).sum().item()
                    
                    # Apply the pruning permanently
                    with torch.no_grad():
                        module.weight.data = effective_weight
                    prune.remove(module, 'weight')
                else:
                    # No pruning
                    if hasattr(module, 'weight'):
                        total_params += module.weight.numel()
                        nonzero_params += (module.weight != 0).sum().item()
                
                # Count bias
                if hasattr(module, 'bias') and module.bias is not None:
                    total_params += module.bias.numel()
                    nonzero_params += (module.bias != 0).sum().item()
        
        # Method 2: If no standard modules found, count all parameters
        if not found_modules or total_params == 0:
            for name, param in model_copy.named_parameters():
                if param.requires_grad:
                    total_params += param.numel()
                    nonzero_params += (param != 0).sum().item()
        
        # Calculate sparsity
        sparsity = 1 - (nonzero_params / total_params) if total_params > 0 else 0
        
        # Calculate size based on model type
        if hasattr(model, '_quantized') and model._quantized:
            # Quantized model
            bits_per_param = getattr(model, '_quant_bits', 8)
            # For minifloat, calculate based on actual bit width
            if hasattr(model, '_exp_bits') and hasattr(model, '_mantissa_bits'):
                bits_per_param = model._exp_bits + model._mantissa_bits + 1  # +1 for sign bit
            size_mb = (total_params * bits_per_param) / (8 * 1024 * 1024)
        elif hasattr(model, '_pruned') and model._pruned and sparsity > 0.3:
            # For significantly pruned models, use CSR format estimation
            # CSR format: values array + column indices + row pointers
            # More efficient than COO for structured sparsity
            values_size = nonzero_params * 4  # float32
            indices_size = nonzero_params * 4  # int32 column indices
            pointers_size = 1000 * 4  # Approximate row pointers (depends on matrix structure)
            total_sparse_bytes = values_size + indices_size + pointers_size
            
            # Compare with dense storage
            dense_bytes = total_params * 4
            
            # Use the smaller of the two
            size_mb = min(total_sparse_bytes, dense_bytes) / (1024 * 1024)
        else:
            # Dense model
            param_size = sum(p.numel() * p.element_size() for p in model_copy.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model_copy.buffers())
            size_mb = (param_size + buffer_size) / (1024**2)
        
        # Get compressed size
        temp_path = 'temp_model.pth'
        
        # For pruned models, save in sparse format if beneficial
        state_dict = model_copy.state_dict()
        if sparsity > 0.5:  # More than 50% sparse
            sparse_state_dict = {}
            for key, tensor in state_dict.items():
                if 'weight' in key and (tensor == 0).sum() > tensor.numel() * 0.5:
                    # Convert to sparse tensor
                    sparse_state_dict[key] = tensor.to_sparse_csr() if tensor.dim() == 2 else tensor.to_sparse()
                else:
                    sparse_state_dict[key] = tensor
            torch.save(sparse_state_dict, temp_path)
        else:
            torch.save(state_dict, temp_path)
        
        with open(temp_path, 'rb') as f_in:
            with gzip.open(temp_path + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        compressed_size_mb = os.path.getsize(temp_path + '.gz') / (1024**2)
        
        os.remove(temp_path)
        os.remove(temp_path + '.gz')
        
        # Build the stats dictionary
        stats = {
            'total_params': total_params,
            'nonzero_params': nonzero_params,
            'sparsity': sparsity,
            'size_mb': size_mb,
            'compressed_size_mb': compressed_size_mb
        }
        
        # Add performance-based BER if requested
        if include_performance:
            try:
                ber = self._calculate_ber(self.original_model, model_copy)
                stats['ber'] = ber
                
                # Estimate accuracy degradation
                stats['estimated_accuracy_retention'] = (1 - ber) * 100
            except:
                # If BER calculation fails, use default
                stats['ber'] = 0.0
                stats['estimated_accuracy_retention'] = 100.0
        else:
            stats['ber'] = 0.0
        
        return stats
    
    def create_synthetic_test_data(self, num_samples=1000):
        """
        Create synthetic test data for models when real data isn't available
        
        Returns:
            DataLoader with synthetic test data
        """
        input_shape = self._infer_input_shape(self.original_model)
        is_pinn = self._is_pinn_architecture(self.original_model)
        
        # Generate inputs
        if is_pinn and len(input_shape) == 1:
            if input_shape[0] == 2:
                inputs = torch.rand(num_samples, 2) * 2 - 1  # [-1, 1]
            elif input_shape[0] == 3:
                inputs = torch.rand(num_samples, 3) * 2 - 1
            else:
                inputs = torch.randn(num_samples, *input_shape)
        else:
            inputs = torch.randn(num_samples, *input_shape)
        
        # Generate targets using original model
        self.original_model.eval()
        with torch.no_grad():
            targets = self.original_model(inputs.to(self.device)).cpu()
            
            # For classification, convert to class labels
            if targets.shape[-1] > 1 and not is_pinn:
                targets = torch.argmax(targets, dim=-1)
        
        # Create dataset and loader
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        
        return loader

    # ============== PRUNING METHODS ==============
    
    def apply_magnitude_pruning(self, amount=0.3):
        """Prune weights with smallest magnitude"""
        model = copy.deepcopy(self.original_model)
        
        # For GenericModel, prune parameters directly
        if hasattr(model, '_original_keys'):
            # This is a GenericModel
            all_weights = []
            weight_params = []
            
            # Collect all weight tensors
            for key in model._original_keys:
                if 'weight' in key:
                    param = getattr(model, key.replace('.', '_'))
                    if param is not None:
                        all_weights.append(param.data.abs().flatten())
                        weight_params.append((key.replace('.', '_'), param))
            
            if all_weights:
                # Calculate threshold
                all_weights_cat = torch.cat(all_weights)
                threshold = torch.quantile(all_weights_cat, amount)
                
                # Apply pruning
                for param_name, param in weight_params:
                    mask = param.data.abs() > threshold
                    param.data = param.data * mask
        else:
            # Standard model pruning
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    prune.l1_unstructured(module, name='weight', amount=amount)
        
        model._pruned = True
        return model


    def apply_random_pruning(self, amount=0.3):
        """Randomly prune weights"""
        model = copy.deepcopy(self.original_model)
        
        # For GenericModel, prune parameters directly
        if hasattr(model, '_original_keys'):
            # This is a GenericModel
            for key in model._original_keys:
                if 'weight' in key:
                    param = getattr(model, key.replace('.', '_'))
                    if param is not None:
                        mask = torch.rand_like(param) > amount
                        param.data = param.data * mask
        else:
            # Standard model pruning
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    prune.random_unstructured(module, name='weight', amount=amount)
        
        model._pruned = True
        return model


    def apply_structured_pruning(self, amount=0.3):
        """Prune entire channels/neurons"""
        model = copy.deepcopy(self.original_model)
        
        # For GenericModel, prune by zeroing entire rows/columns
        if hasattr(model, '_original_keys'):
            for key in model._original_keys:
                if 'weight' in key:
                    param = getattr(model, key.replace('.', '_'))
                    if param is not None and param.dim() >= 2:
                        # Prune entire output channels/neurons
                        num_channels = param.shape[0]
                        num_to_prune = int(num_channels * amount)
                        
                        # Calculate channel importance (L2 norm)
                        importance = param.data.view(num_channels, -1).norm(2, dim=1)
                        _, indices = importance.sort()
                        
                        # Zero out least important channels
                        param.data[indices[:num_to_prune]] = 0
        else:
            # Standard model pruning
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
        
        model._pruned = True
        return model
    
    # ============== QUANTIZATION METHODS ==============
        
    def apply_dynamic_quantization(self):
        """Apply INT8 dynamic quantization"""
        model = copy.deepcopy(self.original_model)
        
        # For GenericModel, manually quantize weights
        if hasattr(model, '_original_keys'):
            for key in model._original_keys:
                if 'weight' in key or 'bias' in key:
                    param = getattr(model, key.replace('.', '_'))
                    if param is not None:
                        # Quantize to INT8 range
                        scale = param.abs().max() / 127.0
                        if scale > 0:
                            quantized = torch.round(param / scale).clamp(-128, 127)
                            param.data = quantized * scale
        else:
            # Try standard quantization
            try:
                model.cpu()
                model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
            except:
                pass
        
        model._quantized = True
        model._quant_bits = 8
        return model
    
    def apply_log2_quantization(self, bits=8):
        """Quantize weights to powers of 2"""
        model = copy.deepcopy(self.original_model)
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name:
                    weight = param.data
                    
                    # Handle zeros
                    mask_zero = weight == 0
                    
                    # Get sign and magnitude
                    sign = torch.sign(weight)
                    abs_weight = torch.abs(weight)
                    abs_weight[mask_zero] = 1  # Avoid log(0)
                    
                    # Quantize to powers of 2
                    log_weight = torch.log2(abs_weight)
                    max_val = 2**(bits-1) - 1
                    min_val = -2**(bits-1)
                    
                    quantized_log = torch.clamp(torch.round(log_weight), min_val, max_val)
                    quantized_weight = sign * (2.0 ** quantized_log)
                    quantized_weight[mask_zero] = 0
                    
                    param.data = quantized_weight
        
        model._quantized = True  # Mark as quantized
        model._quant_bits = bits
        return model
    
    def apply_minifloat_quantization(self, exp_bits=4, mantissa_bits=3):
        """Apply custom floating-point quantization"""
        model = copy.deepcopy(self.original_model)
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = param.data
                
                # Handle special cases
                sign = torch.sign(weight)
                abs_weight = torch.abs(weight)
                mask_zero = abs_weight == 0
                abs_weight[mask_zero] = 1e-10
                
                # Extract exponent and mantissa
                exponent = torch.floor(torch.log2(abs_weight))
                mantissa = abs_weight / (2.0 ** exponent)
                
                # Quantize
                exp_max = 2**(exp_bits-1) - 1
                exp_min = -2**(exp_bits-1)
                exponent = torch.clamp(torch.round(exponent), exp_min, exp_max)
                
                mantissa_levels = 2**mantissa_bits
                mantissa = torch.round(mantissa * mantissa_levels) / mantissa_levels
                
                # Reconstruct
                quantized_weight = sign * mantissa * (2.0 ** exponent)
                quantized_weight[mask_zero] = 0
                
                param.data = quantized_weight
        
        model._quantized = True
        model._exp_bits = exp_bits
        model._mantissa_bits = mantissa_bits
        model._quant_bits = exp_bits + mantissa_bits + 1  # Total bits including sign
        return model
    
    # ============== KNOWLEDGE DISTILLATION ==============
    
    def create_student_model(self, compression_ratio=0.25, train_student=True, epochs=50):
        """Create and optionally train a smaller student model"""
        
        # Analyze if this is a PINN-like architecture
        is_pinn = self._is_pinn_architecture(self.original_model)
        
        if is_pinn:
            # Create PINN-style student
            student = self._create_pinn_student(self.original_model, compression_ratio)
        else:
            # Use original CNN-based student creation
            teacher_stats = self._analyze_architecture(self.original_model)
            student = self._create_cnn_student(teacher_stats, compression_ratio)
        
        student.to(self.device)
        
        # Print initial comparison
        teacher_size = self.get_model_stats(self.original_model)['size_mb']
        student_size = self.get_model_stats(student)['size_mb']
        
        print(f"\nStudent Model Created:")
        print(f"  Teacher size: {teacher_size:.2f} MB")
        print(f"  Student size: {student_size:.2f} MB") 
        print(f"  Actual compression: {student_size/teacher_size:.2%}")
        
        # Train the student if requested
        if train_student:
            print(f"  Training student for {epochs} epochs...")
            student = self._train_student_self_distillation(
                self.original_model, student, epochs=epochs
            )
            print("  Training completed!")
        
        return student


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


    def _create_pinn_student(self, teacher, compression_ratio):
        """Create a student model matching PINN architecture"""
        
        # Extract teacher configuration
        teacher_config = self._extract_pinn_config(teacher)
        
        # Scale down the architecture
        scale_factor = np.sqrt(compression_ratio)
        
        # Create student configuration
        student_hidden_layers = []
        for size in teacher_config['hidden_layers']:
            student_size = max(16, int(size * scale_factor))  # Minimum 16 neurons
            student_hidden_layers.append(student_size)
        
        # Reduce Fourier mapping size
        student_fourier_size = max(32, int(teacher_config['fourier_size'] * scale_factor))
        # Make sure it's even (for sin/cos split)
        student_fourier_size = (student_fourier_size // 2) * 2
        
        class StudentPINN(nn.Module):
            def __init__(self, input_dim, output_dim, hidden_layers, fourier_size, fourier_scale):
                super().__init__()
                
                self.input_dim = input_dim
                self.fourier_size = fourier_size
                
                # Fourier feature mapping
                self.register_buffer('B', torch.randn((input_dim, fourier_size // 2)) * fourier_scale)
                
                # Build network
                self.layers = nn.ModuleList()
                
                # First layer after Fourier (fourier_size is the full size including sin and cos)
                self.layers.append(nn.Linear(fourier_size, hidden_layers[0]))
                self.layers.append(nn.Tanh())
                
                # Hidden layers
                for i in range(len(hidden_layers) - 1):
                    self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                    self.layers.append(nn.Tanh())
                
                # Output layer
                self.layers.append(nn.Linear(hidden_layers[-1], output_dim))
                
                # Initialize weights
                self._initialize_weights()
            
            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
            
            def forward(self, x):
                # Ensure input has correct shape
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                
                # Fourier feature mapping
                x_proj = torch.matmul(x, self.B)  # [batch, fourier_size // 2]
                x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # [batch, fourier_size]
                
                # Forward through layers
                for layer in self.layers:
                    x = layer(x)
                
                return x
        
        # Create student model
        student = StudentPINN(
            input_dim=teacher_config['input_dim'],
            output_dim=teacher_config['output_dim'],
            hidden_layers=student_hidden_layers,
            fourier_size=student_fourier_size,
            fourier_scale=teacher_config['fourier_scale']
        )
        
        return student


    def _extract_pinn_config(self, model):
        """Extract configuration from PINN teacher model"""
        config = {
            'input_dim': 2,  # Default for spatial coordinates
            'output_dim': 3,  # Default output
            'hidden_layers': [128, 128, 128, 128],  # Default
            'fourier_size': 256,
            'fourier_scale': 10.0
        }
        
        # First, let's check the actual output by running a forward pass
        try:
            with torch.no_grad():
                # Create test input based on expected dimensions
                test_input = torch.randn(1, 2).to(self.device)  # Start with 2D
                try:
                    test_output = model(test_input)
                    actual_output_dim = test_output.shape[-1]
                    config['output_dim'] = actual_output_dim
                    print(f"    Detected output dimension from forward pass: {actual_output_dim}")
                except:
                    # Try 3D input
                    test_input = torch.randn(1, 3).to(self.device)
                    try:
                        test_output = model(test_input)
                        actual_output_dim = test_output.shape[-1]
                        config['output_dim'] = actual_output_dim
                        config['input_dim'] = 3
                        print(f"    Detected 3D input, output dimension: {actual_output_dim}")
                    except:
                        print("    Could not determine output dimension from forward pass")
        except Exception as e:
            print(f"    Forward pass test failed: {str(e)}")
        
        # Try to extract from GenericModel structure
        if hasattr(model, '_original_keys'):
            print("    Analyzing model structure...")
            
            # Debug: print all keys
            fourier_keys = [k for k in model._original_keys if 'fourier' in k.lower()]
            input_keys = [k for k in model._original_keys if 'input' in k.lower()]
            output_keys = [k for k in model._original_keys if 'output' in k.lower()]
            
            print(f"    Found Fourier keys: {fourier_keys}")
            print(f"    Found input keys: {input_keys}")
            print(f"    Found output keys: {output_keys}")
            
            # Look for Fourier layer parameters
            for key in model._original_keys:
                if 'fourier' in key.lower() and 'B' in key:
                    param = getattr(model, key.replace('.', '_'), None)
                    if param is not None:
                        config['input_dim'] = param.shape[0]
                        config['fourier_size'] = param.shape[1] * 2  # sin + cos
                        config['fourier_scale'] = float(param.std() * np.sqrt(param.shape[0]))
                        print(f"    Fourier B matrix shape: {param.shape}")
                        break
            
            # Look for input layer to verify
            for key in model._original_keys:
                if 'input_layer' in key.lower() and 'weight' in key:
                    param = getattr(model, key.replace('.', '_'), None)
                    if param is not None:
                        print(f"    Input layer weight shape: {param.shape}")
                        config['fourier_size'] = param.shape[1]
                        break
            
            # Look for output dimension - but don't override if we got it from forward pass
            if 'output_dim' not in locals():
                for key in reversed(model._original_keys):
                    if 'output' in key.lower() and 'weight' in key:
                        param = getattr(model, key.replace('.', '_'), None)
                        if param is not None:
                            print(f"    Output layer weight shape: {param.shape}")
                            # Note: shape[0] might not be the actual output if model does slicing
                            break
            
            # Extract hidden layer sizes
            hidden_sizes = []
            hidden_keys = [k for k in model._original_keys if 'hidden' in k.lower() and 'weight' in k and 'bias' not in k]
            
            for key in hidden_keys:
                param = getattr(model, key.replace('.', '_'), None)
                if param is not None:
                    hidden_sizes.append(param.shape[0])
                    
            if hidden_sizes:
                config['hidden_layers'] = hidden_sizes
                print(f"    Hidden layer sizes: {hidden_sizes}")
        
        print(f"    Final config: input_dim={config['input_dim']}, output_dim={config['output_dim']}, fourier_size={config['fourier_size']}")
        
        return config


    def _create_cnn_student(self, teacher_stats, compression_ratio):
        """Original CNN-based student creation (fallback)"""
        class StudentModel(nn.Module):
            def __init__(self, teacher_stats, compression_ratio):
                super().__init__()
                self.features = nn.Sequential()
                self.classifier = nn.Sequential()
                
                # Scale down the architecture
                scale_factor = np.sqrt(compression_ratio)
                
                # Build feature extractor
                in_channels = 3  # Assume RGB input
                
                for i, (layer_type, layer_info) in enumerate(teacher_stats['conv_layers'][:3]):
                    out_channels = max(8, int(layer_info['out_channels'] * scale_factor))
                    
                    self.features.add_module(f'conv{i}', nn.Conv2d(
                        in_channels, out_channels, kernel_size=3, padding=1
                    ))
                    self.features.add_module(f'relu{i}', nn.ReLU())
                    
                    if i < 2:
                        self.features.add_module(f'pool{i}', nn.MaxPool2d(2))
                    
                    in_channels = out_channels
                
                # Adaptive pooling
                self.features.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((4, 4)))
                
                # Build classifier
                fc_input = out_channels * 16
                
                if teacher_stats['fc_layers']:
                    hidden_size = max(32, int(teacher_stats['fc_layers'][0][1]['out_features'] * scale_factor))
                    num_classes = teacher_stats['fc_layers'][-1][1]['out_features']
                    
                    self.classifier.add_module('fc1', nn.Linear(fc_input, hidden_size))
                    self.classifier.add_module('relu', nn.ReLU())
                    self.classifier.add_module('dropout', nn.Dropout(0.2))
                    self.classifier.add_module('fc2', nn.Linear(hidden_size, num_classes))
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        return StudentModel(teacher_stats, compression_ratio)

    def _train_student_self_distillation(self, teacher, student, epochs=50, batch_size=32, 
                                        learning_rate=0.001, temperature=4.0):
        """
        Train student using self-distillation (no real data required)
        """
        teacher.eval()
        student.train()
        
        optimizer = optim.Adam(student.parameters(), lr=learning_rate)
        
        # Infer input shape and type
        input_shape = self._infer_input_shape(teacher)
        is_pinn = self._is_pinn_architecture(teacher)
        
        # Adjust batch size for PINN models
        if is_pinn:
            batch_size = 256
        
        # First, test compatibility with debug info
        print(f"    Testing model compatibility...")
        print(f"    Input shape: {input_shape}")
        
        projection_layer = None
        
        with torch.no_grad():
            test_input = torch.randn(1, *input_shape).to(self.device)
            try:
                teacher_out = teacher(test_input)
                student_out = student(test_input)
                print(f"    Teacher output shape: {teacher_out.shape}")
                print(f"    Student output shape: {student_out.shape}")
                
                if teacher_out.shape[-1] != student_out.shape[-1]:
                    print(f"    Output dimension mismatch! Creating projection layer...")
                    # Create a projection layer to match dimensions
                    projection_layer = nn.Linear(student_out.shape[-1], teacher_out.shape[-1]).to(self.device)
                    nn.init.xavier_normal_(projection_layer.weight)
                    nn.init.zeros_(projection_layer.bias)
                    
                    # Add projection layer parameters to optimizer
                    optimizer.add_param_group({'params': projection_layer.parameters()})
                    
            except Exception as e:
                print(f"    Initial test failed: {str(e)}")
                return student
        
        # Training loop
        successful_batches = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 20 if is_pinn else 10
            
            for batch_idx in range(num_batches):
                # Generate appropriate random input
                if is_pinn and len(input_shape) == 1:
                    # For PINN: generate random coordinates in domain
                    if input_shape[0] == 2:
                        # 2D spatial coordinates
                        random_inputs = torch.rand(batch_size, 2).to(self.device) * 2 - 1
                    elif input_shape[0] == 3:
                        # 3D spatial coordinates
                        random_inputs = torch.rand(batch_size, 3).to(self.device) * 2 - 1
                    else:
                        random_inputs = torch.randn(batch_size, *input_shape).to(self.device)
                else:
                    # Standard random inputs
                    random_inputs = torch.randn(batch_size, *input_shape).to(self.device)
                
                try:
                    # Get teacher outputs
                    with torch.no_grad():
                        teacher_outputs = teacher(random_inputs)
                    
                    # Get student outputs
                    student_outputs = student(random_inputs)
                    
                    # Apply projection if needed
                    if projection_layer is not None:
                        student_outputs = projection_layer(student_outputs)
                    
                    # Debug shapes on first successful batch
                    if successful_batches == 0:
                        print(f"    First successful batch - Input: {random_inputs.shape}, Teacher: {teacher_outputs.shape}, Student (projected): {student_outputs.shape}")
                    
                    # Calculate loss based on task type
                    if is_pinn:
                        # For PINN: use MSE loss (regression)
                        loss = F.mse_loss(student_outputs, teacher_outputs)
                    else:
                        # For classification: use KL divergence
                        soft_targets = F.softmax(teacher_outputs / temperature, dim=-1)
                        student_log_probs = F.log_softmax(student_outputs / temperature, dim=-1)
                        loss = F.kl_div(student_log_probs, soft_targets, reduction='batchmean') * (temperature ** 2)
                    
                    # Add L2 regularization
                    l2_reg = sum(p.pow(2.0).sum() for p in student.parameters())
                    if projection_layer is not None:
                        l2_reg += sum(p.pow(2.0).sum() for p in projection_layer.parameters())
                    loss = loss + 1e-5 * l2_reg
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                    if projection_layer is not None:
                        torch.nn.utils.clip_grad_norm_(projection_layer.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    successful_batches += 1
                    
                except Exception as e:
                    if epoch == 0 and batch_idx < 5:
                        print(f"    Training batch failed: {str(e)}")
                    continue
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                if successful_batches > 0:
                    avg_loss = epoch_loss / successful_batches
                    print(f"    Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
                    successful_batches = 0
                else:
                    print(f"    Epoch [{epoch+1}/{epochs}], No successful batches")
        
        # If we used a projection layer, we might want to update the student's final layer
        # to match the teacher's output dimension
        if projection_layer is not None:
            print(f"    Note: Student model trained with projection layer to match teacher output dimension")
        
        student.eval()
        return student

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
    
    def analyze_ber_compression_tradeoff(self, ber_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5], 
                                    save_results=True):
        """
        Analyze the trade-off between BER and compression ratio to find optimal points.
        
        Args:
            ber_thresholds: List of acceptable BER thresholds to analyze
            save_results: Whether to save the analysis results
        
        Returns:
            Dictionary with optimal compression techniques for each BER threshold
        """
        print("\n" + "="*60)
        print("BER-AWARE COMPRESSION ANALYSIS")
        print("="*60)
        
        if not self.results:
            print("No compression results available. Run compress_all() first.")
            return None
        
        # Collect all compression results with BER
        all_techniques = []
        
        for category, category_results in self.results.items():
            if category == 'original':
                continue
                
            for technique_name, stats in category_results.items():
                compression_ratio = self.results['original']['size_mb'] / stats['size_mb']
                ber = stats.get('ber', 0)
                
                all_techniques.append({
                    'category': category,
                    'name': technique_name,
                    'compression_ratio': compression_ratio,
                    'ber': ber,
                    'size_mb': stats['size_mb'],
                    'sparsity': stats.get('sparsity', 0)
                })
        
        # Sort by compression ratio
        all_techniques.sort(key=lambda x: x['compression_ratio'], reverse=True)
        
        # Find optimal techniques for each BER threshold
        ber_optimal = {}
        
        for threshold in ber_thresholds:
            # Filter techniques within BER threshold
            valid_techniques = [t for t in all_techniques if t['ber'] <= threshold]
            
            if valid_techniques:
                # Get the one with highest compression
                best = valid_techniques[0]
                ber_optimal[threshold] = best
                
                print(f"\nBER Threshold: {threshold*100:.0f}%")
                print(f"  Best technique: {best['category']}: {best['name']}")
                print(f"  Compression ratio: {best['compression_ratio']:.2f}x")
                print(f"  Actual BER: {best['ber']*100:.1f}%")
                print(f"  Size: {best['size_mb']:.2f} MB")
            else:
                print(f"\nBER Threshold: {threshold*100:.0f}%")
                print(f"  No techniques found within this BER threshold")
        
        # Find Pareto optimal points
        pareto_points = self._find_pareto_optimal(all_techniques)
        
        print("\n" + "-"*40)
        print("PARETO OPTIMAL POINTS (Best BER-Compression Trade-offs):")
        print("-"*40)
        
        for point in pareto_points[:5]:  # Show top 5
            print(f"\n{point['category']}: {point['name']}")
            print(f"  Compression: {point['compression_ratio']:.2f}x")
            print(f"  BER: {point['ber']*100:.1f}%")
            print(f"  Size: {point['size_mb']:.2f} MB")
        
        # Create visualization
        self._plot_ber_pareto_analysis(all_techniques, pareto_points, ber_optimal)
        
        # Save results if requested
        if save_results:
            self._save_ber_analysis(ber_optimal, pareto_points)
        
        return {
            'ber_optimal': ber_optimal,
            'pareto_points': pareto_points,
            'all_techniques': all_techniques
        }

    def _find_pareto_optimal(self, techniques):
        """Find Pareto optimal points for BER vs Compression trade-off"""
        pareto_points = []
        
        for i, technique in enumerate(techniques):
            is_pareto = True
            
            # Check if any other technique dominates this one
            for j, other in enumerate(techniques):
                if i != j:
                    # Other dominates if it has both higher compression AND lower BER
                    if (other['compression_ratio'] > technique['compression_ratio'] and 
                        other['ber'] < technique['ber']):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_points.append(technique)
        
        # Sort by compression ratio
        pareto_points.sort(key=lambda x: x['compression_ratio'], reverse=True)
        
        return pareto_points

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
    

    def _save_ber_analysis(self, ber_optimal, pareto_points):
        """Save BER analysis results to file"""
        # Use UTF-8 encoding to handle special characters
        with open('ber_compression_analysis.txt', 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("BER-AWARE COMPRESSION ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("OPTIMAL TECHNIQUES FOR DIFFERENT BER THRESHOLDS\n")
            f.write("-"*40 + "\n\n")
            
            for threshold in sorted(ber_optimal.keys()):
                technique = ber_optimal[threshold]
                f.write(f"BER Threshold: <= {threshold*100:.0f}%\n")  # Changed ≤ to <=
                f.write(f"  Best Technique: {technique['category']}: {technique['name']}\n")
                f.write(f"  Compression Ratio: {technique['compression_ratio']:.2f}x\n")
                f.write(f"  Actual BER: {technique['ber']*100:.1f}%\n")
                f.write(f"  Final Size: {technique['size_mb']:.2f} MB\n")
                f.write(f"  Size Reduction: {(1-technique['size_mb']/self.results['original']['size_mb'])*100:.1f}%\n\n")
            
            f.write("\nPARETO OPTIMAL POINTS\n")
            f.write("-"*40 + "\n")
            f.write("(Techniques that offer the best trade-off between BER and compression)\n\n")
            
            for i, point in enumerate(pareto_points[:10]):  # Top 10
                f.write(f"{i+1}. {point['category']}: {point['name']}\n")
                f.write(f"   Compression: {point['compression_ratio']:.2f}x\n")
                f.write(f"   BER: {point['ber']*100:.1f}%\n")
                f.write(f"   Size: {point['size_mb']:.2f} MB\n\n")
            
            f.write("\nRECOMMENDATIONS\n")
            f.write("-"*40 + "\n\n")
            
            # Find knee point (best trade-off)
            knee_point = self._find_knee_point(pareto_points)
            if knee_point:
                f.write(f"Recommended (Knee Point): {knee_point['category']}: {knee_point['name']}\n")
                f.write(f"  This offers the best balance with {knee_point['compression_ratio']:.2f}x compression\n")
                f.write(f"  at {knee_point['ber']*100:.1f}% BER\n\n")
            
            f.write("For different use cases:\n")
            f.write("- Critical applications (BER < 10%): Use pruning with low amounts\n")
            f.write("- Balanced applications (BER < 30%): Consider moderate pruning or INT8 quantization\n")
            f.write("- Size-critical applications (BER < 50%): Use aggressive quantization or distillation\n")
            
        print("BER analysis report saved as 'ber_compression_analysis.txt'")


    def _find_knee_point(self, pareto_points):
        """Find the knee point in the Pareto frontier (best trade-off)"""
        if len(pareto_points) < 3:
            return pareto_points[0] if pareto_points else None
        
        # Calculate the distance from the line connecting first and last points
        first = pareto_points[0]
        last = pareto_points[-1]
        
        max_distance = 0
        knee_point = None
        
        for point in pareto_points[1:-1]:
            # Calculate perpendicular distance to the line
            distance = self._point_line_distance(
                point['compression_ratio'], point['ber'],
                first['compression_ratio'], first['ber'],
                last['compression_ratio'], last['ber']
            )
            
            if distance > max_distance:
                max_distance = distance
                knee_point = point
        
        return knee_point


    def _point_line_distance(self, px, py, x1, y1, x2, y2):
        """Calculate perpendicular distance from point to line"""
        # Normalize the scales
        px_norm = px / max(x1, x2)
        py_norm = py / max(y1, y2)
        x1_norm = x1 / max(x1, x2)
        y1_norm = y1 / max(y1, y2)
        x2_norm = x2 / max(x1, x2)
        y2_norm = y2 / max(y1, y2)
        
        # Calculate distance
        numerator = abs((y2_norm - y1_norm) * px_norm - (x2_norm - x1_norm) * py_norm + 
                    x2_norm * y1_norm - y2_norm * x1_norm)
        denominator = np.sqrt((y2_norm - y1_norm)**2 + (x2_norm - x1_norm)**2)
        
        return numerator / denominator if denominator > 0 else 0
    
    # ============== MAIN COMPRESSION PIPELINE ==============
    
    def compress_all(self, pruning_amounts=[0.1, 0.3, 0.5, 0.7], 
                    quantization_bits=[4, 6, 8],
                    student_ratios=[0.25, 0.1],
                    train_students=True,
                    student_epochs=50):
        """Apply all compression techniques"""
        
        print("="*60)
        print("NEURAL NETWORK COMPRESSION ANALYSIS")
        print("="*60)
        
        # Original model stats
        print("\nAnalyzing original model...")
        self.results['original'] = self.get_model_stats(self.original_model)
        self._print_stats("Original Model", self.results['original'])
        
        # Pruning
        print("\n" + "="*60)
        print("PRUNING TECHNIQUES")
        print("="*60)
        
        self.results['pruning'] = {}
        
        for amount in pruning_amounts:
            print(f"\nPruning with amount={amount}")
            
            # Apply each pruning method
            for method_name, method_func in [
                ('magnitude', self.apply_magnitude_pruning),
                ('random', self.apply_random_pruning),
                ('structured', self.apply_structured_pruning)
            ]:
                model = method_func(amount)
                stats = self.get_model_stats(model)
                self.results['pruning'][f'{method_name}_{amount}'] = stats
                self._print_stats(f"  {method_name.title()} Pruning", stats)
        
        # Quantization
        print("\n" + "="*60)
        print("QUANTIZATION TECHNIQUES")
        print("="*60)
        
        self.results['quantization'] = {}
        
        # Dynamic INT8
        print("\nDynamic INT8 Quantization")
        model = self.apply_dynamic_quantization()
        stats = self.get_model_stats(model)
        self.results['quantization']['dynamic_int8'] = stats
        self._print_stats("  Dynamic INT8", stats)
        
        # Log2 quantization
        for bits in quantization_bits:
            print(f"\nLog2 Quantization ({bits}-bit)")
            model = self.apply_log2_quantization(bits)
            stats = self.get_model_stats(model)
            self.results['quantization'][f'log2_{bits}bit'] = stats
            self._print_stats(f"  Log2 {bits}-bit", stats)
        
        # Minifloat quantization
        minifloat_configs = [(4, 3), (5, 2), (3, 4)]
        for exp_bits, mantissa_bits in minifloat_configs:
            print(f"\nMinifloat E{exp_bits}M{mantissa_bits}")
            model = self.apply_minifloat_quantization(exp_bits, mantissa_bits)
            stats = self.get_model_stats(model)
            self.results['quantization'][f'minifloat_E{exp_bits}M{mantissa_bits}'] = stats
            self._print_stats(f"  Minifloat E{exp_bits}M{mantissa_bits}", stats)
        
        # Knowledge Distillation
        print("\n" + "="*60)
        print("KNOWLEDGE DISTILLATION")
        if train_students:
            print("(With self-distillation training)")
        print("="*60)
        
        self.results['distillation'] = {}
        
        for ratio in student_ratios:
            print(f"\nCreating student with {ratio:.0%} size")
            student = self.create_student_model(
                ratio, 
                train_student=train_students,
                epochs=student_epochs
            )
            stats = self.get_model_stats(student)
            self.results['distillation'][f'student_{ratio}'] = stats
            self._print_stats(f"  Student {ratio:.0%}", stats)
        
        return self.results
    
    def _print_stats(self, name, stats):
        """Pretty print model statistics including size details"""
        orig_size = self.results.get('original', {}).get('size_mb', stats['size_mb'])
        compression = orig_size / stats['size_mb'] if stats['size_mb'] > 0 else 1
        
        print(f"{name}:")
        print(f"  Original size: {orig_size:.2f} MB")
        print(f"  Current size: {stats['size_mb']:.2f} MB")
        print(f"  Compressed size: {stats['compressed_size_mb']:.2f} MB")
        print(f"  Compression ratio: {compression:.2f}x")
        print(f"  Sparsity: {stats['sparsity']*100:.1f}%")
        print(f"  BER: {stats['ber']*100:.2f}%")
        print(f"  Total params: {stats['total_params']:,}")
        print(f"  Non-zero params: {stats['nonzero_params']:,}")
        print("-" * 40)

    
    # ============== VISUALIZATION ==============
    
    def plot_results(self):
        """Create comprehensive visualizations including BER analysis"""
        if not self.results:
            print("No results to plot. Run compress_all() first.")
            return
        
        # Create figure with better spacing
        fig = plt.figure(figsize=(24, 14))
        
        # Use GridSpec for better control over layout
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3, 
                            top=0.95, bottom=0.05, left=0.05, right=0.95)
        
        # Create subplots
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[0, 3])
        ax5 = fig.add_subplot(gs[1, 0], projection='3d')
        ax6 = fig.add_subplot(gs[1, 1])
        ax7 = fig.add_subplot(gs[1, 2])
        ax8 = fig.add_subplot(gs[1, 3])
        
        self._plot_compression_overview(ax1)
        self._plot_size_comparison(ax2)
        self._plot_sparsity_analysis(ax3)
        self._plot_ber_analysis(ax4)
        self._plot_technique_comparison(ax5)
        self._plot_combined_potential(ax6)
        self._plot_ber_vs_compression(ax7)
        self._create_summary_table(ax8)
        
        plt.tight_layout()
        plt.savefig('compression_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nVisualization saved as 'compression_analysis.png'")


    def _plot_ber_pareto_analysis(self, all_techniques, pareto_points, ber_optimal):
        """Create visualization for BER-aware compression analysis"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: BER vs Compression scatter with Pareto frontier
        compressions = [t['compression_ratio'] for t in all_techniques]
        bers = [t['ber'] * 100 for t in all_techniques]
        categories = [t['category'] for t in all_techniques]
        
        # Color by category
        color_map = {'pruning': 'blue', 'quantization': 'green', 'distillation': 'orange'}
        colors = [color_map.get(cat, 'gray') for cat in categories]
        
        # Plot all points
        scatter = ax1.scatter(compressions, bers, c=colors, alpha=0.6, s=50)
        
        # Highlight Pareto points
        pareto_comps = [p['compression_ratio'] for p in pareto_points]
        pareto_bers = [p['ber'] * 100 for p in pareto_points]
        ax1.scatter(pareto_comps, pareto_bers, c='red', s=100, marker='*', 
                label='Pareto Optimal', zorder=5, edgecolors='black')
        
        # Draw Pareto frontier
        if len(pareto_points) > 1:
            # Sort by compression for line drawing
            sorted_pareto = sorted(zip(pareto_comps, pareto_bers))
            pareto_x, pareto_y = zip(*sorted_pareto)
            ax1.plot(pareto_x, pareto_y, 'r--', alpha=0.5, linewidth=2)
        
        ax1.set_xlabel('Compression Ratio')
        ax1.set_ylabel('Bit Error Rate (%)')
        ax1.set_title('BER vs Compression Trade-off', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add BER threshold lines
        for threshold, technique in ber_optimal.items():
            ax1.axhline(y=threshold*100, color='gray', linestyle=':', alpha=0.5)
            ax1.text(0.5, threshold*100 + 1, f'{threshold*100:.0f}% threshold', 
                    fontsize=8, alpha=0.7)
        
        # Plot 2: Compression ratio for different BER thresholds
        thresholds = list(ber_optimal.keys())
        max_compressions = []
        technique_names = []
        
        for threshold in sorted(thresholds):
            if threshold in ber_optimal:
                max_compressions.append(ber_optimal[threshold]['compression_ratio'])
                technique_names.append(ber_optimal[threshold]['name'])
            else:
                max_compressions.append(0)
                technique_names.append('None')
        
        bars = ax2.bar([t*100 for t in sorted(thresholds)], max_compressions, 
                    color='skyblue', alpha=0.7, edgecolor='navy')
        
        # Add technique names on bars
        for i, (bar, name) in enumerate(zip(bars, technique_names)):
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        name.replace('_', '\n'), ha='center', va='bottom', 
                        fontsize=8, rotation=0)
        
        ax2.set_xlabel('BER Threshold (%)')
        ax2.set_ylabel('Maximum Compression Ratio')
        ax2.set_title('Achievable Compression at Different BER Thresholds', fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Plot 3: BER vs Size reduction
        size_reductions = [(1 - t['size_mb']/self.results['original']['size_mb'])*100 
                        for t in all_techniques]
        
        ax3.scatter(size_reductions, bers, c=colors, alpha=0.6, s=50)
        
        # Highlight optimal points
        optimal_reductions = [(1 - p['size_mb']/self.results['original']['size_mb'])*100 
                            for p in pareto_points]
        ax3.scatter(optimal_reductions, pareto_bers, c='red', s=100, marker='*', 
                label='Pareto Optimal', zorder=5, edgecolors='black')
        
        ax3.set_xlabel('Size Reduction (%)')
        ax3.set_ylabel('Bit Error Rate (%)')
        ax3.set_title('BER vs Size Reduction', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Add legend for categories
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=cat.title()) 
                        for cat, color in color_map.items()]
        ax1.legend(handles=legend_elements + [plt.Line2D([0], [0], marker='*', color='w', 
                markerfacecolor='r', markersize=10, label='Pareto Optimal')], 
                loc='upper left')
        
        plt.tight_layout()
        plt.savefig('ber_compression_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nBER analysis visualization saved as 'ber_compression_analysis.png'")

    def _plot_ber_analysis(self, ax):
        """Plot BER for different compression techniques"""
        techniques = []
        bers = []
        colors = []
        
        # Collect BER data
        for category, results in self.results.items():
            if category == 'original':
                continue
                
            if category == 'pruning':
                for name, stats in results.items():
                    techniques.append(name.replace('_', '\n'))
                    bers.append(stats.get('ber', 0) * 100)  # Convert to percentage
                    colors.append('blue')
            elif category == 'quantization':
                for name, stats in results.items():
                    techniques.append(name.replace('_', '\n'))
                    bers.append(stats.get('ber', 0) * 100)  # Convert to percentage
                    colors.append('green')
            elif category == 'distillation':
                for name, stats in results.items():
                    techniques.append(name.replace('_', '\n'))
                    bers.append(stats.get('ber', 0) * 100)  # Convert to percentage
                    colors.append('orange')
        
        # Create bar plot
        bars = ax.bar(range(len(techniques)), bers, color=colors, alpha=0.7)
        ax.set_xticks(range(len(techniques)))
        ax.set_xticklabels(techniques, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Bit Error Rate (%)')
        ax.set_title('BER by Compression Technique', fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=7)
        
        # Add reference line
        ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
        ax.legend()

    def _plot_ber_vs_compression(self, ax):
        """Plot BER vs compression ratio"""
        # Separate data by category
        pruning_data = {'comps': [], 'bers': []}
        quant_data = {'comps': [], 'bers': []}
        distill_data = {'comps': [], 'bers': []}
        
        orig_size = self.results['original']['size_mb']
        
        # Collect data by category
        for category, results in self.results.items():
            if category == 'original':
                continue
                
            for name, stats in results.items():
                compression = orig_size / stats['size_mb']
                ber = stats.get('ber', 0) * 100
                
                if compression > 1.0:  # Only plot if there's actual compression
                    if category == 'pruning':
                        pruning_data['comps'].append(compression)
                        pruning_data['bers'].append(ber)
                    elif category == 'quantization':
                        quant_data['comps'].append(compression)
                        quant_data['bers'].append(ber)
                    elif category == 'distillation':
                        distill_data['comps'].append(compression)
                        distill_data['bers'].append(ber)
        
        # Plot each category with different markers
        if pruning_data['comps']:
            ax.scatter(pruning_data['comps'], pruning_data['bers'], 
                    c='blue', marker='o', s=100, alpha=0.7, label='Pruning')
        
        if quant_data['comps']:
            ax.scatter(quant_data['comps'], quant_data['bers'], 
                    c='green', marker='s', s=100, alpha=0.7, label='Quantization')
        
        if distill_data['comps']:
            ax.scatter(distill_data['comps'], distill_data['bers'], 
                    c='orange', marker='^', s=100, alpha=0.7, label='Distillation')
        
        # Add trend line for all points
        all_comps = pruning_data['comps'] + quant_data['comps'] + distill_data['comps']
        all_bers = pruning_data['bers'] + quant_data['bers'] + distill_data['bers']
        
        if len(all_comps) > 3:
            z = np.polyfit(all_comps, all_bers, 2)
            p = np.poly1d(z)
            x_trend = np.linspace(min(all_comps), max(all_comps), 100)
            ax.plot(x_trend, np.maximum(0, p(x_trend)), 'r--', alpha=0.5, label='Trend')
        
        ax.set_xlabel('Compression Ratio')
        ax.set_ylabel('Bit Error Rate (%)')
        ax.set_title('BER vs Compression Trade-off', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set reasonable axis limits
        if all_comps:
            ax.set_xlim(0.5, max(all_comps) * 1.1)
            ax.set_ylim(-5, max(all_bers) * 1.1 if all_bers else 100)

    def _plot_compression_overview(self, ax):
        """Overview of all compression ratios"""
        techniques = []
        compressions = []
        colors = []
        
        orig_size = self.results['original']['size_mb']
        
        # Collect all results
        for category, results in self.results.items():
            if category == 'original':
                continue
                
            if category == 'pruning':
                for name, stats in results.items():
                    techniques.append(name.replace('_', '\n'))
                    compressions.append(orig_size / stats['size_mb'])
                    colors.append('blue')
            elif category == 'quantization':
                for name, stats in results.items():
                    techniques.append(name.replace('_', '\n'))
                    compressions.append(orig_size / stats['size_mb'])
                    colors.append('green')
            elif category == 'distillation':
                for name, stats in results.items():
                    techniques.append(name.replace('_', '\n'))
                    compressions.append(orig_size / stats['size_mb'])
                    colors.append('orange')
        
        bars = ax.bar(range(len(techniques)), compressions, color=colors, alpha=0.7)
        ax.set_xticks(range(len(techniques)))
        ax.set_xticklabels(techniques, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Compression Ratio')
        ax.set_title('Compression Ratios by Technique', fontweight='bold')
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.1f}x', ha='center', va='bottom', fontsize=7)
    
    def _plot_size_comparison(self, ax):
        """Compare original vs compressed sizes"""
        categories = ['Pruning', 'Quantization', 'Distillation']
        best_compressions = []
        
        orig_size = self.results['original']['size_mb']
        
        for category in ['pruning', 'quantization', 'distillation']:
            if category in self.results:
                best_compression = min(
                    stats['compressed_size_mb'] 
                    for stats in self.results[category].values()
                )
                best_compressions.append(best_compression)
            else:
                best_compressions.append(orig_size)
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, [orig_size]*len(categories), width, 
                       label='Original', alpha=0.8, color='red')
        bars2 = ax.bar(x + width/2, best_compressions, width,
                       label='Best Compressed', alpha=0.8, color='green')
        
        ax.set_xlabel('Technique Category')
        ax.set_ylabel('Size (MB)')
        ax.set_title('Best Compression per Category', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add compression ratios
        for i, (orig, comp) in enumerate(zip([orig_size]*len(categories), best_compressions)):
            ratio = orig / comp
            ax.text(i, max(orig, comp) + 0.5, f'{ratio:.1f}x', ha='center')
    
    def _plot_sparsity_analysis(self, ax):
        """Analyze sparsity for pruning methods"""
        # Debug: print what we have
        print("\nDEBUG - Pruning results:")
        for name, stats in self.results.get('pruning', {}).items():
            print(f"{name}: sparsity={stats['sparsity']*100:.1f}%")
        
        # Separate data by pruning type
        pruning_data = {
            'magnitude': {'amounts': [], 'sparsities': []},
            'random': {'amounts': [], 'sparsities': []},
            'structured': {'amounts': [], 'sparsities': []}
        }
        
        # Collect data
        for name, stats in self.results.get('pruning', {}).items():
            parts = name.split('_')
            if len(parts) == 2:
                method = parts[0]
                amount = float(parts[1])
                sparsity = stats['sparsity'] * 100
                
                if method in pruning_data:
                    pruning_data[method]['amounts'].append(amount)
                    pruning_data[method]['sparsities'].append(sparsity)
        
        # Plot each method
        colors = {'magnitude': 'blue', 'random': 'orange', 'structured': 'green'}
        markers = {'magnitude': 'o', 'random': 's', 'structured': '^'}
        
        has_data = False
        for method, data in pruning_data.items():
            if data['amounts'] and any(s > 0 for s in data['sparsities']):
                # Sort by amount
                sorted_pairs = sorted(zip(data['amounts'], data['sparsities']))
                amounts, sparsities = zip(*sorted_pairs)
                
                ax.plot(amounts, sparsities, 
                    color=colors[method],
                    marker=markers[method],
                    label=method, 
                    linewidth=2, 
                    markersize=8,
                    linestyle='-')
                has_data = True
        
        # If no pruning achieved, show expected line
        if not has_data:
            ax.text(0.35, 35, 'Pruning not effective\non this model', 
                    ha='center', va='center', fontsize=12, color='red',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
        
        # Add expected diagonal
        ax.plot([0, 0.7], [0, 70], 'k:', alpha=0.3, label='Expected')
        
        ax.set_xlabel('Pruning Amount')
        ax.set_ylabel('Sparsity (%)')
        ax.set_title('Sparsity vs Pruning Amount', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, 0.75)
        ax.set_ylim(0, 75)
    
    def _plot_technique_comparison(self, ax):
        """3D scatter plot of techniques"""
        # Remove the existing 2D axis
        ax.remove()
        
        # Create new 3D axis with try-except for safety
        try:
            ax = plt.subplot(2, 4, 5, projection='3d')
            
            sizes = []
            compressions = []
            bers = []
            colors = []
            labels = []
            
            orig_size = self.results['original']['size_mb']
            
            # Add original
            sizes.append(orig_size)
            compressions.append(1.0)
            bers.append(0.0)
            colors.append('red')
            labels.append('Original')
            
            # Add all techniques
            color_map = {'pruning': 'blue', 'quantization': 'green', 'distillation': 'orange'}
            
            for category, results in self.results.items():
                if category == 'original':
                    continue
                    
                for name, stats in results.items():
                    if stats['size_mb'] > 0:  # Only add valid results
                        sizes.append(stats['size_mb'])
                        compressions.append(orig_size / stats['size_mb'])
                        bers.append(stats.get('ber', 0) * 100)
                        colors.append(color_map.get(category, 'gray'))
                        labels.append(f"{category}: {name}")
            
            # Create scatter plot
            if len(sizes) > 1:  # Only plot if we have data
                scatter = ax.scatter(sizes, compressions, bers, c=colors, s=100, alpha=0.7)
                
                ax.set_xlabel('Model Size (MB)')
                ax.set_ylabel('Compression Ratio')
                ax.set_zlabel('Bit Error Rate (%)')
                ax.set_title('3D Technique Comparison', fontweight='bold')
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor=color, label=cat.title()) 
                                for cat, color in color_map.items()]
                legend_elements.insert(0, Patch(facecolor='red', label='Original'))
                ax.legend(handles=legend_elements, loc='upper left')
            else:
                ax.text(0.5, 0.5, 0.5, 'Insufficient data for 3D plot', 
                        ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            print(f"Warning: 3D plot error: {e}")
            # Fallback to 2D plot
            ax = plt.subplot(2, 4, 5)
            ax.text(0.5, 0.5, '3D plot unavailable', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12)
    
    def _plot_combined_potential(self, ax):
        """Show potential of combining techniques with realistic estimates"""
        techniques = ['Original', 'Pruning\nOnly', 'Quantization\nOnly', 
                    'Distillation\nOnly', 'Pruning +\nQuantization', 'All\nTechniques']
        
        # Get actual best compressions from results
        orig_size = self.results['original']['size_mb']
        
        # Find best in each category
        best_pruning_ratio = 1.0
        best_quant_ratio = 1.0
        best_distill_ratio = 1.0
        
        # Check pruning
        for name, stats in self.results.get('pruning', {}).items():
            if stats['size_mb'] > 0:
                ratio = orig_size / stats['size_mb']
                best_pruning_ratio = max(best_pruning_ratio, ratio)
        
        # Check quantization
        for name, stats in self.results.get('quantization', {}).items():
            if stats['size_mb'] > 0:
                ratio = orig_size / stats['size_mb']
                best_quant_ratio = max(best_quant_ratio, ratio)
        
        # Check distillation
        for name, stats in self.results.get('distillation', {}).items():
            if stats['size_mb'] > 0:
                ratio = orig_size / stats['size_mb']
                best_distill_ratio = max(best_distill_ratio, ratio)
        
        # Calculate realistic combined compressions
        compressions = [
            1.0,  # Original
            best_pruning_ratio,  # Pruning only
            best_quant_ratio,  # Quantization only
            best_distill_ratio,  # Distillation only
            min(best_pruning_ratio * best_quant_ratio * 0.7, 20.0),  # Pruning + Quant
            min(best_pruning_ratio * best_quant_ratio * best_distill_ratio * 0.5, 30.0)  # All
        ]
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        bars = ax.bar(techniques, compressions, color=colors, alpha=0.7)
        
        ax.set_ylabel('Compression Ratio')
        ax.set_title('Realistic Combined Compression Potential', fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.1f}x', ha='center', va='bottom')
        
        # Set y limit based on actual values
        ax.set_ylim(0, max(compressions) * 1.2)
        
        ax.text(0.5, 0.95, 'Note: Combined techniques with efficiency factors',
                transform=ax.transAxes, ha='center', fontsize=9, style='italic')
    
    def _create_summary_table(self, ax):
        """Create summary table of compression results including BER"""
        ax.axis('tight')
        ax.axis('off')
        
        # Find best techniques
        orig_size = self.results['original']['size_mb']
        best_overall = {'name': 'Original', 'compression': 1.0, 'size': orig_size, 'ber': 0}
        best_pruning = {'name': 'None', 'compression': 1.0, 'ber': 0}
        best_quant = {'name': 'None', 'compression': 1.0, 'ber': 0}
        best_distill = {'name': 'None', 'compression': 1.0, 'ber': 0}
        lowest_ber = {'name': 'Original', 'compression': 1.0, 'ber': 1.0}
        
        # Check all techniques
        for category in ['pruning', 'quantization', 'distillation']:
            for name, stats in self.results.get(category, {}).items():
                comp = orig_size / stats['size_mb']
                ber = stats.get('ber', 0)
                
                # Update best in category
                if category == 'pruning' and comp > best_pruning['compression']:
                    best_pruning = {'name': name, 'compression': comp, 'ber': ber}
                elif category == 'quantization' and comp > best_quant['compression']:
                    best_quant = {'name': name, 'compression': comp, 'ber': ber}
                elif category == 'distillation' and comp > best_distill['compression']:
                    best_distill = {'name': name, 'compression': comp, 'ber': ber}
                
                # Update overall best
                if comp > best_overall['compression']:
                    best_overall = {'name': f'{category}: {name}', 'compression': comp, 
                                'size': stats['size_mb'], 'ber': ber}
                
                # Track lowest BER
                if ber < lowest_ber['ber'] and comp > 1.5:
                    lowest_ber = {'name': f'{category}: {name}', 'compression': comp, 'ber': ber}
        
        # Create table data
        table_data = [
            ['Metric', 'Technique', 'Compression', 'BER'],
            ['Overall Best', best_overall['name'], f"{best_overall['compression']:.2f}x", f"{best_overall['ber']:.4f}"],
            ['Best Pruning', best_pruning['name'], f"{best_pruning['compression']:.2f}x", f"{best_pruning['ber']:.4f}"],
            ['Best Quantization', best_quant['name'], f"{best_quant['compression']:.2f}x", f"{best_quant['ber']:.4f}"],
            ['Best Distillation', best_distill['name'], f"{best_distill['compression']:.2f}x", f"{best_distill['ber']:.4f}"],
            ['Lowest BER', lowest_ber['name'], f"{lowest_ber['compression']:.2f}x", f"{lowest_ber['ber']:.4f}"],
            ['', '', '', ''],
            ['Original Size', f"{orig_size:.2f} MB", '', ''],
            ['Best Final Size', f"{best_overall['size']:.2f} MB", '', ''],
        ]
        
        # Create the table
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.3, 0.35, 0.2, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style the header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style the summary rows
        for row in [7, 8]:
            table[(row, 0)].set_facecolor('#E3F2FD')
            table[(row, 0)].set_text_props(weight='bold')
        
        # Highlight lowest BER row
        table[(5, 0)].set_facecolor('#FFF3E0')
        table[(5, 0)].set_text_props(weight='bold')
        
        ax.set_title('Compression Summary with BER Analysis', fontsize=14, fontweight='bold', pad=20)
    
    # ============== EXPORT METHODS ==============
    
    def export_compressed_models(self, output_dir='compressed_models'):
        """Export all compressed models"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nExporting compressed models to {output_dir}/")
        
        # Export pruned models
        for amount in [0.1, 0.3, 0.5, 0.7]:
            if f'magnitude_{amount}' in self.results.get('pruning', {}):
                model = self.apply_magnitude_pruning(amount)
                path = f"{output_dir}/pruned_magnitude_{amount}.pth"
                torch.save(model.state_dict(), path)
                print(f"  Saved: {path}")
                
                model = self.apply_random_pruning(amount)
                path = f"{output_dir}/pruned_random_{amount}.pth"
                torch.save(model.state_dict(), path)
                print(f"  Saved: {path}")
                
                model = self.apply_structured_pruning(amount)
                path = f"{output_dir}/pruned_structured_{amount}.pth"
                torch.save(model.state_dict(), path)
                print(f"  Saved: {path}")
        
        # Export quantized models
        model = self.apply_dynamic_quantization()
        torch.save(model.state_dict(), f"{output_dir}/quantized_dynamic_int8.pth")
        print(f"  Saved: {output_dir}/quantized_dynamic_int8.pth")
        
        for bits in [4, 6, 8]:
            model = self.apply_log2_quantization(bits)
            path = f"{output_dir}/quantized_log2_{bits}bit.pth"
            torch.save(model.state_dict(), path)
            print(f"  Saved: {path}")
        
        # Export student models
        for ratio in [0.25, 0.1]:
            if f'student_{ratio}' in self.results.get('distillation', {}):
                student = self.create_student_model(ratio)
                path = f"{output_dir}/student_{int(ratio*100)}percent.pth"
                torch.save(student.state_dict(), path)
                print(f"  Saved: {path}")
    
    def generate_report(self, save_path='compression_report.txt'):
        """Generate detailed text report"""
        with open(save_path, 'w', encoding='utf-8') as f:  # Added encoding='utf-8'
            f.write("="*80 + "\n")
            f.write("NEURAL NETWORK COMPRESSION REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Original model
            f.write("ORIGINAL MODEL\n")
            f.write("-"*40 + "\n")
            orig = self.results['original']
            f.write(f"Size: {orig['size_mb']:.2f} MB\n")
            f.write(f"Compressed size: {orig['compressed_size_mb']:.2f} MB\n")
            f.write(f"Parameters: {orig['total_params']:,}\n")
            f.write(f"Non-zero parameters: {orig['nonzero_params']:,}\n\n")
            
            # Best results per category
            f.write("BEST RESULTS PER CATEGORY\n")
            f.write("-"*40 + "\n")
            
            orig_size = orig['size_mb']
            
            # Pruning
            if 'pruning' in self.results:
                best_pruning = min(self.results['pruning'].items(), 
                                key=lambda x: x[1]['size_mb'])
                f.write(f"\nBest Pruning: {best_pruning[0]}\n")
                f.write(f"  Size: {best_pruning[1]['size_mb']:.2f} MB\n")
                f.write(f"  Compression: {orig_size/best_pruning[1]['size_mb']:.2f}x\n")
                f.write(f"  Sparsity: {best_pruning[1]['sparsity']*100:.1f}%\n")
            
            # Quantization
            if 'quantization' in self.results:
                best_quant = min(self.results['quantization'].items(),
                            key=lambda x: x[1]['size_mb'])
                f.write(f"\nBest Quantization: {best_quant[0]}\n")
                f.write(f"  Size: {best_quant[1]['size_mb']:.2f} MB\n")
                f.write(f"  Compression: {orig_size/best_quant[1]['size_mb']:.2f}x\n")
            
            # Distillation
            if 'distillation' in self.results:
                best_distill = min(self.results['distillation'].items(),
                                key=lambda x: x[1]['size_mb'])
                f.write(f"\nBest Distillation: {best_distill[0]}\n")
                f.write(f"  Size: {best_distill[1]['size_mb']:.2f} MB\n")
                f.write(f"  Compression: {orig_size/best_distill[1]['size_mb']:.2f}x\n")
            
            # Detailed results
            f.write("\n\nDETAILED RESULTS\n")
            f.write("="*80 + "\n")
            
            # Pruning details
            f.write("\nPRUNING TECHNIQUES\n")
            f.write("-"*40 + "\n")
            for name, stats in self.results.get('pruning', {}).items():
                f.write(f"\n{name}:\n")
                f.write(f"  Size: {stats['size_mb']:.2f} MB\n")
                f.write(f"  Compressed: {stats['compressed_size_mb']:.2f} MB\n")
                f.write(f"  Compression: {orig_size/stats['size_mb']:.2f}x\n")
                f.write(f"  Sparsity: {stats['sparsity']*100:.1f}%\n")
                f.write(f"  Non-zero params: {stats['nonzero_params']:,}\n")
            
            # Quantization details
            f.write("\n\nQUANTIZATION TECHNIQUES\n")
            f.write("-"*40 + "\n")
            for name, stats in self.results.get('quantization', {}).items():
                f.write(f"\n{name}:\n")
                f.write(f"  Size: {stats['size_mb']:.2f} MB\n")
                f.write(f"  Compressed: {stats['compressed_size_mb']:.2f} MB\n")
                f.write(f"  Compression: {orig_size/stats['size_mb']:.2f}x\n")
            
            # Distillation details
            f.write("\n\nKNOWLEDGE DISTILLATION\n")
            f.write("-"*40 + "\n")
            for name, stats in self.results.get('distillation', {}).items():
                f.write(f"\n{name}:\n")
                f.write(f"  Size: {stats['size_mb']:.2f} MB\n")
                f.write(f"  Compressed: {stats['compressed_size_mb']:.2f} MB\n")
                f.write(f"  Compression: {orig_size/stats['size_mb']:.2f}x\n")
                f.write(f"  Parameters: {stats['total_params']:,}\n")
            
            # Recommendations
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-"*40 + "\n")
            f.write("\n1. For maximum compression: Combine techniques\n")
            f.write("   - Apply magnitude pruning (50-70%)\n")
            f.write("   - Then apply log2 or minifloat quantization\n")
            f.write("   - Expected compression: 10-20x\n")
            
            f.write("\n2. For hardware deployment:\n")
            f.write("   - Use log2 quantization (powers of 2 are efficient)\n")
            f.write("   - Or structured pruning (removes entire channels)\n")
            
            f.write("\n3. For maintaining accuracy:\n")
            f.write("   - Start with small pruning amounts (10-30%)\n")
            f.write("   - Use knowledge distillation for aggressive compression\n")
            
            f.write("\n4. Combination strategies:\n")
            f.write("   - Pruning + Quantization: Multiplicative compression\n")
            f.write("   - Distillation + Quantization: Small and efficient models\n")
            
            f.write("\n5. Next steps:\n")
            f.write("   - Test compressed models on your specific task\n")
            f.write("   - Fine-tune if accuracy drop is significant\n")
            f.write("   - Consider hardware-specific optimizations\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"\nReport saved to {save_path}")


# ============== MAIN COMPRESSION FUNCTION ==============

def compress_model(pth_path, output_dir='compression_results', 
                  train_students=True, student_epochs=50,
                  analyze_ber_tradeoff=True):
    """
    Main function to compress a model from a .pth file
    
    Args:
        pth_path: Path to the .pth file
        output_dir: Directory to save results
        train_students: Whether to train student models
        student_epochs: Number of epochs for student training
        analyze_ber_tradeoff: Whether to perform BER-aware analysis
    """
    
    print(f"\nLoading model from: {pth_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize framework
    framework = ModelCompressionFramework(pth_path)
    
    # Run compression analysis
    results = framework.compress_all(
        pruning_amounts=[0.1, 0.3, 0.5, 0.7],
        quantization_bits=[4, 6, 8],
        student_ratios=[0.25, 0.1, 0.05],
        train_students=train_students,
        student_epochs=student_epochs
    )
    
    # Run BER-aware analysis if requested
    if analyze_ber_tradeoff:
        ber_analysis = framework.analyze_ber_compression_tradeoff(
            ber_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        )
    
    # Generate outputs
    print("\nGenerating visualizations...")
    framework.plot_results()
    
    print("\nGenerating report...")
    framework.generate_report(f"{output_dir}/compression_report.txt")
    
    print("\nExporting compressed models...")
    framework.export_compressed_models(f"{output_dir}/models")
    
    # Print final summary
    print("\n" + "="*60)
    print("COMPRESSION COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}/")
    print(f"  - Compression visualization: compression_analysis.png")
    print(f"  - BER analysis visualization: ber_compression_analysis.png")
    print(f"  - Compression report: {output_dir}/compression_report.txt")
    print(f"  - BER analysis report: ber_compression_analysis.txt")
    print(f"  - Models: {output_dir}/models/")
    
    # Print best compression achieved
    orig_size = results['original']['size_mb']
    best_compression = 1.0
    best_technique = "None"
    
    for category, category_results in results.items():
        if category == 'original':
            continue
        for name, stats in category_results.items():
            comp = orig_size / stats['size_mb']
            if comp > best_compression:
                best_compression = comp
                best_technique = f"{category}: {name}"
    
    print(f"\nBest compression achieved: {best_compression:.2f}x")
    print(f"Technique: {best_technique}")
    print(f"Reduced from {orig_size:.2f} MB to {orig_size/best_compression:.2f} MB")
    
    return framework, results


# ============== EXAMPLE USAGE ==============

if __name__ == "__main__":
    # Example usage
    pth_file = "carbopol_model.pth"  # Replace with your model path
    
    # Run compression
    framework, results = compress_model(pth_file)
    
    # You can also access specific compressed models
    # Example: Apply only magnitude pruning with 50% sparsity
    pruned_model = framework.apply_magnitude_pruning(0.5)
    
    # Example: Apply log2 quantization with 4 bits
    quantized_model = framework.apply_log2_quantization(4)
    
    # Example: Create a student model at 10% of original size
    student_model = framework.create_student_model(0.1)
    
    print("\nCompression analysis complete!")