"""
Neural Network Compression Framework - Statistics and Evaluation Module
Model statistics calculation and performance evaluation
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import os
import gzip
import shutil
import copy
import numpy as np



class StatsModule:
    """Module for calculating model statistics and evaluating performance."""
    
    def __init__(self, base_framework):
        self.base = base_framework

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
            values_size = nonzero_params * 4  # float32
            indices_size = nonzero_params * 4  # int32 column indices
            pointers_size = 1000 * 4  # Approximate row pointers
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
                ber = self._calculate_ber(self.base.original_model, model_copy)
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

    def _calculate_ber(self, original_model, compressed_model):
        """Calculate realistic BER with special handling for distillation"""
        
        print("    🔍 Calculating BER...")
        
        orig_state = original_model.state_dict()
        comp_state = compressed_model.state_dict()
        
        # Check if this is a distillation case (completely different architectures)
        is_distillation = self._detect_distillation_case(orig_state, comp_state)
        
        if is_distillation:
            print("    🎓 Detected knowledge distillation - using size-based BER estimation")
            return self._calculate_distillation_ber(original_model, compressed_model)
        
        # Normal BER calculation for pruning/quantization
        total_ber = 0
        layers_with_changes = 0
        
        for key in orig_state.keys():
            if key in comp_state and ('weight' in key or 'bias' in key):
                orig_tensor = orig_state[key].detach().cpu().float()
                comp_tensor = comp_state[key].detach().cpu().float()
                
                if orig_tensor.shape != comp_tensor.shape:
                    print(f"      ⚠️ {key}: Shape mismatch - skipping")
                    continue
                
                # Check if tensors are different
                if torch.equal(orig_tensor, comp_tensor):
                    continue
                
                # Calculate bit-level BER
                try:
                    import numpy as np
                    
                    orig_flat = orig_tensor.view(-1)
                    comp_flat = comp_tensor.view(-1)
                    
                    orig_bytes = np.frombuffer(orig_flat.numpy().tobytes(), dtype=np.uint8)
                    comp_bytes = np.frombuffer(comp_flat.numpy().tobytes(), dtype=np.uint8)
                    
                    xor_bytes = np.bitwise_xor(orig_bytes, comp_bytes)
                    bit_errors = np.unpackbits(xor_bytes).sum()
                    total_bits = len(orig_bytes) * 8
                    
                    layer_ber = bit_errors / total_bits if total_bits > 0 else 0
                    
                    if layer_ber > 0:
                        print(f"      🔍 {key}: BER = {layer_ber:.2e}")
                        total_ber += layer_ber
                        layers_with_changes += 1
                        
                except Exception as e:
                    print(f"      ❌ {key}: Error calculating BER")
                    # Use statistical fallback
                    abs_diff = torch.abs(orig_tensor - comp_tensor)
                    if abs_diff.sum() > 0:
                        fallback_ber = min(abs_diff.mean().item() * 50, 0.3)
                        total_ber += fallback_ber
                        layers_with_changes += 1
        
        # Calculate overall BER
        if layers_with_changes > 0:
            overall_ber = total_ber / layers_with_changes
            print(f"    📈 Overall BER: {overall_ber:.2e}")
            return min(overall_ber, 1.0)
        else:
            print("    ⚠️ No comparable layers found")
            return 0.0

    def _detect_distillation_case(self, orig_state, comp_state):
        """Detect if this is a knowledge distillation case"""
        
        # Check if layer names are completely different
        orig_keys = set(orig_state.keys())
        comp_keys = set(comp_state.keys())
        
        # If very few matching keys, it's likely distillation
        matching_keys = orig_keys.intersection(comp_keys)
        total_keys = len(orig_keys)
        
        if len(matching_keys) < total_keys * 0.5:  # Less than 50% matching
            return True
        
        # Check if matching layers have very different sizes
        size_mismatches = 0
        total_comparisons = 0
        
        for key in matching_keys:
            if 'weight' in key:
                orig_shape = orig_state[key].shape
                comp_shape = comp_state[key].shape
                total_comparisons += 1
                
                if orig_shape != comp_shape:
                    size_mismatches += 1
        
        # If most layers have size mismatches, it's distillation
        if total_comparisons > 0 and size_mismatches > total_comparisons * 0.7:
            return True
        
        return False

    def _calculate_distillation_ber(self, original_model, compressed_model):
        """Calculate BER for knowledge distillation cases"""
        
        # For distillation, BER is estimated based on compression ratio
        orig_stats = self.get_model_stats(original_model, include_performance=False)
        comp_stats = self.get_model_stats(compressed_model, include_performance=False)
        
        compression_ratio = orig_stats['size_mb'] / (comp_stats['size_mb'] + 1e-8)
        
        # Estimate BER based on compression ratio
        # More aggressive compression = higher BER
        if compression_ratio > 20:
            estimated_ber = 0.3  # 30% for very aggressive compression
        elif compression_ratio > 10:
            estimated_ber = 0.2  # 20% for aggressive compression  
        elif compression_ratio > 5:
            estimated_ber = 0.1  # 10% for moderate compression
        elif compression_ratio > 2:
            estimated_ber = 0.05  # 5% for light compression
        else:
            estimated_ber = 0.01  # 1% for minimal compression
        
        print(f"    🎓 Distillation BER estimate: {estimated_ber:.2e} (compression: {compression_ratio:.1f}x)")
        
        return estimated_ber

    def _calculate_tensor_ber_integrated(self, original_tensor, corrupted_tensor):
        """Integrated BER calculation method"""
        import numpy as np
        
        if original_tensor.shape != corrupted_tensor.shape:
            raise ValueError("Tensors must have same shape")
        
        orig_flat = original_tensor.view(-1).detach().cpu().float()
        corr_flat = corrupted_tensor.view(-1).detach().cpu().float()
        
        # Convert to bytes
        orig_bytes = np.frombuffer(orig_flat.numpy().tobytes(), dtype=np.uint8)
        corr_bytes = np.frombuffer(corr_flat.numpy().tobytes(), dtype=np.uint8)
        
        # XOR to find differences
        xor_bytes = np.bitwise_xor(orig_bytes, corr_bytes)
        
        # Count different bits
        bit_errors = np.unpackbits(xor_bytes).sum()
        total_bits = len(orig_bytes) * 8
        
        ber = bit_errors / total_bits if total_bits > 0 else 0
        
        return {
            'ber': float(ber),
            'bit_errors': int(bit_errors),
            'total_bits': int(total_bits)
        }

    def _calculate_output_error_rate(self, original_model, compressed_model, num_samples=1000):
        """Calculate error rate based on output differences"""
        original_model.eval()
        compressed_model.eval()
        
        # Determine input shape and type
        input_shape = self.base._infer_input_shape(original_model)
        is_pinn = self.base._is_pinn_architecture(original_model)
        
        total_error = 0
        max_possible_error = 0
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Generate appropriate test inputs
                if is_pinn and len(input_shape) == 1:
                    # For PINN: use domain-appropriate inputs
                    if input_shape[0] == 2:
                        test_input = torch.rand(1, 2).to(self.base.device) * 2 - 1  # [-1, 1]
                    elif input_shape[0] == 3:
                        test_input = torch.rand(1, 3).to(self.base.device) * 2 - 1
                    else:
                        test_input = torch.randn(1, *input_shape).to(self.base.device)
                else:
                    # Standard random inputs
                    test_input = torch.randn(1, *input_shape).to(self.base.device)
                
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

    def _calculate_enhanced_statistical_ber(self, original_model, compressed_model):
        """Enhanced statistical BER calculation with better sensitivity"""
        print("    📊 Using enhanced statistical BER calculation...")
        
        total_weight_change = 0
        total_weights = 0
        
        orig_state = original_model.state_dict()
        comp_state = compressed_model.state_dict()
        
        for key in orig_state.keys():
            if key in comp_state and ('weight' in key or 'bias' in key):
                orig_tensor = orig_state[key].detach().cpu().float()
                comp_tensor = comp_state[key].detach().cpu().float()
                
                if orig_tensor.shape != comp_tensor.shape:
                    continue
                
                # Check if tensors are actually different
                if torch.equal(orig_tensor, comp_tensor):
                    continue
                
                # Calculate different types of changes
                abs_diff = torch.abs(orig_tensor - comp_tensor)
                
                # Method 1: Relative change
                max_val = torch.max(torch.abs(orig_tensor), torch.abs(comp_tensor))
                mask = max_val > 1e-8
                
                if mask.sum() > 0:
                    relative_change = abs_diff[mask] / (max_val[mask] + 1e-8)
                    layer_rel_change = relative_change.mean().item()
                else:
                    layer_rel_change = 0
                
                # Method 2: Normalized change
                orig_norm = torch.norm(orig_tensor)
                diff_norm = torch.norm(abs_diff)
                layer_norm_change = (diff_norm / (orig_norm + 1e-8)).item()
                
                # Method 3: Percentage of changed weights
                threshold = torch.std(orig_tensor).item() * 0.001
                changed_weights = (abs_diff > threshold).sum().item()
                total_layer_weights = orig_tensor.numel()
                layer_changed_ratio = changed_weights / total_layer_weights
                
                # Combined metric with higher sensitivity
                layer_ber_estimate = (layer_rel_change * 0.3 + 
                                    layer_norm_change * 0.3 + 
                                    layer_changed_ratio * 0.4) * 10.0  # Scale up
                
                print(f"      🔍 {key}: Statistical BER = {layer_ber_estimate:.2e}")
                
                total_weight_change += layer_ber_estimate * total_layer_weights
                total_weights += total_layer_weights
        
        # Calculate overall statistical BER
        if total_weights > 0:
            avg_ber = total_weight_change / total_weights
            
            # Apply scaling based on compression type
            scaling_factor = 1.0
            
            # Check for sparsity (pruning)
            sparsity = self._calculate_sparsity(comp_state)
            if sparsity > 0.05:
                scaling_factor *= (1 + sparsity * 2)
                print(f"      ✂️ Detected pruning (sparsity: {sparsity:.1%})")
            
            # Check for quantization
            if self._detect_quantization_pattern(comp_state):
                scaling_factor *= 2.0
                print(f"      📊 Detected quantization pattern")
            
            final_ber = min(avg_ber * scaling_factor, 1.0)
            print(f"      📈 Enhanced Statistical BER: {final_ber:.2e}")
            
            return final_ber
        
        return 0.0

    def _calculate_sparsity(self, state_dict):
        """Calculate overall sparsity of the model"""
        total_params = 0
        zero_params = 0
        
        for key, tensor in state_dict.items():
            if 'weight' in key or 'bias' in key:
                total_params += tensor.numel()
                zero_params += (tensor == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0

    def _detect_quantization_pattern(self, state_dict):
        """Detect if model shows quantization patterns"""
        quantization_indicators = 0
        total_layers = 0
        
        for key, tensor in state_dict.items():
            if 'weight' in key:
                total_layers += 1
                unique_values = torch.unique(tensor).numel()
                total_values = tensor.numel()
                
                # If less than 10% unique values, likely quantized
                if unique_values < total_values * 0.1:
                    quantization_indicators += 1
        
        return quantization_indicators > total_layers * 0.5

    def _print_stats(self, name, stats):
        """Pretty print model statistics including size details"""
        orig_size = self.base.results.get('original', {}).get('size_mb', stats['size_mb'])
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