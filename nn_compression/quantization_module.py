"""
Neural Network Compression Framework - Quantization Module
Implementations of different quantization techniques
"""

import torch
import torch.nn as nn
import torch.quantization as quant
import numpy as np
import copy


class QuantizationModule:
    """Module containing all quantization techniques."""
    
    def __init__(self, base_framework):
        self.base = base_framework

    def apply_dynamic_quantization(self):
        """Apply INT8 dynamic quantization"""
        model = copy.deepcopy(self.base.original_model)
        
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
        model = copy.deepcopy(self.base.original_model)
        
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
        
        model._quantized = True
        model._quant_bits = bits
        return model

    def apply_minifloat_quantization(self, exp_bits=4, mantissa_bits=3):
        """Apply custom floating-point quantization"""
        model = copy.deepcopy(self.base.original_model)
        
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

    def apply_all_quantization_techniques(self, quantization_bits=[4, 6, 8]):
        """Apply all quantization techniques"""
        results = {}
        
        print("\n" + "="*60)
        print("QUANTIZATION TECHNIQUES")
        print("="*60)
        
        # Dynamic INT8
        print("\nDynamic INT8 Quantization")
        model = self.apply_dynamic_quantization()
        stats = self.base.get_model_stats(model)
        results['dynamic_int8'] = stats
        self.base._print_stats("  Dynamic INT8", stats)
        
        # Log2 quantization
        for bits in quantization_bits:
            print(f"\nLog2 Quantization ({bits}-bit)")
            model = self.apply_log2_quantization(bits)
            stats = self.base.get_model_stats(model)
            results[f'log2_{bits}bit'] = stats
            self.base._print_stats(f"  Log2 {bits}-bit", stats)
        
        # Minifloat quantization
        minifloat_configs = [(4, 3), (5, 2), (3, 4)]
        for exp_bits, mantissa_bits in minifloat_configs:
            print(f"\nMinifloat E{exp_bits}M{mantissa_bits}")
            model = self.apply_minifloat_quantization(exp_bits, mantissa_bits)
            stats = self.base.get_model_stats(model)
            results[f'minifloat_E{exp_bits}M{mantissa_bits}'] = stats
            self.base._print_stats(f"  Minifloat E{exp_bits}M{mantissa_bits}", stats)
        
        return results