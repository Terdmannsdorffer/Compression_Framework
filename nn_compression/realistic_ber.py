"""
Realistic BER (Bit Error Rate) Calculation with Bit Error Injection
Simulates real-world bit errors and measures their impact
"""

import torch
import numpy as np
import random
from typing import Dict, Tuple, Optional
import struct


class RealisticBERCalculator:
    """
    Realistic BER calculation by injecting actual bit errors into model weights
    and measuring the resulting performance degradation
    """
    
    @staticmethod
    def inject_bit_errors(tensor, ber=1e-5, error_pattern='random', affected_layers=None):
        """
        Inject bit errors into tensor weights to simulate real hardware/transmission errors
        
        Args:
            tensor: Input tensor to corrupt
            ber: Bit error rate (probability of bit flip)
            error_pattern: 'random', 'burst', 'single_event', 'clustered'
            affected_layers: List of layer patterns to affect (None = all layers)
            
        Returns:
            corrupted_tensor: Tensor with injected bit errors
            error_info: Dictionary with error injection statistics
        """
        
        corrupted = tensor.clone().detach()
        original_shape = corrupted.shape
        flat_tensor = corrupted.view(-1)
        
        total_bits = flat_tensor.numel() * 32  # 32 bits per float32
        target_errors = int(total_bits * ber)
        
        if target_errors == 0:
            return corrupted, {'errors_injected': 0, 'target_ber': ber, 'actual_ber': 0.0}
        
        actual_errors = 0
        error_positions = []
        
        if error_pattern == 'random':
            actual_errors, error_positions = RealisticBERCalculator._inject_random_errors(
                flat_tensor, target_errors
            )
        elif error_pattern == 'burst':
            actual_errors, error_positions = RealisticBERCalculator._inject_burst_errors(
                flat_tensor, target_errors
            )
        elif error_pattern == 'single_event':
            actual_errors, error_positions = RealisticBERCalculator._inject_single_event_errors(
                flat_tensor, target_errors
            )
        elif error_pattern == 'clustered':
            actual_errors, error_positions = RealisticBERCalculator._inject_clustered_errors(
                flat_tensor, target_errors
            )
        else:
            raise ValueError(f"Unknown error pattern: {error_pattern}")
        
        actual_ber = actual_errors / total_bits
        
        error_info = {
            'errors_injected': actual_errors,
            'target_errors': target_errors,
            'target_ber': ber,
            'actual_ber': actual_ber,
            'total_bits': total_bits,
            'error_pattern': error_pattern,
            'error_positions': error_positions[:100]  # Store first 100 for analysis
        }
        
        return corrupted.view(original_shape), error_info
    
    @staticmethod
    def _inject_random_errors(flat_tensor, target_errors):
        """Inject randomly distributed bit errors"""
        actual_errors = 0
        error_positions = []
        
        # Generate unique random positions
        tensor_size = flat_tensor.numel()
        if target_errors >= tensor_size * 32:
            # If too many errors requested, sample without replacement
            positions = random.sample(range(tensor_size), min(target_errors, tensor_size))
            bit_positions = [random.randint(0, 31) for _ in positions]
        else:
            positions = torch.randint(0, tensor_size, (target_errors,))
            bit_positions = [random.randint(0, 31) for _ in range(target_errors)]
        
        for i, (idx, bit_pos) in enumerate(zip(positions, bit_positions)):
            try:
                original_val = flat_tensor[idx].item()
                
                # Convert float to uint32 representation
                float_bytes = struct.pack('f', original_val)
                int_val = struct.unpack('I', float_bytes)[0]
                
                # Flip the bit
                int_val ^= (1 << bit_pos)
                
                # Convert back to float
                new_bytes = struct.pack('I', int_val)
                new_float = struct.unpack('f', new_bytes)[0]
                
                # Check for NaN/Inf and skip if corrupted
                if np.isfinite(new_float):
                    flat_tensor[idx] = new_float
                    actual_errors += 1
                    error_positions.append((int(idx), bit_pos))
                
            except (OverflowError, ValueError):
                # Skip corrupted values
                continue
        
        return actual_errors, error_positions
    
    @staticmethod
    def _inject_burst_errors(flat_tensor, target_errors):
        """Inject burst errors (multiple consecutive bits)"""
        actual_errors = 0
        error_positions = []
        
        tensor_size = flat_tensor.numel()
        burst_size = min(8, target_errors)  # Burst of up to 8 bits
        num_bursts = max(1, target_errors // burst_size)
        
        for _ in range(num_bursts):
            if actual_errors >= target_errors:
                break
                
            # Random position for burst
            idx = random.randint(0, tensor_size - 1)
            start_bit = random.randint(0, 32 - burst_size)
            
            try:
                original_val = flat_tensor[idx].item()
                float_bytes = struct.pack('f', original_val)
                int_val = struct.unpack('I', float_bytes)[0]
                
                # Create burst mask
                burst_mask = ((1 << burst_size) - 1) << start_bit
                int_val ^= burst_mask
                
                new_bytes = struct.pack('I', int_val)
                new_float = struct.unpack('f', new_bytes)[0]
                
                if np.isfinite(new_float):
                    flat_tensor[idx] = new_float
                    actual_errors += burst_size
                    error_positions.extend([(int(idx), start_bit + i) for i in range(burst_size)])
                    
            except (OverflowError, ValueError):
                continue
        
        return actual_errors, error_positions
    
    @staticmethod
    def _inject_single_event_errors(flat_tensor, target_errors):
        """Inject single event upsets (SEU) - affects multiple related bits"""
        actual_errors = 0
        error_positions = []
        
        tensor_size = flat_tensor.numel()
        
        # SEU typically affects sign bit and exponent
        for _ in range(min(target_errors, tensor_size)):
            idx = random.randint(0, tensor_size - 1)
            
            try:
                original_val = flat_tensor[idx].item()
                float_bytes = struct.pack('f', original_val)
                int_val = struct.unpack('I', float_bytes)[0]
                
                # Flip sign bit (bit 31) with high probability
                if random.random() < 0.7:
                    int_val ^= (1 << 31)
                    actual_errors += 1
                    error_positions.append((int(idx), 31))
                
                # Flip one exponent bit (bits 23-30)
                if random.random() < 0.5 and actual_errors < target_errors:
                    exp_bit = random.randint(23, 30)
                    int_val ^= (1 << exp_bit)
                    actual_errors += 1
                    error_positions.append((int(idx), exp_bit))
                
                new_bytes = struct.pack('I', int_val)
                new_float = struct.unpack('f', new_bytes)[0]
                
                if np.isfinite(new_float):
                    flat_tensor[idx] = new_float
                    
            except (OverflowError, ValueError):
                continue
        
        return actual_errors, error_positions
    
    @staticmethod
    def _inject_clustered_errors(flat_tensor, target_errors):
        """Inject clustered errors (errors near each other in memory)"""
        actual_errors = 0
        error_positions = []
        
        tensor_size = flat_tensor.numel()
        cluster_size = min(10, tensor_size // 10)  # Cluster size
        num_clusters = max(1, target_errors // cluster_size)
        
        for _ in range(num_clusters):
            if actual_errors >= target_errors:
                break
                
            # Random cluster center
            center_idx = random.randint(cluster_size, tensor_size - cluster_size)
            
            # Apply errors in cluster
            for offset in range(-cluster_size//2, cluster_size//2):
                if actual_errors >= target_errors:
                    break
                    
                idx = center_idx + offset
                if 0 <= idx < tensor_size and random.random() < 0.3:  # 30% chance per position
                    bit_pos = random.randint(0, 31)
                    
                    try:
                        original_val = flat_tensor[idx].item()
                        float_bytes = struct.pack('f', original_val)
                        int_val = struct.unpack('I', float_bytes)[0]
                        
                        int_val ^= (1 << bit_pos)
                        
                        new_bytes = struct.pack('I', int_val)
                        new_float = struct.unpack('f', new_bytes)[0]
                        
                        if np.isfinite(new_float):
                            flat_tensor[idx] = new_float
                            actual_errors += 1
                            error_positions.append((int(idx), bit_pos))
                            
                    except (OverflowError, ValueError):
                        continue
        
        return actual_errors, error_positions
    
    @staticmethod
    def calculate_true_ber(original_tensor, corrupted_tensor):
        """
        Calculate actual BER by comparing bit representations of tensors
        
        Args:
            original_tensor: Original tensor
            corrupted_tensor: Tensor with injected errors
            
        Returns:
            dict: BER statistics
        """
        
        # Ensure same shape and flatten
        assert original_tensor.shape == corrupted_tensor.shape, "Tensors must have same shape"
        
        orig_flat = original_tensor.view(-1).detach().cpu()
        corr_flat = corrupted_tensor.view(-1).detach().cpu()
        
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
            'total_bits': int(total_bits),
            'byte_errors': int(np.count_nonzero(xor_bytes)),
            'total_bytes': len(orig_bytes)
        }


# Convenience functions for easy integration
def inject_realistic_bit_errors(tensor, ber=1e-5, pattern='random'):
    """Simplified interface for bit error injection"""
    corrupted, info = RealisticBERCalculator.inject_bit_errors(tensor, ber, pattern)
    return corrupted

def calculate_realistic_ber(original, corrupted):
    """Simplified interface for BER calculation"""
    return RealisticBERCalculator.calculate_true_ber(original, corrupted)['ber']