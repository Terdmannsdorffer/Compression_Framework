"""
Neural Network Compression Framework - Pruning Module
Implementations of different pruning techniques
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy


class PruningModule:
    """Module containing all pruning techniques."""
    
    def __init__(self, base_framework):
        self.base = base_framework

    def apply_magnitude_pruning(self, amount=0.3):
        """Prune weights with smallest magnitude"""
        model = copy.deepcopy(self.base.original_model)
        
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
        model = copy.deepcopy(self.base.original_model)
        
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
        model = copy.deepcopy(self.base.original_model)
        
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

    def apply_all_pruning_techniques(self, amounts=[0.1, 0.3, 0.5, 0.7]):
        """Apply all pruning techniques with different amounts"""
        results = {}
        
        print("\n" + "="*60)
        print("PRUNING TECHNIQUES")
        print("="*60)
        
        for amount in amounts:
            print(f"\nPruning with amount={amount}")
            
            # Apply each pruning method
            for method_name, method_func in [
                ('magnitude', self.apply_magnitude_pruning),
                ('random', self.apply_random_pruning),
                ('structured', self.apply_structured_pruning)
            ]:
                model = method_func(amount)
                stats = self.base.get_model_stats(model)
                results[f'{method_name}_{amount}'] = stats
                self.base._print_stats(f"  {method_name.title()} Pruning", stats)
        
        return results