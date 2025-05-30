"""
Neural Network Compression Framework
A comprehensive toolkit for compressing neural networks using pruning, quantization, and knowledge distillation
"""

# Import the main framework class and compress function
from .main_framework import ModelCompressionFramework, compress_model

# Import base framework if needed directly
from .base_framework import BaseCompressionFramework

# Import individual modules if needed
from .pruning_module import PruningModule
from .quantization_module import QuantizationModule
from .distillation_module import DistillationModule
from .stats_module import StatsModule
from .analysis_module import AnalysisModule
from .export_module import ExportModule

__version__ = "1.0.0"
__author__ = "Neural Compression Framework Team"

__all__ = [
    'ModelCompressionFramework',
    'compress_model',
    'BaseCompressionFramework',
    'PruningModule',
    'QuantizationModule', 
    'DistillationModule',
    'StatsModule',
    'AnalysisModule',
    'ExportModule'
]

# Quick usage examples in docstring
"""
Basic usage:
    from nn_compression import compress_model
    
    # Full compression analysis
    framework, results = compress_model('model.pth')
    
    # Quick compression targeting 5x reduction
    framework, result = compress_model('model.pth', quick_mode=True, target_compression=5.0)

Advanced usage:
    from nn_compression import ModelCompressionFramework
    
    framework = ModelCompressionFramework('model.pth')
    
    # Individual techniques
    pruned_model = framework.apply_magnitude_pruning(0.5)
    quantized_model = framework.apply_log2_quantization(4)
    student_model = framework.create_student_model(0.25)
    
    # Full analysis
    results = framework.compress_all()
    framework.analyze_ber_compression_tradeoff()
    framework.plot_results()
    framework.export_compressed_models()
"""