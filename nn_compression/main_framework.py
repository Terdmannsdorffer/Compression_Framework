"""
Neural Network Compression Framework - Main Framework with PIV-PINN Precision
Unified interface that orchestrates all compression modules with precision-aware analysis
"""

import torch
import warnings
import os
warnings.filterwarnings('ignore')

# Import all the module classes
from .base_framework import BaseCompressionFramework
from .pruning_module import PruningModule
from .quantization_module import QuantizationModule
from .distillation_module import DistillationModule
from .stats_module import StatsModule
from .analysis_module import AnalysisModule
from .export_module import ExportModule

# Try to import precision calculator
try:
    from precision_calculator import PINNPrecisionCalculator
    PRECISION_AVAILABLE = True
except ImportError:
    PRECISION_AVAILABLE = False


class ModelCompressionFramework(BaseCompressionFramework):
    """
    Comprehensive framework for neural network compression with PIV-PINN precision analysis.
    
    Techniques included:
    - Pruning: magnitude, random, structured
    - Quantization: dynamic INT8, log2, minifloat
    - Knowledge Distillation: automatic student architecture generation
    - PIV-PINN precision-aware analysis and optimization
    """
    
    def __init__(self, model_path: str, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(model_path, device)
        
        # Initialize all modules
        self.stats = StatsModule(self)
        self.pruning = PruningModule(self)
        self.quantization = QuantizationModule(self)
        self.distillation = DistillationModule(self)
        self.analysis = AnalysisModule(self)
        self.export = ExportModule(self)
        
        print(f"Model loaded successfully from: {model_path}")
        print(f"Device: {device}")
        
        if PRECISION_AVAILABLE:
            print("🎯 PIV-PINN precision analysis enabled!")
        else:
            print("⚠️ PIV-PINN precision calculator not available - using BER estimation")

    # Delegate methods to appropriate modules
    def get_model_stats(self, model, include_performance=True):
        """Get comprehensive model statistics with precision"""
        return self.stats.get_model_stats(model, include_performance)

    def _print_stats(self, name, stats):
        """Print model statistics with precision focus"""
        return self.stats._print_stats(name, stats)

    # Pruning methods
    def apply_magnitude_pruning(self, amount=0.3):
        """Apply magnitude-based pruning"""
        return self.pruning.apply_magnitude_pruning(amount)

    def apply_random_pruning(self, amount=0.3):
        """Apply random pruning"""
        return self.pruning.apply_random_pruning(amount)

    def apply_structured_pruning(self, amount=0.3):
        """Apply structured pruning"""
        return self.pruning.apply_structured_pruning(amount)

    # Quantization methods
    def apply_dynamic_quantization(self):
        """Apply dynamic INT8 quantization"""
        return self.quantization.apply_dynamic_quantization()

    def apply_log2_quantization(self, bits=8):
        """Apply log2 quantization"""
        return self.quantization.apply_log2_quantization(bits)

    def apply_minifloat_quantization(self, exp_bits=4, mantissa_bits=3):
        """Apply minifloat quantization"""
        return self.quantization.apply_minifloat_quantization(exp_bits, mantissa_bits)

    # Knowledge distillation methods
    def create_student_model(self, compression_ratio=0.25, train_student=True, epochs=50):
        """Create and train student model"""
        return self.distillation.create_student_model(compression_ratio, train_student, epochs)

    # Analysis methods with precision focus
    def analyze_precision_compression_tradeoff(self, precision_thresholds=[50, 60, 70, 80, 90], 
                                             save_results=True):
        """Analyze precision vs compression trade-off"""
        return self.analysis.analyze_precision_compression_tradeoff(precision_thresholds, save_results)

    def analyze_ber_compression_tradeoff(self, ber_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5], 
                                        save_results=True):
        """Legacy BER analysis - use analyze_precision_compression_tradeoff instead"""
        print("⚠️ analyze_ber_compression_tradeoff is deprecated. Use analyze_precision_compression_tradeoff instead.")
        return self.analysis.analyze_ber_compression_tradeoff(ber_thresholds, save_results)

    def plot_results(self):
        """Create comprehensive visualizations with precision focus"""
        return self.analysis.plot_results()

    def plot_precision_comparison(self):
        """Create detailed precision before/after comparison plots"""
        return self.analysis.plot_precision_detailed_comparison()

    # Export methods
    def export_compressed_models(self, output_dir='compressed_models'):
        """Export all compressed models"""
        return self.export.export_compressed_models(output_dir)

    def generate_report(self, save_path='compression_report.txt'):
        """Generate detailed text report"""
        return self.export.generate_report(save_path)

    def create_model_comparison_table(self):
        """Create comparison table of all techniques"""
        return self.export.create_model_comparison_table()

    def export_best_models_only(self, output_dir='best_models', precision_threshold=70.0):
        """Export only the best models based on precision threshold"""
        return self.export.export_best_models_only(output_dir, precision_threshold)

    # Main compression pipeline
    def compress_all(self, pruning_amounts=[0.1, 0.3, 0.5, 0.7], 
                    quantization_bits=[4, 6, 8],
                    student_ratios=[0.25, 0.1],
                    train_students=True,
                    student_epochs=50):
        """Apply all compression techniques and return comprehensive results with precision analysis"""
        
        print("="*60)
        print("NEURAL NETWORK COMPRESSION ANALYSIS WITH PIV-PINN PRECISION")
        print("="*60)
        
        # Original model stats with precision
        print("\nAnalyzing original model...")
        self.results['original'] = self.get_model_stats(self.original_model)
        self._print_stats("Original Model", self.results['original'])
        
        # Apply all pruning techniques
        self.results['pruning'] = self.pruning.apply_all_pruning_techniques(pruning_amounts)
        
        # Apply all quantization techniques
        self.results['quantization'] = self.quantization.apply_all_quantization_techniques(quantization_bits)
        
        # Apply knowledge distillation
        self.results['distillation'] = self.distillation.apply_all_distillation_techniques(
            student_ratios, train_students, student_epochs)
        
        return self.results

    def quick_compress_precision(self, target_compression=5.0, min_precision=70.0):
        """
        Quick compression targeting a specific compression ratio with precision constraint
        
        Args:
            target_compression: Desired compression ratio (e.g., 5.0 for 5x compression)
            min_precision: Minimum acceptable precision (70.0 = 70%)
        """
        print(f"\nQuick compression targeting {target_compression}x with precision >= {min_precision}%")
        print("="*60)
        
        # Run basic compression techniques
        results = self.compress_all(
            pruning_amounts=[0.1, 0.3, 0.5],
            quantization_bits=[4, 8],
            student_ratios=[0.25],
            train_students=False,
            student_epochs=20
        )
        
        # Find techniques that meet criteria
        orig_size = results['original']['size_mb']
        candidates = []
        
        for category, category_results in results.items():
            if category == 'original':
                continue
            for name, stats in category_results.items():
                compression = orig_size / stats['size_mb']
                precision = stats.get('precision', 50.0)
                
                if compression >= target_compression * 0.8 and precision >= min_precision:
                    candidates.append({
                        'name': f"{category}: {name}",
                        'compression': compression,
                        'precision': precision,
                        'category': category,
                        'technique': name
                    })
        
        if candidates:
            # Sort by compression ratio
            candidates.sort(key=lambda x: x['compression'], reverse=True)
            best = candidates[0]
            
            print(f"\nBest technique found:")
            print(f"  {best['name']}")
            print(f"  Compression: {best['compression']:.2f}x")
            print(f"  Precision: {best['precision']:.1f}%")
            
            # Export the best model
            self.export_best_models_only('quick_compressed_models', min_precision)
            
            return best
        else:
            print(f"\nNo techniques found that achieve {target_compression}x compression with precision >= {min_precision}%")
            print("Try:")
            print("  - Reducing target compression ratio")
            print("  - Lowering precision constraint")
            print("  - Running full compression analysis")
            return None

    def quick_compress(self, target_compression=5.0, max_ber=0.3):
        """
        Legacy quick compress method - redirects to precision-based version
        
        Args:
            target_compression: Desired compression ratio
            max_ber: Maximum acceptable BER (converted to minimum precision)
        """
        print("⚠️ quick_compress is deprecated. Use quick_compress_precision instead.")
        # Convert BER to precision (inverse relationship)
        min_precision = max(0, 100 - max_ber*200)
        return self.quick_compress_precision(target_compression, min_precision)

    def benchmark_compression(self, techniques=['pruning', 'quantization', 'distillation']):
        """
        Benchmark specific compression techniques with precision analysis
        
        Args:
            techniques: List of techniques to benchmark ['pruning', 'quantization', 'distillation']
        """
        print(f"\nBenchmarking compression techniques with precision analysis: {techniques}")
        print("="*60)
        
        # Initialize results
        self.results['original'] = self.get_model_stats(self.original_model)
        
        # Benchmark each requested technique
        if 'pruning' in techniques:
            self.results['pruning'] = self.pruning.apply_all_pruning_techniques([0.1, 0.3, 0.5])
        
        if 'quantization' in techniques:
            self.results['quantization'] = self.quantization.apply_all_quantization_techniques([4, 8])
        
        if 'distillation' in techniques:
            self.results['distillation'] = self.distillation.apply_all_distillation_techniques([0.25], False, 10)
        
        # Create comparison table
        self.create_model_comparison_table()
        
        return self.results

    def find_optimal_precision_compression(self, min_precision=80.0):
        """
        Find the technique that achieves the best compression while maintaining minimum precision
        
        Args:
            min_precision: Minimum acceptable precision (default 80%)
        """
        if not self.results:
            print("No compression results available. Run compress_all() first.")
            return None
        
        print(f"\nFinding optimal compression technique with precision >= {min_precision}%")
        print("="*50)
        
        orig_size = self.results['original']['size_mb']
        best_technique = None
        best_compression = 1.0
        
        for category, category_results in self.results.items():
            if category == 'original':
                continue
            
            for name, stats in category_results.items():
                precision = stats.get('precision', 50.0)
                compression = orig_size / stats['size_mb']
                
                if precision >= min_precision and compression > best_compression:
                    best_compression = compression
                    best_technique = {
                        'category': category,
                        'name': name,
                        'compression': compression,
                        'precision': precision,
                        'size_mb': stats['size_mb'],
                        'stats': stats
                    }
        
        if best_technique:
            print(f"✅ Optimal technique found:")
            print(f"   Category: {best_technique['category']}")
            print(f"   Technique: {best_technique['name']}")
            print(f"   Compression: {best_technique['compression']:.2f}x")
            print(f"   Precision: {best_technique['precision']:.1f}%")
            print(f"   Final size: {best_technique['size_mb']:.2f} MB")
            print(f"   Size reduction: {(1 - best_technique['size_mb']/orig_size)*100:.1f}%")
        else:
            print(f"❌ No technique found that maintains precision >= {min_precision}%")
            print("   Consider:")
            print("   - Lowering precision requirement")
            print("   - Using less aggressive compression")
            print("   - Training with more epochs for distillation")
        
        return best_technique


# Main function for easy usage with precision focus
def compress_model(pth_path, output_dir='compression_results', 
                  train_students=True, student_epochs=50,
                  analyze_precision_tradeoff=True,
                  quick_mode=False, target_compression=5.0, min_precision=70.0):
    """
    Main function to compress a model from a .pth file with precision analysis
    
    Args:
        pth_path: Path to the .pth file
        output_dir: Directory to save results
        train_students: Whether to train student models
        student_epochs: Number of epochs for student training
        analyze_precision_tradeoff: Whether to perform precision-aware analysis
        quick_mode: If True, run quick compression targeting specific ratio
        target_compression: Target compression ratio for quick mode
        min_precision: Minimum precision requirement for quick mode
    """
    
    print(f"\nLoading model from: {pth_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize framework
    framework = ModelCompressionFramework(pth_path)
    
    if quick_mode:
        # Quick compression mode with precision
        result = framework.quick_compress_precision(
            target_compression=target_compression, 
            min_precision=min_precision
        )
        if result:
            print(f"\nQuick compression completed!")
            print(f"Results saved to: {output_dir}/")
        return framework, result
    
    # Run full compression analysis
    results = framework.compress_all(
        pruning_amounts=[0.1, 0.3, 0.5, 0.7],
        quantization_bits=[4, 6, 8],
        student_ratios=[0.25, 0.1, 0.05],
        train_students=train_students,
        student_epochs=student_epochs
    )
    
    # Run precision-aware analysis if requested
    if analyze_precision_tradeoff:
        precision_analysis = framework.analyze_precision_compression_tradeoff(
            precision_thresholds=[50, 60, 70, 80, 90]
        )
    
    # Generate outputs
    print("\nGenerating visualizations...")
    framework.plot_results()
    
    print("\nGenerating report...")
    framework.generate_report(f"{output_dir}/compression_report.txt")
    
    print("\nExporting compressed models...")
    framework.export_compressed_models(f"{output_dir}/models")
    
    print("\nExporting best models...")
    framework.export_best_models_only(f"{output_dir}/best_models", min_precision)
    
    print("\nCreating comparison table...")
    framework.create_model_comparison_table()
    
    # Print final summary with precision focus
    print("\n" + "="*60)
    print("COMPRESSION COMPLETE - PRECISION ANALYSIS")
    print("="*60)
    print(f"Results saved to: {output_dir}/")
    print(f"  - Compression visualization: compression_analysis.png")
    print(f"  - Precision analysis visualization: precision_compression_analysis.png")
    print(f"  - Compression report: {output_dir}/compression_report.txt")
    print(f"  - Precision analysis report: precision_compression_analysis.txt")
    print(f"  - All models: {output_dir}/models/")
    print(f"  - Best models: {output_dir}/best_models/")
    
    # Print best compression achieved with precision info
    orig_size = results['original']['size_mb']
    orig_precision = results['original'].get('precision', 100.0)
    best_compression = 1.0
    best_technique = "None"
    best_precision = orig_precision
    
    for category, category_results in results.items():
        if category == 'original':
            continue
        for name, stats in category_results.items():
            comp = orig_size / stats['size_mb']
            precision = stats.get('precision', 50.0)
            if comp > best_compression:
                best_compression = comp
                best_technique = f"{category}: {name}"
                best_precision = precision
    
    print(f"\nBest compression achieved: {best_compression:.2f}x")
    print(f"Technique: {best_technique}")
    print(f"Final precision: {best_precision:.1f}% (original: {orig_precision:.1f}%)")
    print(f"Precision retention: {(best_precision/orig_precision)*100:.1f}%")
    print(f"Reduced from {orig_size:.2f} MB to {orig_size/best_compression:.2f} MB")
    
    # Find best technique that maintains high precision
    optimal = framework.find_optimal_precision_compression(min_precision=80.0)
    if optimal:
        print(f"\nOptimal high-precision technique:")
        print(f"  {optimal['category']}: {optimal['name']}")
        print(f"  {optimal['compression']:.2f}x compression at {optimal['precision']:.1f}% precision")
    
    return framework, results