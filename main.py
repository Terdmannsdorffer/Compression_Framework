"""
Neural Network Compression Framework - Main Script
Run comprehensive compression analysis on any PyTorch model with PIV-PINN precision
"""

import argparse
import sys
import os
from pathlib import Path

# Add the package to path if running as script
if __name__ == "__main__":
    # Add current directory to path for imports
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))

# Import from the actual module structure based on your directory layout
try:
    # Method 1: Try importing from nn_compression subdirectory (your structure)
    from nn_compression.main_framework import ModelCompressionFramework, compress_model
    print("📦 Imported from nn_compression subdirectory")
except ImportError:
    try:
        # Method 2: Add nn_compression to path and import directly (what worked in helper)
        import sys
        import os
        nn_compression_path = os.path.join(os.getcwd(), "nn_compression")
        if os.path.exists(nn_compression_path) and nn_compression_path not in sys.path:
            sys.path.insert(0, nn_compression_path)
        
        from main_framework import ModelCompressionFramework, compress_model
        print("📦 Imported from nn_compression path (method that worked in helper)")
    except ImportError:
        try:
            # Method 3: Fallback to generation folder if it exists
            from generation.main_framework import ModelCompressionFramework, compress_model
            print("📦 Imported from generation subdirectory")
        except ImportError:
            try:
                # Method 4: Fallback to local import (if files are in same directory)
                from main_framework import ModelCompressionFramework, compress_model
                print("📦 Imported from local directory")
            except ImportError as e:
                print(f"❌ Error importing compression framework: {str(e)}")
                print("💡 Expected directory structure:")
                print("   C:/Users/Usuario/Desktop/Compression framework/")
                print("   ├── main.py")
                print("   └── nn_compression/")
                print("       ├── main_framework.py")
                print("       ├── precision_calculator.py")
                print("       └── [other modules]")
                print("\n🔧 Try running the helper first:")
                print("   python nn_compression/setup_piv_data.py")
                sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Neural Network Compression Framework - Compress PyTorch models with PIV-PINN precision analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic compression analysis with precision
  python main.py model.pth
  
  # Quick compression targeting 5x reduction with 80% precision
  python main.py model.pth --quick --target-compression 5.0 --min-precision 80
  
  # Custom output directory with no student training
  python main.py model.pth --output results/ --no-train-students
  
  # Benchmark specific techniques only
  python main.py model.pth --benchmark pruning quantization
  
  # Custom compression parameters
  python main.py model.pth --pruning-amounts 0.2 0.4 0.6 --quantization-bits 4 8
        """)
    
    # Required arguments
    parser.add_argument('model_path', type=str, 
                        help='Path to the PyTorch model (.pth file)')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, default='compression_results',
                        help='Output directory for results (default: compression_results)')
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--quick', action='store_true',
                           help='Quick compression mode targeting specific compression ratio')
    mode_group.add_argument('--benchmark', nargs='+', 
                           choices=['pruning', 'quantization', 'distillation'],
                           help='Benchmark specific techniques only')
    
    # Quick mode options (updated for precision)
    parser.add_argument('--target-compression', type=float, default=5.0,
                        help='Target compression ratio for quick mode (default: 5.0)')
    parser.add_argument('--min-precision', type=float, default=70.0,
                        help='Minimum acceptable precision for quick mode (default: 70.0)')
    parser.add_argument('--max-ber', type=float, default=0.3,
                        help='DEPRECATED: Use --min-precision instead. Maximum acceptable BER (default: 0.3)')
    
    # Compression technique parameters
    parser.add_argument('--pruning-amounts', nargs='+', type=float, 
                        default=[0.1, 0.3, 0.5, 0.7],
                        help='Pruning amounts to test (default: 0.1 0.3 0.5 0.7)')
    parser.add_argument('--quantization-bits', nargs='+', type=int,
                        default=[4, 6, 8],
                        help='Quantization bit widths to test (default: 4 6 8)')
    parser.add_argument('--student-ratios', nargs='+', type=float,
                        default=[0.25, 0.1, 0.05],
                        help='Student model compression ratios (default: 0.25 0.1 0.05)')
    
    # Training options
    parser.add_argument('--no-train-students', action='store_true',
                        help='Skip training student models (faster but less accurate)')
    parser.add_argument('--student-epochs', type=int, default=50,
                        help='Number of epochs for student training (default: 50)')
    
    # Analysis options (updated for precision)
    parser.add_argument('--no-precision-analysis', action='store_true',
                        help='Skip precision-aware analysis')
    parser.add_argument('--precision-thresholds', nargs='+', type=float,
                        default=[50, 60, 70, 80, 90],
                        help='Precision thresholds for analysis (default: 50 60 70 80 90)')
    parser.add_argument('--no-ber-analysis', action='store_true',
                        help='DEPRECATED: Use --no-precision-analysis instead')
    parser.add_argument('--ber-thresholds', nargs='+', type=float,
                        default=[0.1, 0.2, 0.3, 0.4, 0.5],
                        help='DEPRECATED: Use --precision-thresholds instead')
    
    # Export options
    parser.add_argument('--export-all', action='store_true',
                        help='Export all compressed models (can be large)')
    parser.add_argument('--export-best-only', action='store_true', default=True,
                        help='Export only best models (default behavior)')
    parser.add_argument('--precision-threshold-export', type=float, default=70.0,
                        help='Minimum precision for exported models (default: 70.0)')
    
    # Device selection
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (default: auto)')
    
    # Verbose output
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Handle deprecation warnings
    if args.no_ber_analysis:
        print("⚠️ Warning: --no-ber-analysis is deprecated. Use --no-precision-analysis instead.")
        args.no_precision_analysis = True
    
    if args.ber_thresholds != [0.1, 0.2, 0.3, 0.4, 0.5]:  # User specified BER thresholds
        print("⚠️ Warning: --ber-thresholds is deprecated. Converting to precision thresholds.")
        # Convert BER to precision thresholds
        args.precision_thresholds = [max(0, 100 - ber*200) for ber in args.ber_thresholds]
    
    # Handle special case for demo
    if hasattr(args, 'model_path') and args.model_path == 'demo':
        quick_demo()
        return
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        sys.exit(1)
    
    # Set device
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("Neural Network Compression Framework with PIV-PINN Precision")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output}")
    print(f"Device: {device}")
    
    # Check for PIV-PINN precision availability  
    try:
        # Use the same import strategy that worked in the helper
        nn_compression_path = os.path.join(os.getcwd(), "nn_compression")
        if os.path.exists(nn_compression_path) and nn_compression_path not in sys.path:
            sys.path.insert(0, nn_compression_path)
        
        # Import precision calculator (same way as helper)
        from precision_calculator import PINNPrecisionCalculator
        
        # Check for PIV data in multiple locations based on your structure
        possible_data_paths = [
            # From main directory (where main.py is) - these worked in helper
            "data/averaged_piv_steady_state.txt",
            "data/piv_steady_state.txt",
            
            # Absolute path to your data directory - this also worked
            r"C:\Users\Usuario\Desktop\Compression framework\data\averaged_piv_steady_state.txt",
            r"C:\Users\Usuario\Desktop\Compression framework\data\piv_steady_state.txt",
        ]
        
        found_piv_data = False
        for path in possible_data_paths:
            if os.path.exists(path):
                calculator = PINNPrecisionCalculator(path)
                if calculator.is_available():
                    print(f"🎯 PIV-PINN precision analysis: ENABLED (using {path})")
                    print(f"   PIV data points: {len(calculator.piv_data['x'])}")
                    print(f"   U velocity range: [{calculator.piv_data['u'].min():.6f}, {calculator.piv_data['u'].max():.6f}] m/s")
                    print(f"   V velocity range: [{calculator.piv_data['v'].min():.6f}, {calculator.piv_data['v'].max():.6f}] m/s")
                    found_piv_data = True
                    break
        
        if not found_piv_data:
            print("⚠️ PIV-PINN precision analysis: PIV reference data not found")
            print("   This is unexpected since the helper found the data")
            print("   Check that you're running from the same directory as the helper")
            
    except ImportError as e:
        print(f"⚠️ PIV-PINN precision analysis: Import error - {str(e)}")
        print("   Using BER estimates instead of PIV precision")
        print("   Run: python nn_compression/setup_piv_data.py to debug")
    except Exception as e:
        print(f"⚠️ PIV-PINN precision analysis: Unexpected error - {str(e)}")
        print("   Using BER estimates instead of PIV precision")
    
    print()
    
    try:
        if args.quick:
            # Quick compression mode with precision
            print("Running in QUICK PRECISION mode")
            print(f"Target compression: {args.target_compression}x")
            print(f"Minimum precision: {args.min_precision}%")
            
            framework, result = compress_model(
                args.model_path,
                output_dir=args.output,
                quick_mode=True,
                target_compression=args.target_compression,
                min_precision=args.min_precision  # Updated parameter name
            )
            
            if result:
                print(f"\n✅ Quick compression successful!")
                print(f"  Best technique: {result['name']}")
                print(f"  Compression: {result['compression']:.2f}x")
                print(f"  Precision: {result['precision']:.1f}%")
            else:
                print(f"\n❌ Could not achieve {args.target_compression}x compression with precision >= {args.min_precision}%")
                print("Consider:")
                print("  - Lowering target compression ratio")
                print("  - Reducing minimum precision requirement")
                print("  - Running full analysis with: python main.py", args.model_path)
        
        elif args.benchmark:
            # Benchmark mode
            print(f"Running BENCHMARK mode for: {', '.join(args.benchmark)}")
            
            framework = ModelCompressionFramework(args.model_path)
            results = framework.benchmark_compression(args.benchmark)
            
            # Generate basic outputs
            framework.generate_report(f"{args.output}/benchmark_report.txt")
            if args.export_best_only or args.export_all:
                framework.export_best_models_only(f"{args.output}/best_models", 
                                                args.precision_threshold_export)
            
            print(f"\n✅ Benchmark completed!")
            print(f"Results saved to: {args.output}/")
        
        else:
            # Full compression analysis with precision
            print("Running FULL PRECISION-AWARE compression analysis")
            
            framework, results = compress_model(
                pth_path=args.model_path,
                output_dir=args.output,
                train_students=not args.no_train_students,
                student_epochs=args.student_epochs,
                analyze_precision_tradeoff=not args.no_precision_analysis,  # Updated parameter name
                min_precision=args.precision_threshold_export
            )
            
            # Run precision analysis if not disabled
            if not args.no_precision_analysis:
                print("\n🎯 Running precision trade-off analysis...")
                precision_analysis = framework.analyze_precision_compression_tradeoff(
                    precision_thresholds=args.precision_thresholds,
                    save_results=True
                )
            
            # Export models based on options
            if args.export_all:
                framework.export_compressed_models(f"{args.output}/all_models")
            
            # Find and display optimal techniques
            optimal_high_precision = framework.find_optimal_precision_compression(min_precision=90.0)
            optimal_balanced = framework.find_optimal_precision_compression(min_precision=75.0)
            
            print(f"\n✅ Full compression analysis completed!")
            print(f"📊 Check {args.output}/ for detailed results")
            
            if optimal_high_precision:
                print(f"\n🏆 Best high-precision technique (≥90%):")
                print(f"   {optimal_high_precision['category']}: {optimal_high_precision['name']}")
                print(f"   {optimal_high_precision['compression']:.2f}x compression, {optimal_high_precision['precision']:.1f}% precision")
            
            if optimal_balanced:
                print(f"\n⚖️ Best balanced technique (≥75%):")
                print(f"   {optimal_balanced['category']}: {optimal_balanced['name']}")
                print(f"   {optimal_balanced['compression']:.2f}x compression, {optimal_balanced['precision']:.1f}% precision")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during compression: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        # Provide helpful suggestions
        print("\n🔧 Troubleshooting suggestions:")
        print("  1. Ensure your model file is valid and loadable")
        print("  2. Check that you have sufficient GPU memory")
        print("  3. Try running with --device cpu if GPU issues")
        print("  4. Use --verbose for detailed error information")
        print("  5. For PIV-PINN models, ensure PIV reference data exists")
        
        sys.exit(1)


def quick_demo():
    """Quick demonstration with a simple model and precision analysis"""
    print("Neural Network Compression Framework - Quick Demo with Precision")
    print("="*60)
    
    # Create a simple demo model
    import torch
    import torch.nn as nn
    
    class DemoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(10, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Save demo model
    demo_model = DemoModel()
    torch.save(demo_model.state_dict(), 'demo_model.pth')
    print("📝 Created demo model: demo_model.pth")
    
    # Run compression with precision analysis
    try:
        framework, results = compress_model(
            pth_path='demo_model.pth',
            output_dir='demo_results',
            train_students=False,  # Skip training for demo
            student_epochs=10,
            analyze_precision_tradeoff=True,
            min_precision=50.0  # Lower threshold for demo
        )
        
        print("\n✅ Demo completed successfully!")
        print("📁 Check demo_results/ folder for outputs:")
        print("   - compression_analysis.png")
        print("   - precision_compression_analysis.png") 
        print("   - precision_compression_analysis.txt")
        
        # Show quick results summary
        if results:
            orig_precision = results['original'].get('precision', 100.0)
            print(f"\n📊 Quick Results Summary:")
            print(f"   Original model precision: {orig_precision:.1f}%")
            
            # Find best technique
            best_comp = 1.0
            best_technique = "None"
            best_precision = orig_precision
            
            for category, category_results in results.items():
                if category == 'original':
                    continue
                for name, stats in category_results.items():
                    comp = results['original']['size_mb'] / stats['size_mb']
                    precision = stats.get('precision', 50.0)
                    if comp > best_comp:
                        best_comp = comp
                        best_technique = f"{category}: {name}"
                        best_precision = precision
            
            print(f"   Best compression: {best_comp:.2f}x ({best_technique})")
            print(f"   Resulting precision: {best_precision:.1f}%")
        
        # Cleanup
        os.remove('demo_model.pth')
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        print("This might be normal if PIV-PINN precision calculator is not available")
        print("The demo still shows the framework structure and basic compression")
        
        # Cleanup on failure
        if os.path.exists('demo_model.pth'):
            os.remove('demo_model.pth')


def print_help_precision():
    """Print additional help about precision analysis"""
    print("\n" + "="*60)
    print("PIV-PINN PRECISION ANALYSIS HELP")
    print("="*60)
    print("""
This framework now includes PIV-PINN precision analysis that compares
compressed model predictions with experimental PIV (Particle Image Velocimetry) data.

PRECISION METRICS:
  - 100%: Perfect match with experimental PIV data
  - 90%+: Excellent precision, minimal loss in fluid dynamics accuracy
  - 70-90%: Good precision, acceptable for most CFD applications
  - 50-70%: Moderate precision, suitable for size-critical applications
  - <50%: Poor precision, significant accuracy loss

SETUP FOR PIV-PINN PRECISION:
  1. Ensure precision_calculator.py is in your Python path
  2. Create PIV reference data: python averaged_piv_debug.py
  3. This creates averaged_piv_steady_state.txt for comparison

PRECISION-FOCUSED COMMANDS:
  # Target high precision
  python main.py model.pth --min-precision 85

  # Quick compression with precision constraint
  python main.py model.pth --quick --target-compression 3.0 --min-precision 80

  # Custom precision thresholds for analysis
  python main.py model.pth --precision-thresholds 60 70 80 85 90 95

  # Export only high-precision models
  python main.py model.pth --precision-threshold-export 85

FALLBACK BEHAVIOR:
  If PIV data is not available, the framework automatically converts
  BER (Bit Error Rate) estimates to approximate precision values.

For more information, see the precision integration documentation.
    """)


if __name__ == "__main__":
    # Check if running as demo
    if len(sys.argv) == 2 and sys.argv[1] == 'demo':
        quick_demo()
    elif len(sys.argv) == 2 and sys.argv[1] in ['help-precision', '--help-precision']:
        print_help_precision()
    else:
        main()