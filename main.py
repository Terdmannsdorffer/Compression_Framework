
"""
Neural Network Compression Framework - Main Script
Run comprehensive compression analysis on any PyTorch model
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

# Import from the actual module structure (generation folder)
try:
    from generation.main_framework import ModelCompressionFramework, compress_model
except ImportError:
    # Fallback to nn_compression if generation doesn't exist
    from nn_compression.main_framework import ModelCompressionFramework, compress_model


def main():
    parser = argparse.ArgumentParser(
        description="Neural Network Compression Framework - Compress PyTorch models using pruning, quantization, and distillation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic compression analysis
  python main.py model.pth
  
  # Quick compression targeting 5x reduction
  python main.py model.pth --quick --target-compression 5.0
  
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
    
    # Quick mode options
    parser.add_argument('--target-compression', type=float, default=5.0,
                        help='Target compression ratio for quick mode (default: 5.0)')
    parser.add_argument('--max-ber', type=float, default=0.3,
                        help='Maximum acceptable BER for quick mode (default: 0.3)')
    
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
    
    # Analysis options
    parser.add_argument('--no-ber-analysis', action='store_true',
                        help='Skip BER-aware analysis')
    parser.add_argument('--ber-thresholds', nargs='+', type=float,
                        default=[0.1, 0.2, 0.3, 0.4, 0.5],
                        help='BER thresholds for analysis (default: 0.1 0.2 0.3 0.4 0.5)')
    
    # Export options
    parser.add_argument('--export-all', action='store_true',
                        help='Export all compressed models (can be large)')
    parser.add_argument('--export-best-only', action='store_true', default=True,
                        help='Export only best models (default behavior)')
    
    # Device selection
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (default: auto)')
    
    # Verbose output
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
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
    
    print("Neural Network Compression Framework")
    print("="*50)
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output}")
    print(f"Device: {device}")
    print()
    
    try:
        if args.quick:
            # Quick compression mode
            print("Running in QUICK mode")
            print(f"Target compression: {args.target_compression}x")
            print(f"Max BER: {args.max_ber*100}%")
            
            framework, result = compress_model(
                args.model_path,
                output_dir=args.output,
                quick_mode=True,
                target_compression=args.target_compression
            )
            
            if result:
                print(f"\n✓ Quick compression successful!")
                print(f"  Best technique: {result['name']}")
                print(f"  Compression: {result['compression']:.2f}x")
                print(f"  BER: {result['ber']*100:.1f}%")
            else:
                print(f"\n✗ Could not achieve {args.target_compression}x compression with BER < {args.max_ber*100}%")
                print("Consider running full analysis with: python main.py", args.model_path)
        
        elif args.benchmark:
            # Benchmark mode
            print(f"Running BENCHMARK mode for: {', '.join(args.benchmark)}")
            
            framework = ModelCompressionFramework(args.model_path)
            results = framework.benchmark_compression(args.benchmark)
            
            # Generate basic outputs
            framework.generate_report(f"{args.output}/benchmark_report.txt")
            if args.export_best_only or args.export_all:
                framework.export_best_models_only(f"{args.output}/best_models")
            
            print(f"\n✓ Benchmark completed!")
            print(f"Results saved to: {args.output}/")
        
        else:
            # Full compression analysis
            print("Running FULL compression analysis")
            
            framework, results = compress_model(
                args.model_path,
                output_dir=args.output,
                train_students=not args.no_train_students,
                student_epochs=args.student_epochs,
                analyze_ber_tradeoff=not args.no_ber_analysis
            )
            
            # Export models based on options
            if args.export_all:
                framework.export_compressed_models(f"{args.output}/all_models")
            
            print(f"\n✓ Full compression analysis completed!")
            print(f"Check {args.output}/ for detailed results")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during compression: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def quick_demo():
    """Quick demonstration with a simple model"""
    print("Neural Network Compression Framework - Quick Demo")
    print("="*50)
    
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
    print("Created demo model: demo_model.pth")
    
    # Run compression
    try:
        framework, results = compress_model(
            'demo_model.pth',
            output_dir='demo_results',
            train_students=False,  # Skip training for demo
            student_epochs=10,
            analyze_ber_tradeoff=True
        )
        
        print("\n✓ Demo completed successfully!")
        print("Check demo_results/ folder for outputs")
        
        # Cleanup
        os.remove('demo_model.pth')
        
    except Exception as e:
        print(f"Demo failed: {str(e)}")
        # Cleanup on failure
        if os.path.exists('demo_model.pth'):
            os.remove('demo_model.pth')


if __name__ == "__main__":
    # Check if running as demo
    if len(sys.argv) == 2 and sys.argv[1] == 'demo':
        quick_demo()
    else:
        main()