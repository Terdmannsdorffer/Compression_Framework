
"""
debug_precision.py

Debug script to understand why PIV-PINN precision is always 20%
"""

import sys
import os
import torch
import numpy as np

# Add nn_compression to path
nn_compression_path = os.path.join(os.getcwd(), "nn_compression")
if os.path.exists(nn_compression_path) and nn_compression_path not in sys.path:
    sys.path.insert(0, nn_compression_path)

def debug_piv_data():
    """Debug PIV data loading"""
    print("🔍 Debugging PIV data loading...")
    
    try:
        from precision_calculator import PINNPrecisionCalculator
        
        calculator = PINNPrecisionCalculator()
        
        if calculator.is_available():
            print("✅ PIV data loaded successfully")
            print(f"   Points: {len(calculator.piv_data['x'])}")
            print(f"   X range: [{calculator.piv_data['x'].min():.6f}, {calculator.piv_data['x'].max():.6f}]")
            print(f"   Y range: [{calculator.piv_data['y'].min():.6f}, {calculator.piv_data['y'].max():.6f}]")
            print(f"   U range: [{calculator.piv_data['u'].min():.6f}, {calculator.piv_data['u'].max():.6f}]")
            print(f"   V range: [{calculator.piv_data['v'].min():.6f}, {calculator.piv_data['v'].max():.6f}]")
            
            return calculator
        else:
            print("❌ PIV data not available")
            return None
            
    except Exception as e:
        print(f"❌ Error loading PIV data: {str(e)}")
        return None

def debug_model_loading():
    """Debug model loading"""
    print("\n🔍 Debugging model loading...")
    
    model_path = "carbopol_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    try:
        # Try to load the model state dict
        state_dict = torch.load(model_path, map_location='cpu')
        print("✅ Model state dict loaded successfully")
        print(f"   Keys: {len(state_dict)} layers")
        
        # Show some layer info
        for i, (key, tensor) in enumerate(list(state_dict.items())[:5]):
            print(f"   {key}: {tensor.shape}")
        
        if len(state_dict) > 5:
            print(f"   ... and {len(state_dict) - 5} more layers")
        
        return state_dict
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None

def debug_model_architecture():
    """Try to understand the model architecture"""
    print("\n🔍 Debugging model architecture...")
    
    # We need to create a model to load the state dict
    # Let's try to infer the architecture from the state dict
    
    model_path = "carbopol_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Analyze the architecture
        print("📊 Model architecture analysis:")
        
        # Look for patterns in layer names
        linear_layers = [k for k in state_dict.keys() if 'weight' in k and len(state_dict[k].shape) == 2]
        bias_layers = [k for k in state_dict.keys() if 'bias' in k]
        
        print(f"   Linear layers: {len(linear_layers)}")
        print(f"   Bias layers: {len(bias_layers)}")
        
        # Show input/output dimensions
        if linear_layers:
            first_layer = linear_layers[0]
            last_layer = linear_layers[-1]
            
            input_dim = state_dict[first_layer].shape[1]
            output_dim = state_dict[last_layer].shape[0]
            
            print(f"   Input dimension: {input_dim}")
            print(f"   Output dimension: {output_dim}")
            
            print(f"   First layer: {first_layer} -> {state_dict[first_layer].shape}")
            print(f"   Last layer: {last_layer} -> {state_dict[last_layer].shape}")
            
            # Expected for PINN: input=2 (x,y), output=3 (u,v,p) or similar
            if input_dim == 2 and output_dim >= 2:
                print("✅ Architecture looks like a PINN (2D input, multi-output)")
                return True
            else:
                print(f"⚠️ Unexpected architecture for PINN")
                return False
        
        return False
        
    except Exception as e:
        print(f"❌ Error analyzing architecture: {str(e)}")
        return None

def create_simple_test_model():
    """Create a simple test model that matches expected PINN architecture"""
    print("\n🔧 Creating simple test model...")
    
    import torch.nn as nn
    
    class SimplePINN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(2, 64),  # x, y input
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 3)   # u, v, p output
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = SimplePINN()
    
    # Test with some sample data
    test_input = torch.randn(10, 2)  # 10 points, 2D coordinates
    
    try:
        output = model(test_input)
        print(f"✅ Test model created successfully")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output sample: {output[0].detach().numpy()}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error with test model: {str(e)}")
        return None

def test_precision_calculation():
    """Test precision calculation with simple model"""
    print("\n🧪 Testing precision calculation...")
    
    # Load PIV calculator
    calculator = debug_piv_data()
    if not calculator:
        return
    
    # Create simple test model
    test_model = create_simple_test_model()
    if not test_model:
        return
    
    device = torch.device('cpu')
    
    try:
        # Test precision calculation
        print("🎯 Testing precision calculation...")
        precision = calculator.calculate_precision(test_model, device)
        
        print(f"✅ Precision calculation completed: {precision:.1f}%")
        
        if precision == 20.0:
            print("⚠️ Got minimum precision value (20%) - something is failing")
            print("   This suggests an error in the calculation process")
        elif 20 < precision < 95:
            print("✅ Got reasonable precision value")
        else:
            print("⚠️ Got unexpected precision value")
        
    except Exception as e:
        print(f"❌ Error in precision calculation: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main debugging function"""
    print("🔍 PIV-PINN Precision Debugging")
    print("="*50)
    
    # Debug steps
    print("Current directory:", os.getcwd())
    
    # 1. Debug PIV data
    calculator = debug_piv_data()
    
    # 2. Debug model loading
    state_dict = debug_model_loading()
    
    # 3. Debug model architecture
    arch_ok = debug_model_architecture()
    
    # 4. Test precision calculation
    if calculator:
        test_precision_calculation()
    
    print("\n" + "="*50)
    print("DEBUGGING SUMMARY")
    print("="*50)
    
    if calculator and state_dict and arch_ok:
        print("✅ All components seem to be working")
        print("   The issue might be in the precision calculation logic")
        print("   or in the model-PIV data compatibility")
    else:
        print("❌ Found issues in the debugging process")
        print("   Check the errors above")
    
    print("\n💡 Next steps:")
    print("   1. If you see '20%' precision, the calculation is failing")
    print("   2. Check if your model architecture matches PINN expectations")
    print("   3. Verify PIV data coordinates match model domain")

if __name__ == "__main__":
    main()