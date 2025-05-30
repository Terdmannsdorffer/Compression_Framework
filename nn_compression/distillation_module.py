"""
Neural Network Compression Framework - Knowledge Distillation Module
Implementation of knowledge distillation with automatic student generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy


class DistillationModule:
    """Module containing knowledge distillation techniques."""
    
    def __init__(self, base_framework):
        self.base = base_framework

    def create_student_model(self, compression_ratio=0.25, train_student=True, epochs=50):
        """Create and optionally train a smaller student model"""
        
        # Analyze if this is a PINN-like architecture
        is_pinn = self.base._is_pinn_architecture(self.base.original_model)
        
        if is_pinn:
            # Create PINN-style student
            student = self._create_pinn_student(self.base.original_model, compression_ratio)
        else:
            # Use original CNN-based student creation
            teacher_stats = self.base._analyze_architecture(self.base.original_model)
            student = self._create_cnn_student(teacher_stats, compression_ratio)
        
        student.to(self.base.device)
        
        # Print initial comparison
        teacher_size = self.base.get_model_stats(self.base.original_model)['size_mb']
        student_size = self.base.get_model_stats(student)['size_mb']
        
        print(f"\nStudent Model Created:")
        print(f"  Teacher size: {teacher_size:.2f} MB")
        print(f"  Student size: {student_size:.2f} MB") 
        print(f"  Actual compression: {student_size/teacher_size:.2%}")
        
        # Train the student if requested
        if train_student:
            print(f"  Training student for {epochs} epochs...")
            student = self._train_student_self_distillation(
                self.base.original_model, student, epochs=epochs
            )
            print("  Training completed!")
        
        return student

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
                test_input = torch.randn(1, 2).to(self.base.device)  # Start with 2D
                try:
                    test_output = model(test_input)
                    actual_output_dim = test_output.shape[-1]
                    config['output_dim'] = actual_output_dim
                    print(f"    Detected output dimension from forward pass: {actual_output_dim}")
                except:
                    # Try 3D input
                    test_input = torch.randn(1, 3).to(self.base.device)
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
        """Create CNN-based student model for standard architectures"""
        
        class StudentModel(nn.Module):
            def __init__(self, teacher_stats, compression_ratio):
                super().__init__()
                self.features = nn.Sequential()
                self.classifier = nn.Sequential()
                
                # Scale down the architecture
                scale_factor = np.sqrt(compression_ratio)
                
                # Build feature extractor if conv layers exist
                if teacher_stats['conv_layers']:
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
                else:
                    # Fully connected network
                    fc_input = teacher_stats['fc_layers'][0][1]['in_features'] if teacher_stats['fc_layers'] else 128
                
                # Build classifier
                if teacher_stats['fc_layers']:
                    hidden_size = max(32, int(teacher_stats['fc_layers'][0][1]['out_features'] * scale_factor))
                    num_classes = teacher_stats['fc_layers'][-1][1]['out_features']
                    
                    self.classifier.add_module('fc1', nn.Linear(fc_input, hidden_size))
                    self.classifier.add_module('relu', nn.ReLU())
                    self.classifier.add_module('dropout', nn.Dropout(0.2))
                    self.classifier.add_module('fc2', nn.Linear(hidden_size, num_classes))
                else:
                    # Default classifier
                    self.classifier.add_module('fc', nn.Linear(fc_input, 10))
            
            def forward(self, x):
                if len(self.features) > 0:
                    x = self.features(x)
                    x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        return StudentModel(teacher_stats, compression_ratio)

    def _train_student_self_distillation(self, teacher, student, epochs=50, batch_size=32, 
                                        learning_rate=0.001, temperature=4.0):
        """Train student using self-distillation (no real data required)"""
        teacher.eval()
        student.train()
        
        optimizer = optim.Adam(student.parameters(), lr=learning_rate)
        
        # Infer input shape and type
        input_shape = self.base._infer_input_shape(teacher)
        is_pinn = self.base._is_pinn_architecture(teacher)
        
        # Adjust batch size for PINN models
        if is_pinn:
            batch_size = 256
        
        # Test compatibility
        projection_layer = None
        
        with torch.no_grad():
            test_input = torch.randn(1, *input_shape).to(self.base.device)
            try:
                teacher_out = teacher(test_input)
                student_out = student(test_input)
                
                if teacher_out.shape[-1] != student_out.shape[-1]:
                    print(f"    Output dimension mismatch! Creating projection layer...")
                    # Create a projection layer to match dimensions
                    projection_layer = nn.Linear(student_out.shape[-1], teacher_out.shape[-1]).to(self.base.device)
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
                        random_inputs = torch.rand(batch_size, 2).to(self.base.device) * 2 - 1
                    elif input_shape[0] == 3:
                        random_inputs = torch.rand(batch_size, 3).to(self.base.device) * 2 - 1
                    else:
                        random_inputs = torch.randn(batch_size, *input_shape).to(self.base.device)
                else:
                    # Standard random inputs
                    random_inputs = torch.randn(batch_size, *input_shape).to(self.base.device)
                
                try:
                    # Get teacher outputs
                    with torch.no_grad():
                        teacher_outputs = teacher(random_inputs)
                    
                    # Get student outputs
                    student_outputs = student(random_inputs)
                    
                    # Apply projection if needed
                    if projection_layer is not None:
                        student_outputs = projection_layer(student_outputs)
                    
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
        
        student.eval()
        return student

    def apply_all_distillation_techniques(self, student_ratios=[0.25, 0.1], 
                                        train_students=True, student_epochs=50):
        """Apply knowledge distillation with different compression ratios"""
        results = {}
        
        print("\n" + "="*60)
        print("KNOWLEDGE DISTILLATION")
        if train_students:
            print("(With self-distillation training)")
        print("="*60)
        
        for ratio in student_ratios:
            print(f"\nCreating student with {ratio:.0%} size")
            student = self.create_student_model(
                ratio, 
                train_student=train_students,
                epochs=student_epochs
            )
            stats = self.base.get_model_stats(student)
            results[f'student_{ratio}'] = stats
            self.base._print_stats(f"  Student {ratio:.0%}", stats)
        
        return results