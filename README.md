# Neural Network Compression Framework

A comprehensive PyTorch framework for analyzing and applying various neural network compression techniques without requiring training data. This framework automatically evaluates multiple compression methods, provides detailed visualizations, and helps find the optimal balance between model size and performance.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Compression Techniques](#compression-techniques)
- [Metrics and Calculations](#metrics-and-calculations)
- [Installation](#installation)
- [Usage](#usage)
- [Output Files](#output-files)
- [Understanding the Results](#understanding-the-results)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

This framework provides a unified interface for applying and comparing various neural network compression techniques. It's designed to work with any PyTorch model and automatically handles different architectures including CNNs, fully connected networks, and Physics-Informed Neural Networks (PINNs).

## Features

### 🎯 Compression Techniques

1. **Pruning Methods**
   - **Magnitude Pruning**: Removes weights with smallest absolute values
   - **Random Pruning**: Randomly removes weights
   - **Structured Pruning**: Removes entire channels/neurons for hardware efficiency
   - Supports pruning levels from 10% to 90%

2. **Quantization Methods**
   - **Dynamic INT8**: Standard 8-bit integer quantization
   - **Log2 Quantization**: Quantizes weights to powers of 2 (hardware-friendly)
   - **Minifloat**: Custom floating-point formats (E4M3, E5M2, E3M4)
   - Bit widths from 4 to 8 bits supported

3. **Knowledge Distillation**
   - Automatic student architecture generation
   - Creates students at 25%, 10%, and 5% of teacher size
   - Self-distillation training using teacher outputs on random inputs
   - Handles architecture mismatches automatically

### 📊 Analysis and Visualization

- **Comprehensive Visualizations**:
  - Compression ratios by technique
  - Best compression per category
  - Sparsity vs pruning amount
  - Bit Error Rate (BER) analysis
  - 3D technique comparison
  - Combined compression potential
  - BER vs compression trade-off
  - Summary table with best results

- **BER-Aware Compression Analysis**:
  - Finds optimal compression for given BER thresholds
  - Identifies Pareto optimal points
  - Recommends best trade-off (knee point)
  - Separate visualization showing BER vs compression frontier

## Compression Techniques

### 1. Pruning

Pruning removes unnecessary weights from the network, creating sparse models.

#### Magnitude Pruning
```python
# Algorithm
1. Calculate absolute values of all weights
2. Determine threshold: kth percentile of |weights|
3. Set weights below threshold to zero
4. Store as sparse tensor if sparsity > 50%
```

**Pros**: Simple, often maintains accuracy well  
**Cons**: Requires sparse tensor support for actual speedup

#### Random Pruning
```python
# Algorithm
1. Generate random mask with probability p
2. Apply mask: weight = weight * mask
3. Results in exactly p% sparsity
```

**Pros**: Unbiased, good baseline  
**Cons**: May remove important weights

#### Structured Pruning
```python
# Algorithm
1. Calculate importance scores for channels/neurons
2. Remove entire channels with lowest L2 norm
3. Results in actual architecture reduction
```

**Pros**: Hardware-friendly, real speedup  
**Cons**: Typically lower compression ratios

### 2. Quantization

Quantization reduces the precision of weights, using fewer bits per parameter.

#### Dynamic INT8 Quantization
```python
# Algorithm
1. Find min/max values per layer
2. Map float32 range to int8 range [-128, 127]
3. Store scale and zero-point for dequantization
4. Actual storage: 8 bits per weight
```

**Compression**: 4x (32 bits → 8 bits)  
**Typical BER**: 1-5%

#### Log2 Quantization
```python
# Algorithm
1. For each weight w:
   - sign = sign(w)
   - magnitude = |w|
   - log_val = round(log2(magnitude))
   - quantized = sign * 2^log_val
2. Store only exponent (4-8 bits)
```

**Compression**: 4-8x depending on bit width  
**Hardware benefit**: Multiplication becomes bit shifting

#### Minifloat Quantization
```python
# Format: [sign(1)] [exponent(E)] [mantissa(M)]
# E4M3: 8 bits total (like FP8)
# E5M2: 8 bits total (more range, less precision)
# E3M4: 8 bits total (less range, more precision)

# Algorithm
1. Extract sign, exponent, mantissa from float32
2. Quantize exponent to E bits
3. Quantize mantissa to M bits
4. Reconstruct: sign * (1.mantissa) * 2^exponent
```

### 3. Knowledge Distillation

Creates smaller "student" models that mimic the larger "teacher" model.

```python
# Student Architecture Creation
1. Analyze teacher architecture
2. Scale hidden dimensions by sqrt(compression_ratio)
3. Maintain input/output dimensions
4. Reduce Fourier features (for PINNs)

# Self-Distillation Training
1. Generate random inputs from expected distribution
2. Get teacher outputs (soft targets)
3. Train student to match teacher outputs
4. Loss = MSE(student_output, teacher_output) for regression
        = KL_divergence for classification
```

## Metrics and Calculations

### 1. Model Size Calculation

```python
# Dense Model
size_mb = (num_parameters * bytes_per_param) / (1024^2)

# Sparse Model (>30% sparsity)
sparse_size = nonzero_params * 4  # values
            + nonzero_params * 4  # indices  
            + overhead           # row pointers
size_mb = min(sparse_size, dense_size) / (1024^2)

# Quantized Model
size_mb = (num_parameters * bits_per_param) / (8 * 1024^2)
```

### 2. Compression Ratio

```python
compression_ratio = original_size_mb / compressed_size_mb
```

### 3. Sparsity

```python
sparsity = 1 - (nonzero_parameters / total_parameters)
```

### 4. Bit Error Rate (BER)

Our framework uses **output-based BER**, not weight-based:

```python
# For Regression (e.g., PINNs)
ber = mean(||teacher_output - student_output|| / ||teacher_output||)

# For Classification
ber = mean(teacher_prediction != student_prediction)

# Interpretation:
# 0-5% BER:   Excellent, minimal degradation
# 5-15% BER:  Good, acceptable for most applications
# 15-30% BER: Moderate, noticeable quality loss
# >30% BER:   High, significant degradation
```

### 5. Pareto Optimality

A compression technique is Pareto optimal if no other technique has both:
- Higher compression ratio AND
- Lower BER

The "knee point" is the point on the Pareto frontier with the best trade-off.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/compression-framework.git
cd compression-framework

# Install requirements
pip install torch torchvision numpy matplotlib

# For GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Basic Usage

```python
from compression_framework import compress_model

# Run full compression analysis
framework, results = compress_model(
    "your_model.pth",
    train_students=True,      # Enable student training
    analyze_ber_tradeoff=True # Enable BER analysis
)
```

### Advanced Usage

```python
# Initialize framework
framework = ModelCompressionFramework("your_model.pth")

# Custom compression parameters
results = framework.compress_all(
    pruning_amounts=[0.1, 0.3, 0.5, 0.7],
    quantization_bits=[4, 6, 8],
    student_ratios=[0.25, 0.1, 0.05],
    train_students=True,
    student_epochs=100
)

# BER-aware analysis with custom thresholds
ber_analysis = framework.analyze_ber_compression_tradeoff(
    ber_thresholds=[0.01, 0.05, 0.10, 0.20, 0.30]
)

# Apply specific technique
pruned_model = framework.apply_magnitude_pruning(0.5)
quantized_model = framework.apply_log2_quantization(4)
student_model = framework.create_student_model(0.1, train_student=True)
```

### With Test Data

```python
# Evaluate with real test data
test_loader = your_test_dataloader()

compressed_models = {
    'pruned_50': framework.apply_magnitude_pruning(0.5),
    'quantized_8bit': framework.apply_dynamic_quantization(),
    'student_25': framework.create_student_model(0.25)
}

accuracy_results = framework.evaluate_with_data(
    compressed_models,
    test_loader=test_loader
)
```

## Output Files

1. **`compression_analysis.png`**: 8-panel visualization showing:
   - Compression ratios for all techniques
   - Size comparisons
   - Sparsity analysis
   - BER measurements
   - 3D comparison
   - Combined potential

2. **`ber_compression_analysis.png`**: BER-focused analysis:
   - Pareto frontier plot
   - Achievable compression at BER thresholds
   - Size reduction vs BER scatter

3. **`compression_report.txt`**: Detailed metrics for all techniques

4. **`ber_compression_analysis.txt`**: BER optimization report with:
   - Best technique for each BER threshold
   - Pareto optimal points
   - Recommended knee point

5. **`compressed_models/`**: All compressed model files

## Understanding the Results

### Compression Ratio Interpretation

| Ratio | Interpretation | Typical Use Case |
|-------|----------------|------------------|
| 1-2x | Modest compression | When accuracy is critical |
| 2-4x | Moderate compression | Balanced deployments |
| 4-8x | Good compression | Mobile/edge devices |
| 8-16x | High compression | Resource-constrained devices |
| >16x | Very high compression | Extreme size constraints |

### BER Interpretation

| BER Range | Quality | Recommendation |
|-----------|---------|----------------|
| 0-5% | Excellent | Safe for production |
| 5-15% | Good | Test on your specific task |
| 15-30% | Moderate | May need fine-tuning |
| >30% | Poor | Consider less aggressive compression |

### Technique Selection Guide

**For Maximum Compression**:
- Knowledge distillation (5% student) + quantization
- Expect 20-50x compression with moderate BER

**For Hardware Deployment**:
- Structured pruning + log2 quantization
- Hardware-efficient operations

**For Minimal Accuracy Loss**:
- Magnitude pruning (30%) + INT8 quantization
- Typically <5% accuracy drop

**For Balanced Results**:
- Follow the "knee point" recommendation
- Best trade-off between size and performance

## Best Practices

1. **Start Conservative**: Begin with 30% pruning or INT8 quantization
2. **Combine Techniques**: Pruning + quantization often works well
3. **Validate Results**: Always test compressed models on your task
4. **Consider Hardware**: Some techniques require specific support
5. **Use BER Analysis**: Let data guide your compression choice

## Troubleshooting

### Common Issues

**High BER Values**:
- Old version used weight-based BER (often 90%+)
- New version uses output-based BER (typically 1-30%)
- Update to latest version

**Unicode Errors on Windows**:
- Fixed in latest version
- Uses UTF-8 encoding for all text files

**Memory Issues**:
- Reduce batch size for student training
- Process techniques sequentially
- Use CPU if GPU memory limited

**PINN Compatibility**:
- Framework auto-detects PINN architecture
- Creates appropriate student models
- Uses MSE loss instead of classification loss

### Architecture Support

The framework automatically handles:
- Standard CNNs (Conv2D + Linear layers)
- Fully connected networks
- Physics-Informed Neural Networks (PINNs)
- Models with Fourier features
- Custom architectures via GenericModel wrapper

