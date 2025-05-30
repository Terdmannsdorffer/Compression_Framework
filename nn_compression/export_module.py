"""
Neural Network Compression Framework - Export and Reporting Module
Export compressed models and generate detailed reports
"""

import torch
import os


class ExportModule:
    """Module for exporting compressed models and generating reports."""
    
    def __init__(self, base_framework):
        self.base = base_framework
        # Import modules when needed to avoid circular imports
        self.pruning = None
        self.quantization = None
        self.distillation = None

    def _get_modules(self):
        """Lazy initialization of modules to avoid circular imports"""
        if self.pruning is None:
            from .pruning_module import PruningModule
            from .quantization_module import QuantizationModule
            from .distillation_module import DistillationModule
            
            self.pruning = PruningModule(self.base)
            self.quantization = QuantizationModule(self.base)
            self.distillation = DistillationModule(self.base)

    def export_compressed_models(self, output_dir='compressed_models'):
        """Export all compressed models"""
        self._get_modules()  # Initialize modules
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nExporting compressed models to {output_dir}/")
        
        # Export pruned models
        for amount in [0.1, 0.3, 0.5, 0.7]:
            if f'magnitude_{amount}' in self.base.results.get('pruning', {}):
                model = self.pruning.apply_magnitude_pruning(amount)
                path = f"{output_dir}/pruned_magnitude_{amount}.pth"
                torch.save(model.state_dict(), path)
                print(f"  Saved: {path}")
                
                model = self.pruning.apply_random_pruning(amount)
                path = f"{output_dir}/pruned_random_{amount}.pth"
                torch.save(model.state_dict(), path)
                print(f"  Saved: {path}")
                
                model = self.pruning.apply_structured_pruning(amount)
                path = f"{output_dir}/pruned_structured_{amount}.pth"
                torch.save(model.state_dict(), path)
                print(f"  Saved: {path}")
        
        # Export quantized models
        model = self.quantization.apply_dynamic_quantization()
        torch.save(model.state_dict(), f"{output_dir}/quantized_dynamic_int8.pth")
        print(f"  Saved: {output_dir}/quantized_dynamic_int8.pth")
        
        for bits in [4, 6, 8]:
            model = self.quantization.apply_log2_quantization(bits)
            path = f"{output_dir}/quantized_log2_{bits}bit.pth"
            torch.save(model.state_dict(), path)
            print(f"  Saved: {path}")
        
        # Export student models
        for ratio in [0.25, 0.1]:
            if f'student_{ratio}' in self.base.results.get('distillation', {}):
                student = self.distillation.create_student_model(ratio, train_student=False)
                path = f"{output_dir}/student_{int(ratio*100)}percent.pth"
                torch.save(student.state_dict(), path)
                print(f"  Saved: {path}")
                torch.save(student.state_dict(), path)
                print(f"  Saved: {path}")

    def generate_report(self, save_path='compression_report.txt'):
        """Generate detailed text report"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("NEURAL NETWORK COMPRESSION REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Original model
            f.write("ORIGINAL MODEL\n")
            f.write("-"*40 + "\n")
            orig = self.base.results['original']
            f.write(f"Size: {orig['size_mb']:.2f} MB\n")
            f.write(f"Compressed size: {orig['compressed_size_mb']:.2f} MB\n")
            f.write(f"Parameters: {orig['total_params']:,}\n")
            f.write(f"Non-zero parameters: {orig['nonzero_params']:,}\n\n")
            
            # Best results per category
            f.write("BEST RESULTS PER CATEGORY\n")
            f.write("-"*40 + "\n")
            
            orig_size = orig['size_mb']
            
            # Pruning
            if 'pruning' in self.base.results:
                best_pruning = min(self.base.results['pruning'].items(), 
                                key=lambda x: x[1]['size_mb'])
                f.write(f"\nBest Pruning: {best_pruning[0]}\n")
                f.write(f"  Size: {best_pruning[1]['size_mb']:.2f} MB\n")
                f.write(f"  Compression: {orig_size/best_pruning[1]['size_mb']:.2f}x\n")
                f.write(f"  Sparsity: {best_pruning[1]['sparsity']*100:.1f}%\n")
                f.write(f"  BER: {best_pruning[1]['ber']*100:.2f}%\n")
            
            # Quantization
            if 'quantization' in self.base.results:
                best_quant = min(self.base.results['quantization'].items(),
                            key=lambda x: x[1]['size_mb'])
                f.write(f"\nBest Quantization: {best_quant[0]}\n")
                f.write(f"  Size: {best_quant[1]['size_mb']:.2f} MB\n")
                f.write(f"  Compression: {orig_size/best_quant[1]['size_mb']:.2f}x\n")
                f.write(f"  BER: {best_quant[1]['ber']*100:.2f}%\n")
            
            # Distillation
            if 'distillation' in self.base.results:
                best_distill = min(self.base.results['distillation'].items(),
                                key=lambda x: x[1]['size_mb'])
                f.write(f"\nBest Distillation: {best_distill[0]}\n")
                f.write(f"  Size: {best_distill[1]['size_mb']:.2f} MB\n")
                f.write(f"  Compression: {orig_size/best_distill[1]['size_mb']:.2f}x\n")
                f.write(f"  BER: {best_distill[1]['ber']*100:.2f}%\n")
            
            # Detailed results
            f.write("\n\nDETAILED RESULTS\n")
            f.write("="*80 + "\n")
            
            # Pruning details
            f.write("\nPRUNING TECHNIQUES\n")
            f.write("-"*40 + "\n")
            for name, stats in self.base.results.get('pruning', {}).items():
                f.write(f"\n{name}:\n")
                f.write(f"  Size: {stats['size_mb']:.2f} MB\n")
                f.write(f"  Compressed: {stats['compressed_size_mb']:.2f} MB\n")
                f.write(f"  Compression: {orig_size/stats['size_mb']:.2f}x\n")
                f.write(f"  Sparsity: {stats['sparsity']*100:.1f}%\n")
                f.write(f"  BER: {stats['ber']*100:.2f}%\n")
                f.write(f"  Non-zero params: {stats['nonzero_params']:,}\n")
            
            # Quantization details
            f.write("\n\nQUANTIZATION TECHNIQUES\n")
            f.write("-"*40 + "\n")
            for name, stats in self.base.results.get('quantization', {}).items():
                f.write(f"\n{name}:\n")
                f.write(f"  Size: {stats['size_mb']:.2f} MB\n")
                f.write(f"  Compressed: {stats['compressed_size_mb']:.2f} MB\n")
                f.write(f"  Compression: {orig_size/stats['size_mb']:.2f}x\n")
                f.write(f"  BER: {stats['ber']*100:.2f}%\n")
            
            # Distillation details
            f.write("\n\nKNOWLEDGE DISTILLATION\n")
            f.write("-"*40 + "\n")
            for name, stats in self.base.results.get('distillation', {}).items():
                f.write(f"\n{name}:\n")
                f.write(f"  Size: {stats['size_mb']:.2f} MB\n")
                f.write(f"  Compressed: {stats['compressed_size_mb']:.2f} MB\n")
                f.write(f"  Compression: {orig_size/stats['size_mb']:.2f}x\n")
                f.write(f"  BER: {stats['ber']*100:.2f}%\n")
                f.write(f"  Parameters: {stats['total_params']:,}\n")
            
            # BER Analysis
            f.write("\n\nBER ANALYSIS\n")
            f.write("-"*40 + "\n")
            f.write("Bit Error Rate (BER) represents the performance degradation:\n")
            f.write("- BER < 10%: Excellent preservation of model behavior\n")
            f.write("- BER < 30%: Good compression with acceptable performance loss\n")
            f.write("- BER < 50%: Aggressive compression, may need fine-tuning\n")
            f.write("- BER > 50%: Significant performance degradation expected\n\n")
            
            # Find techniques with lowest BER
            low_ber_techniques = []
            for category, results in self.base.results.items():
                if category == 'original':
                    continue
                for name, stats in results.items():
                    ber = stats.get('ber', 0)
                    compression = orig_size / stats['size_mb']
                    if ber < 0.3 and compression > 1.5:  # BER < 30% and compression > 1.5x
                        low_ber_techniques.append((f"{category}: {name}", ber, compression))
            
            if low_ber_techniques:
                low_ber_techniques.sort(key=lambda x: x[2], reverse=True)  # Sort by compression
                f.write("Recommended techniques (BER < 30%):\n")
                for technique, ber, compression in low_ber_techniques[:5]:
                    f.write(f"  {technique}: {compression:.2f}x compression, {ber*100:.1f}% BER\n")
            
            # Recommendations
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-"*40 + "\n")
            f.write("\n1. For maximum compression: Combine techniques\n")
            f.write("   - Apply magnitude pruning (50-70%)\n")
            f.write("   - Then apply log2 or minifloat quantization\n")
            f.write("   - Expected compression: 10-20x\n")
            
            f.write("\n2. For hardware deployment:\n")
            f.write("   - Use log2 quantization (powers of 2 are efficient)\n")
            f.write("   - Or structured pruning (removes entire channels)\n")
            
            f.write("\n3. For maintaining accuracy:\n")
            f.write("   - Start with small pruning amounts (10-30%)\n")
            f.write("   - Use knowledge distillation for aggressive compression\n")
            f.write("   - Monitor BER and keep it below 30% for production use\n")
            
            f.write("\n4. Combination strategies:\n")
            f.write("   - Pruning + Quantization: Multiplicative compression\n")
            f.write("   - Distillation + Quantization: Small and efficient models\n")
            f.write("   - Progressive compression: Apply techniques sequentially\n")
            
            f.write("\n5. Next steps:\n")
            f.write("   - Test compressed models on your specific task\n")
            f.write("   - Fine-tune if BER is above acceptable threshold\n")
            f.write("   - Consider hardware-specific optimizations\n")
            f.write("   - Validate performance on real data if available\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"\nReport saved to {save_path}")

    def create_model_comparison_table(self):
        """Create a comparison table of all compression techniques"""
        print("\n" + "="*100)
        print("COMPRESSION TECHNIQUES COMPARISON TABLE")
        print("="*100)
        
        # Header
        header = f"{'Technique':<25} {'Size (MB)':<12} {'Compression':<12} {'BER (%)':<10} {'Sparsity (%)':<12} {'Recommendation':<20}"
        print(header)
        print("-"*100)
        
        # Original model
        orig = self.base.results['original']
        print(f"{'Original':<25} {orig['size_mb']:<12.2f} {'1.00x':<12} {0:<10.1f} {orig['sparsity']*100:<12.1f} {'Baseline':<20}")
        
        # All techniques
        all_results = []
        for category, results in self.base.results.items():
            if category == 'original':
                continue
            for name, stats in results.items():
                compression = orig['size_mb'] / stats['size_mb']
                ber = stats.get('ber', 0) * 100
                sparsity = stats.get('sparsity', 0) * 100
                
                # Generate recommendation
                if ber < 10 and compression > 2:
                    recommendation = "Excellent"
                elif ber < 20 and compression > 1.5:
                    recommendation = "Very Good"
                elif ber < 30:
                    recommendation = "Good"
                elif ber < 50:
                    recommendation = "Acceptable"
                else:
                    recommendation = "Needs tuning"
                
                all_results.append({
                    'name': f"{category}: {name}",
                    'size': stats['size_mb'],
                    'compression': compression,
                    'ber': ber,
                    'sparsity': sparsity,
                    'recommendation': recommendation
                })
        
        # Sort by compression ratio
        all_results.sort(key=lambda x: x['compression'], reverse=True)
        
        # Print results
        for result in all_results:
            print(f"{result['name']:<25} {result['size']:<12.2f} {result['compression']:<12.2f}x {result['ber']:<10.1f} {result['sparsity']:<12.1f} {result['recommendation']:<20}")
        
        print("-"*100)
        
        # Summary statistics
        best_compression = max(all_results, key=lambda x: x['compression'])
        best_ber = min(all_results, key=lambda x: x['ber'])
        
        print(f"\nBest compression: {best_compression['name']} ({best_compression['compression']:.2f}x)")
        print(f"Lowest BER: {best_ber['name']} ({best_ber['ber']:.1f}%)")
        
        # Find sweet spot (good compression with low BER)
        sweet_spot_candidates = [r for r in all_results if r['ber'] < 20 and r['compression'] > 2]
        if sweet_spot_candidates:
            sweet_spot = max(sweet_spot_candidates, key=lambda x: x['compression'])
            print(f"Sweet spot: {sweet_spot['name']} ({sweet_spot['compression']:.2f}x, {sweet_spot['ber']:.1f}% BER)")
        
        print("="*100)

    def export_best_models_only(self, output_dir='best_models', ber_threshold=0.3):
        """Export only the best models based on compression ratio and BER threshold"""
        self._get_modules()  # Initialize modules
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nExporting best models (BER < {ber_threshold*100:.0f}%) to {output_dir}/")
        
        orig_size = self.base.results['original']['size_mb']
        candidates = []
        
        # Collect all candidates
        for category, results in self.base.results.items():
            if category == 'original':
                continue
            for name, stats in results.items():
                ber = stats.get('ber', 0)
                compression = orig_size / stats['size_mb']
                
                if ber <= ber_threshold and compression > 1.2:  # Must have some compression
                    candidates.append({
                        'category': category,
                        'name': name,
                        'compression': compression,
                        'ber': ber,
                        'stats': stats
                    })
        
        if not candidates:
            print(f"No models found within BER threshold of {ber_threshold*100:.0f}%")
            return
        
        # Sort by compression ratio and take top models
        candidates.sort(key=lambda x: x['compression'], reverse=True)
        
        # Export top models from each category
        categories_exported = set()
        models_exported = 0
        
        for candidate in candidates:
            category = candidate['category']
            name = candidate['name']
            
            # Export best from each category, plus top 3 overall
            if category not in categories_exported or models_exported < 3:
                # Recreate the model
                try:
                    if category == 'pruning':
                        parts = name.split('_')
                        method = parts[0]
                        amount = float(parts[1])
                        
                        if method == 'magnitude':
                            model = self.pruning.apply_magnitude_pruning(amount)
                        elif method == 'random':
                            model = self.pruning.apply_random_pruning(amount)
                        elif method == 'structured':
                            model = self.pruning.apply_structured_pruning(amount)
                        
                    elif category == 'quantization':
                        if 'dynamic' in name:
                            model = self.quantization.apply_dynamic_quantization()
                        elif 'log2' in name:
                            bits = int(name.split('_')[1].replace('bit', ''))
                            model = self.quantization.apply_log2_quantization(bits)
                        elif 'minifloat' in name:
                            # Extract exp and mantissa bits from name like 'minifloat_E4M3'
                            parts = name.split('_')[1]  # E4M3
                            exp_bits = int(parts[1])
                            mantissa_bits = int(parts[3])
                            model = self.quantization.apply_minifloat_quantization(exp_bits, mantissa_bits)
                    
                    elif category == 'distillation':
                        ratio = float(name.split('_')[1])
                        model = self.distillation.create_student_model(ratio, train_student=False)
                    
                    # Save model
                    filename = f"best_{category}_{name}.pth"
                    torch.save(model.state_dict(), os.path.join(output_dir, filename))
                    
                    print(f"  Exported: {filename}")
                    print(f"    Compression: {candidate['compression']:.2f}x")
                    print(f"    BER: {candidate['ber']*100:.1f}%")
                    
                    categories_exported.add(category)
                    models_exported += 1
                    
                    if models_exported >= 5:  # Limit to top 5 models
                        break
                        
                except Exception as e:
                    print(f"  Failed to export {category}:{name} - {str(e)}")
                    continue
        
        print(f"\nExported {models_exported} best models to {output_dir}/")