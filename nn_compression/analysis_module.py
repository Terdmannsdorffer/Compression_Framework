"""
Neural Network Compression Framework - Analysis and Visualization Module
BER-aware analysis and comprehensive visualizations
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class AnalysisModule:
    """Module for advanced analysis and visualization of compression results."""
    
    def __init__(self, base_framework):
        self.base = base_framework

    def analyze_ber_compression_tradeoff(self, ber_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5], 
                                        save_results=True):
        """Analyze the trade-off between BER and compression ratio to find optimal points."""
        print("\n" + "="*60)
        print("BER-AWARE COMPRESSION ANALYSIS")
        print("="*60)
        
        if not self.base.results:
            print("No compression results available. Run compress_all() first.")
            return None
        
        # Collect all compression results with BER
        all_techniques = []
        
        for category, category_results in self.base.results.items():
            if category == 'original':
                continue
                
            for technique_name, stats in category_results.items():
                compression_ratio = self.base.results['original']['size_mb'] / stats['size_mb']
                ber = stats.get('ber', 0)
                
                all_techniques.append({
                    'category': category,
                    'name': technique_name,
                    'compression_ratio': compression_ratio,
                    'ber': ber,
                    'size_mb': stats['size_mb'],
                    'sparsity': stats.get('sparsity', 0)
                })
        
        # Sort by compression ratio
        all_techniques.sort(key=lambda x: x['compression_ratio'], reverse=True)
        
        # Find optimal techniques for each BER threshold
        ber_optimal = {}
        
        for threshold in ber_thresholds:
            # Filter techniques within BER threshold
            valid_techniques = [t for t in all_techniques if t['ber'] <= threshold]
            
            if valid_techniques:
                # Get the one with highest compression
                best = valid_techniques[0]
                ber_optimal[threshold] = best
                
                print(f"\nBER Threshold: {threshold*100:.0f}%")
                print(f"  Best technique: {best['category']}: {best['name']}")
                print(f"  Compression ratio: {best['compression_ratio']:.2f}x")
                print(f"  Actual BER: {best['ber']*100:.1f}%")
                print(f"  Size: {best['size_mb']:.2f} MB")
            else:
                print(f"\nBER Threshold: {threshold*100:.0f}%")
                print(f"  No techniques found within this BER threshold")
        
        # Find Pareto optimal points
        pareto_points = self._find_pareto_optimal(all_techniques)
        
        print("\n" + "-"*40)
        print("PARETO OPTIMAL POINTS (Best BER-Compression Trade-offs):")
        print("-"*40)
        
        for point in pareto_points[:5]:  # Show top 5
            print(f"\n{point['category']}: {point['name']}")
            print(f"  Compression: {point['compression_ratio']:.2f}x")
            print(f"  BER: {point['ber']*100:.1f}%")
            print(f"  Size: {point['size_mb']:.2f} MB")
        
        # Create visualization
        self._plot_ber_pareto_analysis(all_techniques, pareto_points, ber_optimal)
        
        # Save results if requested
        if save_results:
            self._save_ber_analysis(ber_optimal, pareto_points)
        
        return {
            'ber_optimal': ber_optimal,
            'pareto_points': pareto_points,
            'all_techniques': all_techniques
        }

    def _find_pareto_optimal(self, techniques):
        """Find Pareto optimal points for BER vs Compression trade-off"""
        pareto_points = []
        
        for i, technique in enumerate(techniques):
            is_pareto = True
            
            # Check if any other technique dominates this one
            for j, other in enumerate(techniques):
                if i != j:
                    # Other dominates if it has both higher compression AND lower BER
                    if (other['compression_ratio'] > technique['compression_ratio'] and 
                        other['ber'] < technique['ber']):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_points.append(technique)
        
        # Sort by compression ratio
        pareto_points.sort(key=lambda x: x['compression_ratio'], reverse=True)
        
        return pareto_points

    def _plot_ber_pareto_analysis(self, all_techniques, pareto_points, ber_optimal):
        """Create visualization for BER-aware compression analysis"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: BER vs Compression scatter with Pareto frontier
        compressions = [t['compression_ratio'] for t in all_techniques]
        bers = [t['ber'] * 100 for t in all_techniques]
        categories = [t['category'] for t in all_techniques]
        
        # Color by category
        color_map = {'pruning': 'blue', 'quantization': 'green', 'distillation': 'orange'}
        colors = [color_map.get(cat, 'gray') for cat in categories]
        
        # Plot all points
        ax1.scatter(compressions, bers, c=colors, alpha=0.6, s=50)
        
        # Highlight Pareto points
        pareto_comps = [p['compression_ratio'] for p in pareto_points]
        pareto_bers = [p['ber'] * 100 for p in pareto_points]
        ax1.scatter(pareto_comps, pareto_bers, c='red', s=100, marker='*', 
                label='Pareto Optimal', zorder=5, edgecolors='black')
        
        # Draw Pareto frontier
        if len(pareto_points) > 1:
            # Sort by compression for line drawing
            sorted_pareto = sorted(zip(pareto_comps, pareto_bers))
            pareto_x, pareto_y = zip(*sorted_pareto)
            ax1.plot(pareto_x, pareto_y, 'r--', alpha=0.5, linewidth=2)
        
        ax1.set_xlabel('Compression Ratio')
        ax1.set_ylabel('Bit Error Rate (%)')
        ax1.set_title('BER vs Compression Trade-off', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add BER threshold lines
        for threshold, technique in ber_optimal.items():
            ax1.axhline(y=threshold*100, color='gray', linestyle=':', alpha=0.5)
            ax1.text(0.5, threshold*100 + 1, f'{threshold*100:.0f}% threshold', 
                    fontsize=8, alpha=0.7)
        
        # Plot 2: Compression ratio for different BER thresholds
        thresholds = list(ber_optimal.keys())
        max_compressions = []
        technique_names = []
        
        for threshold in sorted(thresholds):
            if threshold in ber_optimal:
                max_compressions.append(ber_optimal[threshold]['compression_ratio'])
                technique_names.append(ber_optimal[threshold]['name'])
            else:
                max_compressions.append(0)
                technique_names.append('None')
        
        bars = ax2.bar([t*100 for t in sorted(thresholds)], max_compressions, 
                    color='skyblue', alpha=0.7, edgecolor='navy')
        
        # Add technique names on bars
        for i, (bar, name) in enumerate(zip(bars, technique_names)):
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        name.replace('_', '\n'), ha='center', va='bottom', 
                        fontsize=8, rotation=0)
        
        ax2.set_xlabel('BER Threshold (%)')
        ax2.set_ylabel('Maximum Compression Ratio')
        ax2.set_title('Achievable Compression at Different BER Thresholds', fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Plot 3: BER vs Size reduction
        size_reductions = [(1 - t['size_mb']/self.base.results['original']['size_mb'])*100 
                        for t in all_techniques]
        
        ax3.scatter(size_reductions, bers, c=colors, alpha=0.6, s=50)
        
        # Highlight optimal points
        optimal_reductions = [(1 - p['size_mb']/self.base.results['original']['size_mb'])*100 
                            for p in pareto_points]
        ax3.scatter(optimal_reductions, pareto_bers, c='red', s=100, marker='*', 
                label='Pareto Optimal', zorder=5, edgecolors='black')
        
        ax3.set_xlabel('Size Reduction (%)')
        ax3.set_ylabel('Bit Error Rate (%)')
        ax3.set_title('BER vs Size Reduction', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Add legend for categories
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=cat.title()) 
                        for cat, color in color_map.items()]
        ax1.legend(handles=legend_elements + [plt.Line2D([0], [0], marker='*', color='w', 
                markerfacecolor='r', markersize=10, label='Pareto Optimal')], 
                loc='upper left')
        
        plt.tight_layout()
        plt.savefig('ber_compression_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nBER analysis visualization saved as 'ber_compression_analysis.png'")

    def _save_ber_analysis(self, ber_optimal, pareto_points):
        """Save BER analysis results to file"""
        # Use UTF-8 encoding to handle special characters
        with open('ber_compression_analysis.txt', 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("BER-AWARE COMPRESSION ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("OPTIMAL TECHNIQUES FOR DIFFERENT BER THRESHOLDS\n")
            f.write("-"*40 + "\n\n")
            
            for threshold in sorted(ber_optimal.keys()):
                technique = ber_optimal[threshold]
                f.write(f"BER Threshold: <= {threshold*100:.0f}%\n")
                f.write(f"  Best Technique: {technique['category']}: {technique['name']}\n")
                f.write(f"  Compression Ratio: {technique['compression_ratio']:.2f}x\n")
                f.write(f"  Actual BER: {technique['ber']*100:.1f}%\n")
                f.write(f"  Final Size: {technique['size_mb']:.2f} MB\n")
                f.write(f"  Size Reduction: {(1-technique['size_mb']/self.base.results['original']['size_mb'])*100:.1f}%\n\n")
            
            f.write("\nPARETO OPTIMAL POINTS\n")
            f.write("-"*40 + "\n")
            f.write("(Techniques that offer the best trade-off between BER and compression)\n\n")
            
            for i, point in enumerate(pareto_points[:10]):  # Top 10
                f.write(f"{i+1}. {point['category']}: {point['name']}\n")
                f.write(f"   Compression: {point['compression_ratio']:.2f}x\n")
                f.write(f"   BER: {point['ber']*100:.1f}%\n")
                f.write(f"   Size: {point['size_mb']:.2f} MB\n\n")
            
            f.write("\nRECOMMENDATIONS\n")
            f.write("-"*40 + "\n\n")
            
            # Find knee point (best trade-off)
            knee_point = self._find_knee_point(pareto_points)
            if knee_point:
                f.write(f"Recommended (Knee Point): {knee_point['category']}: {knee_point['name']}\n")
                f.write(f"  This offers the best balance with {knee_point['compression_ratio']:.2f}x compression\n")
                f.write(f"  at {knee_point['ber']*100:.1f}% BER\n\n")
            
            f.write("For different use cases:\n")
            f.write("- Critical applications (BER < 10%): Use pruning with low amounts\n")
            f.write("- Balanced applications (BER < 30%): Consider moderate pruning or INT8 quantization\n")
            f.write("- Size-critical applications (BER < 50%): Use aggressive quantization or distillation\n")
            
        print("BER analysis report saved as 'ber_compression_analysis.txt'")

    def _find_knee_point(self, pareto_points):
        """Find the knee point in the Pareto frontier (best trade-off)"""
        if len(pareto_points) < 3:
            return pareto_points[0] if pareto_points else None
        
        # Calculate the distance from the line connecting first and last points
        first = pareto_points[0]
        last = pareto_points[-1]
        
        max_distance = 0
        knee_point = None
        
        for point in pareto_points[1:-1]:
            # Calculate perpendicular distance to the line
            distance = self._point_line_distance(
                point['compression_ratio'], point['ber'],
                first['compression_ratio'], first['ber'],
                last['compression_ratio'], last['ber']
            )
            
            if distance > max_distance:
                max_distance = distance
                knee_point = point
        
        return knee_point

    def _point_line_distance(self, px, py, x1, y1, x2, y2):
        """Calculate perpendicular distance from point to line"""
        # Normalize the scales
        px_norm = px / max(x1, x2)
        py_norm = py / max(y1, y2)
        x1_norm = x1 / max(x1, x2)
        y1_norm = y1 / max(y1, y2)
        x2_norm = x2 / max(x1, x2)
        y2_norm = y2 / max(y1, y2)
        
        # Calculate distance
        numerator = abs((y2_norm - y1_norm) * px_norm - (x2_norm - x1_norm) * py_norm + 
                    x2_norm * y1_norm - y2_norm * x1_norm)
        denominator = np.sqrt((y2_norm - y1_norm)**2 + (x2_norm - x1_norm)**2)
        
        return numerator / denominator if denominator > 0 else 0

    def plot_results(self):
        """Create comprehensive visualizations including BER analysis"""
        if not self.base.results:
            print("No results to plot. Run compress_all() first.")
            return
        
        # Create figure with better spacing
        fig = plt.figure(figsize=(24, 14))
        
        # Use GridSpec for better control over layout
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3, 
                            top=0.95, bottom=0.05, left=0.05, right=0.95)
        
        # Create subplots
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[0, 3])
        ax5 = fig.add_subplot(gs[1, 0], projection='3d')
        ax6 = fig.add_subplot(gs[1, 1])
        ax7 = fig.add_subplot(gs[1, 2])
        ax8 = fig.add_subplot(gs[1, 3])
        
        self._plot_compression_overview(ax1)
        self._plot_size_comparison(ax2)
        self._plot_sparsity_analysis(ax3)
        self._plot_ber_analysis(ax4)
        self._plot_technique_comparison(ax5)
        self._plot_combined_potential(ax6)
        self._plot_ber_vs_compression(ax7)
        self._create_summary_table(ax8)
        
        plt.tight_layout()
        plt.savefig('compression_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nVisualization saved as 'compression_analysis.png'")

    def _plot_compression_overview(self, ax):
        """Overview of all compression ratios"""
        techniques = []
        compressions = []
        colors = []
        
        orig_size = self.base.results['original']['size_mb']
        
        # Collect all results
        for category, results in self.base.results.items():
            if category == 'original':
                continue
                
            if category == 'pruning':
                for name, stats in results.items():
                    techniques.append(name.replace('_', '\n'))
                    compressions.append(orig_size / stats['size_mb'])
                    colors.append('blue')
            elif category == 'quantization':
                for name, stats in results.items():
                    techniques.append(name.replace('_', '\n'))
                    compressions.append(orig_size / stats['size_mb'])
                    colors.append('green')
            elif category == 'distillation':
                for name, stats in results.items():
                    techniques.append(name.replace('_', '\n'))
                    compressions.append(orig_size / stats['size_mb'])
                    colors.append('orange')
        
        bars = ax.bar(range(len(techniques)), compressions, color=colors, alpha=0.7)
        ax.set_xticks(range(len(techniques)))
        ax.set_xticklabels(techniques, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Compression Ratio')
        ax.set_title('Compression Ratios by Technique', fontweight='bold')
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.1f}x', ha='center', va='bottom', fontsize=7)

    def _plot_size_comparison(self, ax):
        """Compare original vs compressed sizes"""
        categories = ['Pruning', 'Quantization', 'Distillation']
        best_compressions = []
        
        orig_size = self.base.results['original']['size_mb']
        
        for category in ['pruning', 'quantization', 'distillation']:
            if category in self.base.results:
                best_compression = min(
                    stats['compressed_size_mb'] 
                    for stats in self.base.results[category].values()
                )
                best_compressions.append(best_compression)
            else:
                best_compressions.append(orig_size)
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, [orig_size]*len(categories), width, 
                       label='Original', alpha=0.8, color='red')
        bars2 = ax.bar(x + width/2, best_compressions, width,
                       label='Best Compressed', alpha=0.8, color='green')
        
        ax.set_xlabel('Technique Category')
        ax.set_ylabel('Size (MB)')
        ax.set_title('Best Compression per Category', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add compression ratios
        for i, (orig, comp) in enumerate(zip([orig_size]*len(categories), best_compressions)):
            ratio = orig / comp
            ax.text(i, max(orig, comp) + 0.5, f'{ratio:.1f}x', ha='center')

    def _plot_sparsity_analysis(self, ax):
        """Analyze sparsity for pruning methods"""
        # Separate data by pruning type
        pruning_data = {
            'magnitude': {'amounts': [], 'sparsities': []},
            'random': {'amounts': [], 'sparsities': []},
            'structured': {'amounts': [], 'sparsities': []}
        }
        
        # Collect data
        for name, stats in self.base.results.get('pruning', {}).items():
            parts = name.split('_')
            if len(parts) == 2:
                method = parts[0]
                amount = float(parts[1])
                sparsity = stats['sparsity'] * 100
                
                if method in pruning_data:
                    pruning_data[method]['amounts'].append(amount)
                    pruning_data[method]['sparsities'].append(sparsity)
        
        # Plot each method
        colors = {'magnitude': 'blue', 'random': 'orange', 'structured': 'green'}
        markers = {'magnitude': 'o', 'random': 's', 'structured': '^'}
        
        has_data = False
        for method, data in pruning_data.items():
            if data['amounts'] and any(s > 0 for s in data['sparsities']):
                # Sort by amount
                sorted_pairs = sorted(zip(data['amounts'], data['sparsities']))
                amounts, sparsities = zip(*sorted_pairs)
                
                ax.plot(amounts, sparsities, 
                    color=colors[method],
                    marker=markers[method],
                    label=method, 
                    linewidth=2, 
                    markersize=8,
                    linestyle='-')
                has_data = True
        
        # If no pruning achieved, show expected line
        if not has_data:
            ax.text(0.35, 35, 'Pruning not effective\non this model', 
                    ha='center', va='center', fontsize=12, color='red',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
        
        # Add expected diagonal
        ax.plot([0, 0.7], [0, 70], 'k:', alpha=0.3, label='Expected')
        
        ax.set_xlabel('Pruning Amount')
        ax.set_ylabel('Sparsity (%)')
        ax.set_title('Sparsity vs Pruning Amount', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, 0.75)
        ax.set_ylim(0, 75)

    def _plot_ber_analysis(self, ax):
        """Plot BER for different compression techniques"""
        techniques = []
        bers = []
        colors = []
        
        # Collect BER data
        for category, results in self.base.results.items():
            if category == 'original':
                continue
                
            if category == 'pruning':
                for name, stats in results.items():
                    techniques.append(name.replace('_', '\n'))
                    bers.append(stats.get('ber', 0) * 100)
                    colors.append('blue')
            elif category == 'quantization':
                for name, stats in results.items():
                    techniques.append(name.replace('_', '\n'))
                    bers.append(stats.get('ber', 0) * 100)
                    colors.append('green')
            elif category == 'distillation':
                for name, stats in results.items():
                    techniques.append(name.replace('_', '\n'))
                    bers.append(stats.get('ber', 0) * 100)
                    colors.append('orange')
        
        # Create bar plot
        bars = ax.bar(range(len(techniques)), bers, color=colors, alpha=0.7)
        ax.set_xticks(range(len(techniques)))
        ax.set_xticklabels(techniques, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Bit Error Rate (%)')
        ax.set_title('BER by Compression Technique', fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=7)
        
        # Add reference line
        ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
        ax.legend()

    def _plot_technique_comparison(self, ax):
        """3D scatter plot of techniques"""
        try:
            sizes = []
            compressions = []
            bers = []
            colors = []
            labels = []
            
            orig_size = self.base.results['original']['size_mb']
            
            # Add original
            sizes.append(orig_size)
            compressions.append(1.0)
            bers.append(0.0)
            colors.append('red')
            labels.append('Original')
            
            # Add all techniques
            color_map = {'pruning': 'blue', 'quantization': 'green', 'distillation': 'orange'}
            
            for category, results in self.base.results.items():
                if category == 'original':
                    continue
                    
                for name, stats in results.items():
                    if stats['size_mb'] > 0:
                        sizes.append(stats['size_mb'])
                        compressions.append(orig_size / stats['size_mb'])
                        bers.append(stats.get('ber', 0) * 100)
                        colors.append(color_map.get(category, 'gray'))
                        labels.append(f"{category}: {name}")
            
            # Create scatter plot
            if len(sizes) > 1:
                scatter = ax.scatter(sizes, compressions, bers, c=colors, s=100, alpha=0.7)
                
                ax.set_xlabel('Model Size (MB)')
                ax.set_ylabel('Compression Ratio')
                ax.set_zlabel('Bit Error Rate (%)')
                ax.set_title('3D Technique Comparison', fontweight='bold')
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor=color, label=cat.title()) 
                                for cat, color in color_map.items()]
                legend_elements.insert(0, Patch(facecolor='red', label='Original'))
                ax.legend(handles=legend_elements, loc='upper left')
            else:
                ax.text(0.5, 0.5, 0.5, 'Insufficient data for 3D plot', 
                        ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            print(f"Warning: 3D plot error: {e}")
            ax.text(0.5, 0.5, '3D plot unavailable', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12)

    def _plot_combined_potential(self, ax):
        """Show potential of combining techniques with realistic estimates"""
        techniques = ['Original', 'Pruning\nOnly', 'Quantization\nOnly', 
                    'Distillation\nOnly', 'Pruning +\nQuantization', 'All\nTechniques']
        
        # Get actual best compressions from results
        orig_size = self.base.results['original']['size_mb']
        
        # Find best in each category
        best_pruning_ratio = 1.0
        best_quant_ratio = 1.0
        best_distill_ratio = 1.0
        
        # Check pruning
        for name, stats in self.base.results.get('pruning', {}).items():
            if stats['size_mb'] > 0:
                ratio = orig_size / stats['size_mb']
                best_pruning_ratio = max(best_pruning_ratio, ratio)
        
        # Check quantization
        for name, stats in self.base.results.get('quantization', {}).items():
            if stats['size_mb'] > 0:
                ratio = orig_size / stats['size_mb']
                best_quant_ratio = max(best_quant_ratio, ratio)
        
        # Check distillation
        for name, stats in self.base.results.get('distillation', {}).items():
            if stats['size_mb'] > 0:
                ratio = orig_size / stats['size_mb']
                best_distill_ratio = max(best_distill_ratio, ratio)
        
        # Calculate realistic combined compressions
        compressions = [
            1.0,  # Original
            best_pruning_ratio,  # Pruning only
            best_quant_ratio,  # Quantization only
            best_distill_ratio,  # Distillation only
            min(best_pruning_ratio * best_quant_ratio * 0.7, 20.0),  # Pruning + Quant
            min(best_pruning_ratio * best_quant_ratio * best_distill_ratio * 0.5, 30.0)  # All
        ]
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        bars = ax.bar(techniques, compressions, color=colors, alpha=0.7)
        
        ax.set_ylabel('Compression Ratio')
        ax.set_title('Realistic Combined Compression Potential', fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.1f}x', ha='center', va='bottom')
        
        # Set y limit based on actual values
        ax.set_ylim(0, max(compressions) * 1.2)
        
        ax.text(0.5, 0.95, 'Note: Combined techniques with efficiency factors',
                transform=ax.transAxes, ha='center', fontsize=9, style='italic')

    def _plot_ber_vs_compression(self, ax):
        """Plot BER vs compression ratio"""
        # Separate data by category
        pruning_data = {'comps': [], 'bers': []}
        quant_data = {'comps': [], 'bers': []}
        distill_data = {'comps': [], 'bers': []}
        
        orig_size = self.base.results['original']['size_mb']
        
        # Collect data by category
        for category, results in self.base.results.items():
            if category == 'original':
                continue
                
            for name, stats in results.items():
                compression = orig_size / stats['size_mb']
                ber = stats.get('ber', 0) * 100
                
                if compression > 1.0:  # Only plot if there's actual compression
                    if category == 'pruning':
                        pruning_data['comps'].append(compression)
                        pruning_data['bers'].append(ber)
                    elif category == 'quantization':
                        quant_data['comps'].append(compression)
                        quant_data['bers'].append(ber)
                    elif category == 'distillation':
                        distill_data['comps'].append(compression)
                        distill_data['bers'].append(ber)
        
        # Plot each category with different markers
        if pruning_data['comps']:
            ax.scatter(pruning_data['comps'], pruning_data['bers'], 
                    c='blue', marker='o', s=100, alpha=0.7, label='Pruning')
        
        if quant_data['comps']:
            ax.scatter(quant_data['comps'], quant_data['bers'], 
                    c='green', marker='s', s=100, alpha=0.7, label='Quantization')
        
        if distill_data['comps']:
            ax.scatter(distill_data['comps'], distill_data['bers'], 
                    c='orange', marker='^', s=100, alpha=0.7, label='Distillation')
        
        # Add trend line for all points
        all_comps = pruning_data['comps'] + quant_data['comps'] + distill_data['comps']
        all_bers = pruning_data['bers'] + quant_data['bers'] + distill_data['bers']
        
        if len(all_comps) > 3:
            z = np.polyfit(all_comps, all_bers, 2)
            p = np.poly1d(z)
            x_trend = np.linspace(min(all_comps), max(all_comps), 100)
            ax.plot(x_trend, np.maximum(0, p(x_trend)), 'r--', alpha=0.5, label='Trend')
        
        ax.set_xlabel('Compression Ratio')
        ax.set_ylabel('Bit Error Rate (%)')
        ax.set_title('BER vs Compression Trade-off', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set reasonable axis limits
        if all_comps:
            ax.set_xlim(0.5, max(all_comps) * 1.1)
            ax.set_ylim(-5, max(all_bers) * 1.1 if all_bers else 100)

    def _create_summary_table(self, ax):
        """Create summary table of compression results including BER"""
        ax.axis('tight')
        ax.axis('off')
        
        # Find best techniques
        orig_size = self.base.results['original']['size_mb']
        best_overall = {'name': 'Original', 'compression': 1.0, 'size': orig_size, 'ber': 0}
        best_pruning = {'name': 'None', 'compression': 1.0, 'ber': 0}
        best_quant = {'name': 'None', 'compression': 1.0, 'ber': 0}
        best_distill = {'name': 'None', 'compression': 1.0, 'ber': 0}
        lowest_ber = {'name': 'Original', 'compression': 1.0, 'ber': 1.0}
        
        # Check all techniques
        for category in ['pruning', 'quantization', 'distillation']:
            for name, stats in self.base.results.get(category, {}).items():
                comp = orig_size / stats['size_mb']
                ber = stats.get('ber', 0)
                
                # Update best in category
                if category == 'pruning' and comp > best_pruning['compression']:
                    best_pruning = {'name': name, 'compression': comp, 'ber': ber}
                elif category == 'quantization' and comp > best_quant['compression']:
                    best_quant = {'name': name, 'compression': comp, 'ber': ber}
                elif category == 'distillation' and comp > best_distill['compression']:
                    best_distill = {'name': name, 'compression': comp, 'ber': ber}
                
                # Update overall best
                if comp > best_overall['compression']:
                    best_overall = {'name': f'{category}: {name}', 'compression': comp, 
                                'size': stats['size_mb'], 'ber': ber}
                
                # Track lowest BER
                if ber < lowest_ber['ber'] and comp > 1.5:
                    lowest_ber = {'name': f'{category}: {name}', 'compression': comp, 'ber': ber}
        
        # Create table data
        table_data = [
            ['Metric', 'Technique', 'Compression', 'BER'],
            ['Overall Best', best_overall['name'], f"{best_overall['compression']:.2f}x", f"{best_overall['ber']:.4f}"],
            ['Best Pruning', best_pruning['name'], f"{best_pruning['compression']:.2f}x", f"{best_pruning['ber']:.4f}"],
            ['Best Quantization', best_quant['name'], f"{best_quant['compression']:.2f}x", f"{best_quant['ber']:.4f}"],
            ['Best Distillation', best_distill['name'], f"{best_distill['compression']:.2f}x", f"{best_distill['ber']:.4f}"],
            ['Lowest BER', lowest_ber['name'], f"{lowest_ber['compression']:.2f}x", f"{lowest_ber['ber']:.4f}"],
            ['', '', '', ''],
            ['Original Size', f"{orig_size:.2f} MB", '', ''],
            ['Best Final Size', f"{best_overall['size']:.2f} MB", '', ''],
        ]
        
        # Create the table
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.3, 0.35, 0.2, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style the header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style the summary rows
        for row in [7, 8]:
            table[(row, 0)].set_facecolor('#E3F2FD')
            table[(row, 0)].set_text_props(weight='bold')
        
        # Highlight lowest BER row
        table[(5, 0)].set_facecolor('#FFF3E0')
        table[(5, 0)].set_text_props(weight='bold')
        
        ax.set_title('Compression Summary with BER Analysis', fontsize=14, fontweight='bold', pad=20)