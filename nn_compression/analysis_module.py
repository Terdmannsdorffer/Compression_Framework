"""
Neural Network Compression Framework - Analysis and Visualization Module
Precision-aware analysis with PIV-PINN metrics and comprehensive visualizations
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class AnalysisModule:
    """Module for advanced analysis and visualization of compression results with precision focus."""
    
    def __init__(self, base_framework):
        self.base = base_framework

    # =============================================================================
    # MAIN PRECISION ANALYSIS METHODS
    # =============================================================================

    def analyze_precision_compression_tradeoff(self, precision_thresholds=[50, 60, 70, 80, 90], 
                                             save_results=True):
        """Analyze the trade-off between precision and compression ratio to find optimal points."""
        print("\n" + "="*60)
        print("PRECISION-AWARE COMPRESSION ANALYSIS")
        print("="*60)
        
        if not self.base.results:
            print("No compression results available. Run compress_all() first.")
            return None
        
        # Collect all compression results with precision
        all_techniques = []
        
        for category, category_results in self.base.results.items():
            if category == 'original':
                continue
                
            for technique_name, stats in category_results.items():
                compression_ratio = self.base.results['original']['size_mb'] / stats['size_mb']
                precision = stats.get('precision', 50.0)  # Default precision if not available
                
                all_techniques.append({
                    'category': category,
                    'name': technique_name,
                    'compression_ratio': compression_ratio,
                    'precision': precision,
                    'size_mb': stats['size_mb'],
                    'sparsity': stats.get('sparsity', 0),
                    'ber': stats.get('ber', 0)
                })
        
        # Sort by compression ratio
        all_techniques.sort(key=lambda x: x['compression_ratio'], reverse=True)
        
        # Find optimal techniques for each precision threshold
        precision_optimal = {}
        
        for threshold in precision_thresholds:
            # Filter techniques within precision threshold
            valid_techniques = [t for t in all_techniques if t['precision'] >= threshold]
            
            if valid_techniques:
                # Get the one with highest compression
                best = valid_techniques[0]
                precision_optimal[threshold] = best
                
                print(f"\nPrecision Threshold: >= {threshold:.0f}%")
                print(f"  Best technique: {best['category']}: {best['name']}")
                print(f"  Compression ratio: {best['compression_ratio']:.2f}x")
                print(f"  Actual precision: {best['precision']:.1f}%")
                print(f"  Size: {best['size_mb']:.2f} MB")
            else:
                print(f"\nPrecision Threshold: >= {threshold:.0f}%")
                print(f"  No techniques found above this precision threshold")
        
        # Find Pareto optimal points (high precision AND high compression)
        pareto_points = self._find_pareto_optimal_precision(all_techniques)
        
        print("\n" + "-"*40)
        print("PARETO OPTIMAL POINTS (Best Precision-Compression Trade-offs):")
        print("-"*40)
        
        for point in pareto_points[:5]:  # Show top 5
            print(f"\n{point['category']}: {point['name']}")
            print(f"  Compression: {point['compression_ratio']:.2f}x")
            print(f"  Precision: {point['precision']:.1f}%")
            print(f"  Size: {point['size_mb']:.2f} MB")
        
        # Create visualization
        self._plot_precision_pareto_analysis(all_techniques, pareto_points, precision_optimal)
        
        # Save results if requested
        if save_results:
            self._save_precision_analysis(precision_optimal, pareto_points)
        
        return {
            'precision_optimal': precision_optimal,
            'pareto_points': pareto_points,
            'all_techniques': all_techniques
        }

    def plot_results(self):
        """Create comprehensive visualizations with precision focus"""
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
        self._plot_precision_comparison(ax2)
        self._plot_sparsity_analysis(ax3)
        self._plot_precision_analysis(ax4)
        self._plot_technique_comparison_precision(ax5)
        self._plot_precision_before_after(ax6)  # New precision comparison plot
        self._plot_precision_vs_compression(ax7)
        self._create_summary_table_precision(ax8)
        
        plt.tight_layout()
        plt.savefig('compression_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nVisualization saved as 'compression_analysis.png'")

    # =============================================================================
    # PRECISION ANALYSIS HELPER METHODS
    # =============================================================================

    def _find_pareto_optimal_precision(self, techniques):
        """Find Pareto optimal points for Precision vs Compression trade-off"""
        pareto_points = []
        
        for i, technique in enumerate(techniques):
            is_pareto = True
            
            # Check if any other technique dominates this one
            for j, other in enumerate(techniques):
                if i != j:
                    # Other dominates if it has both higher compression AND higher precision
                    if (other['compression_ratio'] > technique['compression_ratio'] and 
                        other['precision'] > technique['precision']):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_points.append(technique)
        
        # Sort by compression ratio
        pareto_points.sort(key=lambda x: x['compression_ratio'], reverse=True)
        
        return pareto_points

    def _find_knee_point_precision(self, pareto_points):
        """Find the knee point in the Pareto frontier for precision (best trade-off)"""
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
                point['compression_ratio'], point['precision'],
                first['compression_ratio'], first['precision'],
                last['compression_ratio'], last['precision']
            )
            
            if distance > max_distance:
                max_distance = distance
                knee_point = point
        
        return knee_point

    def _point_line_distance(self, px, py, x1, y1, x2, y2):
        """Calculate perpendicular distance from point to line"""
        # Normalize the scales
        px_norm = px / max(x1, x2) if max(x1, x2) > 0 else px
        py_norm = py / max(y1, y2) if max(y1, y2) > 0 else py
        x1_norm = x1 / max(x1, x2) if max(x1, x2) > 0 else x1
        y1_norm = y1 / max(y1, y2) if max(y1, y2) > 0 else y1
        x2_norm = x2 / max(x1, x2) if max(x1, x2) > 0 else x2
        y2_norm = y2 / max(y1, y2) if max(y1, y2) > 0 else y2
        
        # Calculate distance
        numerator = abs((y2_norm - y1_norm) * px_norm - (x2_norm - x1_norm) * py_norm + 
                    x2_norm * y1_norm - y2_norm * x1_norm)
        denominator = np.sqrt((y2_norm - y1_norm)**2 + (x2_norm - x1_norm)**2)
        
        return numerator / denominator if denominator > 0 else 0

    # =============================================================================
    # PRECISION VISUALIZATION METHODS
    # =============================================================================

    def _plot_precision_pareto_analysis(self, all_techniques, pareto_points, precision_optimal):
        """Create visualization for precision-aware compression analysis"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Precision vs Compression scatter with Pareto frontier
        compressions = [t['compression_ratio'] for t in all_techniques]
        precisions = [t['precision'] for t in all_techniques]
        categories = [t['category'] for t in all_techniques]
        
        # Color by category
        color_map = {'pruning': 'blue', 'quantization': 'green', 'distillation': 'orange'}
        colors = [color_map.get(cat, 'gray') for cat in categories]
        
        # Plot all points
        ax1.scatter(compressions, precisions, c=colors, alpha=0.6, s=50)
        
        # Highlight Pareto points
        pareto_comps = [p['compression_ratio'] for p in pareto_points]
        pareto_precisions = [p['precision'] for p in pareto_points]
        ax1.scatter(pareto_comps, pareto_precisions, c='red', s=100, marker='*', 
                label='Pareto Optimal', zorder=5, edgecolors='black')
        
        # Draw Pareto frontier
        if len(pareto_points) > 1:
            # Sort by compression for line drawing
            sorted_pareto = sorted(zip(pareto_comps, pareto_precisions))
            pareto_x, pareto_y = zip(*sorted_pareto)
            ax1.plot(pareto_x, pareto_y, 'r--', alpha=0.5, linewidth=2)
        
        ax1.set_xlabel('Compression Ratio')
        ax1.set_ylabel('PIV-PINN Precision (%)')
        ax1.set_title('Precision vs Compression Trade-off', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add precision threshold lines
        for threshold, technique in precision_optimal.items():
            ax1.axhline(y=threshold, color='gray', linestyle=':', alpha=0.5)
            ax1.text(0.5, threshold + 1, f'{threshold:.0f}% threshold', 
                    fontsize=8, alpha=0.7)
        
        # Plot 2: Compression ratio for different precision thresholds
        thresholds = list(precision_optimal.keys())
        max_compressions = []
        technique_names = []
        
        for threshold in sorted(thresholds):
            if threshold in precision_optimal:
                max_compressions.append(precision_optimal[threshold]['compression_ratio'])
                technique_names.append(precision_optimal[threshold]['name'])
            else:
                max_compressions.append(0)
                technique_names.append('None')
        
        bars = ax2.bar([t for t in sorted(thresholds)], max_compressions, 
                    color='skyblue', alpha=0.7, edgecolor='navy')
        
        # Add technique names on bars
        for i, (bar, name) in enumerate(zip(bars, technique_names)):
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        name.replace('_', '\n'), ha='center', va='bottom', 
                        fontsize=8, rotation=0)
        
        ax2.set_xlabel('Precision Threshold (%)')
        ax2.set_ylabel('Maximum Compression Ratio')
        ax2.set_title('Achievable Compression at Different Precision Thresholds', fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Plot 3: Precision vs Size reduction
        size_reductions = [(1 - t['size_mb']/self.base.results['original']['size_mb'])*100 
                        for t in all_techniques]
        
        ax3.scatter(size_reductions, precisions, c=colors, alpha=0.6, s=50)
        
        # Highlight optimal points
        optimal_reductions = [(1 - p['size_mb']/self.base.results['original']['size_mb'])*100 
                            for p in pareto_points]
        ax3.scatter(optimal_reductions, pareto_precisions, c='red', s=100, marker='*', 
                label='Pareto Optimal', zorder=5, edgecolors='black')
        
        ax3.set_xlabel('Size Reduction (%)')
        ax3.set_ylabel('PIV-PINN Precision (%)')
        ax3.set_title('Precision vs Size Reduction', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Add legend for categories
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=cat.title()) 
                        for cat, color in color_map.items()]
        ax1.legend(handles=legend_elements + [plt.Line2D([0], [0], marker='*', color='w', 
                markerfacecolor='r', markersize=10, label='Pareto Optimal')], 
                loc='lower left')
        
        plt.tight_layout()
        plt.savefig('precision_compression_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nPrecision analysis visualization saved as 'precision_compression_analysis.png'")

    def _save_precision_analysis(self, precision_optimal, pareto_points):
        """Save precision analysis results to file"""
        with open('precision_compression_analysis.txt', 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PRECISION-AWARE COMPRESSION ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("OPTIMAL TECHNIQUES FOR DIFFERENT PRECISION THRESHOLDS\n")
            f.write("-"*40 + "\n\n")
            
            for threshold in sorted(precision_optimal.keys(), reverse=True):
                technique = precision_optimal[threshold]
                f.write(f"Precision Threshold: >= {threshold:.0f}%\n")
                f.write(f"  Best Technique: {technique['category']}: {technique['name']}\n")
                f.write(f"  Compression Ratio: {technique['compression_ratio']:.2f}x\n")
                f.write(f"  Actual Precision: {technique['precision']:.1f}%\n")
                f.write(f"  Final Size: {technique['size_mb']:.2f} MB\n")
                f.write(f"  Size Reduction: {(1-technique['size_mb']/self.base.results['original']['size_mb'])*100:.1f}%\n")
                if 'ber' in technique:
                    f.write(f"  BER: {technique['ber']*100:.2f}%\n")
                f.write("\n")
            
            f.write("\nPARETO OPTIMAL POINTS\n")
            f.write("-"*40 + "\n")
            f.write("(Techniques that offer the best trade-off between precision and compression)\n\n")
            
            for i, point in enumerate(pareto_points[:10]):  # Top 10
                f.write(f"{i+1}. {point['category']}: {point['name']}\n")
                f.write(f"   Compression: {point['compression_ratio']:.2f}x\n")
                f.write(f"   Precision: {point['precision']:.1f}%\n")
                f.write(f"   Size: {point['size_mb']:.2f} MB\n\n")
            
            f.write("\nRECOMMENDATIONS\n")
            f.write("-"*40 + "\n\n")
            
            # Find knee point (best trade-off)
            knee_point = self._find_knee_point_precision(pareto_points)
            if knee_point:
                f.write(f"Recommended (Knee Point): {knee_point['category']}: {knee_point['name']}\n")
                f.write(f"  This offers the best balance with {knee_point['compression_ratio']:.2f}x compression\n")
                f.write(f"  at {knee_point['precision']:.1f}% precision\n\n")
            
            f.write("For different use cases:\n")
            f.write("- Critical applications (precision > 90%): Use light pruning with low amounts\n")
            f.write("- Balanced applications (precision > 70%): Consider moderate pruning or INT8 quantization\n")
            f.write("- Size-critical applications (precision > 50%): Use aggressive quantization or distillation\n")
            
        print("Precision analysis report saved as 'precision_compression_analysis.txt'")

    # =============================================================================
    # INDIVIDUAL PLOT METHODS
    # =============================================================================

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
                
            color = {'pruning': 'blue', 'quantization': 'green', 'distillation': 'orange'}[category]
            for name, stats in results.items():
                techniques.append(name.replace('_', '\n'))
                compressions.append(orig_size / stats['size_mb'])
                colors.append(color)
        
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

    def _plot_precision_comparison(self, ax):
        """Compare original vs compressed model precision"""
        categories = ['Pruning', 'Quantization', 'Distillation']
        best_precisions = []
        
        # Use ACTUAL original precision from PIV-PINN comparison
        orig_precision = self.base.results['original'].get('precision', 100.0)
        
        for category in ['pruning', 'quantization', 'distillation']:
            if category in self.base.results:
                best_precision = max(
                    stats.get('precision', 50.0) 
                    for stats in self.base.results[category].values()
                )
                best_precisions.append(best_precision)
            else:
                best_precisions.append(orig_precision)
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, [orig_precision]*len(categories), width, 
                       label=f'Original ({orig_precision:.1f}%)', alpha=0.8, color='red')
        bars2 = ax.bar(x + width/2, best_precisions, width,
                       label='Best Compressed', alpha=0.8, color='green')
        
        ax.set_xlabel('Technique Category')
        ax.set_ylabel('PIV-PINN Precision (%)')
        ax.set_title('Best Precision per Category\n(vs Original PIV-PINN Baseline)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # Adjust y-axis to actual precision range
        all_precisions = [orig_precision] + best_precisions
        y_min = max(0, min(all_precisions) - 5)
        y_max = min(100, max(all_precisions) + 5)
        ax.set_ylim(y_min, y_max)
        
        # Add precision retention ratios (not multiplication factors)
        for i, (orig, comp) in enumerate(zip([orig_precision]*len(categories), best_precisions)):
            retention = comp / orig if orig > 0 else 0
            change = comp - orig
            ax.text(i, max(orig, comp) + 1, f'Retention: {retention:.2f}\nChange: {change:+.1f}%', 
                   ha='center', va='bottom', fontsize=8)

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

    def _plot_precision_analysis(self, ax):
        """Plot precision for different compression techniques"""
        techniques = []
        precisions = []
        colors = []
        
        # Collect precision data
        color_map = {'pruning': 'blue', 'quantization': 'green', 'distillation': 'orange'}
        
        for category, results in self.base.results.items():
            if category == 'original':
                continue
                
            for name, stats in results.items():
                techniques.append(name.replace('_', '\n'))
                precisions.append(stats.get('precision', 50.0))
                colors.append(color_map[category])
        
        # Create bar plot
        bars = ax.bar(range(len(techniques)), precisions, color=colors, alpha=0.7)
        ax.set_xticks(range(len(techniques)))
        ax.set_xticklabels(techniques, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('PIV-PINN Precision (%)')
        ax.set_title('Precision by Compression Technique', fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 105)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=7)
        
        # Add reference lines
        orig_precision = self.base.results['original'].get('precision', 100.0)
        ax.axhline(y=orig_precision, color='red', linestyle='--', alpha=0.5, label='Original')
        ax.axhline(y=90, color='orange', linestyle=':', alpha=0.5, label='90% threshold')
        ax.legend()

    def _plot_technique_comparison_precision(self, ax):
        """3D scatter plot of techniques with precision"""
        try:
            compressions = []
            precisions = []
            bers = []
            colors = []
            labels = []
            
            orig_size = self.base.results['original']['size_mb']
            orig_precision = self.base.results['original'].get('precision', 100.0)
            
            # Add original
            compressions.append(1.0)
            precisions.append(orig_precision)
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
                        compressions.append(orig_size / stats['size_mb'])
                        precisions.append(stats.get('precision', 50.0))
                        bers.append(stats.get('ber', 0) * 100)
                        colors.append(color_map.get(category, 'gray'))
                        labels.append(f"{category}: {name}")
            
            # Create scatter plot
            if len(compressions) > 1:
                scatter = ax.scatter(compressions, precisions, bers, c=colors, s=100, alpha=0.7)
                
                ax.set_xlabel('Compression Ratio')
                ax.set_ylabel('PIV-PINN Precision (%)')
                ax.set_zlabel('Bit Error Rate (%)')
                ax.set_title('3D Technique Comparison\n(Compression vs Precision vs BER)', fontweight='bold')
                
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

    def _plot_precision_vs_compression(self, ax):
        """Plot precision vs compression ratio"""
        # Separate data by category
        pruning_data = {'comps': [], 'precisions': []}
        quant_data = {'comps': [], 'precisions': []}
        distill_data = {'comps': [], 'precisions': []}
        
        orig_size = self.base.results['original']['size_mb']
        
        # Collect data by category
        for category, results in self.base.results.items():
            if category == 'original':
                continue
                
            for name, stats in results.items():
                compression = orig_size / stats['size_mb']
                precision = stats.get('precision', 50.0)
                
                if compression > 1.0:  # Only plot if there's actual compression
                    if category == 'pruning':
                        pruning_data['comps'].append(compression)
                        pruning_data['precisions'].append(precision)
                    elif category == 'quantization':
                        quant_data['comps'].append(compression)
                        quant_data['precisions'].append(precision)
                    elif category == 'distillation':
                        distill_data['comps'].append(compression)
                        distill_data['precisions'].append(precision)
        
        # Plot each category with different markers
        if pruning_data['comps']:
            ax.scatter(pruning_data['comps'], pruning_data['precisions'], 
                    c='blue', marker='o', s=100, alpha=0.7, label='Pruning')
        
        if quant_data['comps']:
            ax.scatter(quant_data['comps'], quant_data['precisions'], 
                    c='green', marker='s', s=100, alpha=0.7, label='Quantization')
        
        if distill_data['comps']:
            ax.scatter(distill_data['comps'], distill_data['precisions'], 
                    c='orange', marker='^', s=100, alpha=0.7, label='Distillation')
        
        # Add trend line for all points
        all_comps = pruning_data['comps'] + quant_data['comps'] + distill_data['comps']
        all_precisions = pruning_data['precisions'] + quant_data['precisions'] + distill_data['precisions']
        
        if len(all_comps) > 3:
            z = np.polyfit(all_comps, all_precisions, 2)
            p = np.poly1d(z)
            x_trend = np.linspace(min(all_comps), max(all_comps), 100)
            ax.plot(x_trend, np.maximum(0, p(x_trend)), 'r--', alpha=0.5, label='Trend')
        
        ax.set_xlabel('Compression Ratio')
        ax.set_ylabel('PIV-PINN Precision (%)')
        ax.set_title('Precision vs Compression Trade-off', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 105)
        
        # Set reasonable axis limits
        if all_comps:
            ax.set_xlim(0.5, max(all_comps) * 1.1)

    def _create_summary_table_precision(self, ax):
        """Create summary table of compression results with precision focus"""
        ax.axis('tight')
        ax.axis('off')
        
        # Find best techniques
        orig_size = self.base.results['original']['size_mb']
        orig_precision = self.base.results['original'].get('precision', 100.0)
        
        best_overall = {'name': 'Original', 'compression': 1.0, 'size': orig_size, 'precision': orig_precision}
        best_pruning = {'name': 'None', 'compression': 1.0, 'precision': orig_precision}
        best_quant = {'name': 'None', 'compression': 1.0, 'precision': orig_precision}
        best_distill = {'name': 'None', 'compression': 1.0, 'precision': orig_precision}
        highest_precision = {'name': 'Original', 'compression': 1.0, 'precision': orig_precision}
        
        # Check all techniques
        for category in ['pruning', 'quantization', 'distillation']:
            for name, stats in self.base.results.get(category, {}).items():
                comp = orig_size / stats['size_mb']
                precision = stats.get('precision', 50.0)
                
                # Update best in category
                if category == 'pruning' and comp > best_pruning['compression']:
                    best_pruning = {'name': name, 'compression': comp, 'precision': precision}
                elif category == 'quantization' and comp > best_quant['compression']:
                    best_quant = {'name': name, 'compression': comp, 'precision': precision}
                elif category == 'distillation' and comp > best_distill['compression']:
                    best_distill = {'name': name, 'compression': comp, 'precision': precision}
                
                # Update overall best compression
                if comp > best_overall['compression']:
                    best_overall = {'name': f'{category}: {name}', 'compression': comp, 
                                'size': stats['size_mb'], 'precision': precision}
                
                # Track highest precision (among compressed models)
                if precision > highest_precision['precision'] and comp > 1.1:
                    highest_precision = {'name': f'{category}: {name}', 'compression': comp, 'precision': precision}
        
        # Create table data
        table_data = [
            ['Metric', 'Technique', 'Compression', 'Precision'],
            ['Overall Best', best_overall['name'], f"{best_overall['compression']:.2f}x", f"{best_overall['precision']:.1f}%"],
            ['Best Pruning', best_pruning['name'], f"{best_pruning['compression']:.2f}x", f"{best_pruning['precision']:.1f}%"],
            ['Best Quantization', best_quant['name'], f"{best_quant['compression']:.2f}x", f"{best_quant['precision']:.1f}%"],
            ['Best Distillation', best_distill['name'], f"{best_distill['compression']:.2f}x", f"{best_distill['precision']:.1f}%"],
            ['Highest Precision', highest_precision['name'], f"{highest_precision['compression']:.2f}x", f"{highest_precision['precision']:.1f}%"],
            ['', '', '', ''],
            ['Original Size', f"{orig_size:.2f} MB", '', ''],
            ['Best Final Size', f"{best_overall['size']:.2f} MB", '', ''],
            ['Original Precision', f"{orig_precision:.1f}%", '', ''],
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
        for row in [7, 8, 9]:
            table[(row, 0)].set_facecolor('#E3F2FD')
            table[(row, 0)].set_text_props(weight='bold')
        
        # Highlight highest precision row
        table[(5, 0)].set_facecolor('#FFF3E0')
        table[(5, 0)].set_text_props(weight='bold')
        
        ax.set_title('Compression Summary with PIV-PINN Precision Analysis', fontsize=14, fontweight='bold', pad=20)

    def _plot_precision_before_after(self, ax):
        """Plot precision comparison before and after compression techniques"""
        techniques = []
        original_precisions = []
        compressed_precisions = []
        precision_changes = []
        colors = []
        
        # Get ACTUAL original precision from PIV-PINN comparison (not assumed 100%)
        orig_precision = self.base.results['original'].get('precision', 100.0)
        print(f"DEBUG: Using original model precision: {orig_precision:.1f}% (from PIV-PINN comparison)")
        
        # Collect precision data for all techniques
        color_map = {'pruning': 'blue', 'quantization': 'green', 'distillation': 'orange'}
        
        for category, results in self.base.results.items():
            if category == 'original':
                continue
                
            for name, stats in results.items():
                compressed_precision = stats.get('precision', 50.0)
                precision_change = compressed_precision - orig_precision
                
                techniques.append(f"{category}\n{name.replace('_', ' ')}")
                original_precisions.append(orig_precision)
                compressed_precisions.append(compressed_precision)
                precision_changes.append(precision_change)
                colors.append(color_map[category])
        
        if not techniques:
            ax.text(0.5, 0.5, 'No compression techniques\navailable for comparison', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Precision Before vs After Compression', fontweight='bold')
            return
        
        # Create the before/after comparison plot
        x = np.arange(len(techniques))
        width = 0.35
        
        # Plot original and compressed precision bars
        bars1 = ax.bar(x - width/2, original_precisions, width, 
                      label=f'Original Model ({orig_precision:.1f}%)', alpha=0.8, color='lightgray', edgecolor='black')
        bars2 = ax.bar(x + width/2, compressed_precisions, width,
                      label='Compressed', alpha=0.8, color=colors, edgecolor='black')
        
        # Add precision change indicators
        for i, (orig, comp, change) in enumerate(zip(original_precisions, compressed_precisions, precision_changes)):
            # Arrow indicating change (only if significant)
            if abs(change) > 0.5:  # Show arrow if change > 0.5%
                arrow_color = 'red' if change < 0 else 'darkgreen'
                arrow_style = '↓' if change < 0 else '↑'
                
                # Position arrow between bars
                ax.annotate(arrow_style, 
                           xy=(i, max(orig, comp) + 1), 
                           ha='center', va='bottom',
                           fontsize=14, color=arrow_color, weight='bold')
                
                # Add change value
                ax.text(i, max(orig, comp) + 3, f'{change:+.1f}%', 
                       ha='center', va='bottom', fontsize=8, 
                       color=arrow_color, weight='bold')
        
        # Styling
        ax.set_xlabel('Compression Techniques')
        ax.set_ylabel('PIV-PINN Precision (%)')
        ax.set_title('Precision: Before vs After Compression\n(Based on PIV Experimental Comparison)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(techniques, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # Set y-axis limits based on actual precision values (not assuming 0-100)
        all_precisions = original_precisions + compressed_precisions
        y_min = max(0, min(all_precisions) - 5)
        y_max = min(100, max(all_precisions) + 10)
        ax.set_ylim(y_min, y_max)
        
        # Add reference lines based on original precision
        ax.axhline(y=orig_precision, color='red', linestyle='--', alpha=0.5, label='Original Baseline')
        
        # Add meaningful thresholds relative to original
        high_threshold = orig_precision * 0.95  # 95% of original
        low_threshold = orig_precision * 0.80   # 80% of original
        
        if high_threshold < y_max:
            ax.axhline(y=high_threshold, color='orange', linestyle=':', alpha=0.5, 
                      label=f'95% of Original ({high_threshold:.1f}%)')
        if low_threshold > y_min:
            ax.axhline(y=low_threshold, color='yellow', linestyle=':', alpha=0.5, 
                      label=f'80% of Original ({low_threshold:.1f}%)')
        
        # Add statistics text box with realistic expectations
        avg_change = np.mean(precision_changes)
        min_precision = min(compressed_precisions)
        max_precision = max(compressed_precisions)
        
        # Count techniques meeting different retention thresholds
        techniques_95_plus = sum(1 for p in compressed_precisions if p >= orig_precision * 0.95)
        techniques_90_plus = sum(1 for p in compressed_precisions if p >= orig_precision * 0.90)
        techniques_80_plus = sum(1 for p in compressed_precisions if p >= orig_precision * 0.80)
        
        stats_text = f"""Precision Statistics:
Original (PIV comparison): {orig_precision:.1f}%
Min after compression: {min_precision:.1f}%
Max after compression: {max_precision:.1f}%
Average change: {avg_change:+.1f}%

Retention Analysis:
≥95% of original: {techniques_95_plus}/{len(techniques)}
≥90% of original: {techniques_90_plus}/{len(techniques)}
≥80% of original: {techniques_80_plus}/{len(techniques)}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

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

    def plot_precision_detailed_comparison(self):
        """Create a standalone detailed precision comparison plot"""
        if not self.base.results:
            print("No results to plot. Run compress_all() first.")
            return
        
        # Create a comprehensive precision comparison figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Before vs After bars
        self._plot_precision_before_after(ax1)
        
        # Plot 2: Precision retention percentage
        self._plot_precision_retention(ax2)
        
        # Plot 3: Precision vs compression scatter with trend
        self._plot_precision_vs_compression_detailed(ax3)
        
        # Plot 4: Precision degradation by technique type
        self._plot_precision_degradation_by_type(ax4)
        
        plt.tight_layout()
        plt.savefig('precision_detailed_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Detailed precision comparison saved as 'precision_detailed_comparison.png'")

    def _plot_precision_retention(self, ax):
        """Plot precision retention percentage for each technique"""
        techniques = []
        retentions = []
        colors = []
        compressions = []
        
        # Use ACTUAL original precision from PIV-PINN comparison
        orig_precision = self.base.results['original'].get('precision', 100.0)
        orig_size = self.base.results['original']['size_mb']
        color_map = {'pruning': 'blue', 'quantization': 'green', 'distillation': 'orange'}
        
        for category, results in self.base.results.items():
            if category == 'original':
                continue
                
            for name, stats in results.items():
                compressed_precision = stats.get('precision', 50.0)
                # Calculate retention as percentage (compressed/original * 100)
                retention = (compressed_precision / orig_precision * 100) if orig_precision > 0 else 0
                compression = orig_size / stats['size_mb']
                
                techniques.append(f"{name.replace('_', ' ')}")
                retentions.append(retention)
                compressions.append(compression)
                colors.append(color_map[category])
        
        if not techniques:
            ax.text(0.5, 0.5, 'No techniques available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return
        
        # Create horizontal bar chart with compression info
        y_pos = np.arange(len(techniques))
        bars = ax.barh(y_pos, retentions, color=colors, alpha=0.7)
        
        # Add compression ratio labels on bars
        for i, (bar, comp) in enumerate(zip(bars, compressions)):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                   f'{comp:.1f}x', ha='left', va='center', fontsize=8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(techniques, fontsize=9)
        ax.set_xlabel('Precision Retention (%)')
        ax.set_title(f'Precision Retention by Technique\n(Original baseline: {orig_precision:.1f}%)', fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add reference lines based on retention percentages
        ax.axvline(x=100, color='red', linestyle='--', alpha=0.5, label='Perfect Retention (100%)')
        ax.axvline(x=95, color='orange', linestyle=':', alpha=0.5, label='95% Retention')
        ax.axvline(x=90, color='yellow', linestyle=':', alpha=0.5, label='90% Retention')
        ax.legend()
        
        # Set reasonable x-axis limits
        max_retention = max(retentions) if retentions else 100
        ax.set_xlim(0, min(110, max_retention * 1.1))

    def _plot_precision_vs_compression_detailed(self, ax):
        """Detailed precision vs compression with trend analysis"""
        # Collect data
        all_data = {'pruning': {'comp': [], 'prec': [], 'names': []},
                   'quantization': {'comp': [], 'prec': [], 'names': []},
                   'distillation': {'comp': [], 'prec': [], 'names': []}}
        
        orig_size = self.base.results['original']['size_mb']
        
        for category, results in self.base.results.items():
            if category == 'original':
                continue
                
            for name, stats in results.items():
                compression = orig_size / stats['size_mb']
                precision = stats.get('precision', 50.0)
                
                if compression > 1.0:
                    all_data[category]['comp'].append(compression)
                    all_data[category]['prec'].append(precision)
                    all_data[category]['names'].append(name)
        
        # Plot each category
        colors = {'pruning': 'blue', 'quantization': 'green', 'distillation': 'orange'}
        markers = {'pruning': 'o', 'quantization': 's', 'distillation': '^'}
        
        for category, data in all_data.items():
            if data['comp']:
                scatter = ax.scatter(data['comp'], data['prec'], 
                                   c=colors[category], marker=markers[category], 
                                   s=100, alpha=0.7, label=category.title(), 
                                   edgecolors='black', linewidth=0.5)
                
                # Add labels for each point
                for i, (comp, prec, name) in enumerate(zip(data['comp'], data['prec'], data['names'])):
                    ax.annotate(name.replace('_', '\n'), (comp, prec), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=7, alpha=0.8)
        
        # Add trend lines for each category
        for category, data in all_data.items():
            if len(data['comp']) > 2:
                # Fit polynomial trend
                z = np.polyfit(data['comp'], data['prec'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(data['comp']), max(data['comp']), 100)
                ax.plot(x_trend, p(x_trend), '--', color=colors[category], 
                       alpha=0.5, linewidth=2, label=f'{category.title()} trend')
        
        ax.set_xlabel('Compression Ratio')
        ax.set_ylabel('PIV-PINN Precision (%)')
        ax.set_title('Precision vs Compression Detailed Analysis', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 105)

    def _plot_precision_degradation_by_type(self, ax):
        """Plot precision degradation grouped by technique type"""
        categories = []
        degradations = []
        colors_list = []
        
        # Use ACTUAL original precision from PIV-PINN comparison
        orig_precision = self.base.results['original'].get('precision', 100.0)
        color_map = {'pruning': 'blue', 'quantization': 'green', 'distillation': 'orange'}
        
        # Calculate average degradation per category
        for category, results in self.base.results.items():
            if category == 'original':
                continue
            
            category_degradations = []
            for name, stats in results.items():
                compressed_precision = stats.get('precision', 50.0)
                degradation = orig_precision - compressed_precision
                category_degradations.append(degradation)
            
            if category_degradations:
                avg_degradation = np.mean(category_degradations)
                min_degradation = np.min(category_degradations)
                max_degradation = np.max(category_degradations)
                
                categories.append(category.title())
                degradations.append(avg_degradation)
                colors_list.append(color_map[category])
                
                # Add error bars showing range
                ax.bar(category.title(), avg_degradation, 
                      color=color_map[category], alpha=0.7,
                      yerr=[[avg_degradation - min_degradation], [max_degradation - avg_degradation]],
                      capsize=5, error_kw={'alpha': 0.8})
        
        # Manual bar creation since we already plotted with error bars
        if not categories:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            return
        
        ax.set_ylabel('Precision Degradation (percentage points)')
        ax.set_title(f'Average Precision Loss by Technique Type\n(Original baseline: {orig_precision:.1f}%)', fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add reference lines based on original precision
        ax.axhline(y=0, color='green', linestyle='-', alpha=0.8, label='No degradation')
        
        # Set meaningful thresholds as percentage points, not absolute values
        threshold_5pct = orig_precision * 0.05  # 5% of original precision as threshold
        threshold_10pct = orig_precision * 0.10  # 10% of original precision as threshold
        
        ax.axhline(y=threshold_5pct, color='orange', linestyle='--', alpha=0.5, 
                  label=f'5% loss threshold ({threshold_5pct:.1f} pts)')
        ax.axhline(y=threshold_10pct, color='red', linestyle='--', alpha=0.5, 
                  label=f'10% loss threshold ({threshold_10pct:.1f} pts)')
        ax.legend()
        
        # Add value labels on bars with context
        for i, (cat, deg) in enumerate(zip(categories, degradations)):
            percentage_loss = (deg / orig_precision * 100) if orig_precision > 0 else 0
            ax.text(i, deg + 0.2, f'{deg:.1f} pts\n({percentage_loss:.1f}%)', 
                   ha='center', va='bottom', fontweight='bold', fontsize=8)

    def analyze_ber_compression_tradeoff(self, ber_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5], 
                                        save_results=True):
        """Legacy BER analysis - use analyze_precision_compression_tradeoff instead"""
        print("⚠️ analyze_ber_compression_tradeoff is deprecated. Use analyze_precision_compression_tradeoff instead.")
        # Convert BER thresholds to precision thresholds (inverse relationship)
        precision_thresholds = [max(0, 100 - ber*200) for ber in ber_thresholds]
        return self.analyze_precision_compression_tradeoff(precision_thresholds, save_results)

    def _plot_ber_analysis(self, ax):
        """Legacy method - redirects to precision analysis"""
        return self._plot_precision_analysis(ax)
    
    def _plot_ber_vs_compression(self, ax):
        """Legacy method - redirects to precision vs compression"""
        return self._plot_precision_vs_compression(ax)
    
    def _create_summary_table(self, ax):
        """Legacy method - redirects to precision summary table"""
        return self._create_summary_table_precision(ax)

    def _plot_ber_pareto_analysis(self, all_techniques, pareto_points, ber_optimal):
        """Legacy BER Pareto analysis - converts to precision"""
        print("⚠️ Converting BER analysis to precision analysis...")
        
        # Convert BER optimal to precision optimal
        precision_optimal = {}
        for ber_threshold, technique in ber_optimal.items():
            precision_threshold = max(0, 100 - ber_threshold*200)
            precision_optimal[precision_threshold] = technique
        
        return self._plot_precision_pareto_analysis(all_techniques, pareto_points, precision_optimal)

    def _save_ber_analysis(self, ber_optimal, pareto_points):
        """Legacy BER analysis save - converts to precision"""
        print("⚠️ Converting BER analysis to precision analysis...")
        
        # Convert BER optimal to precision optimal
        precision_optimal = {}
        for ber_threshold, technique in ber_optimal.items():
            precision_threshold = max(0, 100 - ber_threshold*200)
            precision_optimal[precision_threshold] = technique
        
        return self._save_precision_analysis(precision_optimal, pareto_points)

    def _find_knee_point(self, pareto_points):
        """Legacy method - redirects to precision knee point"""
        return self._find_knee_point_precision(pareto_points)