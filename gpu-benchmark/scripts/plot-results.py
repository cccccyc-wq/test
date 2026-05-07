#!/usr/bin/env python3
"""
Results Visualization Script
Generates plots and charts from benchmark results
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResultsVisualizer:
    """Visualize benchmark results"""
    
    def __init__(self, metrics_list: List[Dict], output_dir: str = './plots'):
        """Initialize visualizer"""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib and seaborn are required for visualization")
        
        self.metrics_list = metrics_list
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        sns.set_theme(style="darkgrid")
    
    def plot_latency_vs_batch_size(self) -> None:
        """Plot latency vs batch size"""
        try:
            # Group by sequence length and precision
            data_by_config = {}
            
            for metric in self.metrics_list:
                key = (metric['sequence_length'], metric['precision'])
                if key not in data_by_config:
                    data_by_config[key] = {'batch_sizes': [], 'latencies': []}
                
                data_by_config[key]['batch_sizes'].append(metric['batch_size'])
                data_by_config[key]['latencies'].append(metric['latency_ms'])
            
            fig, axes = plt.subplots(len(data_by_config), 1, figsize=(10, 5*len(data_by_config)))
            if len(data_by_config) == 1:
                axes = [axes]
            
            for idx, ((seq_len, precision), data) in enumerate(data_by_config.items()):
                axes[idx].plot(data['batch_sizes'], data['latencies'], marker='o', linewidth=2)
                axes[idx].set_xlabel('Batch Size')
                axes[idx].set_ylabel('Latency (ms)')
                axes[idx].set_title(f'Latency vs Batch Size (Seq Len={seq_len}, Precision={precision})')
                axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'latency_vs_batch_size.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: latency_vs_batch_size.png")
            plt.close()
        
        except Exception as e:
            logger.error(f"Failed to plot latency vs batch size: {e}")
    
    def plot_throughput_vs_batch_size(self) -> None:
        """Plot throughput vs batch size"""
        try:
            # Group by sequence length and precision
            data_by_config = {}
            
            for metric in self.metrics_list:
                key = (metric['sequence_length'], metric['precision'])
                if key not in data_by_config:
                    data_by_config[key] = {'batch_sizes': [], 'throughputs': []}
                
                data_by_config[key]['batch_sizes'].append(metric['batch_size'])
                data_by_config[key]['throughputs'].append(metric['throughput_tokens_per_sec'])
            
            fig, axes = plt.subplots(len(data_by_config), 1, figsize=(10, 5*len(data_by_config)))
            if len(data_by_config) == 1:
                axes = [axes]
            
            for idx, ((seq_len, precision), data) in enumerate(data_by_config.items()):
                axes[idx].plot(data['batch_sizes'], data['throughputs'], marker='s', linewidth=2, color='green')
                axes[idx].set_xlabel('Batch Size')
                axes[idx].set_ylabel('Throughput (tokens/sec)')
                axes[idx].set_title(f'Throughput vs Batch Size (Seq Len={seq_len}, Precision={precision})')
                axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'throughput_vs_batch_size.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: throughput_vs_batch_size.png")
            plt.close()
        
        except Exception as e:
            logger.error(f"Failed to plot throughput vs batch size: {e}")
    
    def plot_gpu_utilization(self) -> None:
        """Plot GPU utilization"""
        try:
            batch_sizes = [m['batch_size'] for m in self.metrics_list]
            gpu_utils = [m['gpu_utilization_%'] for m in self.metrics_list]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(batch_sizes, gpu_utils, s=100, alpha=0.6, color='orange')
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('GPU Utilization (%)')
            ax.set_title('GPU Utilization vs Batch Size')
            ax.set_ylim([0, 100])
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'gpu_utilization.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: gpu_utilization.png")
            plt.close()
        
        except Exception as e:
            logger.error(f"Failed to plot GPU utilization: {e}")
    
    def plot_precision_comparison(self) -> None:
        """Plot precision comparison"""
        try:
            # Group by precision
            data_by_precision = {}
            
            for metric in self.metrics_list:
                precision = metric['precision']
                if precision not in data_by_precision:
                    data_by_precision[precision] = {'throughputs': []}
                data_by_precision[precision]['throughputs'].append(metric['throughput_tokens_per_sec'])
            
            precisions = list(data_by_precision.keys())
            avg_throughputs = [np.mean(data_by_precision[p]['throughputs']) for p in precisions]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(precisions, avg_throughputs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_ylabel('Average Throughput (tokens/sec)')
            ax.set_title('Average Throughput by Precision')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}',
                       ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'precision_comparison.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: precision_comparison.png")
            plt.close()
        
        except Exception as e:
            logger.error(f"Failed to plot precision comparison: {e}")
    
    def generate_all_plots(self) -> None:
        """Generate all plots"""
        logger.info("Generating plots...")
        self.plot_latency_vs_batch_size()
        self.plot_throughput_vs_batch_size()
        self.plot_gpu_utilization()
        self.plot_precision_comparison()
        logger.info(f"All plots saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Plot benchmark results')
    parser.add_argument('--json-file', required=True, help='Results JSON file')
    parser.add_argument('--output-dir', default='./plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib and seaborn are required. Install with: pip install matplotlib seaborn")
        return 1
    
    try:
        with open(args.json_file, 'r') as f:
            data = json.load(f)
        
        metrics_list = data.get('metrics', [])
        if not metrics_list:
            logger.error("No metrics found in JSON file")
            return 1
        
        visualizer = ResultsVisualizer(metrics_list, output_dir=args.output_dir)
        visualizer.generate_all_plots()
        
        logger.info("Visualization completed")
        return 0
    
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
