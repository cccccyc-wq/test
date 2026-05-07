#!/usr/bin/env python3
"""
Results Comparison Script
Compares benchmark results across different configurations
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResultsComparator:
    """Compare benchmark results"""
    
    def __init__(self):
        """Initialize comparator"""
        self.results: Dict[str, Dict] = {}
    
    def load_results(self, name: str, json_path: str) -> bool:
        """Load benchmark results from JSON file"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            self.results[name] = {
                'metrics': data.get('metrics', []),
                'summary': data.get('summary', {}),
            }
            logger.info(f"Loaded results '{name}' from {json_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load results from {json_path}: {e}")
            return False
    
    def compare(self) -> Dict:
        """Compare all loaded results"""
        if len(self.results) < 2:
            logger.warning("Need at least 2 results to compare")
            return {}
        
        comparison = {}
        result_names = list(self.results.keys())
        
        # Compare latency
        latencies = {}
        for name, data in self.results.items():
            if 'latency_ms' in data['summary']:
                latencies[name] = data['summary']['latency_ms'].get('mean', 0)
        
        if latencies:
            min_latency = min(latencies.values())
            comparison['latency_comparison'] = {
                name: {
                    'mean_ms': lat,
                    'improvement_%': ((lat - min_latency) / lat * 100) if lat > 0 else 0,
                }
                for name, lat in latencies.items()
            }
        
        # Compare throughput
        throughputs = {}
        for name, data in self.results.items():
            if 'throughput_tokens_per_sec' in data['summary']:
                throughputs[name] = data['summary']['throughput_tokens_per_sec'].get('mean', 0)
        
        if throughputs:
            max_throughput = max(throughputs.values())
            comparison['throughput_comparison'] = {
                name: {
                    'mean_tokens_per_sec': tps,
                    'performance_ratio': (tps / max_throughput) if max_throughput > 0 else 0,
                }
                for name, tps in throughputs.items()
            }
        
        return comparison
    
    def print_comparison(self) -> None:
        """Print comparison results"""
        comparison = self.compare()
        
        print("\n" + "="*60)
        print("BENCHMARK RESULTS COMPARISON")
        print("="*60 + "\n")
        
        if 'latency_comparison' in comparison:
            print("LATENCY COMPARISON (ms)")
            print("-"*60)
            for name, data in comparison['latency_comparison'].items():
                print(f"{name:20s}: {data['mean_ms']:10.2f}ms "
                      f"(Δ {data['improvement_%']:+6.1f}%)")
            print()
        
        if 'throughput_comparison' in comparison:
            print("THROUGHPUT COMPARISON (tokens/sec)")
            print("-"*60)
            for name, data in comparison['throughput_comparison'].items():
                print(f"{name:20s}: {data['mean_tokens_per_sec']:12.0f} "
                      f"({data['performance_ratio']*100:5.1f}%)")
            print()
        
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Compare benchmark results')
    parser.add_argument('--baseline', required=True, help='Baseline results JSON file')
    parser.add_argument('--compare', required=True, nargs='+', 
                       help='Results to compare (JSON files)')
    
    args = parser.parse_args()
    
    comparator = ResultsComparator()
    
    # Load baseline
    if not comparator.load_results('baseline', args.baseline):
        return 1
    
    # Load comparison results
    for i, compare_file in enumerate(args.compare):
        name = f"config_{i+1}"
        if not comparator.load_results(name, compare_file):
            logger.warning(f"Skipped {compare_file}")
            continue
    
    # Print comparison
    comparator.print_comparison()
    
    return 0


if __name__ == '__main__':
    exit(main())
