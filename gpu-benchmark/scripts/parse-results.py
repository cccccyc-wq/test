#!/usr/bin/env python3
"""
Results Parsing and Analysis Script
Parses benchmark results and generates reports
"""

import json
import csv
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyze and parse benchmark results"""
    
    def __init__(self, metrics_list: List[Dict], summary: Dict = None):
        """Initialize analyzer"""
        self.metrics_list = metrics_list
        self.summary = summary or {}
    
    def save_to_csv(self, csv_path: str) -> None:
        """Save results to CSV file"""
        if not self.metrics_list:
            logger.warning("No metrics to save")
            return
        
        try:
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.metrics_list[0].keys())
                writer.writeheader()
                writer.writerows(self.metrics_list)
            
            logger.info(f"Results saved to CSV: {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
    
    def save_to_json(self, json_path: str) -> None:
        """Save results to JSON file"""
        try:
            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': self.metrics_list,
                'summary': self.summary,
            }
            
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Results saved to JSON: {json_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
    
    def save_summary(self, summary_path: str) -> None:
        """Save summary report to text file"""
        try:
            Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(summary_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("GPU BENCHMARK TEST REPORT\n")
                f.write("="*60 + "\n\n")
                
                f.write(f"Report Generated: {datetime.now().isoformat()}\n\n")
                
                if self.summary:
                    f.write("SUMMARY STATISTICS\n")
                    f.write("-"*60 + "\n")
                    
                    if 'latency_ms' in self.summary:
                        f.write(f"Latency (ms):\n")
                        f.write(f"  Mean: {self.summary['latency_ms'].get('mean', 0):.2f}\n")
                        f.write(f"  Min:  {self.summary['latency_ms'].get('min', 0):.2f}\n")
                        f.write(f"  Max:  {self.summary['latency_ms'].get('max', 0):.2f}\n")
                        f.write(f"  Std:  {self.summary['latency_ms'].get('std', 0):.2f}\n\n")
                    
                    if 'throughput_tokens_per_sec' in self.summary:
                        f.write(f"Throughput (tokens/sec):\n")
                        f.write(f"  Mean: {self.summary['throughput_tokens_per_sec'].get('mean', 0):.0f}\n")
                        f.write(f"  Min:  {self.summary['throughput_tokens_per_sec'].get('min', 0):.0f}\n")
                        f.write(f"  Max:  {self.summary['throughput_tokens_per_sec'].get('max', 0):.0f}\n\n")
                    
                    if 'gpu_utilization_%' in self.summary:
                        f.write(f"GPU Utilization (%):\n")
                        f.write(f"  Mean: {self.summary['gpu_utilization_%'].get('mean', 0):.1f}%\n")
                        f.write(f"  Max:  {self.summary['gpu_utilization_%'].get('max', 0):.1f}%\n\n")
                    
                    if 'gpu_power_watts' in self.summary:
                        f.write(f"GPU Power (watts):\n")
                        f.write(f"  Mean: {self.summary['gpu_power_watts'].get('mean', 0):.1f}W\n")
                        f.write(f"  Max:  {self.summary['gpu_power_watts'].get('max', 0):.1f}W\n\n")
                
                f.write("="*60 + "\n")
            
            logger.info(f"Summary saved to: {summary_path}")
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")


def main():
    parser = argparse.ArgumentParser(description='Parse benchmark results')
    parser.add_argument('--results-dir', help='Results directory')
    parser.add_argument('--json-file', help='JSON results file')
    
    args = parser.parse_args()
    
    if args.json_file:
        # Load from JSON file
        try:
            with open(args.json_file, 'r') as f:
                data = json.load(f)
            
            analyzer = ResultsAnalyzer(data.get('metrics', []), data.get('summary', {}))
            
            # Generate CSV
            csv_path = Path(args.json_file).parent / 'results_detailed.csv'
            analyzer.save_to_csv(str(csv_path))
            
            # Generate summary
            summary_path = Path(args.json_file).parent / 'report.txt'
            analyzer.save_summary(str(summary_path))
            
            logger.info("Results parsing completed")
        
        except Exception as e:
            logger.error(f"Failed to parse results: {e}")
            return 1
    
    elif args.results_dir:
        logger.info(f"Analyzing results in {args.results_dir}")
        # Find and analyze all result files
        results_path = Path(args.results_dir)
        
        # Look for JSON files
        json_files = list(results_path.glob('*.json'))
        if json_files:
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    analyzer = ResultsAnalyzer(data.get('metrics', []), data.get('summary', {}))
                    
                    # Generate reports
                    csv_path = json_file.parent / f"{json_file.stem}.csv"
                    analyzer.save_to_csv(str(csv_path))
                    
                    summary_path = json_file.parent / f"{json_file.stem}_report.txt"
                    analyzer.save_summary(str(summary_path))
                
                except Exception as e:
                    logger.error(f"Failed to process {json_file}: {e}")
        else:
            logger.warning("No JSON files found in results directory")
    
    else:
        logger.error("Please provide either --json-file or --results-dir")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
