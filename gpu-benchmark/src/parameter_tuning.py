#!/usr/bin/env python3
"""
Parameter Tuning Module
Analyzes impact of different parameters on performance
"""

import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from .bert_benchmark import BertBenchmark
from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class ParameterImpactResult:
    """Parameter impact analysis result"""
    parameter_name: str
    parameter_values: List
    latency_values: List[float]
    throughput_values: List[float]
    gpu_utilization_values: List[float]
    
    def calculate_sensitivity(self) -> float:
        """Calculate parameter sensitivity (throughput improvement %)"""
        if not self.throughput_values or len(self.throughput_values) < 2:
            return 0.0
        
        min_throughput = min(self.throughput_values)
        max_throughput = max(self.throughput_values)
        
        if min_throughput == 0:
            return 0.0
        
        return ((max_throughput - min_throughput) / min_throughput) * 100


class ParameterTuning:
    """Analyze parameter impact on performance"""
    
    def __init__(self, benchmark: BertBenchmark, config: Config):
        """Initialize parameter tuning"""
        self.benchmark = benchmark
        self.config = config
        self.results: Dict[str, ParameterImpactResult] = {}
    
    def analyze_batch_size_impact(self, 
                                  test_batch_sizes: List[int],
                                  sequence_length: int = 128,
                                  precision: str = 'fp16') -> ParameterImpactResult:
        """Analyze impact of batch size on performance"""
        logger.info(f"Analyzing batch size impact (seq_len={sequence_length}, precision={precision})")
        
        latencies = []
        throughputs = []
        gpu_utils = []
        
        for batch_size in test_batch_sizes:
            try:
                metrics = self.benchmark.run_benchmark(batch_size, sequence_length, precision)
                latencies.append(metrics.latency_ms)
                throughputs.append(metrics.throughput_tokens_per_sec)
                gpu_utils.append(metrics.gpu_utilization)
            except Exception as e:
                logger.warning(f"Failed to test batch_size={batch_size}: {e}")
        
        result = ParameterImpactResult(
            parameter_name='batch_size',
            parameter_values=test_batch_sizes[:len(latencies)],
            latency_values=latencies,
            throughput_values=throughputs,
            gpu_utilization_values=gpu_utils
        )
        
        self.results['batch_size'] = result
        return result
    
    def analyze_sequence_length_impact(self,
                                       test_sequence_lengths: List[int],
                                       batch_size: int = 32,
                                       precision: str = 'fp16') -> ParameterImpactResult:
        """Analyze impact of sequence length on performance"""
        logger.info(f"Analyzing sequence length impact (batch_size={batch_size}, precision={precision})")
        
        latencies = []
        throughputs = []
        gpu_utils = []
        
        for seq_len in test_sequence_lengths:
            try:
                metrics = self.benchmark.run_benchmark(batch_size, seq_len, precision)
                latencies.append(metrics.latency_ms)
                throughputs.append(metrics.throughput_tokens_per_sec)
                gpu_utils.append(metrics.gpu_utilization)
            except Exception as e:
                logger.warning(f"Failed to test sequence_length={seq_len}: {e}")
        
        result = ParameterImpactResult(
            parameter_name='sequence_length',
            parameter_values=test_sequence_lengths[:len(latencies)],
            latency_values=latencies,
            throughput_values=throughputs,
            gpu_utilization_values=gpu_utils
        )
        
        self.results['sequence_length'] = result
        return result
    
    def analyze_precision_impact(self,
                                test_precisions: List[str],
                                batch_size: int = 32,
                                sequence_length: int = 128) -> ParameterImpactResult:
        """Analyze impact of precision on performance"""
        logger.info(f"Analyzing precision impact (batch_size={batch_size}, seq_len={sequence_length})")
        
        latencies = []
        throughputs = []
        gpu_utils = []
        
        for precision in test_precisions:
            try:
                metrics = self.benchmark.run_benchmark(batch_size, sequence_length, precision)
                latencies.append(metrics.latency_ms)
                throughputs.append(metrics.throughput_tokens_per_sec)
                gpu_utils.append(metrics.gpu_utilization)
            except Exception as e:
                logger.warning(f"Failed to test precision={precision}: {e}")
        
        result = ParameterImpactResult(
            parameter_name='precision',
            parameter_values=test_precisions[:len(latencies)],
            latency_values=latencies,
            throughput_values=throughputs,
            gpu_utilization_values=gpu_utils
        )
        
        self.results['precision'] = result
        return result
    
    def get_recommendation(self) -> Dict:
        """Get performance optimization recommendations"""
        recommendations = {
            'batch_size': self._get_batch_size_recommendation(),
            'sequence_length': self._get_sequence_length_recommendation(),
            'precision': self._get_precision_recommendation(),
        }
        return recommendations
    
    def _get_batch_size_recommendation(self) -> str:
        """Recommend optimal batch size"""
        if 'batch_size' not in self.results:
            return "No batch size analysis performed"
        
        result = self.results['batch_size']
        best_idx = np.argmax(result.throughput_values)
        best_batch_size = result.parameter_values[best_idx]
        best_throughput = result.throughput_values[best_idx]
        
        return (f"Optimal batch size: {best_batch_size} "
                f"(throughput: {best_throughput:.0f} tokens/sec)")
    
    def _get_sequence_length_recommendation(self) -> str:
        """Recommend optimal sequence length"""
        if 'sequence_length' not in self.results:
            return "No sequence length analysis performed"
        
        result = self.results['sequence_length']
        best_idx = np.argmax(result.throughput_values)
        best_seq_len = result.parameter_values[best_idx]
        best_throughput = result.throughput_values[best_idx]
        
        return (f"Optimal sequence length: {best_seq_len} "
                f"(throughput: {best_throughput:.0f} tokens/sec)")
    
    def _get_precision_recommendation(self) -> str:
        """Recommend optimal precision"""
        if 'precision' not in self.results:
            return "No precision analysis performed"
        
        result = self.results['precision']
        best_idx = np.argmax(result.throughput_values)
        best_precision = result.parameter_values[best_idx]
        best_throughput = result.throughput_values[best_idx]
        
        return (f"Optimal precision: {best_precision} "
                f"(throughput: {best_throughput:.0f} tokens/sec)")
