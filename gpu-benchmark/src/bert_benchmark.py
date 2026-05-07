#!/usr/bin/env python3
"""
BERT Benchmark Module
Core BERT model benchmarking functionality
"""

import torch
import time
import logging
from typing import Tuple, Dict, List
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertModel
from tqdm import tqdm

from .config import Config
from .gpu_validator import GPUValidator
from .performance_monitor import PerformanceMonitor, PerformanceMetrics

logger = logging.getLogger(__name__)


class BertBenchmark:
    """BERT model benchmarking"""
    
    def __init__(self, config: Config, device_id: int = 0):
        """Initialize BERT benchmark"""
        self.config = config
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        
        # Initialize validators and monitors
        self.gpu_validator = GPUValidator()
        self.performance_monitor = PerformanceMonitor(device_id=device_id)
        
        # Model and tokenizer
        self.model = None
        self.tokenizer = None
        
        logger.info(f"BertBenchmark initialized on device: {self.device}")
    
    def setup(self) -> bool:
        """Setup and validate GPU, load model"""
        # Validate GPU
        if not self.gpu_validator.validate():
            logger.error("GPU validation failed")
            return False
        
        self.gpu_validator.print_info(self.device_id)
        
        try:
            # Load model and tokenizer
            logger.info(f"Loading model: {self.config.model.name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
            self.model = AutoModelForMaskedLM.from_pretrained(self.config.model.name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def prepare_input(self, batch_size: int, sequence_length: int) -> Dict:
        """Prepare random input for benchmarking"""
        input_ids = torch.randint(0, self.config.model.vocab_size, 
                                  (batch_size, sequence_length)).to(self.device)
        attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long).to(self.device)
        token_type_ids = torch.zeros(batch_size, sequence_length, dtype=torch.long).to(self.device)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }
    
    def run_benchmark(self, batch_size: int, sequence_length: int, precision: str) -> PerformanceMetrics:
        """Run single benchmark"""
        if precision == 'fp16':
            self.model = self.model.half()
        elif precision == 'fp32':
            self.model = self.model.float()
        
        # Warmup
        logger.debug(f"Warmup: batch_size={batch_size}, seq_len={sequence_length}, precision={precision}")
        for _ in range(self.config.benchmark.num_warmup_steps):
            inputs = self.prepare_input(batch_size, sequence_length)
            if precision == 'fp16':
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        _ = self.model(**inputs)
            else:
                with torch.no_grad():
                    _ = self.model(**inputs)
        
        # Benchmark
        logger.debug(f"Benchmarking: batch_size={batch_size}, seq_len={sequence_length}, precision={precision}")
        torch.cuda.synchronize()
        
        latencies = []
        for _ in range(self.config.benchmark.num_bench_steps):
            inputs = self.prepare_input(batch_size, sequence_length)
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            if precision == 'fp16':
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        _ = self.model(**inputs)
            else:
                with torch.no_grad():
                    _ = self.model(**inputs)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate metrics
        avg_latency = sum(latencies) / len(latencies)
        metrics = self.performance_monitor.record_metrics(
            batch_size=batch_size,
            sequence_length=sequence_length,
            precision=precision,
            latency_ms=avg_latency
        )
        
        return metrics
    
    def run_comprehensive_benchmark(self) -> List[PerformanceMetrics]:
        """Run comprehensive benchmark with all parameter combinations"""
        all_metrics = []
        
        total_combinations = (len(self.config.benchmark.batch_sizes) * 
                             len(self.config.benchmark.sequence_lengths) * 
                             len(self.config.benchmark.precisions) *
                             self.config.benchmark.num_iterations)
        
        pbar = tqdm(total=total_combinations, desc="Benchmarking")
        
        for precision in self.config.benchmark.precisions:
            for batch_size in self.config.benchmark.batch_sizes:
                for sequence_length in self.config.benchmark.sequence_lengths:
                    for iteration in range(self.config.benchmark.num_iterations):
                        try:
                            metrics = self.run_benchmark(batch_size, sequence_length, precision)
                            all_metrics.append(metrics)
                            
                            log_msg = (f"BS={batch_size}, SL={sequence_length}, "
                                      f"{precision}: {metrics.latency_ms:.2f}ms, "
                                      f"Throughput: {metrics.throughput_tokens_per_sec:.0f} tokens/sec")
                            logger.info(log_msg)
                        
                        except Exception as e:
                            logger.error(f"Benchmark failed: {e}")
                        
                        finally:
                            pbar.update(1)
        
        pbar.close()
        return all_metrics
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.performance_monitor.stop_monitoring()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Benchmark cleanup completed")
