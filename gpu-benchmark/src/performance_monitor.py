#!/usr/bin/env python3
"""
Performance Monitoring Module
Monitors GPU performance metrics during benchmark execution
"""

import torch
import time
import threading
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    timestamp: str = ""
    batch_size: int = 0
    sequence_length: int = 0
    precision: str = ""
    latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_power_watts: float = 0.0
    gpu_temperature_c: float = 0.0
    cpu_memory_used_mb: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'batch_size': self.batch_size,
            'sequence_length': self.sequence_length,
            'precision': self.precision,
            'latency_ms': round(self.latency_ms, 4),
            'throughput_tokens_per_sec': round(self.throughput_tokens_per_sec, 2),
            'gpu_utilization_%': round(self.gpu_utilization, 2),
            'gpu_memory_used_mb': round(self.gpu_memory_used_mb, 2),
            'gpu_memory_total_mb': round(self.gpu_memory_total_mb, 2),
            'gpu_power_watts': round(self.gpu_power_watts, 2),
            'gpu_temperature_c': round(self.gpu_temperature_c, 2),
            'cpu_memory_used_mb': round(self.cpu_memory_used_mb, 2),
        }


class PerformanceMonitor:
    """Monitor GPU and system performance"""
    
    def __init__(self, device_id: int = 0, sample_interval: float = 0.1):
        """Initialize performance monitor"""
        self.device_id = device_id
        self.sample_interval = sample_interval
        self.metrics_list: List[PerformanceMetrics] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.current_metrics: Optional[PerformanceMetrics] = None
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.pynvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                self.pynvml_available = True
            except Exception as e:
                logger.warning(f"Failed to initialize PYNVML: {e}")
                self.pynvml_available = False
        else:
            self.pynvml_available = False
    
    def start_monitoring(self) -> None:
        """Start background monitoring thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring thread"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop"""
        while self.monitoring:
            try:
                self._update_metrics()
                time.sleep(self.sample_interval)
            except Exception as e:
                logger.warning(f"Error in monitoring loop: {e}")
    
    def _update_metrics(self) -> None:
        """Update current metrics"""
        if self.current_metrics is None:
            return
        
        try:
            # GPU memory
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu_memory_used = torch.cuda.memory_allocated(self.device_id) / 1024**2
                gpu_memory_total = torch.cuda.get_device_properties(self.device_id).total_memory / 1024**2
                self.current_metrics.gpu_memory_used_mb = gpu_memory_used
                self.current_metrics.gpu_memory_total_mb = gpu_memory_total
            
            # PYNVML metrics
            if self.pynvml_available:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.pynvml_handle)
                self.current_metrics.gpu_utilization = util.gpu
                
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(self.pynvml_handle) / 1000.0
                    self.current_metrics.gpu_power_watts = power
                except:
                    pass
                
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(self.pynvml_handle, 0)
                    self.current_metrics.gpu_temperature_c = temp
                except:
                    pass
        
        except Exception as e:
            logger.debug(f"Error updating metrics: {e}")
    
    def record_metrics(self, 
                      batch_size: int,
                      sequence_length: int,
                      precision: str,
                      latency_ms: float) -> PerformanceMetrics:
        """Record performance metrics for a benchmark run"""
        metrics = PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            batch_size=batch_size,
            sequence_length=sequence_length,
            precision=precision,
            latency_ms=latency_ms,
            throughput_tokens_per_sec=(batch_size * sequence_length) / (latency_ms / 1000.0),
        )
        
        # Update with current metrics
        self.current_metrics = metrics
        self._update_metrics()
        
        self.metrics_list.append(metrics)
        return metrics
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of all recorded metrics"""
        if not self.metrics_list:
            return {}
        
        latencies = [m.latency_ms for m in self.metrics_list]
        throughputs = [m.throughput_tokens_per_sec for m in self.metrics_list]
        gpu_utils = [m.gpu_utilization for m in self.metrics_list if m.gpu_utilization > 0]
        powers = [m.gpu_power_watts for m in self.metrics_list if m.gpu_power_watts > 0]
        
        return {
            'count': len(self.metrics_list),
            'latency_ms': {
                'mean': np.mean(latencies),
                'min': np.min(latencies),
                'max': np.max(latencies),
                'std': np.std(latencies),
            },
            'throughput_tokens_per_sec': {
                'mean': np.mean(throughputs),
                'min': np.min(throughputs),
                'max': np.max(throughputs),
            },
            'gpu_utilization_%': {
                'mean': np.mean(gpu_utils) if gpu_utils else 0,
                'max': np.max(gpu_utils) if gpu_utils else 0,
            },
            'gpu_power_watts': {
                'mean': np.mean(powers) if powers else 0,
                'max': np.max(powers) if powers else 0,
            },
        }
    
    def get_metrics_list(self) -> List[Dict]:
        """Get all recorded metrics as list of dicts"""
        return [m.to_dict() for m in self.metrics_list]
    
    def clear_metrics(self) -> None:
        """Clear recorded metrics"""
        self.metrics_list.clear()
