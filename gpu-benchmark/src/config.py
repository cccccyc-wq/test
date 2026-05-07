#!/usr/bin/env python3
"""
Configuration Management Module
Handles loading, validation, and management of benchmark configurations
"""

import os
import yaml
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    type: str = "bert"
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    gradient_checkpointing: bool = False


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    batch_sizes: List[int]
    sequence_lengths: List[int]
    precisions: List[str]
    num_warmup_steps: int = 100
    num_bench_steps: int = 1000
    num_iterations: int = 3


@dataclass
class DeviceConfig:
    """Device configuration"""
    device_type: str = "cuda"
    device_id: int = 0
    enable_multi_gpu: bool = False
    enable_tf32: bool = False


@dataclass
class ProfilingConfig:
    """Profiling configuration"""
    collect_gpu_metrics: bool = True
    profile_memory: bool = True
    profile_flops: bool = True
    enable_cudnn_benchmark: bool = True
    sample_interval: float = 0.1


@dataclass
class OutputConfig:
    """Output configuration"""
    output_dir: str = "./results"
    save_csv: bool = True
    save_html: bool = True
    save_json: bool = True
    log_level: str = "INFO"
    verbose: bool = True


class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from YAML file or defaults"""
        self.model: ModelConfig = ModelConfig(name="bert-base-uncased")
        self.benchmark: BenchmarkConfig = BenchmarkConfig(
            batch_sizes=[32],
            sequence_lengths=[128],
            precisions=["fp16"]
        )
        self.device: DeviceConfig = DeviceConfig()
        self.profiling: ProfilingConfig = ProfilingConfig()
        self.output: OutputConfig = OutputConfig()
        
        if config_path and os.path.exists(config_path):
            self.load_from_yaml(config_path)
    
    def load_from_yaml(self, yaml_path: str) -> None:
        """Load configuration from YAML file"""
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            if 'model' in config_dict:
                self.model = ModelConfig(**config_dict['model'])
            
            if 'benchmark' in config_dict:
                bench_cfg = config_dict['benchmark']
                self.benchmark = BenchmarkConfig(
                    batch_sizes=bench_cfg.get('batch_sizes', [32]),
                    sequence_lengths=bench_cfg.get('sequence_lengths', [128]),
                    precisions=bench_cfg.get('precisions', ['fp16']),
                    num_warmup_steps=bench_cfg.get('num_warmup_steps', 100),
                    num_bench_steps=bench_cfg.get('num_bench_steps', 1000),
                    num_iterations=bench_cfg.get('num_iterations', 3)
                )
            
            if 'device' in config_dict:
                self.device = DeviceConfig(**config_dict['device'])
            
            if 'profiling' in config_dict:
                self.profiling = ProfilingConfig(**config_dict['profiling'])
            
            if 'output' in config_dict:
                self.output = OutputConfig(**config_dict['output'])
            
            logger.info(f"Configuration loaded from {yaml_path}")
        
        except Exception as e:
            logger.error(f"Failed to load configuration from {yaml_path}: {e}")
            raise
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': asdict(self.model),
            'benchmark': asdict(self.benchmark),
            'device': asdict(self.device),
            'profiling': asdict(self.profiling),
            'output': asdict(self.output)
        }
    
    def save_to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file"""
        try:
            Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
            with open(yaml_path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            logger.info(f"Configuration saved to {yaml_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {yaml_path}: {e}")
            raise
    
    def validate(self) -> bool:
        """Validate configuration"""
        try:
            assert self.device.device_id >= 0, "device_id must be >= 0"
            assert len(self.benchmark.batch_sizes) > 0, "batch_sizes cannot be empty"
            assert len(self.benchmark.sequence_lengths) > 0, "sequence_lengths cannot be empty"
            assert len(self.benchmark.precisions) > 0, "precisions cannot be empty"
            assert self.benchmark.num_warmup_steps >= 0, "num_warmup_steps must be >= 0"
            assert self.benchmark.num_bench_steps > 0, "num_bench_steps must be > 0"
            assert self.benchmark.num_iterations > 0, "num_iterations must be > 0"
            
            valid_precisions = ['fp32', 'fp16', 'int8']
            for precision in self.benchmark.precisions:
                assert precision in valid_precisions, f"Invalid precision: {precision}"
            
            logger.info("Configuration validation passed")
            return True
        except AssertionError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
