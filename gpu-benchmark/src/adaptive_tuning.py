#!/usr/bin/env python3
"""
Adaptive Tuning Module
Adaptively adjusts parameters based on GPU capabilities and constraints
"""

import logging
from typing import Dict, Tuple
import torch

logger = logging.getLogger(__name__)


class AdaptiveTuning:
    """Adaptive parameter tuning based on GPU capabilities"""
    
    def __init__(self, device_id: int = 0):
        """Initialize adaptive tuning"""
        self.device_id = device_id
        self.device_properties = torch.cuda.get_device_properties(device_id)
        self.total_memory = self.device_properties.total_memory / 1024**3  # GB
    
    def recommend_batch_size(self, sequence_length: int, model_name: str = 'bert-base') -> int:
        """Recommend batch size based on available GPU memory"""
        # Estimate memory per sample
        memory_per_sample_gb = self._estimate_memory_per_sample(sequence_length, model_name)
        
        # Leave 10% headroom for GPU overhead
        available_memory = self.total_memory * 0.9
        
        # Max batch size
        max_batch_size = int(available_memory / memory_per_sample_gb)
        
        # Recommend powers of 2 for efficiency
        recommended_batch_size = 2 ** (max_batch_size.bit_length() - 1)
        recommended_batch_size = max(1, min(recommended_batch_size, 128))
        
        logger.info(f"Recommended batch size for {model_name} (seq_len={sequence_length}): {recommended_batch_size}")
        return recommended_batch_size
    
    def recommend_sequence_length(self, batch_size: int, model_name: str = 'bert-base') -> int:
        """Recommend sequence length based on available GPU memory"""
        # Get base sequence length for model
        base_seq_len = 128
        
        # Estimate memory per sample
        memory_per_token_gb = self._estimate_memory_per_token(model_name)
        available_memory = self.total_memory * 0.9
        memory_per_sample = memory_per_token_gb * base_seq_len * batch_size
        
        if memory_per_sample > available_memory:
            # Need to reduce sequence length
            reduction_factor = available_memory / memory_per_sample
            recommended_seq_len = int(base_seq_len * reduction_factor)
        else:
            # Can potentially increase sequence length
            increase_factor = available_memory / memory_per_sample
            recommended_seq_len = int(base_seq_len * increase_factor)
        
        # Round to nearest multiple of 64
        recommended_seq_len = max(64, (recommended_seq_len // 64) * 64)
        recommended_seq_len = min(512, recommended_seq_len)  # BERT max is 512
        
        logger.info(f"Recommended sequence length for {model_name} (batch_size={batch_size}): {recommended_seq_len}")
        return recommended_seq_len
    
    def recommend_precision(self) -> str:
        """Recommend precision based on GPU compute capability"""
        compute_capability = self.device_properties.major * 10 + self.device_properties.minor
        
        if compute_capability >= 70:  # Volta and newer have tensor cores
            recommended_precision = 'fp16'
            reason = "GPU supports tensor cores (Volta+), FP16 recommended for performance"
        else:
            recommended_precision = 'fp32'
            reason = "GPU does not support tensor cores, FP32 recommended"
        
        logger.info(f"Recommended precision: {recommended_precision} - {reason}")
        return recommended_precision
    
    def get_optimal_config(self, model_name: str = 'bert-base') -> Dict:
        """Get complete optimal configuration for the GPU"""
        seq_len = 128  # Start with default
        batch_size = self.recommend_batch_size(seq_len, model_name)
        precision = self.recommend_precision()
        
        config = {
            'batch_size': batch_size,
            'sequence_length': seq_len,
            'precision': precision,
            'gpu_memory_gb': self.total_memory,
            'compute_capability': f"{self.device_properties.major}.{self.device_properties.minor}",
        }
        
        logger.info(f"Optimal configuration: {config}")
        return config
    
    def _estimate_memory_per_sample(self, sequence_length: int, model_name: str) -> float:
        """Estimate memory consumption per sample in GB"""
        # Simplified estimation based on model size
        if 'base' in model_name.lower():
            hidden_size = 768
            num_layers = 12
        elif 'large' in model_name.lower():
            hidden_size = 1024
            num_layers = 24
        else:
            hidden_size = 768
            num_layers = 12
        
        # Model parameters + activations + gradients (roughly)
        bytes_per_param = 4  # FP32
        model_memory = (hidden_size * sequence_length * num_layers * bytes_per_param) / (1024**3)
        activation_memory = model_memory * 2  # Approximate activations
        
        return model_memory + activation_memory
    
    def _estimate_memory_per_token(self, model_name: str) -> float:
        """Estimate memory per token for a model in GB"""
        if 'base' in model_name.lower():
            return 1e-6  # Rough estimate
        elif 'large' in model_name.lower():
            return 1.5e-6
        else:
            return 1e-6
