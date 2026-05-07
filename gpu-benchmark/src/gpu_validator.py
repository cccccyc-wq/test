#!/usr/bin/env python3
"""
GPU Validator Module
Validates GPU availability, properties, and capabilities
"""

import torch
import logging
from typing import Dict, List, Optional

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)


class GPUValidator:
    """GPU validation and information retrieval"""
    
    def __init__(self):
        """Initialize GPU validator"""
        self.device_available = torch.cuda.is_available()
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.pynvml_available = True
            except Exception as e:
                logger.warning(f"Failed to initialize PYNVML: {e}")
                self.pynvml_available = False
        else:
            self.pynvml_available = False
    
    def validate(self) -> bool:
        """Validate GPU availability"""
        if not self.device_available:
            logger.error("CUDA is not available. Please check GPU drivers and CUDA installation.")
            return False
        
        try:
            # Test GPU access
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            logger.info("GPU validation passed")
            return True
        except Exception as e:
            logger.error(f"GPU validation failed: {e}")
            return False
    
    def get_device_info(self, device_id: int = 0) -> Dict:
        """Get GPU device information"""
        info = {
            'available': self.device_available,
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'device_name': torch.cuda.get_device_name(device_id) if self.device_available else 'N/A',
            'compute_capability': self._get_compute_capability(device_id),
            'total_memory': torch.cuda.get_device_properties(device_id).total_memory / 1024**3 if self.device_available else 0,
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
        }
        
        if self.pynvml_available:
            pynvml_info = self._get_pynvml_info(device_id)
            info.update(pynvml_info)
        
        return info
    
    def _get_compute_capability(self, device_id: int) -> str:
        """Get GPU compute capability"""
        if not self.device_available:
            return "N/A"
        try:
            props = torch.cuda.get_device_properties(device_id)
            return f"{props.major}.{props.minor}"
        except Exception as e:
            logger.warning(f"Failed to get compute capability: {e}")
            return "Unknown"
    
    def _get_pynvml_info(self, device_id: int) -> Dict:
        """Get additional GPU info using PYNVML"""
        if not self.pynvml_available:
            return {}
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            driver_version = pynvml.nvmlSystemGetDriverVersion().decode()
            
            return {
                'driver_version': driver_version,
                'gpu_utilization': pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                'memory_used': mem_info.used / 1024**3,
                'memory_free': mem_info.free / 1024**3,
            }
        except Exception as e:
            logger.warning(f"Failed to get PYNVML info: {e}")
            return {}
    
    def print_info(self, device_id: int = 0) -> None:
        """Print GPU information"""
        info = self.get_device_info(device_id)
        
        print("\n" + "="*60)
        print("GPU DEVICE INFORMATION")
        print("="*60)
        for key, value in info.items():
            print(f"{key:<25}: {value}")
        print("="*60 + "\n")
    
    def check_precision_support(self, device_id: int = 0) -> Dict[str, bool]:
        """Check precision support on GPU"""
        support = {
            'fp32': True,  # Always supported
            'fp16': torch.cuda.is_available(),
            'int8': torch.cuda.is_available(),
        }
        
        # Check for tensor cores (needed for efficient FP16)
        try:
            props = torch.cuda.get_device_properties(device_id)
            # Tensor cores available on Volta (7.0) and newer
            support['tensor_cores'] = props.major >= 7
        except Exception as e:
            logger.warning(f"Failed to check tensor core support: {e}")
            support['tensor_cores'] = False
        
        return support
