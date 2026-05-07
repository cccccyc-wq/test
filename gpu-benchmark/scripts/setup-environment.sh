#!/bin/bash

# GPU Benchmark - Environment Setup Script
# Sets up project structure, validates installation, and configures environment

set -e

echo "================================================"
echo "GPU Benchmark - Environment Setup"
echo "================================================"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${YELLOW}Project directory: $PROJECT_DIR${NC}"

# Create necessary directories
echo -e "\n${YELLOW}Creating project directories...${NC}"

mkdir -p "$PROJECT_DIR/results"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/results/parameter_sweep"
mkdir -p "$PROJECT_DIR/results/comparison"

echo -e "${GREEN}Directories created${NC}"

# Create .env file if not exists
echo -e "\n${YELLOW}Setting up environment variables...${NC}"

if [ ! -f "$PROJECT_DIR/.env" ]; then
    cat > "$PROJECT_DIR/.env" << EOF
# GPU Benchmark Environment Variables

# CUDA Settings
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda
export PATH=\${CUDA_HOME}/bin:\$PATH
export LD_LIBRARY_PATH=\${CUDA_HOME}/lib64:\$LD_LIBRARY_PATH

# PyTorch Settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export PYTHONUNBUFFERED=1

# Project Settings
export PROJECT_ROOT=$PROJECT_DIR
export RESULTS_DIR=$PROJECT_DIR/results
export LOGS_DIR=$PROJECT_DIR/logs

# Benchmark Settings
export NUM_THREADS=8
export BATCH_SIZE=32
export SEQUENCE_LENGTH=128
export MODEL_NAME=bert-base-uncased
EOF
    echo -e "${GREEN}Environment file created: $PROJECT_DIR/.env${NC}"
fi

# Load environment variables
if [ -f "$PROJECT_DIR/.env" ]; then
    source "$PROJECT_DIR/.env"
    echo -e "${GREEN}Environment variables loaded${NC}"
fi

# Verify GPU availability
echo -e "\n${YELLOW}Verifying GPU availability...${NC}"

python3 << 'EOF'
import torch

if torch.cuda.is_available():
    print("\033[0;32m✓ CUDA is available\033[0m")
    print(f"  GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  Device {i}: {props.name}")
        print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
else:
    print("\033[0;31m✗ CUDA is not available\033[0m")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}GPU verification failed${NC}"
    exit 1
fi

# Verify required Python packages
echo -e "\n${YELLOW}Verifying Python packages...${NC}"

python3 << 'EOF'
import sys

required_packages = [
    'torch',
    'transformers',
    'pynvml',
    'pyyaml',
    'pandas',
    'numpy',
    'matplotlib',
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
        print(f"\033[0;32m✓ {package}\033[0m")
    except ImportError:
        print(f"\033[0;31m✗ {package} (missing)\033[0m")
        missing_packages.append(package)

if missing_packages:
    print(f"\n\033[0;31mMissing packages: {', '.join(missing_packages)}\033[0m")
    print("Please run: pip install " + " ".join(missing_packages))
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}Package verification failed${NC}"
    exit 1
fi

# Test benchmark setup
echo -e "\n${YELLOW}Testing benchmark setup...${NC}"

python3 << 'EOF'
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, '$PROJECT_DIR')

try:
    from src.config import Config
    from src.gpu_validator import GPUValidator
    
    # Test config loading
    config = Config('$PROJECT_DIR/config/bert-base.yaml')
    print("\033[0;32m✓ Configuration loading works\033[0m")
    
    # Test GPU validator
    validator = GPUValidator()
    if validator.validate():
        print("\033[0;32m✓ GPU validation works\033[0m")
    else:
        print("\033[0;31m✗ GPU validation failed\033[0m")
        sys.exit(1)
    
    print("\n✓ Benchmark setup test passed")
except Exception as e:
    print(f"\033[0;31m✗ Benchmark setup test failed: {e}\033[0m")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}Benchmark setup test failed${NC}"
    exit 1
fi

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Environment setup completed successfully!${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e "\nNext steps:"
echo -e "  1. Source environment: source $PROJECT_DIR/.env"
echo -e "  2. Run benchmark: bash $SCRIPT_DIR/run-benchmark.sh"
