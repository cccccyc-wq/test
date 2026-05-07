#!/bin/bash

# GPU Benchmark - One-click Installation Script
# Installs NVIDIA drivers, CUDA, cuDNN, and Python dependencies

set -e

echo "================================================"
echo "GPU Benchmark - Installation Script"
echo "================================================"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Linux
if [[ ! "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "${RED}Error: This script only supports Linux${NC}"
    exit 1
fi

# Detect Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
else
    echo -e "${RED}Error: Cannot detect Linux distribution${NC}"
    exit 1
fi

echo -e "${GREEN}Detected distribution: $DISTRO${NC}"

# Install system dependencies based on distribution
echo -e "\n${YELLOW}Step 1: Installing system dependencies...${NC}"

if [[ "$DISTRO" == "ubuntu" || "$DISTRO" == "debian" ]]; then
    sudo apt-get update
    sudo apt-get install -y build-essential curl wget git python3 python3-pip python3-dev
    sudo apt-get install -y libcudnn8 libcudnn8-dev
elif [[ "$DISTRO" == "centos" || "$DISTRO" == "rhel" || "$DISTRO" == "fedora" ]]; then
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y curl wget git python3 python3-devel
else
    echo -e "${YELLOW}Warning: Unsupported distribution. Please install dependencies manually.${NC}"
fi

# Check NVIDIA GPU
echo -e "\n${YELLOW}Step 2: Checking NVIDIA GPU...${NC}"

if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}Warning: NVIDIA GPU or drivers not found.${NC}"
    echo -e "${YELLOW}Please install NVIDIA drivers manually from: https://www.nvidia.com/Download/driverDetails.aspx${NC}"
fi

# Install CUDA toolkit (if not already installed)
echo -e "\n${YELLOW}Step 3: Checking CUDA installation...${NC}"

if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}CUDA is already installed:${NC}"
    nvcc --version
else
    echo -e "${YELLOW}CUDA not found. Please download and install from: https://developer.nvidia.com/cuda-11-8-0-download-archive${NC}"
    echo -e "${YELLOW}After installation, add to ~/.bashrc:${NC}"
    echo -e "${YELLOW}  export PATH=/usr/local/cuda/bin:\$PATH${NC}"
    echo -e "${YELLOW}  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH${NC}"
fi

# Install Python dependencies
echo -e "\n${YELLOW}Step 4: Installing Python dependencies...${NC}"

if [ -f "requirements.txt" ]; then
    pip3 install --upgrade pip setuptools wheel
    pip3 install -r requirements.txt
    echo -e "${GREEN}Python dependencies installed successfully${NC}"
else
    echo -e "${RED}Error: requirements.txt not found${NC}"
    exit 1
fi

# Verify installation
echo -e "\n${YELLOW}Step 5: Verifying installation...${NC}"

python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}PyTorch verification passed${NC}"
else
    echo -e "${RED}PyTorch verification failed${NC}"
    exit 1
fi

python3 -c "from transformers import AutoTokenizer; print('Transformers installed successfully')"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Transformers verification passed${NC}"
else
    echo -e "${RED}Transformers verification failed${NC}"
    exit 1
fi

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Installation completed successfully!${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e "\nNext steps:"
echo -e "  1. Run: bash setup-environment.sh"
echo -e "  2. Run: bash run-benchmark.sh"
