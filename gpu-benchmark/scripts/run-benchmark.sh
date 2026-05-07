#!/bin/bash

# GPU Benchmark - Main Benchmark Execution Script
# Runs comprehensive BERT GPU benchmarks

set -e

echo "================================================"
echo "GPU Benchmark - BERT Benchmark Execution"
echo "================================================"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment
if [ -f "$PROJECT_DIR/.env" ]; then
    source "$PROJECT_DIR/.env"
fi

# Parse command line arguments
CONFIG_FILE="$PROJECT_DIR/config/bert-base.yaml"
BENCHMARK_TYPE="comprehensive"
OUTPUT_PREFIX="benchmark"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --type)
            BENCHMARK_TYPE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_PREFIX="$2"
            shift 2
            ;;
        --device)
            export CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        --help)
            echo "Usage: run-benchmark.sh [OPTIONS]"
            echo "Options:"
            echo "  --config FILE       Configuration file (default: config/bert-base.yaml)"
            echo "  --type TYPE         Benchmark type: comprehensive|quick|custom (default: comprehensive)"
            echo "  --output PREFIX     Output file prefix (default: benchmark)"
            echo "  --device ID         GPU device ID (default: 0)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}Configuration:${NC}"
echo "  Config file: $CONFIG_FILE"
echo "  Benchmark type: $BENCHMARK_TYPE"
echo "  Output prefix: $OUTPUT_PREFIX"
echo "  GPU device: ${CUDA_VISIBLE_DEVICES:-0}"
echo ""

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_DIR/results/$OUTPUT_PREFIX/$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}Output directory: $OUTPUT_DIR${NC}\n"

# Run benchmark based on type
echo -e "${YELLOW}Starting benchmark execution...${NC}\n"

PYTHON_SCRIPT="$PROJECT_DIR/scripts/run_benchmark_main.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    # Create Python benchmark runner
    cat > "$PYTHON_SCRIPT" << 'PYTHON_EOF'
#!/usr/bin/env python3

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.bert_benchmark import BertBenchmark
from src.performance_monitor import PerformanceMonitor
from scripts.parse_results import ResultsAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='GPU Benchmark Runner')
    parser.add_argument('--config', required=True, help='Configuration file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--benchmark-type', default='comprehensive', help='Benchmark type')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = Config(args.config)
    
    if not config.validate():
        logger.error("Configuration validation failed")
        return 1
    
    # Initialize benchmark
    logger.info("Initializing BERT benchmark")
    benchmark = BertBenchmark(config, device_id=args.device)
    
    # Setup
    if not benchmark.setup():
        logger.error("Benchmark setup failed")
        return 1
    
    # Run benchmark
    logger.info(f"Running {args.benchmark_type} benchmark")
    metrics = benchmark.run_comprehensive_benchmark()
    
    # Save results
    logger.info(f"Saving results to {args.output_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get summary
    summary = benchmark.performance_monitor.get_summary_statistics()
    metrics_list = benchmark.performance_monitor.get_metrics_list()
    
    # Analyze and save
    analyzer = ResultsAnalyzer(metrics_list, summary)
    analyzer.save_to_csv(f"{args.output_dir}/benchmark_results.csv")
    analyzer.save_to_json(f"{args.output_dir}/benchmark_results.json")
    analyzer.save_summary(f"{args.output_dir}/summary.txt")
    
    # Cleanup
    benchmark.cleanup()
    
    logger.info("Benchmark completed successfully")
    return 0

if __name__ == '__main__':
    sys.exit(main())
PYTHON_EOF
fi

python3 "$PYTHON_SCRIPT" \
    --config "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --benchmark-type "$BENCHMARK_TYPE" \
    --device "${CUDA_VISIBLE_DEVICES:-0}"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}Benchmark completed successfully${NC}"
    echo -e "${GREEN}Results saved to: $OUTPUT_DIR${NC}\n"
    
    # Generate report
    if [ -f "$PROJECT_DIR/scripts/parse-results.py" ]; then
        echo -e "${YELLOW}Generating analysis report...${NC}\n"
        python3 "$PROJECT_DIR/scripts/parse-results.py" --results-dir "$OUTPUT_DIR"
    fi
else
    echo -e "${RED}Benchmark execution failed${NC}"
    exit 1
fi

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Benchmark execution completed!${NC}"
echo -e "${GREEN}================================================${NC}"
