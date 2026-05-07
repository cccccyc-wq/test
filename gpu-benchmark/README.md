# GPU Benchmark Testing Framework

## 项目概述

GPU Benchmark是一套完整的GPU性能和功能测试框架，专为工厂生产测试设计。通过BERT模型基准测试，验证GPU的性能指标、功能可靠性和稳定性。

## 主要特性

✅ **完整的文档体系** - 从安装到部署全流程覆盖
✅ **自动化测试程序** - 支持多参数扫描和性能监控
✅ **实时性能监控** - GPU利用率、功耗、温度等指标实时采集
✅ **结果分析导出** - CSV/HTML/可视化图表导出
✅ **参数调优指南** - 性能影响因子分析和最优配置建议

## 快速开始

### 1. 环境安装
```bash
cd scripts
bash install-requirements.sh
```

### 2. 环境配置
```bash
bash setup-environment.sh
```

### 3. 运行基准测试
```bash
bash run-benchmark.sh
```

### 4. 分析测试结果
```bash
python parse-results.py
python plot-results.py
```

## 文档结构

| 文档 | 描述 |
|------|------|
| [01-installation-guide.md](docs/01-installation-guide.md) | NVIDIA驱动、CUDA、cuDNN、Python环境一键安装 |
| [02-dependencies-setup.md](docs/02-dependencies-setup.md) | PyTorch、Transformers、监控工具完整配置 |
| [03-deployment-guide.md](docs/03-deployment-guide.md) | 项目结构、配置管理、GPU验证模块 |
| [04-running-tests.md](docs/04-running-tests.md) | BERT基准测试程序、多参数扫描 |
| [05-log-analysis.md](docs/05-log-analysis.md) | 结果解析、CSV/HTML导出、可视化脚本 |
| [06-performance-tuning.md](docs/06-performance-tuning.md) | 参数影响分析、最优配置建议 |

## 测试指标

### 性能指标
- **延迟 (Latency)** - 单次推理时间（ms）
- **吞吐量 (Throughput)** - 每秒处理token数
- **GPU利用率** - GPU计算资源使用率（%）
- **功耗** - 实时功耗（W）

### 功能指标
- **精度验证** - FP32/FP16/INT8精度计算正确性
- **稳定性** - 长时间运行无错误
- **内存管理** - 显存使用效率

## 系统要求

- **GPU**: NVIDIA显卡（Compute Capability >= 7.0）
- **CUDA**: 11.8+
- **Python**: 3.8+
- **OS**: Linux（CentOS/Ubuntu推荐）

## 配置示例

### 基础配置
```yaml
model: "bert-base-uncased"
batch_size: 32
sequence_length: 128
precision: "fp16"
warmup_steps: 100
num_steps: 1000
```

### 高性能配置
```yaml
model: "bert-large-uncased"
batch_size: 64
sequence_length: 512
precision: "fp16"
gradient_checkpointing: true
```

## 联系方式

如有问题或建议，请提交Issue或联系技术支持。
