# MathOrCup 2025 - YOLO Training Utilities

YOLO training utilities for MathOrCup 2025 competition - Surface defect detection project.

## Overview

This project provides a command-line tool for training YOLO models to detect surface defects including:
- **Dent**: Surface dents/凹痕
- **Hole**: Holes in the surface/孔洞
- **Rusty**: Rust damage/锈蚀

## Features

- Easy-to-use command-line interface for YOLO model training
- Support for time-limited training sessions
- Automatic logging of training progress
- Resume capability for interrupted training
- Customizable batch size and caching options
- Automatic timestamp-based result naming

## Installation

### Prerequisites

- Python >= 3.10
- Poetry (recommended) or pip

### Install with Poetry

```bash
poetry install
```

### Install with pip

```bash
pip install -e .
```

## Usage

### Basic Training

```bash
authorcup-train --time 2 --project my-project
```

### Advanced Training with Custom Parameters

```bash
authorcup-train \
  --model yolo11n.pt \
  --data data-bin/yaml-data/data.yaml \
  --batch 0.9 \
  --cache True \
  --time 2.5 \
  --project defect-detection \
  --name experiment-001
```

### Resume Training

```bash
authorcup-train \
  --time 1 \
  --project my-project \
  --resume path/to/last.pt
```

## Command-Line Arguments

| Argument | Short | Type | Default | Required | Description |
|----------|-------|------|---------|----------|-------------|
| `--model` | `-m` | str | `yolo11n.pt` | No | Path to the YOLO model file |
| `--data` | - | str | `data.yaml` | No | Path to the dataset YAML file |
| `--batch` | - | float | `0.9` | No | Batch size percentage for training |
| `--cache` | - | bool | `True` | No | Whether to cache images for faster training |
| `--time` | `-t` | float | - | **Yes** | Training time limit (in hours) |
| `--name` | - | str | `{timestamp}` | No | Name to save training results |
| `--project` | `-P` | str | - | **Yes** | Project directory to save results |
| `--resume` | `-R` | str | `False` | No | Path to checkpoint to resume training |

## Project Structure

```
mathorcup_25/
├── src/
│   └── authorcup_25/
│       ├── __init__.py
│       └── train.py          # Main training script
├── data-bin/                 # Dataset directory
│   └── yaml-data/
│       └── data.yaml         # Dataset configuration
├── output-bin/               # Training outputs
├── Question/                 # Competition materials
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## Dataset Configuration

The dataset is configured in `data.yaml`:

```yaml
path: .
train: train/images
val: validation/images
nc: 3
names: ["Dent", "Hole", "Rusty"]
```

## Training Output

Training results are saved to:
```
output-bin/{project}/{name}/
```

Logs are written to:
```
training-log.log
```

## Examples

### Quick Start (2 hours)

```bash
authorcup-train -t 2 -P quick-test
```

### Production Training (8 hours with custom model)

```bash
authorcup-train \
  -m yolo11m.pt \
  --data data-bin/yaml-data/data.yaml \
  -t 8 \
  -P production \
  --name run-001
```

### Resume Previous Training

```bash
authorcup-train \
  -t 2 \
  -P production \
  -R output-bin/production/run-001/weights/last.pt
```

## Requirements

- ultralytics >= 8.3.221, < 9.0.0

## Author

**chendaile**
- Email: 3185854290@qq.com

## License

This project is created for MathOrCup 2025 competition.

## Notes

- Training logs are automatically saved to `training-log.log`
- Model profiling is enabled by default for performance analysis
- Results are saved with automatic timestamp if no name is specified
- The tool uses `exist_ok=True` to avoid errors when reusing project names
