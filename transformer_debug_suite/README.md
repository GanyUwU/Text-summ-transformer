# Universal Transformer Debug Suite

A Python-based debugging tool for analyzing ONNX Transformer models.

## Features

| Analyzer | What It Detects |
|----------|-----------------|
| **Weight Analyzer** | Dead neurons, layer imbalance |
| **Attention Analyzer** | Collapsed/uniform heads, entropy |
| **Activation Analyzer** | Sparsity, vanishing signals |
| **Hallucination Analyzer** | Low confidence, repetition, high entropy |

## Installation

```bash
pip install onnx onnxruntime scipy numpy
```

## Usage

```bash
# Basic usage
python debug_suite/main.py your_model.onnx --output report.html

# With custom sequence length
python debug_suite/main.py model.onnx --seq-len 128 --batch-size 1
```

## Output

Generates an HTML report with:
- Health Score (0-100)
- Issue alerts (Critical/Warning/Info)
- Detailed diagnostics per analyzer

## Project Structure

```
debug_suite/
├── core/
│   ├── model_loader.py       # ONNX loading
│   ├── graph_instrumenter.py # Expose internal tensors
│   └── inference_engine.py   # ONNX Runtime wrapper
├── analyzers/
│   ├── weights.py            # Static weight analysis
│   ├── attention.py          # Attention entropy
│   ├── activations.py        # Activation sparsity
│   └── hallucination.py      # Output confidence
├── visualization/
│   └── html_generator.py     # Report generation
└── main.py                   # CLI entry point
```
