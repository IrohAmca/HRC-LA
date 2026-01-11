# Hybrid Real-Complex Linear Attention (HRC-LA)

Unofficial PyTorch implementation of **Hybrid Real-Complex Linear Attention**.

Official Paper: [Hybrid Real-Complex Linear Attention](https://zenodo.org/records/17822274)

## Overview

This repository provides a clean and efficient implementation of the HRC-LA mechanism, which reduces the computational complexity of the standard attention mechanism from $O(N^2)$ to $O(N)$ using complex-valued random feature maps.

## Key Features

- **Linear Complexity**: Scales linearly with sequence length $N$.
- **Hybrid Real-Complex Mapping**: Utilizes Euler's formula to map real-valued queries and keys into a complex feature space.
- **Orthogonal Random Features**: Uses orthogonal matrix initialization for random features to stabilize variance and improve approximation quality.
- **PyTorch Implementation**: Fully compatible with standard PyTorch modules.

## Benchmark Results

The following benchmark compares the standard $O(N^2)$ Multihead Attention with our HRC-LA $O(N)$ implementation (both Fixed and Learnable $\Omega$ variants).

### Performance Comparison (GTX 1650 Laptop GPU)

| Sequence Length (N) | Standard Time (s) | HRC-LA (Learnable) Time (s) | Speedup          | Standard Mem (MB) | HRC-LA Mem (MB) | Memory Saving | Loss (Cross-Entropy) |
| ------------------- | ----------------- | --------------------------- | ---------------- | ----------------- | --------------- | ------------- | -------------------- |
| 1024                | 0.0020            | 0.0031                      | 0.6x             | 29.53             | 11.56           | 61%           | 4.28                 |
| 2048                | 0.0075            | 0.0020                      | **3.7x**   | 89.98             | 13.98           | **84%** | 4.29                 |
| 4096                | 0.0260            | 0.0030                      | **8.6x**   | 330.80            | 18.81           | **94%** | 4.38                 |
| 8192                | 0.0897            | 0.0047                      | **19x**    | 1292.46           | 28.46           | **97%** | 4.18                 |
| 16384               | 7.0447            | 0.0194                      | **363x** | 5135.77           | 47.77           | **99%** | 4.20                 |

*Note: Standard Attention Loss consistently stays around ~4.17. HRC-LA achieves very similar loss values with significantly less resource usage.*

### Key Findings

1.  **Crossover Point**: HRC-LA becomes faster than standard attention starting around **N=1024-2048**, but is **always** more memory efficient.
2.  **Extreme Scaling**: At **N=16,384**, HRC-LA is **~363x faster** and uses **~99% less memory**.
3.  **Accuracy**: The Learnable $\Omega$ variant maintains a Cross-Entropy Loss very close to the standard attention mechanism (e.g., 4.20 vs 4.17 at 16k), demonstrating its capability to approximate the dense attention matrix effectively.

### Visualization

![Benchmark Results](benchmark_results.png)

The benchmark can be run in two modes:
- **MSE Mode**: Directly compares the output tensors (Default).
- **Loss Mode**: Compares reconstruction loss on a synthetic task (`--mode loss`).

## Optimization & Scenarios

We provide advanced tools for benchmarking and hyperparameter optimization, specifically designed for the **Copy Task** scenario to test long-range dependency capabilities.

### Optuna Integration

Optimize hyperparameters automatically, including searching for the maximum effective sequence length (`seq_len`) while minimizing feature map size (`m_features`).

```bash
# Basic optimization
uv run python tests/scenarios/copy_test/optuna_search.py --optimize_all --n_trials 50

# Optimize specifically for maximum sequence length
uv run python tests/scenarios/copy_test/seq_len_optimizer.py --mode single --n_trials 50
```

### Copy Task Benchmark

A dedicated scenario to evaluate the model's ability to recall information over long sequences.

```bash
uv run python tests/scenarios/copy_test/main.py --seq_len 2048 --epochs 10
```

## Project Structure

```
HRC-LA/
├── hrc_la/                # Core library
│   ├── attention.py       # HRC-LA implementation
│   └── utils.py           # Helper functions and adapters
├── benchmarks/            # Performance and error analysis
│   ├── benchmark.py       # Main benchmark script
├── tests/                 # Unit tests & Scenarios
│   ├── scenarios/         # Task-specific benchmarks (Copy Task, etc.)
│   └── test_attention.py  # Attention mechanism tests
├── benchmark_results.png  # Benchmark visualization
├── pyproject.toml         # Project dependencies
└── README.md              # Documentation
```

## Usage

```python
import torch
from hrc_la import HRCMultiheadAttention

# Initialize model (Standard Fixed Omega)
model = HRCMultiheadAttention(
    d_model=64, 
    num_heads=4, 
    m_features=256
)

# Initialize model with Learnable Omega (Higher Accuracy)
model_learnable = HRCMultiheadAttention(
    d_model=64, 
    num_heads=4, 
    m_features=256,
    learnable_omega=True
)

# Forward pass
x = torch.randn(1, 1024, 64)
output = model(x)
```

## Testing

To run the unit tests, ensure you have `pytest` installed and run:

```bash
pytest tests/
```

## Citation

If you find this work useful, please cite the original paper:

```bibtex
@misc{hrc_la_2024,
  title={Hybrid Real-Complex Linear Attention},
  author={Emre Fırıl},
  year={2025},
  howpublished={\url{https://zenodo.org/records/17822274}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
