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

The following benchmark compares the standard $O(N^2)$ Multihead Attention with our HRC-LA $O(N)$ implementation.

### Performance Comparison

| Sequence Length (N) | Standard Time (s) | HRC-LA Time (s) | MSE Error |
|---------------------|-------------------|-----------------|-----------|
| 128                 | 0.0015            | 0.0055          | 0.002383  |
| 256                 | Fail              | 0.0076          | 0.000740  |
| 512                 | 0.0025            | 0.0121          | 0.001041  |
| 1024                | 0.0075            | 0.0249          | 0.007950  |
| 2048                | 0.0341            | 0.0431          | 0.907808  |
| 4096                | 0.1460            | 0.1255          | 0.004921  |
| 8192                | 0.4416            | 0.1693          | 0.002242  |
| 16384               | 2.3723            | 0.3558          | 0.051956  |
| 32768               | 10.3643           | 0.8442          | 0.050090  |

### Visualization

![Benchmark Results](results.png)

The plot above demonstrates the linear scaling of HRC-LA compared to the quadratic scaling of standard attention, while maintaining a very low approximation error (MSE).

## Project Structure

```
HRC-LA/
├── hrc_la/                # Core library
│   ├── attention.py       # HRC-LA implementation
│   └── utils.py           # Helper functions and adapters
├── benchmarks/            # Performance and error analysis
│   ├── benchmark.py       # Main benchmark script
│   └── decomposition.py   # Step-by-step implementation analysis
├── tests/                 # Unit tests
│   └── test_attention.py  # Attention mechanism tests
├── results.png            # Benchmark visualization
├── pyproject.toml         # Project dependencies
└── README.md              # Documentation
```

## Usage

```python
import torch
from hrc_la import HRCMultiheadAttention

# Initialize model
model = HRCMultiheadAttention(
    d_model=64, 
    num_heads=4, 
    m_features=256
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
