# GPT Implementation from Scratch 🚀

<div align="center">

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-FF6F00.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

A ground-up implementation of the GPT (Generative Pre-trained Transformer) architecture, focusing on understanding and analyzing the model's core components and performance characteristics. This implementation features a detailed attention mechanism, positional embeddings, and transformer blocks built from first principles.

## 🌟 Features

- **Pure Python Implementation**: Built from scratch using PyTorch, implementing core components:
  - Multi-head self-attention mechanism with scaled dot-product attention
  - Position embeddings for sequence understanding
  - Layer normalization and residual connections
  - GELU activation function in feed-forward networks
- **FLOP Analysis**: Advanced computational analysis including:
  - Model FLOP Utilization (MFU) measurement
  - GPU-specific performance metrics
  - Automatic batch size optimization
- **Modular Architecture**: Highly modular design with:
  - Configurable model sizes (small: 124M to XL: 1558M parameters)
  - Customizable attention heads and embedding dimensions
  - Flexible context length and vocabulary size
- **Performance Metrics**: Comprehensive evaluation tools for:
  - Tokens per second throughput measurement
  - Memory usage optimization
  - Hardware-specific performance analysis

## 🏗️ Technical Implementation

### Model Architecture
- **Token & Position Embeddings**: Converts input tokens into learned embeddings and adds positional information
- **Transformer Blocks**: Multiple layers of self-attention and feed-forward networks
  - Multi-head attention splits input into parallel heads for broader context understanding
  - Layer normalization and residual connections for stable training
  - Feed-forward networks with GELU activation for non-linear transformations

### Attention Mechanism
```python
# Scaled dot-product attention with parallel heads
keys = self.W_key(x)      # Project input to key space
queries = self.W_query(x) # Project input to query space
values = self.W_value(x)  # Project input to value space

# Compute attention scores and apply causal mask
attn_scores = queries @ keys.transpose(-2, -1)
attn_scores.masked_fill_(mask_bool, -torch.inf)

# Get weighted sum of values
context_vec = (attn_weights @ values)
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/pjmreddy/GPT-from-scratch.git
cd GPT-from-scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📚 Project Structure

```
├── gpt.py           # Core GPT implementation with attention mechanism
├── flop_analysis.py # Performance analysis and optimization
├── tests.py         # Comprehensive test suite
└── requirements.txt # Project dependencies
```

## 🚀 Usage

```python
from gpt import GPT

# Initialize model with specific configuration
config = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "drop_rate": 0.1
}
model = GPT(config)

# Generate text
output = model.generate("Your prompt here")
print(output)

# Analyze model performance
from flop_analysis import calculate_flops, measure_throughput
flops = calculate_flops(model, config)
tokens_per_second, mfu = measure_throughput(model, config, batch_size=32)
```

## 📊 Performance Analysis

The implementation includes sophisticated FLOP analysis capabilities to understand and optimize computational requirements:

### Model Configurations
- **GPT-Small (124M)**: 768 embedding dim, 12 layers, 12 attention heads
- **GPT-Medium (355M)**: 1024 embedding dim, 24 layers, 16 attention heads
- **GPT-Large (774M)**: 1280 embedding dim, 36 layers, 20 attention heads
- **GPT-XL (1558M)**: 1600 embedding dim, 48 layers, 25 attention heads

### Performance Metrics
- **Attention Mechanism**: Scaled dot-product attention with causal masking
  - Parallel computation across multiple attention heads
  - Efficient key-query-value projections
- **Feed-Forward Networks**: 4x dimension expansion with GELU activation
- **Hardware Utilization**: Automatic optimization for different GPU models
  - A100: 312 TFLOPS theoretical peak
  - V100: 125 TFLOPS theoretical peak
  - T4: 65 TFLOPS theoretical peak
  - P100: 21 TFLOPS theoretical peak

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Thanks to the PyTorch and TensorFlow communities
- Inspired by the original GPT papers and implementations

---

⭐️ If you find this implementation helpful, please consider giving it a star!

<div align="center">

Developed with ❤️ by [Jagan Reddy](mailto:peravali810@gmail.com)

</div>
