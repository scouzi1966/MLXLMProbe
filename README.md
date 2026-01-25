# MLXLMProbe

A visual probing and interpretability tool for MLX language models on Apple Silicon.

## Features

- **Universal MLX-LM Support**: Works with any model supported by mlx-lm (Llama, Mistral, Phi, Qwen, Gemma, etc.)
- **Layer Analysis**: Visualize activation norms and patterns across all layers
- **FFN Analysis**: Gate sparsity and activation patterns in feed-forward networks
- **Embedding Visualization**: PCA plots and heatmaps of token embeddings
- **Logits Analysis**: Token probability distributions with histograms
- **Layer Similarity**: Cosine similarity heatmaps between layer representations
- **Residual Stream**: Track information flow through the transformer
- **AI Interpretation**: Optional AI-powered analysis of probe results
- **Export**: PDF reports and interactive HTML exports

## Requirements

- macOS 15.0+ (Apple Silicon)
- Python 3.10+
- 8GB+ unified memory (16GB+ recommended for larger models)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MLXLMProbe.git
cd MLXLMProbe

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Run with any MLX model
streamlit run probe.py

# Or specify a model path
streamlit run probe.py -- --model mlx-community/Llama-3.2-1B-Instruct-4bit
```

### Supported Models

Any model compatible with `mlx-lm`, including:
- Llama 2/3 family
- Mistral/Mixtral
- Phi-2/3
- Qwen/Qwen2
- Gemma
- And many more from [mlx-community](https://huggingface.co/mlx-community)

### Command Line Options

```bash
streamlit run probe.py -- --help

Options:
  --model PATH    Path or HuggingFace ID of MLX model
  --port PORT     Streamlit port (default: 8501)
```

## Screenshots

*(Coming soon)*

## How It Works

MLXLMProbe intercepts the forward pass of transformer models to capture:

1. **Embeddings**: Initial token representations
2. **Layer Outputs**: Hidden states after each transformer block
3. **FFN Activations**: Gate values in feed-forward networks
4. **Final Logits**: Output distribution over vocabulary

These are then visualized using Plotly for interactive exploration.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built on [MLX](https://github.com/ml-explore/mlx) by Apple
- Uses [mlx-lm](https://github.com/ml-explore/mlx-examples) for model loading
- Inspired by transformer interpretability research
