# MLXLMProbe

A visual probing and interpretability tool for MLX language models on Apple Silicon.

> **Status:** Work in Progress - Currently testing with GPT-OSS and other MoE models

## Features

- **Universal MLX-LM Support**: TESTED ONLY on GPT-OSS so far
- **MoE Analysis**: Mixture-of-Experts routing visualization, expert load distribution, top-k selection patterns
- **Layer Analysis**: Visualize activation norms and patterns across all layers
- **FFN Analysis**: Gate sparsity and activation patterns in feed-forward networks
- **Embedding Visualization**: PCA plots with section-based coloring (System/User/Reasoning/Response)
- **Logits Analysis**: Token probability distributions with histograms
- **Layer Similarity**: Cosine similarity heatmaps between layer representations
- **Residual Stream**: Track information flow through the transformer
- **Token Alternatives**: See what other tokens the model considered at each position
- **Reasoning Model Support**: Detects and separates reasoning loops from final responses
- **AI Interpretation**: Optional AI-powered analysis using local model or Claude
- **Export**: PDF reports and interactive HTML exports

## Requirements

- **Mac with Apple Silicon** (M1, M2, M3, M4, or later)
- **macOS 15.0+** (Sequoia or later recommended)
- **Python 3.10+**
- **8GB+ unified memory** (16GB+ recommended for larger models, 32GB+ for 30B+ models)

## Quick Start (From Scratch)

### Step 1: Verify Your System

```bash
# Check you have Apple Silicon
uname -m
# Should output: arm64

# Check macOS version
sw_vers
# ProductVersion should be 15.0 or higher

# Check Python version
python3 --version
# Should be 3.10 or higher
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/scouzi1966/MLXLMProbe.git
cd MLXLMProbe
```

### Step 3: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Verify activation (should show path to venv)
which python
```

### Step 4: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

This installs:
- `mlx` - Apple's ML framework for Apple Silicon
- `mlx-lm` - Language model utilities for MLX
- `streamlit` - Web UI framework
- `plotly` - Interactive charts
- `pandas` - Data manipulation
- `scikit-learn` - PCA for embeddings
- `huggingface-hub` - Model downloading
- `fpdf2` - PDF export

### Step 5: Run MLXLMProbe

```bash
# Start the web UI (will open in browser)
streamlit run probe.py
```

The app will open at `http://localhost:8501`

### Step 6: Load a Model

**Option A: Use the sidebar to enter a HuggingFace model ID**

Popular MLX models from [mlx-community](https://huggingface.co/mlx-community):

- `mlx-community/gpt-oss-20b-MXFP4-Q8` (TESTED)
- `mlx-community/Llama-3.2-3B-Instruct-4bit` (small, fast)
- `mlx-community/Mistral-7B-Instruct-v0.3-4bit` (good quality)
- `mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit` (MoE model)
- `mlx-community/Qwen2.5-7B-Instruct-4bit` (multilingual)
- `mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit` (reasoning model)

**Option B: Specify model on command line**

```bash
streamlit run probe.py -- --model mlx-community/Llama-3.2-1B-Instruct-4bit
```

**Option C: Use a local model path**

```bash
streamlit run probe.py -- --model /path/to/your/mlx-model
```

## Usage Guide

### Basic Workflow

1. **Enter a prompt** in the text area
2. **Click "Run Probe"** to generate and analyze
3. **Explore tabs**: Layer Activations, FFN Analysis, Tokens, Embeddings, Logits, etc.
4. **For MoE models**: Check the "MoE Routing" tab for expert analysis

### Understanding MoE Visualizations

For Mixture-of-Experts models (like Mixtral), the MoE tab shows:

- **Top-K Expert Weights**: Stacked bars showing which experts were selected
  - 🟡 Gold = Top-1 (highest weight)
  - 🟣 Magenta = Top-2
  - 🔵 Cyan = Top-3
  - 🟠 Orange = Top-4
  - Bar length = router probability assigned to that expert
  - Labels inside bars = Expert ID (E0, E1, etc.)

- **Expert Load**: How many tokens each expert processed
- **Router Probabilities**: Heatmap of all expert weights

### Command Line Options

```bash
streamlit run probe.py -- --help

Options:
  --model PATH         Path or HuggingFace ID of MLX model
  --port PORT          Streamlit port (default: 8501)
  --max-tokens N       Maximum tokens to generate (default: 100)
  --max-context N      Maximum context length (default: model's max)
```

### Keyboard Shortcuts

- `Ctrl+Enter` / `Cmd+Enter` - Run probe
- `R` - Refresh page

## Troubleshooting

### "No module named 'mlx'"
MLX only works on Apple Silicon Macs. Verify with `uname -m` (should be `arm64`).

### Model download fails
- Check internet connection
- Verify the model ID exists on HuggingFace
- Try a smaller model first

### Out of memory
- Try a smaller/more quantized model (4bit instead of 8bit)
- Reduce max tokens to generate
- Close other applications

### Streamlit won't start
```bash
# Kill any existing Streamlit processes
pkill -f streamlit

# Try a different port
streamlit run probe.py --server.port 8502
```

## How It Works

MLXLMProbe intercepts the forward pass of transformer models to capture:

1. **Embeddings**: Initial token representations
2. **Layer Outputs**: Hidden states after each transformer block
3. **FFN/MoE Activations**: Gate values and expert routing decisions
4. **Final Logits**: Output distribution over vocabulary
5. **Per-token Alternatives**: What other tokens were considered

These are visualized using Plotly for interactive exploration.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built on [MLX](https://github.com/ml-explore/mlx) by Apple
- Uses [mlx-lm](https://github.com/ml-explore/mlx-examples) for model loading
- Inspired by transformer interpretability research

## Contributing

This is a work in progress. Issues and PRs welcome!
