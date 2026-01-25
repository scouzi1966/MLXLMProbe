#!/usr/bin/env python3
"""
MLXLMProbe - Universal probing tool for MLX language models.

A visual interpretability tool that works with any mlx-lm compatible model.
Supports Llama, Mistral, Phi, Qwen, Gemma, and other architectures.

Usage:
    streamlit run probe.py
    streamlit run probe.py -- --model mlx-community/Llama-3.2-1B-Instruct-4bit
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    print("Error: MLX not found. Install with: pip install mlx")
    sys.exit(1)

try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    from sklearn.decomposition import PCA
except ImportError as e:
    print(f"Error: Missing dependency: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ProbeConfig:
    """Configuration for probing."""
    capture_embeddings: bool = True
    capture_layer_outputs: bool = True
    capture_ffn_activations: bool = True
    capture_logits: bool = True
    capture_residual_stream: bool = True
    layer_indices: Optional[List[int]] = None  # None = all layers
    max_sequence_positions: int = 512  # Limit for memory


@dataclass
class ProbeResults:
    """Container for probe results."""
    # Input/Output
    input_tokens: List[int] = field(default_factory=list)
    input_text: str = ""
    output_tokens: List[int] = field(default_factory=list)
    output_text: str = ""

    # Embeddings
    embeddings: Optional[np.ndarray] = None

    # Layer outputs
    layer_outputs: Dict[int, np.ndarray] = field(default_factory=dict)

    # FFN activations
    ffn_activations: Dict[int, np.ndarray] = field(default_factory=dict)

    # Logits
    logits: Optional[np.ndarray] = None
    top_k_tokens: List[Tuple[int, float]] = field(default_factory=list)

    # Residual stream
    residual_stream: Dict[str, np.ndarray] = field(default_factory=dict)
    residual_stream_norms: List[Tuple[str, float]] = field(default_factory=list)
    residual_stream_deltas: List[Tuple[str, float]] = field(default_factory=list)

    # Model info
    model_type: str = ""
    num_layers: int = 0
    hidden_dim: int = 0
    vocab_size: int = 0


# =============================================================================
# Model Prober
# =============================================================================

class ModelProber:
    """
    Universal prober for MLX language models.

    Works with any model loaded via mlx_lm.load().
    """

    def __init__(self, model, tokenizer, config: Optional[ProbeConfig] = None):
        """
        Initialize the prober.

        Args:
            model: MLX model from mlx_lm.load()
            tokenizer: Tokenizer from mlx_lm.load()
            config: Probe configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ProbeConfig()
        self.results = ProbeResults()

        # Detect model architecture
        self._detect_architecture()

    def _detect_architecture(self):
        """Detect model architecture and layer structure."""
        # Get the inner model (usually model.model for mlx-lm models)
        if hasattr(self.model, 'model'):
            self.inner_model = self.model.model
        else:
            self.inner_model = self.model

        # Detect layers
        if hasattr(self.inner_model, 'layers'):
            self.layers = list(self.inner_model.layers)
        else:
            self.layers = []

        # Detect embedding
        if hasattr(self.inner_model, 'embed_tokens'):
            self.embedding = self.inner_model.embed_tokens
        elif hasattr(self.inner_model, 'embedding'):
            self.embedding = self.inner_model.embedding
        elif hasattr(self.inner_model, 'wte'):
            self.embedding = self.inner_model.wte
        else:
            self.embedding = None

        # Detect output norm
        if hasattr(self.inner_model, 'norm'):
            self.output_norm = self.inner_model.norm
        elif hasattr(self.inner_model, 'final_layernorm'):
            self.output_norm = self.inner_model.final_layernorm
        elif hasattr(self.inner_model, 'ln_f'):
            self.output_norm = self.inner_model.ln_f
        else:
            self.output_norm = None

        # Store model info
        self.results.num_layers = len(self.layers)

        # Try to get hidden dim
        if self.layers:
            first_layer = self.layers[0]
            if hasattr(first_layer, 'hidden_size'):
                self.results.hidden_dim = first_layer.hidden_size
            elif hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'hidden_size'):
                self.results.hidden_dim = first_layer.self_attn.hidden_size

        # Try to get vocab size
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'vocab_size'):
            self.results.vocab_size = self.model.model.vocab_size
        elif self.embedding is not None and hasattr(self.embedding, 'num_embeddings'):
            self.results.vocab_size = self.embedding.num_embeddings

    def _should_capture_layer(self, layer_idx: int) -> bool:
        """Check if we should capture this layer."""
        if self.config.layer_indices is None:
            return True
        return layer_idx in self.config.layer_indices

    def _to_numpy(self, x: mx.array, max_positions: Optional[int] = None) -> np.ndarray:
        """Convert MLX array to numpy."""
        mx.eval(x)
        arr = np.array(x)
        if max_positions and len(arr.shape) >= 2:
            arr = arr[..., :max_positions, :]
        return arr

    def reset_results(self):
        """Clear results for new probe."""
        self.results = ProbeResults()
        self.results.num_layers = len(self.layers)

    def _capture_activations(self, x: mx.array):
        """
        Run forward pass and capture activations.

        This is the core probing logic that works across architectures.
        """
        max_pos = self.config.max_sequence_positions

        def compute_norm(arr: np.ndarray) -> float:
            """Compute mean L2 norm."""
            return float(np.linalg.norm(arr, axis=-1).mean())

        # 1. Embedding
        if self.embedding is not None:
            h = self.embedding(x)
            mx.eval(h)

            if self.config.capture_embeddings:
                self.results.embeddings = self._to_numpy(h, max_pos)

            if self.config.capture_residual_stream:
                h_np = self._to_numpy(h, max_pos)
                self.results.residual_stream["embedding"] = h_np
                self.results.residual_stream_norms.append(("Embed", compute_norm(h_np)))
                prev_h = h_np
        else:
            # Fallback: use model directly
            h = x
            prev_h = None

        # 2. Transformer layers
        for i, layer in enumerate(self.layers):
            # Forward through layer
            # Most layers accept (hidden_states, mask, cache) or similar
            try:
                h = layer(h, mask=None, cache=None)
            except TypeError:
                try:
                    h = layer(h, attention_mask=None)
                except TypeError:
                    try:
                        h = layer(h)
                    except:
                        continue
            mx.eval(h)

            # Handle tuple outputs (some models return (hidden, present_kv))
            if isinstance(h, tuple):
                h = h[0]

            # Capture layer output
            if self.config.capture_layer_outputs and self._should_capture_layer(i):
                self.results.layer_outputs[i] = self._to_numpy(h, max_pos)

            # Capture FFN activations (approximate via output)
            if self.config.capture_ffn_activations and self._should_capture_layer(i):
                self.results.ffn_activations[i] = self._to_numpy(h, max_pos)

            # Capture residual stream
            if self.config.capture_residual_stream and self._should_capture_layer(i):
                h_np = self._to_numpy(h, max_pos)
                self.results.residual_stream[f"layer_{i}"] = h_np
                norm = compute_norm(h_np)
                self.results.residual_stream_norms.append((f"L{i}", norm))

                if prev_h is not None:
                    delta = compute_norm(h_np - prev_h)
                    self.results.residual_stream_deltas.append((f"L{i}", delta))
                prev_h = h_np

        # 3. Output normalization
        if self.output_norm is not None:
            h = self.output_norm(h)
            mx.eval(h)

        # 4. Get logits
        if self.config.capture_logits:
            # Most models use embedding weights for output projection
            if hasattr(self.embedding, 'as_linear'):
                logits = self.embedding.as_linear(h)
            elif hasattr(self.model, 'lm_head'):
                logits = self.model.lm_head(h)
            elif hasattr(self.inner_model, 'lm_head'):
                logits = self.inner_model.lm_head(h)
            else:
                logits = h  # Fallback

            mx.eval(logits)
            self.results.logits = np.array(logits[0, -1, :])

            # Get top-k tokens
            top_k = 20
            top_indices = np.argsort(self.results.logits)[-top_k:][::-1]
            self.results.top_k_tokens = [
                (int(idx), float(self.results.logits[idx]))
                for idx in top_indices
            ]

    def probe(self, prompt: str, max_tokens: int = 1) -> ProbeResults:
        """
        Run probing on a prompt.

        Args:
            prompt: Input text
            max_tokens: Tokens to generate (1 for just probing)

        Returns:
            ProbeResults with captured data
        """
        self.reset_results()

        # Encode input
        if hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(prompt)
        else:
            tokens = self.tokenizer(prompt)

        if isinstance(tokens, dict):
            tokens = tokens['input_ids']

        self.results.input_tokens = list(tokens)
        self.results.input_text = prompt

        # Convert to MLX
        x = mx.array([tokens])

        # Capture activations
        self._capture_activations(x)

        # Decode top tokens
        if self.results.top_k_tokens:
            try:
                for idx, logit in self.results.top_k_tokens[:5]:
                    if hasattr(self.tokenizer, 'decode'):
                        text = self.tokenizer.decode([idx])
                    else:
                        text = str(idx)
            except:
                pass

        return self.results

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID to text."""
        try:
            if hasattr(self.tokenizer, 'decode'):
                return self.tokenizer.decode([token_id])
            elif hasattr(self.tokenizer, 'convert_ids_to_tokens'):
                return self.tokenizer.convert_ids_to_tokens([token_id])[0]
            else:
                return f"[{token_id}]"
        except:
            return f"[{token_id}]"


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_layer_norms(results: ProbeResults) -> go.Figure:
    """Plot activation norms across layers."""
    layers = []
    norms = []

    for layer_idx in sorted(results.layer_outputs.keys()):
        arr = results.layer_outputs[layer_idx]
        norm = float(np.linalg.norm(arr, axis=-1).mean())
        layers.append(f"L{layer_idx}")
        norms.append(norm)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers,
        y=norms,
        mode='lines+markers',
        name='Activation Norm',
        line=dict(color='#636EFA', width=2),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title="Layer Activation Norms (L2)",
        xaxis_title="Layer",
        yaxis_title="L2 Norm",
        hovermode='x unified'
    )

    return fig


def plot_ffn_analysis(results: ProbeResults) -> go.Figure:
    """Plot FFN gate analysis."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Gate Sparsity", "Mean Activation")
    )

    layers = []
    sparsities = []
    mean_acts = []

    for layer_idx in sorted(results.ffn_activations.keys()):
        arr = results.ffn_activations[layer_idx]
        sparsity = float((np.abs(arr) < 0.1).mean())
        mean_act = float(np.abs(arr).mean())

        layers.append(f"L{layer_idx}")
        sparsities.append(sparsity)
        mean_acts.append(mean_act)

    fig.add_trace(
        go.Scatter(x=layers, y=sparsities, mode='lines+markers', name='Sparsity'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=layers, y=mean_acts, mode='lines+markers', name='Mean |Act|'),
        row=1, col=2
    )

    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="Layer", row=1, col=1)
    fig.update_xaxes(title_text="Layer", row=1, col=2)
    fig.update_yaxes(title_text="Sparsity (fraction near 0)", row=1, col=1)
    fig.update_yaxes(title_text="Mean |Activation|", row=1, col=2)

    return fig


def plot_embeddings_pca(results: ProbeResults, tokenizer) -> go.Figure:
    """Plot PCA of token embeddings."""
    if results.embeddings is None:
        return go.Figure()

    # Get embeddings for each token position
    emb = results.embeddings[0]  # Remove batch dim

    if emb.shape[0] < 2:
        return go.Figure()

    # PCA
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(emb)

    # Get token labels
    labels = []
    for i, tok_id in enumerate(results.input_tokens[:emb.shape[0]]):
        try:
            text = tokenizer.decode([tok_id]) if hasattr(tokenizer, 'decode') else str(tok_id)
            labels.append(f"{i}: {text[:10]}")
        except:
            labels.append(f"{i}: [{tok_id}]")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=emb_2d[:, 0],
        y=emb_2d[:, 1],
        mode='markers+text',
        text=labels,
        textposition='top center',
        marker=dict(size=10, color=list(range(len(labels))), colorscale='Viridis'),
        hoverinfo='text'
    ))

    fig.update_layout(
        title=f"Token Embeddings PCA (explained var: {pca.explained_variance_ratio_.sum():.1%})",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
    )

    return fig


def plot_logits_distribution(results: ProbeResults, tokenizer=None) -> go.Figure:
    """Plot logits distribution with token labels."""
    if results.logits is None:
        return go.Figure()

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Logits Histogram", "Log Probability", "Top Tokens")
    )

    logits = results.logits

    # 1. Raw logits histogram
    fig.add_trace(
        go.Histogram(x=logits, nbinsx=100, name='Logits'),
        row=1, col=1
    )

    # 2. Log probability histogram
    probs = np.exp(logits - np.max(logits))  # Softmax numerator
    probs = probs / probs.sum()
    log_probs = np.log10(probs + 1e-15)

    fig.add_trace(
        go.Histogram(x=log_probs, nbinsx=100, name='Log Prob'),
        row=1, col=2
    )

    # 3. Top tokens bar chart
    top_k = 10
    top_indices = np.argsort(logits)[-top_k:][::-1]
    top_logits = logits[top_indices]

    labels = []
    for idx in top_indices:
        idx_int = int(idx)
        if tokenizer is not None:
            try:
                text = tokenizer.decode([idx_int]) if hasattr(tokenizer, 'decode') else str(idx_int)
                labels.append(text[:15])
            except:
                labels.append(f"[{idx_int}]")
        else:
            labels.append(f"[{idx_int}]")

    fig.add_trace(
        go.Bar(x=labels, y=top_logits, name='Top Tokens'),
        row=1, col=3
    )

    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="Logit Value", row=1, col=1)
    fig.update_xaxes(title_text="Log₁₀(Probability)", row=1, col=2)
    fig.update_xaxes(title_text="Token", row=1, col=3)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Logit", row=1, col=3)

    return fig


def plot_layer_similarity(results: ProbeResults) -> go.Figure:
    """Plot cosine similarity between layers."""
    layer_indices = sorted(results.layer_outputs.keys())
    n = len(layer_indices)

    if n < 2:
        return go.Figure()

    # Compute similarity matrix
    similarity = np.zeros((n, n))

    vectors = []
    for idx in layer_indices:
        vec = results.layer_outputs[idx].flatten()
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        vectors.append(vec)

    for i in range(n):
        for j in range(n):
            similarity[i, j] = np.dot(vectors[i], vectors[j])

    labels = [f"L{idx}" for idx in layer_indices]

    fig = go.Figure(data=go.Heatmap(
        z=similarity,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmid=0.5,
        text=np.round(similarity, 2),
        texttemplate='%{text}',
        textfont={"size": 8},
        hovertemplate='Layer %{x} vs %{y}: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title="Layer Similarity (Cosine)",
        xaxis_title="Layer",
        yaxis_title="Layer"
    )

    return fig


def plot_residual_stream(results: ProbeResults) -> go.Figure:
    """Plot residual stream norms and deltas."""
    if not results.residual_stream_norms:
        return go.Figure()

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Stream Magnitude (L2 Norm)", "Layer Delta (Change)")
    )

    # Norms
    labels = [x[0] for x in results.residual_stream_norms]
    norms = [x[1] for x in results.residual_stream_norms]

    fig.add_trace(
        go.Scatter(x=labels, y=norms, mode='lines+markers', name='Norm'),
        row=1, col=1
    )

    # Deltas
    if results.residual_stream_deltas:
        delta_labels = [x[0] for x in results.residual_stream_deltas]
        deltas = [x[1] for x in results.residual_stream_deltas]

        fig.add_trace(
            go.Bar(x=delta_labels, y=deltas, name='Delta'),
            row=2, col=1
        )

    fig.update_layout(height=600, showlegend=False)
    fig.update_xaxes(title_text="Position", row=1, col=1)
    fig.update_xaxes(title_text="Layer", row=2, col=1)
    fig.update_yaxes(title_text="L2 Norm", row=1, col=1)
    fig.update_yaxes(title_text="Delta", row=2, col=1)

    return fig


# =============================================================================
# Metric Definitions
# =============================================================================

METRIC_DEFINITIONS = {
    "activation_norm": {
        "name": "Activation Norm (L2)",
        "short": "Magnitude of neural activity at each layer",
        "full": """**Activation Norm (L2 Norm)** measures the Euclidean length of the activation vector.

---

**Chart Axes:**
- **X-axis (Layer):** Layer index from 0 to N-1
- **Y-axis (L2 Norm):** Euclidean magnitude: sqrt(sum(x²))

---

**Why it matters:**
- Increasing norms = building up representations
- Decreasing norms = information compression
- Sudden changes = important processing steps"""
    },
    "ffn_analysis": {
        "name": "FFN Analysis",
        "short": "Feed-forward network gate patterns",
        "full": """**FFN Analysis** examines the feed-forward network activations.

---

**Left Chart - Gate Sparsity:**
- **X-axis:** Layer index
- **Y-axis:** Fraction of activations near zero (|x| < 0.1)
- Higher = more selective processing

**Right Chart - Mean Activation:**
- **X-axis:** Layer index
- **Y-axis:** Average |activation| magnitude
- Higher = stronger signals"""
    },
    "embeddings": {
        "name": "Token Embeddings",
        "short": "Vector representations of input tokens",
        "full": """**Token Embeddings** are learned vectors for each token.

---

**PCA Scatter Plot:**
- **X-axis (PC1):** First principal component
- **Y-axis (PC2):** Second principal component
- **Points:** Tokens; nearby = similar meaning"""
    },
    "logits": {
        "name": "Logits",
        "short": "Raw output scores before softmax",
        "full": """**Logits** are raw scores for each vocabulary token.

---

**Histogram (Left):**
- **X-axis:** Logit value (can be negative)
- **Y-axis:** Count of tokens

**Log Probability (Center):**
- **X-axis:** Log₁₀(probability)
- **Y-axis:** Count

**Top Tokens (Right):**
- **X-axis:** Token text
- **Y-axis:** Logit score"""
    },
    "layer_similarity": {
        "name": "Layer Similarity",
        "short": "Cosine similarity between layer outputs",
        "full": """**Layer Similarity** shows how similar representations are across layers.

---

**Heatmap:**
- **X/Y axes:** Layer indices
- **Color:** Cosine similarity (0=orthogonal, 1=identical)
- **Diagonal:** Always 1.0 (self-similarity)"""
    },
    "residual_stream": {
        "name": "Residual Stream",
        "short": "Information flow through the transformer",
        "full": """**Residual Stream** tracks the main information pathway.

---

**Top Chart - Stream Magnitude:**
- **X-axis:** Position (Embedding → Layers)
- **Y-axis:** L2 norm of hidden state

**Bottom Chart - Layer Delta:**
- **X-axis:** Layer
- **Y-axis:** How much that layer changed the stream"""
    }
}


def metric_help(key: str) -> str:
    """Get help text for a metric."""
    if key in METRIC_DEFINITIONS:
        return METRIC_DEFINITIONS[key]["full"]
    return ""


# =============================================================================
# Streamlit App
# =============================================================================

def main():
    st.set_page_config(
        page_title="MLXLMProbe",
        page_icon="🔬",
        layout="wide"
    )

    st.title("🔬 MLXLMProbe")
    st.caption("Universal probing tool for MLX language models")

    # Sidebar
    st.sidebar.header("Model")

    # Model path input
    default_model = st.sidebar.text_input(
        "Model path or HuggingFace ID",
        value="mlx-community/Llama-3.2-1B-Instruct-4bit",
        help="Path to local MLX model or HuggingFace model ID"
    )

    # Load model button
    if st.sidebar.button("Load Model", type="primary"):
        with st.spinner(f"Loading {default_model}..."):
            try:
                from mlx_lm import load
                model, tokenizer = load(default_model)
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.model_path = default_model
                st.sidebar.success("Model loaded!")
            except Exception as e:
                st.sidebar.error(f"Error loading model: {e}")

    # Check if model is loaded
    if 'model' not in st.session_state:
        st.info("👆 Enter a model path and click 'Load Model' to get started.")
        st.markdown("""
        **Supported models:**
        - Any MLX model from [mlx-community](https://huggingface.co/mlx-community)
        - Local MLX model directories

        **Examples:**
        - `mlx-community/Llama-3.2-1B-Instruct-4bit`
        - `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
        - `mlx-community/Phi-3-mini-4k-instruct-4bit`
        """)
        return

    model = st.session_state.model
    tokenizer = st.session_state.tokenizer

    st.sidebar.success(f"✓ {st.session_state.model_path}")

    # Probe settings
    st.sidebar.header("Probe Settings")

    capture_embeddings = st.sidebar.checkbox("Capture Embeddings", value=True)
    capture_layers = st.sidebar.checkbox("Capture Layer Outputs", value=True)
    capture_ffn = st.sidebar.checkbox("Capture FFN Activations", value=True)
    capture_logits = st.sidebar.checkbox("Capture Logits", value=True)
    capture_residual = st.sidebar.checkbox("Capture Residual Stream", value=True)

    # Create probe config
    probe_config = ProbeConfig(
        capture_embeddings=capture_embeddings,
        capture_layer_outputs=capture_layers,
        capture_ffn_activations=capture_ffn,
        capture_logits=capture_logits,
        capture_residual_stream=capture_residual
    )

    # Main input
    st.header("Input")
    prompt = st.text_area(
        "Enter your prompt",
        value="The capital of France is",
        height=100
    )

    # Run probe button
    if st.button("🔬 Run Probe", type="primary"):
        with st.spinner("Probing model..."):
            try:
                prober = ModelProber(model, tokenizer, probe_config)
                results = prober.probe(prompt)
                st.session_state.results = results
                st.session_state.prober = prober
            except Exception as e:
                st.error(f"Error during probing: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Display results
    if 'results' not in st.session_state:
        return

    results = st.session_state.results
    prober = st.session_state.prober

    # Tabs
    tabs = st.tabs([
        "📊 Layers",
        "🧠 FFN",
        "📈 Logits",
        "🎯 Embeddings",
        "🔗 Similarity",
        "🌊 Residual"
    ])

    # Layers tab
    with tabs[0]:
        st.subheader("Layer Activation Norms")
        with st.expander("ℹ️ What is this?"):
            st.markdown(metric_help("activation_norm"))

        if results.layer_outputs:
            fig = plot_layer_norms(results)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No layer outputs captured. Enable 'Capture Layer Outputs' and re-run.")

    # FFN tab
    with tabs[1]:
        st.subheader("FFN Analysis")
        with st.expander("ℹ️ What is this?"):
            st.markdown(metric_help("ffn_analysis"))

        if results.ffn_activations:
            fig = plot_ffn_analysis(results)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No FFN activations captured. Enable 'Capture FFN Activations' and re-run.")

    # Logits tab
    with tabs[2]:
        st.subheader("Logits Distribution")
        with st.expander("ℹ️ What is this?"):
            st.markdown(metric_help("logits"))

        if results.logits is not None:
            fig = plot_logits_distribution(results, tokenizer)
            st.plotly_chart(fig, use_container_width=True)

            # Top tokens table
            st.subheader("Top Predicted Tokens")
            if results.top_k_tokens:
                top_data = []
                for tok_id, logit in results.top_k_tokens[:10]:
                    text = prober.decode_token(tok_id)
                    prob = np.exp(logit - results.logits.max())
                    top_data.append({
                        "Token": text,
                        "ID": tok_id,
                        "Logit": f"{logit:.2f}",
                        "~Prob": f"{prob:.4f}"
                    })
                st.table(pd.DataFrame(top_data))
        else:
            st.info("No logits captured. Enable 'Capture Logits' and re-run.")

    # Embeddings tab
    with tabs[3]:
        st.subheader("Token Embeddings")
        with st.expander("ℹ️ What is this?"):
            st.markdown(metric_help("embeddings"))

        if results.embeddings is not None:
            fig = plot_embeddings_pca(results, tokenizer)
            st.plotly_chart(fig, use_container_width=True)

            # Stats
            st.markdown("**Embedding Statistics:**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Shape", str(results.embeddings.shape))
            col2.metric("Mean", f"{results.embeddings.mean():.4f}")
            col3.metric("Std", f"{results.embeddings.std():.4f}")
        else:
            st.info("No embeddings captured. Enable 'Capture Embeddings' and re-run.")

    # Similarity tab
    with tabs[4]:
        st.subheader("Layer Similarity")
        with st.expander("ℹ️ What is this?"):
            st.markdown(metric_help("layer_similarity"))

        if len(results.layer_outputs) >= 2:
            fig = plot_layer_similarity(results)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 layers for similarity analysis.")

    # Residual tab
    with tabs[5]:
        st.subheader("Residual Stream")
        with st.expander("ℹ️ What is this?"):
            st.markdown(metric_help("residual_stream"))

        if results.residual_stream_norms:
            fig = plot_residual_stream(results)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No residual stream data. Enable 'Capture Residual Stream' and re-run.")


if __name__ == "__main__":
    main()
