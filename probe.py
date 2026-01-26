#!/usr/bin/env python3
"""
MLXLMProbe - Universal probing tool for MLX language models.

A visual interpretability tool that works with any mlx-lm compatible model.
Supports Llama, Mistral, Phi, Qwen, Gemma, and other architectures.

Features:
    - Layer activation analysis
    - FFN gate patterns
    - Token probability distributions
    - Embedding visualization (PCA)
    - Layer similarity heatmaps
    - Residual stream tracking
    - AI interpretation
    - PDF and HTML export

Usage:
    streamlit run probe.py
    streamlit run probe.py -- --model mlx-community/Llama-3.2-1B-Instruct-4bit
"""

import argparse
import colorsys
import sys
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime

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

try:
    from huggingface_hub import HfApi, list_models, snapshot_download, hf_hub_download
    import time
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# PDF support (optional)
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

import shutil
import subprocess


# =============================================================================
# Performance Utilities - Caching and Background Processing
# =============================================================================

# Global thread pool for background tasks
_thread_pool = ThreadPoolExecutor(max_workers=4)
_background_tasks: Dict[str, Future] = {}


def get_results_hash(results) -> str:
    """Generate a hash of probe results for cache invalidation."""
    # Create a simple hash based on key result attributes
    hash_data = f"{results.input_text[:100]}_{len(results.input_tokens)}_{results.num_layers}_{results.is_moe}"
    if results.moe_expert_load:
        hash_data += f"_{len(results.moe_expert_load)}"
    return hashlib.md5(hash_data.encode()).hexdigest()[:16]


def run_in_background(task_id: str, func, *args, **kwargs) -> Future:
    """Run a function in background thread. Returns Future."""
    future = _thread_pool.submit(func, *args, **kwargs)
    _background_tasks[task_id] = future
    return future


def get_background_result(task_id: str, timeout: float = 0.1):
    """Get result from background task if ready, else None."""
    if task_id not in _background_tasks:
        return None
    future = _background_tasks[task_id]
    if future.done():
        try:
            return future.result()
        except Exception as e:
            return f"Error: {e}"
    return None  # Not ready yet


def is_task_running(task_id: str) -> bool:
    """Check if a background task is still running."""
    if task_id not in _background_tasks:
        return False
    return not _background_tasks[task_id].done()


def cancel_task(task_id: str):
    """Cancel a background task if possible."""
    if task_id in _background_tasks:
        _background_tasks[task_id].cancel()
        del _background_tasks[task_id]


def get_cached_plot(cache_key: str, plot_func, *args, **kwargs):
    """Get a cached plot or compute and cache it."""
    if cache_key not in st.session_state:
        st.session_state[cache_key] = plot_func(*args, **kwargs)
    return st.session_state[cache_key]


def invalidate_plot_cache():
    """Invalidate all cached plots (call when results change)."""
    keys_to_delete = [k for k in st.session_state.keys() if k.startswith('_plot_')]
    for key in keys_to_delete:
        del st.session_state[key]


def start_background_interpretation(task_id: str, model, tokenizer, context: str, question: str, interpreter: str):
    """Start AI interpretation in background thread."""
    if is_task_running(task_id):
        return  # Already running

    def run_interpretation():
        return generate_ai_interpretation(model, tokenizer, context, question, interpreter)

    run_in_background(task_id, run_interpretation)


def render_background_interpretation(task_id: str, label: str = "AI Interpretation"):
    """Render interpretation result if ready, or show spinner if still computing."""
    if is_task_running(task_id):
        st.info(f"⏳ {label} is being generated in background...")
        return None

    result = get_background_result(task_id)
    if result:
        if isinstance(result, str) and result.startswith("Error:"):
            st.warning(result)
        else:
            st.markdown(f"**Analysis:**\n\n{result}")
        return result
    return None


def get_cached_interpretation(
    cache_key: str,
    results_hash: str,
    interpret_func,
    *args,
    force_refresh: bool = False,
    **kwargs
):
    """
    Get cached AI interpretation or compute it.

    Args:
        cache_key: Unique key for this interpretation
        results_hash: Hash of the probe results for invalidation
        interpret_func: Function to call for interpretation
        force_refresh: If True, recompute even if cached
        *args, **kwargs: Arguments to pass to interpret_func
    """
    full_key = f"_interp_{cache_key}"
    hash_key = f"_interp_hash_{cache_key}"

    # Check cache validity
    if not force_refresh and full_key in st.session_state:
        if hash_key in st.session_state and st.session_state[hash_key] == results_hash:
            return st.session_state[full_key]

    # Compute and cache
    result = interpret_func(*args, **kwargs)
    st.session_state[full_key] = result
    st.session_state[hash_key] = results_hash
    return result


def render_cached_interpretation(
    cache_key: str,
    results,
    interpret_func,
    interp_label: str,
    *args,
    **kwargs
):
    """Render AI interpretation with caching and manual trigger for Claude."""
    results_hash = get_results_hash(results)
    is_claude = interp_label == "Claude"

    # Cache keys include interpreter to separate Claude vs Local results
    full_key = f"_interp_{cache_key}"
    hash_key = f"_interp_hash_{cache_key}"
    source_key = f"_interp_source_{cache_key}"  # Track which interpreter generated it

    has_cache = (
        full_key in st.session_state and
        hash_key in st.session_state and
        st.session_state[hash_key] == results_hash
    )

    if has_cache:
        # Get the source that generated this cache
        cached_source = st.session_state.get(source_key, "Unknown")

        # Show cached result with refresh button
        col1, col2 = st.columns([6, 1])
        with col2:
            force_refresh = st.button("🔄", key=f"refresh_{cache_key}", help=f"Regenerate with {interp_label}")

        if force_refresh:
            # User requested refresh
            if is_claude:
                st.warning("⚠️ **Token Usage Warning**: This will consume Claude API or Claude Code subscription tokens.")
            with st.spinner(f"Generating interpretation via {interp_label}..."):
                interpretation = interpret_func(*args, **kwargs)
                st.session_state[full_key] = interpretation
                st.session_state[hash_key] = results_hash
                st.session_state[source_key] = interp_label  # Store source
                st.markdown(f"**Analysis via {interp_label}:**\n\n{interpretation}")
        else:
            # Display cached - show which interpreter generated it
            interpretation = st.session_state[full_key]
            st.markdown(f"**Analysis via {cached_source}:** *(cached)*\n\n{interpretation}")
    else:
        # No cache - require user action for Claude, auto-generate for local LLM
        if is_claude:
            st.info("💡 No Claude analysis yet. Click the button below to generate.")
            st.warning("⚠️ **Token Usage Warning**: This will consume Claude API or Claude Code subscription tokens.")
            if st.button(f"🧠 Generate with Claude", key=f"generate_{cache_key}", type="primary"):
                with st.spinner(f"Generating interpretation via Claude..."):
                    interpretation = interpret_func(*args, **kwargs)
                    st.session_state[full_key] = interpretation
                    st.session_state[hash_key] = results_hash
                    st.session_state[source_key] = "Claude"  # Store source
                    st.rerun()
        else:
            # Local LLM - auto-generate (free)
            with st.spinner(f"Generating interpretation via Local LLM..."):
                interpretation = interpret_func(*args, **kwargs)
                st.session_state[full_key] = interpretation
                st.session_state[hash_key] = results_hash
                st.session_state[source_key] = "Local LLM"  # Store source
                st.markdown(f"**Analysis via Local LLM:**\n\n{interpretation}")


# Cached computation decorators
def cache_with_hash(func):
    """Decorator to cache function results in session state based on input hash."""
    def wrapper(*args, **kwargs):
        # Generate cache key from function name and args
        cache_key = f"_cache_{func.__name__}"
        hash_key = f"_hash_{func.__name__}"

        # Try to get results object from args for hashing
        results = None
        for arg in args:
            if hasattr(arg, 'input_tokens') and hasattr(arg, 'moe_expert_load'):
                results = arg
                break

        current_hash = get_results_hash(results) if results else str(args)[:100]

        # Check if cached and hash matches
        if cache_key in st.session_state and hash_key in st.session_state:
            if st.session_state[hash_key] == current_hash:
                return st.session_state[cache_key]

        # Compute and cache
        result = func(*args, **kwargs)
        st.session_state[cache_key] = result
        st.session_state[hash_key] = current_hash
        return result

    return wrapper


# =============================================================================
# Claude Code Detection and Integration
# =============================================================================

def detect_claude_code() -> bool:
    """Detect if Claude Code CLI is available."""
    if shutil.which("claude") is not None:
        return True
    common_paths = [
        Path.home() / ".claude" / "local" / "claude",
        Path("/usr/local/bin/claude"),
        Path.home() / ".local" / "bin" / "claude",
    ]
    for path in common_paths:
        if path.exists():
            return True
    return False


def get_claude_code_path() -> Optional[str]:
    """Get the path to Claude Code CLI."""
    claude_path = shutil.which("claude")
    if claude_path:
        return claude_path
    common_paths = [
        Path.home() / ".claude" / "local" / "claude",
        Path("/usr/local/bin/claude"),
        Path.home() / ".local" / "bin" / "claude",
    ]
    for path in common_paths:
        if path.exists():
            return str(path)
    return None


def generate_claude_interpretation(context: str, question: str, max_tokens: int = 300) -> str:
    """
    Use Claude Code CLI to generate interpretation.
    """
    claude_path = get_claude_code_path()
    if not claude_path:
        return "Claude Code not found."

    prompt = f"""You are an expert in neural network interpretability analyzing probe data from an MLX language model.

Based on this probe data:

{context}

{question}

Provide a clear, concise interpretation (2-3 sentences)."""

    try:
        result = subprocess.run(
            [claude_path, "--print", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=60,
            env={**subprocess.os.environ, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            result = subprocess.run(
                [claude_path, "-p", prompt, "--output-format", "text"],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return f"Claude Code error: {result.stderr[:100]}"

    except subprocess.TimeoutExpired:
        return "Claude Code timed out."
    except Exception as e:
        return f"Claude Code error: {str(e)[:50]}"


# =============================================================================
# HuggingFace Model Browser
# =============================================================================

MLX_LM_SUPPORTED_ARCHITECTURES = {
    "llama", "mistral", "mixtral", "phi", "phi3", "phimoe",
    "qwen", "qwen2", "qwen2_moe", "gemma", "gemma2",
    "starcoder", "starcoder2", "cohere", "dbrx", "deepseek",
    "falcon", "gpt2", "gpt_bigcode", "gpt_neox", "internlm2",
    "mamba", "minicpm", "nemotron", "olmo", "openelm",
    "stablelm", "plamo", "recurrentgemma"
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_mlx_community_models(limit: int = 200) -> List[Dict]:
    """Fetch models from mlx-community on HuggingFace."""
    if not HF_HUB_AVAILABLE:
        return []

    try:
        api = HfApi()
        models = list(api.list_models(
            author="mlx-community",
            sort="downloads",
            direction=-1,
            limit=limit
        ))

        model_list = []
        for model in models:
            model_id = model.id
            model_name_lower = model_id.lower()

            skip_keywords = ['embed', 'clip', 'image', 'vision', 'audio', 'whisper', 'encoder-only']
            if any(kw in model_name_lower for kw in skip_keywords):
                continue

            model_info = {
                'id': model_id,
                'name': model_id.split('/')[-1],
                'downloads': getattr(model, 'downloads', 0) or 0,
                'likes': getattr(model, 'likes', 0) or 0,
            }
            model_list.append(model_info)

        return model_list
    except Exception as e:
        return []


def format_model_option(model: Dict) -> str:
    """Format model info for display in selectbox."""
    name = model['name']
    downloads = model['downloads']

    if downloads >= 1_000_000:
        dl_str = f"{downloads/1_000_000:.1f}M"
    elif downloads >= 1_000:
        dl_str = f"{downloads/1_000:.1f}K"
    else:
        dl_str = str(downloads)

    return f"{name} ({dl_str} downloads)"


def format_bytes(size: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"


def download_model_with_progress(model_id: str, progress_bar, status_text) -> str:
    """Download a model from HuggingFace with progress tracking."""
    if not HF_HUB_AVAILABLE:
        raise ImportError("huggingface_hub not available")

    from huggingface_hub import snapshot_download, HfApi, hf_hub_download
    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

    api = HfApi()
    status_text.text("Fetching model info...")
    progress_bar.progress(0)

    try:
        repo_info = api.repo_info(repo_id=model_id, files_metadata=True)
        files = []
        total_size = 0

        for sibling in repo_info.siblings:
            size = getattr(sibling, 'size', None) or 0
            files.append({'name': sibling.rfilename, 'size': size})
            total_size += size

        important_extensions = {'.safetensors', '.json', '.model', '.txt', '.py', '.bin'}
        files = [f for f in files if any(f['name'].endswith(ext) for ext in important_extensions)]
        total_size = sum(f['size'] for f in files)

    except Exception as e:
        status_text.text(f"Downloading model...")
        disable_progress_bars()
        try:
            local_path = snapshot_download(repo_id=model_id)
            progress_bar.progress(1.0)
            status_text.text("Download complete!")
            return local_path
        finally:
            enable_progress_bars()

    downloaded_size = 0
    num_files = len(files)
    disable_progress_bars()

    try:
        for i, file_info in enumerate(files):
            filename = file_info['name']
            file_size = file_info['size']

            short_name = filename.split('/')[-1]
            if len(short_name) > 30:
                short_name = short_name[:27] + "..."

            pct = downloaded_size / total_size if total_size > 0 else 0
            status_text.text(
                f"Downloading ({i+1}/{num_files}): {short_name} | "
                f"{format_bytes(downloaded_size)}/{format_bytes(total_size)} ({pct*100:.0f}%)"
            )
            progress_bar.progress(pct)

            try:
                hf_hub_download(repo_id=model_id, filename=filename, local_files_only=False)
            except Exception:
                pass

            downloaded_size += file_size

        progress_bar.progress(1.0)
        status_text.text("Download complete!")
        local_path = snapshot_download(repo_id=model_id, local_files_only=True)
        return local_path

    except Exception as e:
        status_text.text("Finalizing download...")
        local_path = snapshot_download(repo_id=model_id)
        progress_bar.progress(1.0)
        status_text.text("Download complete!")
        return local_path

    finally:
        enable_progress_bars()


# =============================================================================
# Reasoning Model Formats (extensible for different architectures)
# =============================================================================

@dataclass
class ReasoningFormat:
    """Defines a reasoning model's output format for detection and parsing."""
    name: str
    # Markers that indicate this format is being used
    detection_markers: List[str]
    # Markers that indicate reasoning section
    reasoning_start: List[str]
    reasoning_end: List[str]
    # Markers that indicate final response section
    response_start: List[str]
    # Stop generation when we see response_start followed by any of these
    stop_after_response: List[str]

    def detect(self, text: str) -> bool:
        """Check if text uses this reasoning format."""
        return any(marker in text for marker in self.detection_markers)

    def should_stop(self, text: str) -> bool:
        """Check if generation should stop (response complete)."""
        for resp_marker in self.response_start:
            if resp_marker in text:
                resp_pos = text.find(resp_marker)
                for stop_marker in self.stop_after_response:
                    if text.find(stop_marker, resp_pos) != -1:
                        return True
        return False

    def parse(self, text: str) -> Tuple[str, str]:
        """Extract (reasoning, answer) from text."""
        reasoning = ""
        answer = ""

        # Find reasoning section
        for start_marker in self.reasoning_start:
            if start_marker in text:
                start_pos = text.find(start_marker) + len(start_marker)
                # Find end of reasoning
                end_pos = len(text)
                for end_marker in self.reasoning_end:
                    pos = text.find(end_marker, start_pos)
                    if pos != -1 and pos < end_pos:
                        end_pos = pos
                reasoning = text[start_pos:end_pos].strip()
                break

        # Find response section
        for resp_marker in self.response_start:
            if resp_marker in text:
                resp_pos = text.find(resp_marker) + len(resp_marker)
                answer = text[resp_pos:].strip()
                # Clean any trailing stop markers
                for stop_marker in self.stop_after_response:
                    if stop_marker in answer:
                        answer = answer[:answer.find(stop_marker)].strip()
                break

        return reasoning, answer


# Registry of known reasoning formats - add new architectures here
REASONING_FORMATS = [
    # GPT-OSS / Channel-based format
    ReasoningFormat(
        name="oss_channel",
        detection_markers=['<|channel|>analysis<|message|>', '<|channel|>final<|message|>'],
        reasoning_start=['<|channel|>analysis<|message|>'],
        reasoning_end=['<|end|>', '<|channel|>'],
        response_start=['<|channel|>final<|message|>', '<|channel|>response<|message|>'],
        stop_after_response=['<|end|>'],
    ),
    # DeepSeek / Think format
    ReasoningFormat(
        name="deepseek_think",
        detection_markers=['<think>', '</think>'],
        reasoning_start=['<think>'],
        reasoning_end=['</think>'],
        response_start=['</think>'],
        stop_after_response=['<|end|>', '<|endoftext|>', '</s>'],
    ),
    # Generic reasoning tags
    ReasoningFormat(
        name="generic_reasoning",
        detection_markers=['<reasoning>', '</reasoning>'],
        reasoning_start=['<reasoning>'],
        reasoning_end=['</reasoning>'],
        response_start=['</reasoning>'],
        stop_after_response=['<|end|>', '<|endoftext|>', '</s>'],
    ),
    # Analysis tags
    ReasoningFormat(
        name="generic_analysis",
        detection_markers=['<analysis>', '</analysis>'],
        reasoning_start=['<analysis>'],
        reasoning_end=['</analysis>'],
        response_start=['</analysis>'],
        stop_after_response=['<|end|>', '<|endoftext|>', '</s>'],
    ),
]


def detect_reasoning_format(text: str) -> Optional[ReasoningFormat]:
    """Detect which reasoning format (if any) the text uses."""
    for fmt in REASONING_FORMATS:
        if fmt.detect(text):
            return fmt
    return None


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
    capture_token_probs: bool = True
    capture_attention: bool = True  # Capture attention patterns
    layer_indices: Optional[List[int]] = None
    max_sequence_positions: int = 512


def clean_special_tokens(text: str) -> str:
    """Remove special tokens from text."""
    import re
    # Common special tokens to remove
    special_tokens = [
        r'<\|channel\|>\w*<\|message\|>',  # <|channel|>xxx<|message|>
        r'<\|end\|>',
        r'<\|start\|>',
        r'<\|assistant\|>',
        r'<\|user\|>',
        r'<\|system\|>',
        r'<\|im_start\|>',
        r'<\|im_end\|>',
        r'<\|endoftext\|>',
        r'<\|pad\|>',
        r'assistant<\|channel\|>\w*<\|message\|>',  # assistant<|channel|>final<|message|>
    ]

    result = text
    for pattern in special_tokens:
        result = re.sub(pattern, '', result)

    # Clean up extra whitespace
    result = re.sub(r'\n\s*\n', '\n\n', result)
    return result.strip()


def parse_reasoning_output(text: str) -> Tuple[str, str]:
    """
    Parse model output to separate reasoning from final answer.
    Uses the REASONING_FORMATS registry for extensible format support.

    Returns:
        Tuple of (reasoning, answer). If no reasoning found, reasoning is empty.
    """
    # Try each registered format
    fmt = detect_reasoning_format(text)
    if fmt:
        reasoning, answer = fmt.parse(text)
        answer = clean_special_tokens(answer)

        # If we have reasoning but no answer, generation was likely incomplete
        if reasoning and not answer:
            answer = "(Response generation incomplete - increase max tokens)"

        return reasoning, answer

    # No reasoning pattern found - still clean special tokens
    return "", clean_special_tokens(text)


@dataclass
class ProbeResults:
    """Container for probe results."""
    # Input/Output
    input_tokens: List[int] = field(default_factory=list)
    input_text: str = ""

    # Generation
    generated_tokens: List[int] = field(default_factory=list)
    generated_text: str = ""
    reasoning_text: str = ""  # Separated reasoning/thinking
    answer_text: str = ""     # Final answer after reasoning
    per_token_alternatives: List[List[Tuple]] = field(default_factory=list)  # Top alternatives at each generation step

    # Embeddings
    embeddings: Optional[np.ndarray] = None

    # Layer outputs
    layer_outputs: Dict[int, np.ndarray] = field(default_factory=dict)

    # FFN activations
    ffn_activations: Dict[int, np.ndarray] = field(default_factory=dict)

    # Logits and token probabilities
    logits: Optional[np.ndarray] = None
    token_probs: Optional[np.ndarray] = None
    top_k_tokens: List[Tuple[int, float, str]] = field(default_factory=list)

    # Residual stream
    residual_stream: Dict[str, np.ndarray] = field(default_factory=dict)
    residual_stream_norms: List[Tuple[str, float]] = field(default_factory=list)
    residual_stream_deltas: List[Tuple[str, float]] = field(default_factory=list)

    # Model info
    model_type: str = ""
    num_layers: int = 0
    hidden_dim: int = 0
    vocab_size: int = 0

    # MoE (Mixture of Experts) specific
    is_moe: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0  # top-k experts selected
    # Per-layer router outputs: {layer_idx: {"probs": np.array, "selected": np.array}}
    moe_router_outputs: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    # Expert load: {layer_idx: {expert_idx: token_count}}
    moe_expert_load: Dict[int, Dict[int, int]] = field(default_factory=dict)
    # Expert token assignments: {expert_idx: [(token_id, position, layer_idx, router_prob), ...]}
    moe_expert_tokens: Dict[int, List[Tuple[int, int, int, float]]] = field(default_factory=dict)

    # Attention patterns
    # Per-layer attention weights: {layer_idx: np.array of shape (n_heads, seq_len, seq_len)}
    attention_patterns: Dict[int, np.ndarray] = field(default_factory=dict)
    n_attention_heads: int = 0
    n_kv_heads: int = 0

    # Generation Replay timeline (for replay feature)
    generation_timeline: Optional["GenerationTimeline"] = None


# =============================================================================
# Causal Tracing Data Structures (Knowledge Localization)
# =============================================================================

@dataclass
class CausalTraceConfig:
    """Configuration for causal tracing experiments."""
    noise_multiplier: float = 3.0  # Noise scale relative to embedding std
    subject_positions: Optional[List[int]] = None  # Positions to corrupt (auto-detect if None)
    target_token_id: Optional[int] = None  # Token to measure probability for
    sample_layers: bool = True  # Sample layers instead of all (for speed)
    layer_sample_rate: int = 2  # Every Nth layer when sampling
    sample_positions: bool = True  # Sample positions instead of all
    position_sample_rate: int = 2  # Every Nth position when sampling


@dataclass
class CausalTraceResults:
    """Results from causal tracing experiment."""
    query_text: str = ""
    subject_text: str = ""
    target_token: str = ""
    target_token_id: int = 0

    # Core measurements
    clean_prob: float = 0.0  # P(target) with no corruption
    corrupted_prob: float = 0.0  # P(target) with subject corrupted
    indirect_effects: Optional[np.ndarray] = None  # (num_layers, seq_len) recovery matrix

    # Critical site (max recovery)
    critical_layer: int = 0
    critical_position: int = 0
    max_recovery: float = 0.0

    # Metadata
    subject_positions: List[int] = field(default_factory=list)
    layers_tested: List[int] = field(default_factory=list)
    positions_tested: List[int] = field(default_factory=list)
    num_forward_passes: int = 0


# =============================================================================
# Generation Replay Data Structures
# =============================================================================

@dataclass
class GenerationStepState:
    """State captured at a single generation step."""
    step_index: int
    token_id: int
    token_text: str
    logits: Optional[np.ndarray] = None  # Full logits at this step (optional, memory-intensive)
    top_k_alternatives: List[Tuple[int, float, str]] = field(default_factory=list)  # (id, prob, text)
    layer_outputs: Optional[Dict[int, np.ndarray]] = None  # Sampled layer outputs
    cumulative_tokens: List[int] = field(default_factory=list)  # All tokens up to this point
    probability: float = 0.0  # Probability of chosen token


@dataclass
class GenerationTimeline:
    """Timeline of generation steps, supporting branching."""
    timeline_id: str = ""
    steps: List[GenerationStepState] = field(default_factory=list)
    parent_timeline_id: Optional[str] = None  # For branches
    branch_point_step: Optional[int] = None  # Step index where this branch diverged
    input_tokens: List[int] = field(default_factory=list)
    input_text: str = ""
    debug_info: List[str] = field(default_factory=list)  # Debug information for branches


@dataclass
class ReplayCaptureConfig:
    """Configuration for generation replay capture."""
    mode: str = "minimal"  # minimal/standard/full
    layer_sample_rate: int = 4  # Capture every Nth layer in standard mode
    max_memory_mb: float = 100.0  # Memory limit for capture
    capture_logits: bool = False  # Only in full mode
    capture_layer_outputs: bool = False  # Only in standard/full mode
    top_k_alternatives: int = 10  # Number of alternatives to capture


# =============================================================================
# Model Prober
# =============================================================================

def get_model_topology(model, tokenizer, model_path: str = None) -> Dict[str, Any]:
    """
    Extract model topology/architecture information.

    Reads from config.json if model_path is provided, otherwise infers from model structure.
    Returns a dictionary with model structure details.
    """
    topology = {
        "model_type": "Unknown",
        "architecture": [],
        "config": {},
        "layer_details": [],
        "total_parameters": 0,
        "is_moe": False,
    }

    # Get inner model
    inner_model = model.model if hasattr(model, 'model') else model

    # Try to get model type from class name
    model_class = type(inner_model).__name__
    topology["model_type"] = model_class

    # Detect MoE from class name
    if 'moe' in model_class.lower() or 'mixture' in model_class.lower():
        topology["is_moe"] = True

    # Try to read config.json directly from model path
    config_dict = {}
    if model_path:
        import os
        config_paths = [
            os.path.join(model_path, "config.json"),
            os.path.join(os.path.expanduser(model_path), "config.json"),
        ]
        # Also check HuggingFace cache
        if "/" in model_path and not os.path.exists(model_path):
            from huggingface_hub import hf_hub_download
            try:
                config_file = hf_hub_download(repo_id=model_path, filename="config.json")
                config_paths.insert(0, config_file)
            except:
                pass

        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    import json
                    with open(config_path, 'r') as f:
                        config_dict = json.load(f)
                    break
                except:
                    pass

    # Fallback to model.config attributes
    if hasattr(model, 'config'):
        config = model.config
    elif hasattr(inner_model, 'config'):
        config = inner_model.config
    else:
        config = None

    # Priority: config.json > model.config attributes
    config_attrs = [
        'model_type', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
        'num_attention_heads', 'num_key_value_heads', 'vocab_size',
        'max_position_embeddings', 'rms_norm_eps', 'rope_theta',
        'head_dim', 'tie_word_embeddings', 'hidden_act',
        # MoE specific
        'num_experts', 'num_experts_per_tok', 'num_local_experts',
        'num_experts_per_token', 'moe_intermediate_size', 'router_aux_loss_coef',
        'num_selected_experts', 'n_routed_experts'
    ]

    # First from config.json
    for attr in config_attrs:
        if attr in config_dict:
            val = config_dict[attr]
            if val is not None:
                topology["config"][attr] = val

    # Then from model.config (if not already set)
    if config:
        for attr in config_attrs:
            if attr not in topology["config"] and hasattr(config, attr):
                val = getattr(config, attr)
                if val is not None:
                    topology["config"][attr] = val

    # Detect MoE from config
    moe_indicators = ['num_experts', 'num_local_experts', 'num_experts_per_tok',
                      'num_selected_experts', 'n_routed_experts', 'moe_intermediate_size']
    for indicator in moe_indicators:
        if topology["config"].get(indicator):
            topology["is_moe"] = True
            break

    # Build architecture list
    arch = []

    # Embedding layer
    if hasattr(inner_model, 'embed_tokens'):
        emb = inner_model.embed_tokens
        if hasattr(emb, 'weight'):
            shape = emb.weight.shape
            arch.append(f"Embedding: {shape[0]:,} tokens × {shape[1]:,} dim")
            topology["total_parameters"] += shape[0] * shape[1]
    elif hasattr(inner_model, 'wte'):
        emb = inner_model.wte
        if hasattr(emb, 'weight'):
            shape = emb.weight.shape
            arch.append(f"Embedding (wte): {shape[0]:,} tokens × {shape[1]:,} dim")
            topology["total_parameters"] += shape[0] * shape[1]

    # Transformer layers
    if hasattr(inner_model, 'layers'):
        layers = list(inner_model.layers)
        num_layers = len(layers)
        arch.append(f"Transformer Layers: {num_layers}")

        # Analyze first layer for structure
        if layers:
            first_layer = layers[0]
            layer_info = {"index": 0, "components": []}

            # Self attention
            if hasattr(first_layer, 'self_attn'):
                attn = first_layer.self_attn
                attn_info = "Self-Attention"

                if hasattr(attn, 'num_heads'):
                    attn_info += f" ({attn.num_heads} heads"
                elif hasattr(attn, 'n_heads'):
                    attn_info += f" ({attn.n_heads} heads"

                if hasattr(attn, 'num_kv_heads'):
                    attn_info += f", {attn.num_kv_heads} KV heads"
                elif hasattr(attn, 'n_kv_heads'):
                    attn_info += f", {attn.n_kv_heads} KV heads"

                if hasattr(attn, 'head_dim'):
                    attn_info += f", {attn.head_dim} head_dim"

                if '(' in attn_info:
                    attn_info += ")"

                layer_info["components"].append(attn_info)

                # Count attention parameters
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'qkv_proj']:
                    if hasattr(attn, proj_name):
                        proj = getattr(attn, proj_name)
                        if hasattr(proj, 'weight'):
                            shape = proj.weight.shape
                            topology["total_parameters"] += shape[0] * shape[1] * num_layers

            # MLP/FFN
            if hasattr(first_layer, 'mlp'):
                mlp = first_layer.mlp
                mlp_info = "MLP/FFN"

                if hasattr(mlp, 'gate_proj') and hasattr(mlp.gate_proj, 'weight'):
                    shape = mlp.gate_proj.weight.shape
                    mlp_info += f" (hidden: {shape[0]:,})"
                    # gate_proj, up_proj, down_proj
                    topology["total_parameters"] += shape[0] * shape[1] * 3 * num_layers
                elif hasattr(mlp, 'fc1') and hasattr(mlp.fc1, 'weight'):
                    shape = mlp.fc1.weight.shape
                    mlp_info += f" (hidden: {shape[0]:,})"
                    topology["total_parameters"] += shape[0] * shape[1] * 2 * num_layers

                layer_info["components"].append(mlp_info)

            # MoE (Mixture of Experts) detection
            moe_module = None
            moe_attr_names = ['block_sparse_moe', 'moe', 'sparse_moe', 'experts', 'switch_mlp']
            for attr_name in moe_attr_names:
                if hasattr(first_layer, attr_name):
                    moe_module = getattr(first_layer, attr_name)
                    break

            # Also check inside layer.mlp for MoE (e.g., GPT-OSS structure)
            if moe_module is None and hasattr(first_layer, 'mlp'):
                mlp = first_layer.mlp
                # MoE inside MLP has experts and router attributes
                if hasattr(mlp, 'experts') and hasattr(mlp, 'router'):
                    moe_module = mlp

            if moe_module is not None:
                topology["is_moe"] = True
                moe_info = "MoE"

                # Get number of experts (priority: config_dict > module attributes > model.config)
                num_experts = None
                # First check config_dict (from config.json)
                if config_dict.get('num_local_experts'):
                    num_experts = config_dict['num_local_experts']
                elif config_dict.get('num_experts'):
                    num_experts = config_dict['num_experts']
                elif config_dict.get('n_routed_experts'):
                    num_experts = config_dict['n_routed_experts']
                # Then check module attributes
                elif hasattr(moe_module, 'experts'):
                    if hasattr(moe_module.experts, '__len__'):
                        num_experts = len(moe_module.experts)
                elif hasattr(moe_module, 'num_experts'):
                    num_experts = moe_module.num_experts
                # Finally check model.config
                elif config and hasattr(config, 'num_local_experts'):
                    num_experts = config.num_local_experts
                elif config and hasattr(config, 'num_experts'):
                    num_experts = config.num_experts

                if num_experts:
                    topology["num_experts"] = num_experts
                    moe_info += f" ({num_experts} experts"

                # Get top-k experts per token (priority: config_dict > module attributes > model.config)
                top_k = None
                # First check config_dict (from config.json)
                if config_dict.get('num_experts_per_tok'):
                    top_k = config_dict['num_experts_per_tok']
                elif config_dict.get('num_experts_per_token'):
                    top_k = config_dict['num_experts_per_token']
                elif config_dict.get('num_selected_experts'):
                    top_k = config_dict['num_selected_experts']
                # Then check module attributes
                elif hasattr(moe_module, 'num_experts_per_tok'):
                    top_k = moe_module.num_experts_per_tok
                elif hasattr(moe_module, 'top_k'):
                    top_k = moe_module.top_k
                # Finally check model.config
                elif config and hasattr(config, 'num_experts_per_tok'):
                    top_k = config.num_experts_per_tok
                elif config and hasattr(config, 'num_experts_per_token'):
                    top_k = config.num_experts_per_token

                if top_k:
                    topology["num_experts_per_tok"] = top_k
                    moe_info += f", top-{top_k}"

                # Check for router/gate
                if hasattr(moe_module, 'gate') or hasattr(moe_module, 'router'):
                    moe_info += ", gated"

                if '(' in moe_info:
                    moe_info += ")"

                layer_info["components"].append(moe_info)
                arch.append(f"  └─ {moe_info}")

            # Layer norms
            norm_count = 0
            for norm_name in ['input_layernorm', 'post_attention_layernorm', 'ln_1', 'ln_2']:
                if hasattr(first_layer, norm_name):
                    norm_count += 1

            if norm_count > 0:
                layer_info["components"].append(f"LayerNorm × {norm_count}")

            topology["layer_details"].append(layer_info)

            # Add layer structure to arch
            arch.append(f"  └─ Per layer: {' → '.join(layer_info['components'])}")

    # Output norm
    if hasattr(inner_model, 'norm'):
        arch.append("Output LayerNorm")
    elif hasattr(inner_model, 'ln_f'):
        arch.append("Output LayerNorm (ln_f)")

    # LM head
    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head
        if hasattr(lm_head, 'weight'):
            shape = lm_head.weight.shape
            arch.append(f"LM Head: {shape[1]:,} → {shape[0]:,} (vocab)")
            topology["total_parameters"] += shape[0] * shape[1]
    elif hasattr(inner_model, 'embed_tokens') and hasattr(inner_model.embed_tokens, 'as_linear'):
        arch.append("LM Head: tied to embeddings")

    topology["architecture"] = arch

    # Calculate total parameters from config if available (more accurate)
    cfg = topology["config"]
    if cfg.get("hidden_size") and cfg.get("num_hidden_layers"):
        hidden = cfg["hidden_size"]
        n_layers = cfg["num_hidden_layers"]
        vocab = cfg.get("vocab_size", 0)
        intermediate = cfg.get("intermediate_size", hidden * 4)
        n_heads = cfg.get("num_attention_heads", 1)
        n_kv_heads = cfg.get("num_key_value_heads", n_heads)
        head_dim = cfg.get("head_dim", hidden // n_heads)

        # Embedding
        params = vocab * hidden

        # Per layer: attention + FFN + norms
        # Attention: Q, K, V, O projections
        attn_params = (hidden * n_heads * head_dim) + (hidden * n_kv_heads * head_dim) * 2 + (n_heads * head_dim * hidden)

        # FFN
        if topology["is_moe"]:
            num_experts = cfg.get("num_experts") or cfg.get("num_local_experts") or cfg.get("n_routed_experts", 8)
            moe_intermediate = cfg.get("moe_intermediate_size", intermediate)
            # Each expert has gate, up, down projections
            ffn_params = num_experts * (hidden * moe_intermediate * 3)
            # Plus router
            ffn_params += hidden * num_experts
            topology["num_experts"] = num_experts
            topology["num_experts_per_tok"] = cfg.get("num_experts_per_tok") or cfg.get("num_selected_experts", 2)
        else:
            # Standard FFN: gate, up, down
            ffn_params = hidden * intermediate * 3

        # Layer norms (2 per layer typically)
        norm_params = hidden * 4

        layer_params = attn_params + ffn_params + norm_params
        params += layer_params * n_layers

        # Output norm + LM head (if not tied)
        params += hidden  # final norm
        if not cfg.get("tie_word_embeddings", True):
            params += vocab * hidden

        topology["total_parameters"] = int(params)

    # Tokenizer info
    if tokenizer:
        topology["tokenizer"] = {
            "type": type(tokenizer).__name__,
        }
        if hasattr(tokenizer, 'vocab_size'):
            topology["tokenizer"]["vocab_size"] = tokenizer.vocab_size

        # Handle max_length carefully - some tokenizers have absurdly large defaults
        max_len = None
        if hasattr(tokenizer, 'model_max_length'):
            max_len = tokenizer.model_max_length
            # Sanity check - if it's > 1M, use max_position_embeddings from config instead
            if max_len and max_len > 1_000_000:
                max_len = cfg.get("max_position_embeddings")
        if max_len is None:
            max_len = cfg.get("max_position_embeddings")
        if max_len:
            topology["tokenizer"]["max_length"] = max_len

        if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token:
            topology["tokenizer"]["bos_token"] = repr(tokenizer.bos_token)
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
            topology["tokenizer"]["eos_token"] = repr(tokenizer.eos_token)

    return topology


def format_topology_display(topology: Dict[str, Any]) -> str:
    """Format topology dict for display."""
    lines = []

    lines.append(f"**Model Type:** `{topology['model_type']}`")
    lines.append("")

    # Config table
    if topology.get("config"):
        lines.append("**Configuration:**")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for key, val in topology["config"].items():
            # Format key nicely
            display_key = key.replace('_', ' ').title()
            # Format value
            if isinstance(val, int) and val > 1000:
                display_val = f"{val:,}"
            else:
                display_val = str(val)
            lines.append(f"| {display_key} | {display_val} |")
        lines.append("")

    # Architecture
    if topology.get("architecture"):
        lines.append("**Architecture:**")
        lines.append("```")
        for item in topology["architecture"]:
            lines.append(item)
        lines.append("```")
        lines.append("")

    # MoE info
    if topology.get("is_moe"):
        lines.append("**🔀 Mixture of Experts:**")
        moe_info = []
        if topology.get("num_experts"):
            moe_info.append(f"{topology['num_experts']} experts")
        if topology.get("num_experts_per_tok"):
            moe_info.append(f"top-{topology['num_experts_per_tok']} selection")
        if moe_info:
            lines.append(" | ".join(moe_info))
        lines.append("")

    # Parameters
    if topology.get("total_parameters") > 0:
        params = topology["total_parameters"]
        if params >= 1e9:
            param_str = f"{params/1e9:.2f}B"
        elif params >= 1e6:
            param_str = f"{params/1e6:.1f}M"
        else:
            param_str = f"{params:,}"
        lines.append(f"**Estimated Parameters:** ~{param_str}")
        lines.append("")

    # Tokenizer
    if topology.get("tokenizer"):
        lines.append("**Tokenizer:**")
        tok = topology["tokenizer"]
        tok_info = [f"Type: `{tok.get('type', 'Unknown')}`"]
        if 'vocab_size' in tok:
            tok_info.append(f"Vocab: {tok['vocab_size']:,}")
        if 'max_length' in tok:
            tok_info.append(f"Max length: {tok['max_length']:,}")
        lines.append(" | ".join(tok_info))

    return "\n".join(lines)


def calculate_transformer_flops(
    topology: Dict[str, Any],
    seq_len: int,
    is_prefill: bool = True,
    num_new_tokens: int = 1
) -> Dict[str, int]:
    """
    Calculate FLOPs for transformer inference.

    Args:
        topology: Model topology dict with config
        seq_len: Current sequence length (including KV cache)
        is_prefill: True for initial prompt processing, False for generation
        num_new_tokens: Number of new tokens being processed (1 for generation)

    Returns:
        Dict with breakdown of FLOPs by component
    """
    cfg = topology.get("config", {})

    # Extract model dimensions
    d_model = cfg.get("hidden_size", 4096)
    n_layers = cfg.get("num_hidden_layers", 32)
    n_heads = cfg.get("num_attention_heads", 32)
    n_kv_heads = cfg.get("num_key_value_heads", n_heads)
    d_head = cfg.get("head_dim", d_model // n_heads)
    d_ff = cfg.get("intermediate_size", d_model * 4)
    vocab_size = cfg.get("vocab_size", 32000)

    # MoE parameters
    is_moe = topology.get("is_moe", False)
    num_experts = cfg.get("num_local_experts") or cfg.get("num_experts") or cfg.get("n_routed_experts") or 0
    top_k = cfg.get("num_experts_per_tok") or cfg.get("num_selected_experts") or 2
    moe_intermediate = cfg.get("moe_intermediate_size", d_ff)

    flops = {
        "attention_qkv": 0,
        "attention_scores": 0,
        "attention_output": 0,
        "ffn": 0,
        "router": 0,
        "lm_head": 0,
        "total": 0
    }

    # For prefill: process all tokens
    # For generation: process only new token(s) but attention spans full seq_len
    tokens_to_process = seq_len if is_prefill else num_new_tokens

    # Per layer calculations
    for _ in range(n_layers):
        # QKV projections (2 for multiply-add)
        # Q: always computed for new tokens
        q_flops = 2 * tokens_to_process * d_model * (n_heads * d_head)

        if is_prefill:
            # K, V computed for all tokens
            k_flops = 2 * tokens_to_process * d_model * (n_kv_heads * d_head)
            v_flops = 2 * tokens_to_process * d_model * (n_kv_heads * d_head)
        else:
            # K, V only for new token (cached for previous)
            k_flops = 2 * num_new_tokens * d_model * (n_kv_heads * d_head)
            v_flops = 2 * num_new_tokens * d_model * (n_kv_heads * d_head)

        flops["attention_qkv"] += q_flops + k_flops + v_flops

        # Attention scores: Q @ K^T
        # Shape: (tokens_to_process, n_heads, d_head) @ (n_heads, d_head, seq_len)
        # Each query attends to all keys (full seq_len for KV cache)
        attn_seq_len = seq_len  # Always attend to full sequence
        flops["attention_scores"] += 2 * tokens_to_process * n_heads * d_head * attn_seq_len

        # Attention output: attn_weights @ V
        flops["attention_output"] += 2 * tokens_to_process * n_heads * attn_seq_len * d_head

        # Output projection
        flops["attention_output"] += 2 * tokens_to_process * (n_heads * d_head) * d_model

        # FFN / MoE
        if is_moe and num_experts > 0:
            # Router: compute scores for all experts
            flops["router"] += 2 * tokens_to_process * d_model * num_experts

            # Only top_k experts are activated per token
            # SwiGLU: gate, up, down projections
            expert_flops = 2 * tokens_to_process * d_model * moe_intermediate  # gate
            expert_flops += 2 * tokens_to_process * d_model * moe_intermediate  # up
            expert_flops += 2 * tokens_to_process * moe_intermediate * d_model  # down
            flops["ffn"] += expert_flops * top_k
        else:
            # Standard FFN (SwiGLU)
            flops["ffn"] += 2 * tokens_to_process * d_model * d_ff  # gate
            flops["ffn"] += 2 * tokens_to_process * d_model * d_ff  # up
            flops["ffn"] += 2 * tokens_to_process * d_ff * d_model  # down

    # LM head projection
    flops["lm_head"] = 2 * tokens_to_process * d_model * vocab_size

    # Total
    flops["total"] = sum(flops.values())

    return flops


def format_flops(flops: int) -> str:
    """Format FLOPs with appropriate unit."""
    if flops >= 1e15:
        return f"{flops/1e15:.2f} PFLOPs"
    elif flops >= 1e12:
        return f"{flops/1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f} MFLOPs"
    elif flops >= 1e3:
        return f"{flops/1e3:.2f} KFLOPs"
    else:
        return f"{flops} FLOPs"


class ModelProber:
    """Universal prober for MLX language models."""

    def __init__(self, model, tokenizer, config: Optional[ProbeConfig] = None, topology: Optional[Dict] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ProbeConfig()
        self.topology = topology or {}
        self.results = ProbeResults()
        self._detect_architecture()

    def _detect_architecture(self):
        """Detect model architecture and layer structure."""
        if hasattr(self.model, 'model'):
            self.inner_model = self.model.model
        else:
            self.inner_model = self.model

        if hasattr(self.inner_model, 'layers'):
            self.layers = list(self.inner_model.layers)
        else:
            self.layers = []

        if hasattr(self.inner_model, 'embed_tokens'):
            self.embedding = self.inner_model.embed_tokens
        elif hasattr(self.inner_model, 'embedding'):
            self.embedding = self.inner_model.embedding
        elif hasattr(self.inner_model, 'wte'):
            self.embedding = self.inner_model.wte
        else:
            self.embedding = None

        if hasattr(self.inner_model, 'norm'):
            self.output_norm = self.inner_model.norm
        elif hasattr(self.inner_model, 'final_layernorm'):
            self.output_norm = self.inner_model.final_layernorm
        elif hasattr(self.inner_model, 'ln_f'):
            self.output_norm = self.inner_model.ln_f
        else:
            self.output_norm = None

        self.results.num_layers = len(self.layers)

        if self.layers:
            first_layer = self.layers[0]
            if hasattr(first_layer, 'hidden_size'):
                self.results.hidden_dim = first_layer.hidden_size
            elif hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'hidden_size'):
                self.results.hidden_dim = first_layer.self_attn.hidden_size

        if hasattr(self.model, 'model') and hasattr(self.model.model, 'vocab_size'):
            self.results.vocab_size = self.model.model.vocab_size
        elif self.embedding is not None and hasattr(self.embedding, 'num_embeddings'):
            self.results.vocab_size = self.embedding.num_embeddings

        # MoE detection - use topology if available, then check class name
        if self.topology.get("is_moe"):
            self.results.is_moe = True
            # Get num_experts from topology or config dict
            num_experts = self.topology.get("num_experts", 0)
            if not num_experts:
                cfg = self.topology.get("config", {})
                num_experts = cfg.get("num_local_experts") or cfg.get("num_experts") or cfg.get("n_routed_experts") or 0
            self.results.num_experts = num_experts

            # Get num_experts_per_tok from topology or config dict
            num_per_tok = self.topology.get("num_experts_per_tok", 0)
            if not num_per_tok:
                cfg = self.topology.get("config", {})
                num_per_tok = cfg.get("num_experts_per_tok") or cfg.get("num_experts_per_token") or cfg.get("num_selected_experts") or 2
            self.results.num_experts_per_tok = num_per_tok
        else:
            model_class = type(self.inner_model).__name__
            if 'moe' in model_class.lower() or 'mixture' in model_class.lower():
                self.results.is_moe = True

        # Also check layer structure for MoE modules
        self.moe_modules = {}  # {layer_idx: moe_module}
        if self.layers:
            for i, layer in enumerate(self.layers):
                moe_found = False
                # Check direct layer attributes first
                moe_attr_names = ['block_sparse_moe', 'moe', 'sparse_moe', 'switch_mlp', 'experts']
                for attr_name in moe_attr_names:
                    if hasattr(layer, attr_name):
                        self.moe_modules[i] = getattr(layer, attr_name)
                        moe_found = True
                        break

                # Also check inside layer.mlp for MoE (e.g., GPT-OSS structure)
                if not moe_found and hasattr(layer, 'mlp'):
                    mlp = layer.mlp
                    if hasattr(mlp, 'experts') and hasattr(mlp, 'router'):
                        self.moe_modules[i] = mlp

        if self.moe_modules:
            self.results.is_moe = True
            # Get expert count from first MoE module (if not already set from topology)
            if not self.results.num_experts:
                first_moe = list(self.moe_modules.values())[0]
                if hasattr(first_moe, 'experts') and hasattr(first_moe.experts, '__len__'):
                    self.results.num_experts = len(first_moe.experts)
                elif hasattr(first_moe, 'num_experts'):
                    self.results.num_experts = first_moe.num_experts

            # If still no expert count, try topology config (for SwitchGLU-style MoE)
            if not self.results.num_experts:
                cfg = self.topology.get("config", {})
                self.results.num_experts = cfg.get("num_local_experts") or cfg.get("num_experts") or cfg.get("n_routed_experts") or 0

            # Get top-k (if not already set from topology)
            if not self.results.num_experts_per_tok:
                first_moe = list(self.moe_modules.values())[0]
                if hasattr(first_moe, 'num_experts_per_tok'):
                    self.results.num_experts_per_tok = first_moe.num_experts_per_tok
                elif hasattr(first_moe, 'top_k'):
                    self.results.num_experts_per_tok = first_moe.top_k

            # If still no top-k, try topology config
            if not self.results.num_experts_per_tok:
                cfg = self.topology.get("config", {})
                self.results.num_experts_per_tok = cfg.get("num_experts_per_tok") or cfg.get("num_experts_per_token") or cfg.get("num_selected_experts") or 2

    def _should_capture_layer(self, layer_idx: int) -> bool:
        if self.config.layer_indices is None:
            return True
        return layer_idx in self.config.layer_indices

    def _to_numpy(self, x: mx.array, max_positions: Optional[int] = None) -> np.ndarray:
        """Convert MLX array to numpy, handling bfloat16 and other dtypes."""
        mx.eval(x)
        # Always convert to float32 first for numpy compatibility
        x_f32 = x.astype(mx.float32)
        mx.eval(x_f32)
        try:
            # Try direct conversion first (faster)
            arr = np.array(x_f32)
        except (RuntimeError, TypeError, ValueError):
            # Fall back to tolist (slower but always works)
            arr = np.array(x_f32.tolist(), dtype=np.float32)
        if max_positions and len(arr.shape) >= 2:
            arr = arr[..., :max_positions, :]
        return arr

    def reset_results(self):
        # Preserve MoE info before reset
        old_is_moe = getattr(self.results, 'is_moe', False) if hasattr(self, 'results') else False
        old_num_experts = getattr(self.results, 'num_experts', 0) if hasattr(self, 'results') else 0
        old_num_per_tok = getattr(self.results, 'num_experts_per_tok', 0) if hasattr(self, 'results') else 0

        self.results = ProbeResults()
        self.results.num_layers = len(self.layers)

        # Restore MoE info
        if hasattr(self, 'moe_modules') and self.moe_modules:
            self.results.is_moe = True
            self.results.num_experts = old_num_experts
            self.results.num_experts_per_tok = old_num_per_tok

    def _install_router_hooks(self):
        """
        Note: MLX doesn't support runtime hooks like PyTorch.
        Router capture during generation would require modifying the model architecture.
        For now, we only capture router outputs during the initial probe (input tokens).
        """
        self._router_hooks_installed = False
        # Router hooks are not implemented for MLX - generation routing not captured

    def _remove_router_hooks(self):
        """Remove router hooks (no-op for MLX)."""
        self._router_hooks_installed = False

    def _collect_router_captures(self, gen_step: int):
        """Collect router captures (no-op - MLX doesn't support generation-time hooks)."""
        pass  # Router outputs during generation not captured in MLX

    def _capture_moe_router(self, layer_idx: int, layer, hidden_state: mx.array):
        """Capture MoE router outputs for a layer."""
        try:
            moe = self.moe_modules.get(layer_idx)
            if moe is None:
                return

            # Get the router/gate module
            router = None
            if hasattr(moe, 'gate'):
                router = moe.gate
            elif hasattr(moe, 'router'):
                router = moe.router

            if router is None:
                return

            # Get hidden state before MoE (need pre-FFN state)
            # For most architectures, we need the state after attention + norm
            pre_moe_state = hidden_state

            # Some layers have input_layernorm before MLP
            if hasattr(layer, 'post_attention_layernorm'):
                # State should be normalized before going to MoE
                pre_moe_state = layer.post_attention_layernorm(hidden_state)
            elif hasattr(layer, 'ffn_norm'):
                pre_moe_state = layer.ffn_norm(hidden_state)

            mx.eval(pre_moe_state)

            # Run router to get expert selection
            # Router input shape: (batch, seq, hidden) -> (batch * seq, hidden)
            batch_size, seq_len, hidden_dim = pre_moe_state.shape
            router_input = pre_moe_state.reshape(-1, hidden_dim)

            # Get router logits
            router_logits = router(router_input)
            mx.eval(router_logits)

            # Convert to probabilities
            router_probs = mx.softmax(router_logits, axis=-1)
            mx.eval(router_probs)

            # Get top-k selections
            num_experts_per_tok = self.results.num_experts_per_tok
            if not num_experts_per_tok:
                # Try topology config as fallback
                cfg = self.topology.get("config", {})
                num_experts_per_tok = cfg.get("num_experts_per_tok") or cfg.get("num_experts_per_token") or cfg.get("num_selected_experts") or 2
                self.results.num_experts_per_tok = num_experts_per_tok
            top_k_indices = mx.argsort(router_probs, axis=-1)[:, -num_experts_per_tok:]
            mx.eval(top_k_indices)

            # Store results
            self.results.moe_router_outputs[layer_idx] = {
                "probs": self._to_numpy(router_probs, None).reshape(batch_size, seq_len, -1),
                "selected": self._to_numpy(top_k_indices, None).reshape(batch_size, seq_len, -1).astype(np.int32)
            }

            # Compute expert load (how many tokens go to each expert)
            selected_flat = top_k_indices.reshape(-1)
            mx.eval(selected_flat)
            selected_np = np.array(selected_flat.tolist())

            # Get num_experts - fallback to router output shape if not set
            num_experts = self.results.num_experts
            if not num_experts:
                # Infer from router output shape (router outputs probabilities over all experts)
                num_experts = router_probs.shape[-1]
                self.results.num_experts = num_experts

            expert_counts = {}
            for expert_idx in range(num_experts):
                expert_counts[expert_idx] = int(np.sum(selected_np == expert_idx))
            self.results.moe_expert_load[layer_idx] = expert_counts

            # Collect expert-to-token mappings (which tokens each expert processed)
            # Initialize if needed
            if not self.results.moe_expert_tokens:
                self.results.moe_expert_tokens = {e: [] for e in range(num_experts)}

            # Get selected experts and their probabilities for each token
            selected_2d = top_k_indices.reshape(seq_len, -1)  # (seq_len, top_k)
            probs_2d = router_probs.reshape(seq_len, -1)  # (seq_len, num_experts)

            for pos in range(seq_len):
                # Get token_id from input or generated tokens
                if pos < len(self.results.input_tokens):
                    token_id = self.results.input_tokens[pos]
                elif hasattr(self.results, 'generated_tokens') and self.results.generated_tokens:
                    gen_idx = pos - len(self.results.input_tokens)
                    token_id = self.results.generated_tokens[gen_idx] if gen_idx < len(self.results.generated_tokens) else -1
                else:
                    token_id = -1
                selected_experts = selected_2d[pos].tolist()
                expert_probs = probs_2d[pos].tolist()

                for expert_id in selected_experts:
                    if isinstance(expert_id, (list, np.ndarray)):
                        expert_id = int(expert_id[0]) if len(expert_id) > 0 else 0
                    expert_id = int(expert_id)
                    prob = float(expert_probs[expert_id]) if expert_id < len(expert_probs) else 0.0
                    if expert_id in self.results.moe_expert_tokens:
                        self.results.moe_expert_tokens[expert_id].append((token_id, pos, layer_idx, prob))

        except Exception as e:
            # MoE capture failed - not critical
            pass

    def _capture_attention_patterns(self, layer_idx: int, layer, hidden_state: mx.array):
        """Capture attention patterns for a layer by computing Q @ K^T."""
        try:
            # Find the attention module
            attn = None
            for name in ['self_attn', 'attention', 'attn']:
                if hasattr(layer, name):
                    attn = getattr(layer, name)
                    break

            if attn is None:
                return

            # Get Q, K projections
            q_proj = getattr(attn, 'q_proj', None)
            k_proj = getattr(attn, 'k_proj', None)

            if q_proj is None or k_proj is None:
                return

            # Get attention config - check multiple attribute names used by different model architectures
            n_heads = (getattr(attn, 'n_heads', None) or
                       getattr(attn, 'num_heads', None) or
                       getattr(attn, 'num_attention_heads', None) or 8)
            n_kv_heads = (getattr(attn, 'n_kv_heads', None) or
                          getattr(attn, 'num_kv_heads', None) or
                          getattr(attn, 'num_key_value_heads', None) or n_heads)
            scale = (getattr(attn, 'scale', None) or
                     getattr(attn, 'sm_scale', None))
            head_dim_attr = getattr(attn, 'head_dim', None)

            # Store head counts in results
            if self.results.n_attention_heads == 0:
                self.results.n_attention_heads = n_heads
                self.results.n_kv_heads = n_kv_heads

            # Apply input layernorm if present (attention expects normalized input)
            h = hidden_state
            if hasattr(layer, 'input_layernorm'):
                h = layer.input_layernorm(h)
            elif hasattr(layer, 'ln_1'):
                h = layer.ln_1(h)

            # Compute Q and K
            batch_size, seq_len, hidden_dim = h.shape
            q = q_proj(h)  # (batch, seq, n_heads * head_dim)
            k = k_proj(h)  # (batch, seq, n_kv_heads * head_dim)
            mx.eval(q)
            mx.eval(k)

            # Compute head_dim (use attribute if available, otherwise calculate)
            head_dim = head_dim_attr if head_dim_attr else q.shape[-1] // n_heads

            # Reshape for multi-head attention
            # Q: (batch, seq, n_heads, head_dim) -> (batch, n_heads, seq, head_dim)
            q = q.reshape(batch_size, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
            # K: (batch, seq, n_kv_heads, head_dim) -> (batch, n_kv_heads, seq, head_dim)
            k = k.reshape(batch_size, seq_len, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

            # Handle GQA (Grouped Query Attention) - repeat K for each group
            if n_kv_heads < n_heads:
                n_rep = n_heads // n_kv_heads
                k = mx.repeat(k, n_rep, axis=1)  # (batch, n_heads, seq, head_dim)

            # Compute attention scores: Q @ K^T / sqrt(d_k)
            if scale is None:
                scale = 1.0 / (head_dim ** 0.5)

            # (batch, n_heads, seq, head_dim) @ (batch, n_heads, head_dim, seq) -> (batch, n_heads, seq, seq)
            attn_scores = (q @ k.transpose(0, 1, 3, 2)) * scale
            mx.eval(attn_scores)

            # Apply causal mask
            mask = mx.triu(mx.full((seq_len, seq_len), float('-inf')), k=1)
            attn_scores = attn_scores + mask

            # Softmax to get attention weights
            attn_weights = mx.softmax(attn_scores, axis=-1)
            mx.eval(attn_weights)

            # Store attention patterns (average across batch, keep heads)
            # Shape: (n_heads, seq_len, seq_len)
            attn_np = self._to_numpy(attn_weights[0], None)  # Take first batch item
            self.results.attention_patterns[layer_idx] = attn_np

        except Exception as e:
            # Attention capture failed - store error for debugging
            if not hasattr(self.results, '_attention_error'):
                self.results._attention_error = f"Layer {layer_idx}: {type(e).__name__}: {e}"
            import sys
            print(f"Attention capture failed for layer {layer_idx}: {type(e).__name__}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    def _capture_activations(self, x: mx.array):
        """Run forward pass and capture activations."""
        max_pos = self.config.max_sequence_positions

        def compute_norm(arr: np.ndarray) -> float:
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
            h = x
            prev_h = None

        # 2. Transformer layers
        for i, layer in enumerate(self.layers):
            # Save pre-layer hidden state for attention capture
            h_pre_layer = h

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

            if isinstance(h, tuple):
                h = h[0]

            if self.config.capture_layer_outputs and self._should_capture_layer(i):
                self.results.layer_outputs[i] = self._to_numpy(h, max_pos)

            if self.config.capture_ffn_activations and self._should_capture_layer(i):
                self.results.ffn_activations[i] = self._to_numpy(h, max_pos)

            if self.config.capture_residual_stream and self._should_capture_layer(i):
                h_np = self._to_numpy(h, max_pos)
                self.results.residual_stream[f"layer_{i}"] = h_np
                norm = compute_norm(h_np)
                self.results.residual_stream_norms.append((f"L{i}", norm))

                if prev_h is not None:
                    delta = compute_norm(h_np - prev_h)
                    self.results.residual_stream_deltas.append((f"L{i}", delta))
                prev_h = h_np

            # MoE router capturing
            if i in self.moe_modules and self._should_capture_layer(i):
                self._capture_moe_router(i, layer, h)

            # Attention pattern capturing (uses pre-layer hidden state)
            if self.config.capture_attention and self._should_capture_layer(i):
                self._capture_attention_patterns(i, layer, h_pre_layer)

        # 3. Output normalization
        if self.output_norm is not None:
            h = self.output_norm(h)
            mx.eval(h)

        # 4. Get logits
        if self.config.capture_logits:
            if hasattr(self.embedding, 'as_linear'):
                logits = self.embedding.as_linear(h)
            elif hasattr(self.model, 'lm_head'):
                logits = self.model.lm_head(h)
            elif hasattr(self.inner_model, 'lm_head'):
                logits = self.inner_model.lm_head(h)
            else:
                logits = h

            mx.eval(logits)
            logits_slice = logits[0, -1, :].astype(mx.float32)
            mx.eval(logits_slice)
            try:
                self.results.logits = np.array(logits_slice)
            except (RuntimeError, TypeError, ValueError):
                self.results.logits = np.array(logits_slice.tolist(), dtype=np.float32)

            # Compute probabilities
            if self.config.capture_token_probs:
                max_logit = self.results.logits.max()
                exp_logits = np.exp(self.results.logits - max_logit)
                self.results.token_probs = exp_logits / exp_logits.sum()

            # Get top-k tokens with text
            top_k = 20
            top_indices = np.argsort(self.results.logits)[-top_k:][::-1]
            self.results.top_k_tokens = []
            for idx in top_indices:
                idx_int = int(idx)
                logit_val = float(self.results.logits[idx])
                prob = float(self.results.token_probs[idx]) if self.results.token_probs is not None else 0.0
                text = self.decode_token(idx_int)
                # Handle empty/whitespace tokens
                if not text or text.isspace():
                    text = f"<{idx_int}>"
                self.results.top_k_tokens.append((idx_int, prob, text))

    def probe(self, prompt: str, max_tokens: int = 1) -> ProbeResults:
        """Run probing on a prompt."""
        self.reset_results()

        if hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(prompt)
        else:
            tokens = self.tokenizer(prompt)

        if isinstance(tokens, dict):
            tokens = tokens['input_ids']

        self.results.input_tokens = list(tokens)
        self.results.input_text = prompt

        x = mx.array([tokens])
        self._capture_activations(x)

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

    def _safe_decode(self, tokens: List[int]) -> str:
        """Safely decode a list of tokens to text."""
        try:
            if hasattr(self.tokenizer, 'decode'):
                return self.tokenizer.decode(tokens)
            else:
                return "".join(self.decode_token(t) for t in tokens)
        except:
            return f"[{len(tokens)} tokens]"

    def probe_with_generation(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.0
    ) -> ProbeResults:
        """
        Run inference with generation, capturing per-token alternatives,
        then probe layer activations.

        This mirrors how AFM7 does it:
        1. Manual token-by-token generation with alternatives capture
        2. Separate probing pass for layer activations
        """
        self.reset_results()

        # Tokenize input
        if hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(prompt)
        else:
            tokens = self.tokenizer(prompt)

        if isinstance(tokens, dict):
            tokens = tokens['input_ids']

        self.results.input_tokens = list(tokens)
        self.results.input_text = prompt

        # Manual generation with per-token alternatives capture
        try:
            x = mx.array([tokens])

            # Initial forward pass
            logits = self.model(x)
            mx.eval(logits)

            # Get vocab size from logits
            vocab_size = logits.shape[-1]

            # Capture initial top-k predictions (for the first generated token)
            initial_logits = logits[0, -1, :]
            probs = mx.softmax(initial_logits, axis=-1)
            mx.eval(probs)

            # Get top-k indices - convert to numpy for reliable indexing
            top_indices = mx.argsort(probs)[-20:][::-1]
            mx.eval(top_indices)

            # Convert probs to numpy for reliable indexing
            probs_np = np.array(probs.astype(mx.float32).tolist(), dtype=np.float32)

            self.results.top_k_tokens = []
            top_indices_list = top_indices.tolist()
            for idx in top_indices_list:
                idx_int = int(idx)
                if 0 <= idx_int < vocab_size:
                    prob_val = float(probs_np[idx_int])
                    tok_text = self.decode_token(idx_int)
                    # Handle empty/whitespace tokens
                    if not tok_text or tok_text.isspace():
                        tok_text = f"<{idx_int}>"
                    self.results.top_k_tokens.append((idx_int, prob_val, tok_text))

            # Store logits and probs - ensure float32 for numpy compatibility
            initial_logits_f32 = initial_logits.astype(mx.float32)
            probs_f32 = probs.astype(mx.float32)
            mx.eval(initial_logits_f32, probs_f32)
            self.results.logits = np.array(initial_logits_f32.tolist(), dtype=np.float32)
            self.results.token_probs = np.array(probs_f32.tolist(), dtype=np.float32)

            # Generate tokens step by step
            generated = []
            per_token_alts = []

            # Install MoE router hooks to capture during generation
            if self.results.is_moe:
                self._install_router_hooks()

            # Create cache for efficient generation
            cache = None
            if hasattr(self.model, 'make_cache'):
                cache = self.model.make_cache()
                # Re-run with cache
                logits = self.model(x, cache=cache)
                mx.eval(logits)

            for gen_step in range(max_tokens):
                # Get probabilities at current position
                step_logits = logits[0, -1, :]
                step_probs = mx.softmax(step_logits, axis=-1)
                mx.eval(step_probs)

                # Capture top-10 alternatives at this position
                top_idx = mx.argsort(step_probs)[-10:][::-1]
                mx.eval(top_idx)

                alternatives = []
                for idx in top_idx:
                    idx_int = int(idx)
                    if 0 <= idx_int < vocab_size:
                        prob_val = float(step_probs[idx_int])
                        tok_text = self.decode_token(idx_int)
                        alternatives.append((idx_int, prob_val, tok_text))

                per_token_alts.append(alternatives)

                # Sample next token
                if temperature == 0:
                    next_token = mx.argmax(step_logits)
                else:
                    scaled_logits = step_logits / temperature
                    next_token = mx.random.categorical(scaled_logits.reshape(1, -1))[0]

                mx.eval(next_token)
                next_token_int = int(next_token)

                # Validate token
                if next_token_int < 0 or next_token_int >= vocab_size:
                    break

                generated.append(next_token_int)

                # Check for EOS tokens
                eos_tokens = []
                if hasattr(self.tokenizer, 'eos_token_id'):
                    eos_id = self.tokenizer.eos_token_id
                    if isinstance(eos_id, int):
                        eos_tokens.append(eos_id)
                    elif isinstance(eos_id, list):
                        eos_tokens.extend(eos_id)

                if next_token_int in eos_tokens:
                    break

                # Check for reasoning model stop conditions (extensible for different architectures)
                current_text = self._safe_decode(generated)
                reasoning_fmt = detect_reasoning_format(current_text)
                if reasoning_fmt and reasoning_fmt.should_stop(current_text):
                    break

                # Repetition detection - break if last N tokens are repeating
                if len(generated) >= 10:
                    last_5 = generated[-5:]
                    prev_5 = generated[-10:-5]
                    if last_5 == prev_5:
                        break  # Stuck in a loop

                # Next forward pass
                next_x = next_token.reshape(1, 1)
                if cache is not None:
                    logits = self.model(next_x, cache=cache)
                else:
                    # Append to sequence and re-run (slower but works without cache)
                    x = mx.concatenate([x, next_x], axis=1)
                    logits = self.model(x)
                mx.eval(logits)

                # Collect MoE router captures for this generation step
                if self.results.is_moe:
                    self._collect_router_captures(gen_step)

            # Remove MoE router hooks
            if self.results.is_moe:
                self._remove_router_hooks()

            # Store generation results
            self.results.generated_tokens = generated
            self.results.generated_text = self._safe_decode(generated)
            self.results.per_token_alternatives = per_token_alts

            # Parse reasoning vs answer
            reasoning, answer = parse_reasoning_output(self.results.generated_text)
            self.results.reasoning_text = reasoning
            self.results.answer_text = answer if answer else self.results.generated_text

        except Exception as e:
            import traceback
            # Clean up hooks on error
            if self.results.is_moe:
                self._remove_router_hooks()
            self.results.generated_text = f"(Generation error: {str(e)})"
            self.results.reasoning_text = ""
            self.results.answer_text = ""
            self.results.generated_tokens = []

        # Now capture layer activations on the FULL sequence (input + generated)
        # This captures MoE router outputs for ALL positions including generated tokens
        # But preserve the logits/probs captured during generation (they're meaningful)
        try:
            # Save generation-time logits (these are for predicting the first generated token)
            saved_logits = self.results.logits
            saved_token_probs = self.results.token_probs
            saved_top_k_tokens = self.results.top_k_tokens

            full_tokens = list(tokens) + self.results.generated_tokens
            x = mx.array([full_tokens])
            self._capture_activations(x)

            # Restore generation-time logits (not the "what comes after everything" logits)
            self.results.logits = saved_logits
            self.results.token_probs = saved_token_probs
            self.results.top_k_tokens = saved_top_k_tokens
        except Exception as e:
            pass  # Probing failed but generation may have succeeded

        return self.results

    def probe_with_generation_replay(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.0,
        capture_config: Optional["ReplayCaptureConfig"] = None
    ) -> ProbeResults:
        """
        Run inference with generation, capturing per-step states for replay.

        This method extends probe_with_generation to store detailed state
        at each generation step, enabling timeline replay and branching.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            capture_config: Configuration for what to capture

        Returns:
            ProbeResults with generation_timeline populated
        """
        import uuid

        capture_config = capture_config or ReplayCaptureConfig()
        self.reset_results()

        # Tokenize input
        if hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(prompt)
        else:
            tokens = self.tokenizer(prompt)

        if isinstance(tokens, dict):
            tokens = tokens['input_ids']

        self.results.input_tokens = list(tokens)
        self.results.input_text = prompt

        # Initialize timeline
        timeline = GenerationTimeline(
            timeline_id=str(uuid.uuid4())[:8],
            input_tokens=list(tokens),
            input_text=prompt
        )

        try:
            x = mx.array([tokens])

            # Initial forward pass
            logits = self.model(x)
            mx.eval(logits)
            vocab_size = logits.shape[-1]

            # Create cache for efficient generation
            cache = None
            if hasattr(self.model, 'make_cache'):
                cache = self.model.make_cache()
                logits = self.model(x, cache=cache)
                mx.eval(logits)

            # Capture initial top-k predictions
            initial_logits = logits[0, -1, :]
            probs = mx.softmax(initial_logits, axis=-1)
            mx.eval(probs)
            probs_np = np.array(probs.astype(mx.float32).tolist(), dtype=np.float32)

            # Store logits for results
            self.results.logits = np.array(initial_logits.astype(mx.float32).tolist(), dtype=np.float32)
            self.results.token_probs = probs_np

            # Get top-k for results
            top_indices = mx.argsort(probs)[-20:][::-1]
            mx.eval(top_indices)
            self.results.top_k_tokens = []
            for idx in top_indices.tolist():
                idx_int = int(idx)
                if 0 <= idx_int < vocab_size:
                    prob_val = float(probs_np[idx_int])
                    tok_text = self.decode_token(idx_int)
                    if not tok_text or tok_text.isspace():
                        tok_text = f"<{idx_int}>"
                    self.results.top_k_tokens.append((idx_int, prob_val, tok_text))

            # Generate tokens step by step with state capture
            generated = []
            per_token_alts = []
            current_tokens = list(tokens)

            for gen_step in range(max_tokens):
                # Get probabilities at current position
                step_logits = logits[0, -1, :]
                step_probs = mx.softmax(step_logits, axis=-1)
                mx.eval(step_probs)
                step_probs_np = np.array(step_probs.astype(mx.float32).tolist(), dtype=np.float32)

                # Capture top-k alternatives
                top_k_count = capture_config.top_k_alternatives
                top_idx = mx.argsort(step_probs)[-top_k_count:][::-1]
                mx.eval(top_idx)

                alternatives = []
                for idx in top_idx.tolist():
                    idx_int = int(idx)
                    if 0 <= idx_int < vocab_size:
                        prob_val = float(step_probs_np[idx_int])
                        tok_text = self.decode_token(idx_int)
                        alternatives.append((idx_int, prob_val, tok_text))

                per_token_alts.append(alternatives)

                # Sample next token
                if temperature == 0:
                    next_token = mx.argmax(step_logits)
                else:
                    scaled_logits = step_logits / temperature
                    next_token = mx.random.categorical(scaled_logits.reshape(1, -1))[0]

                mx.eval(next_token)
                next_token_int = int(next_token)

                if next_token_int < 0 or next_token_int >= vocab_size:
                    break

                generated.append(next_token_int)
                current_tokens.append(next_token_int)

                # Create step state
                step_state = GenerationStepState(
                    step_index=gen_step,
                    token_id=next_token_int,
                    token_text=self.decode_token(next_token_int),
                    top_k_alternatives=alternatives,
                    cumulative_tokens=list(current_tokens),
                    probability=float(step_probs_np[next_token_int])
                )

                # Capture layer outputs if in standard/full mode
                # NOTE: Currently self.results.layer_outputs is empty during generation
                # because _capture_activations runs after the loop. Per-step layer outputs
                # would require capturing during each forward pass, which adds overhead.
                # TODO: Consider post-filling step layer_outputs using position mapping
                # after _capture_activations completes.
                if capture_config.capture_layer_outputs and capture_config.mode in ['standard', 'full']:
                    # Sample layers based on config
                    layer_outputs = {}
                    for layer_idx in range(0, len(self.layers), capture_config.layer_sample_rate):
                        if layer_idx in self.results.layer_outputs:
                            # Take last position only to save memory
                            layer_outputs[layer_idx] = self.results.layer_outputs[layer_idx][..., -1:, :]
                    step_state.layer_outputs = layer_outputs if layer_outputs else None

                # Capture logits if in full mode
                if capture_config.capture_logits and capture_config.mode == 'full':
                    step_state.logits = step_probs_np.copy()

                timeline.steps.append(step_state)

                # Check for EOS tokens
                eos_tokens = []
                if hasattr(self.tokenizer, 'eos_token_id'):
                    eos_id = self.tokenizer.eos_token_id
                    if isinstance(eos_id, int):
                        eos_tokens.append(eos_id)
                    elif isinstance(eos_id, list):
                        eos_tokens.extend(eos_id)

                if next_token_int in eos_tokens:
                    break

                # Check for reasoning model stop conditions
                current_text = self._safe_decode(generated)
                reasoning_fmt = detect_reasoning_format(current_text)
                if reasoning_fmt and reasoning_fmt.should_stop(current_text):
                    break

                # Repetition detection
                if len(generated) >= 10:
                    last_5 = generated[-5:]
                    prev_5 = generated[-10:-5]
                    if last_5 == prev_5:
                        break

                # Next forward pass
                next_x = next_token.reshape(1, 1)
                if cache is not None:
                    logits = self.model(next_x, cache=cache)
                else:
                    x = mx.concatenate([x, next_x], axis=1)
                    logits = self.model(x)
                mx.eval(logits)

            # Store generation results
            self.results.generated_tokens = generated
            self.results.generated_text = self._safe_decode(generated)
            self.results.per_token_alternatives = per_token_alts
            self.results.generation_timeline = timeline

            # Parse reasoning vs answer
            reasoning, answer = parse_reasoning_output(self.results.generated_text)
            self.results.reasoning_text = reasoning
            self.results.answer_text = answer if answer else self.results.generated_text

        except Exception as e:
            import traceback
            self.results.generated_text = f"(Generation error: {str(e)})"
            self.results.reasoning_text = ""
            self.results.answer_text = ""
            self.results.generated_tokens = []

        # Capture layer activations on full sequence
        try:
            saved_logits = self.results.logits
            saved_token_probs = self.results.token_probs
            saved_top_k_tokens = self.results.top_k_tokens
            saved_timeline = self.results.generation_timeline

            full_tokens = list(tokens) + self.results.generated_tokens
            x = mx.array([full_tokens])
            self._capture_activations(x)

            self.results.logits = saved_logits
            self.results.token_probs = saved_token_probs
            self.results.top_k_tokens = saved_top_k_tokens
            self.results.generation_timeline = saved_timeline
        except:
            pass

        return self.results

    def generate_from_branch(
        self,
        timeline: "GenerationTimeline",
        branch_step: int,
        alternative_token_id: int,
        max_tokens: int = 50,
        temperature: float = 0.0,
        capture_config: Optional["ReplayCaptureConfig"] = None
    ) -> "GenerationTimeline":
        """
        Generate a new timeline branch from a specific step with an alternative token.

        Args:
            timeline: The original timeline to branch from
            branch_step: Step index where to branch
            alternative_token_id: Alternative token to use instead of original
            max_tokens: Maximum additional tokens to generate
            temperature: Sampling temperature
            capture_config: Capture configuration

        Returns:
            New GenerationTimeline representing the branch
        """
        import uuid

        capture_config = capture_config or ReplayCaptureConfig()

        # Build tokens up to branch point, then substitute
        tokens_to_branch = timeline.input_tokens.copy()
        for i, step in enumerate(timeline.steps):
            if i < branch_step:
                tokens_to_branch.append(step.token_id)
            elif i == branch_step:
                tokens_to_branch.append(alternative_token_id)
                break

        # Create new timeline
        new_timeline = GenerationTimeline(
            timeline_id=str(uuid.uuid4())[:8],
            parent_timeline_id=timeline.timeline_id,
            branch_point_step=branch_step,
            input_tokens=timeline.input_tokens.copy(),
            input_text=timeline.input_text
        )

        # Copy steps up to branch point (shallow copy to avoid shared mutable state)
        import copy
        for i, step in enumerate(timeline.steps):
            if i < branch_step:
                new_timeline.steps.append(copy.copy(step))

        # Add the branch step with alternative token
        branch_step_state = GenerationStepState(
            step_index=branch_step,
            token_id=alternative_token_id,
            token_text=self.decode_token(alternative_token_id),
            cumulative_tokens=tokens_to_branch.copy()
        )
        new_timeline.steps.append(branch_step_state)

        # Store debug info in the timeline for UI display
        new_timeline.debug_info = []
        new_timeline.debug_info.append(f"Branch step: {branch_step}")
        new_timeline.debug_info.append(f"Original token: '{timeline.steps[branch_step].token_text}' (ID: {timeline.steps[branch_step].token_id})")
        new_timeline.debug_info.append(f"Alternative token: '{self.decode_token(alternative_token_id)}' (ID: {alternative_token_id})")
        new_timeline.debug_info.append(f"Full context length: {len(tokens_to_branch)} tokens")

        # Verify the alternative token is in the context
        if tokens_to_branch[-1] == alternative_token_id:
            new_timeline.debug_info.append("✓ Alternative token correctly placed at end of context")
        else:
            new_timeline.debug_info.append(f"⚠️ Context ends with token ID {tokens_to_branch[-1]}, not {alternative_token_id}")

        # Show last 5 tokens of context for debugging
        last_tokens = tokens_to_branch[-5:] if len(tokens_to_branch) >= 5 else tokens_to_branch
        last_tokens_text = [self.decode_token(t) for t in last_tokens]
        new_timeline.debug_info.append(f"Last 5 context tokens: {last_tokens_text}")

        # Show what the original had at this position vs what we're using
        if branch_step < len(timeline.steps):
            orig_tok = timeline.steps[branch_step].token_id
            new_timeline.debug_info.append(f"Original context had: '{self.decode_token(orig_tok)}' (ID: {orig_tok})")
            new_timeline.debug_info.append(f"Branch context now has: '{self.decode_token(alternative_token_id)}' (ID: {alternative_token_id})")

        # Generate continuation - treat as FRESH inference with full context
        try:
            # Create fresh array - this is a NEW conversation context
            x = mx.array([tokens_to_branch])

            # Fresh forward pass on complete context (input + generated up to branch + alternative)
            # No cache reuse - clean slate
            logits = self.model(x)
            mx.eval(logits)
            vocab_size = logits.shape[-1]

            # Create NEW cache for this branch's generation
            cache = None
            if hasattr(self.model, 'make_cache'):
                cache = self.model.make_cache()
                # Populate fresh cache with the branched context
                logits = self.model(x, cache=cache)
                mx.eval(logits)

            # Debug: Show what model predicts for step N+1 (given injected token at step N)
            first_pred = int(mx.argmax(logits[0, -1, :]))
            first_pred_text = self.decode_token(first_pred)
            new_timeline.debug_info.append(f"Step {branch_step}: INJECTED '{self.decode_token(alternative_token_id)}' (replacing '{timeline.steps[branch_step].token_text}')")
            new_timeline.debug_info.append(f"Step {branch_step + 1}: Model predicts '{first_pred_text}' (ID: {first_pred})")

            # Compare with what original timeline had at step N+1
            if branch_step + 1 < len(timeline.steps):
                orig_next = timeline.steps[branch_step + 1]
                new_timeline.debug_info.append(f"Step {branch_step + 1} in original was: '{orig_next.token_text}' (ID: {orig_next.token_id})")
                if first_pred == orig_next.token_id:
                    new_timeline.debug_info.append("⚠️ Despite different input, model predicts SAME continuation")
                else:
                    new_timeline.debug_info.append("✓ Different input leads to DIFFERENT continuation")

            current_tokens = tokens_to_branch.copy()

            for gen_step in range(max_tokens):
                step_logits = logits[0, -1, :]
                step_probs = mx.softmax(step_logits, axis=-1)
                mx.eval(step_probs)
                step_probs_np = np.array(step_probs.astype(mx.float32).tolist(), dtype=np.float32)

                # Get alternatives
                top_idx = mx.argsort(step_probs)[-capture_config.top_k_alternatives:][::-1]
                mx.eval(top_idx)
                alternatives = []
                for idx in top_idx.tolist():
                    idx_int = int(idx)
                    if 0 <= idx_int < vocab_size:
                        alternatives.append((idx_int, float(step_probs_np[idx_int]), self.decode_token(idx_int)))

                # Sample next token
                if temperature == 0:
                    next_token = mx.argmax(step_logits)
                else:
                    scaled_logits = step_logits / temperature
                    next_token = mx.random.categorical(scaled_logits.reshape(1, -1))[0]

                mx.eval(next_token)
                next_token_int = int(next_token)

                if next_token_int < 0 or next_token_int >= vocab_size:
                    break

                current_tokens.append(next_token_int)

                step_state = GenerationStepState(
                    step_index=branch_step + 1 + gen_step,
                    token_id=next_token_int,
                    token_text=self.decode_token(next_token_int),
                    top_k_alternatives=alternatives,
                    cumulative_tokens=current_tokens.copy(),
                    probability=float(step_probs_np[next_token_int])
                )
                new_timeline.steps.append(step_state)

                # Check for EOS
                if hasattr(self.tokenizer, 'eos_token_id'):
                    eos_id = self.tokenizer.eos_token_id
                    if isinstance(eos_id, int) and next_token_int == eos_id:
                        break
                    elif isinstance(eos_id, list) and next_token_int in eos_id:
                        break

                # Next pass
                next_x = next_token.reshape(1, 1)
                if cache is not None:
                    logits = self.model(next_x, cache=cache)
                else:
                    x = mx.concatenate([x, next_x], axis=1)
                    logits = self.model(x)
                mx.eval(logits)

        except Exception as e:
            new_timeline.debug_info.append(f"❌ ERROR during generation: {str(e)}")
            import traceback
            new_timeline.debug_info.append(traceback.format_exc())

        # Add summary
        gen_count = len(new_timeline.steps) - branch_step - 1
        new_timeline.debug_info.append(f"Generated {gen_count} continuation tokens")

        return new_timeline


# =============================================================================
# Causal Tracer (Knowledge Localization)
# =============================================================================

class CausalTracer:
    """
    Implements causal tracing for knowledge localization (ROME-style).

    Given a factual query like "The capital of France is [MASK]", identifies
    which layers and positions store that knowledge by:
    1. Clean run: Get P(target) with no corruption
    2. Corrupted run: Add noise to subject tokens, P(target) drops
    3. Restore run: For each (layer, position), restore clean state and measure recovery

    Reference: https://rome.baulab.info/
    """

    def __init__(self, model, tokenizer, topology: Optional[Dict] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.topology = topology or {}

        # Get inner model
        if hasattr(model, 'model'):
            self.inner_model = model.model
        else:
            self.inner_model = model

        # Get layers
        if hasattr(self.inner_model, 'layers'):
            self.layers = list(self.inner_model.layers)
        else:
            self.layers = []

        # Get embedding
        if hasattr(self.inner_model, 'embed_tokens'):
            self.embedding = self.inner_model.embed_tokens
        elif hasattr(self.inner_model, 'wte'):
            self.embedding = self.inner_model.wte
        else:
            self.embedding = None

        # Get output norm
        if hasattr(self.inner_model, 'norm'):
            self.output_norm = self.inner_model.norm
        elif hasattr(self.inner_model, 'ln_f'):
            self.output_norm = self.inner_model.ln_f
        else:
            self.output_norm = None

        self.num_layers = len(self.layers)

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text."""
        if hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(text)
        else:
            tokens = self.tokenizer(text)
        if isinstance(tokens, dict):
            tokens = tokens['input_ids']
        return list(tokens)

    def _decode(self, token_id: int) -> str:
        """Decode single token."""
        try:
            if hasattr(self.tokenizer, 'decode'):
                return self.tokenizer.decode([token_id])
            return f"[{token_id}]"
        except:
            return f"[{token_id}]"

    def _get_embedding_std(self) -> float:
        """Get standard deviation of embeddings for noise calibration."""
        if self.embedding is None:
            return 1.0
        try:
            weight = self.embedding.weight
            mx.eval(weight)
            # Sample a subset of embeddings for efficiency
            sample_indices = mx.arange(0, min(1000, weight.shape[0]))
            sample = weight[sample_indices]
            std = float(mx.std(sample.astype(mx.float32)))
            return std if std > 0 else 1.0
        except:
            return 1.0

    def _forward_with_intervention(
        self,
        clean_embeddings: mx.array,
        corrupted_embeddings: mx.array,
        restore_layer: Optional[int] = None,
        restore_positions: Optional[List[int]] = None
    ) -> mx.array:
        """
        Run forward pass with optional intervention.

        If restore_layer/positions specified, start with corrupted embeddings
        but restore clean activations at specified layer and positions.

        Note: This implementation recomputes clean activations up to restore_layer
        on each call, making actual cost O(L^2 * positions) rather than O(L * positions).
        A future optimization could cache per-layer activations to avoid recomputation.
        """
        # Start with corrupted or clean embeddings
        if restore_layer is not None:
            h = corrupted_embeddings
        else:
            h = clean_embeddings

        # Process through layers
        for i, layer in enumerate(self.layers):
            # At the restore layer, blend in clean activations for specific positions
            if restore_layer is not None and i == restore_layer and restore_positions:
                # We need the clean activation at this layer
                # Run clean path up to this layer
                h_clean = clean_embeddings
                for j in range(i):
                    try:
                        h_clean = self.layers[j](h_clean, mask=None, cache=None)
                    except TypeError:
                        try:
                            h_clean = self.layers[j](h_clean)
                        except:
                            pass
                    if isinstance(h_clean, tuple):
                        h_clean = h_clean[0]
                mx.eval(h_clean)

                # Restore clean activations at specified positions using mask
                # More efficient than repeated concatenation
                seq_len = h.shape[1]
                mask = mx.zeros((1, seq_len, 1))
                for pos in restore_positions:
                    if pos < seq_len:
                        mask = mask.at[:, pos:pos+1, :].add(mx.ones((1, 1, 1)))
                mx.eval(mask)
                # Blend: h = h * (1 - mask) + h_clean * mask
                h = h * (1 - mask) + h_clean * mask
                mx.eval(h)

            # Forward through layer
            try:
                h = layer(h, mask=None, cache=None)
            except TypeError:
                try:
                    h = layer(h)
                except:
                    continue

            if isinstance(h, tuple):
                h = h[0]
            mx.eval(h)

        # Output normalization
        if self.output_norm is not None:
            h = self.output_norm(h)
            mx.eval(h)

        # Get logits
        if hasattr(self.embedding, 'as_linear'):
            logits = self.embedding.as_linear(h)
        elif hasattr(self.model, 'lm_head'):
            logits = self.model.lm_head(h)
        else:
            logits = h

        mx.eval(logits)
        return logits

    def _get_target_prob(self, logits: mx.array, target_token_id: int, position: int = -1) -> float:
        """Get probability of target token at specified position."""
        # Get logits at position (default: last)
        if position == -1:
            step_logits = logits[0, -1, :]
        else:
            step_logits = logits[0, position, :]

        # Softmax to get probabilities
        probs = mx.softmax(step_logits, axis=-1)
        mx.eval(probs)

        # Get target probability
        target_prob = float(probs[target_token_id])
        return target_prob

    def detect_subject_positions(self, text: str, subject: str) -> List[int]:
        """Auto-detect subject token positions in the text."""
        # Tokenize full text and subject
        full_tokens = self._tokenize(text)
        subject_tokens = self._tokenize(subject)

        # Find subject tokens in full sequence
        positions = []
        for i in range(len(full_tokens) - len(subject_tokens) + 1):
            if full_tokens[i:i+len(subject_tokens)] == subject_tokens:
                positions.extend(range(i, i + len(subject_tokens)))
                break

        # If no exact match, try to find partial overlap
        if not positions:
            # Look for individual subject tokens
            for i, tok in enumerate(full_tokens):
                tok_text = self._decode(tok).lower().strip()
                if subject.lower() in tok_text or tok_text in subject.lower():
                    positions.append(i)

        return positions

    def run_causal_trace(
        self,
        query_text: str,
        subject: str,
        target_token: Optional[str] = None,
        config: Optional[CausalTraceConfig] = None,
        progress_callback: Optional[callable] = None
    ) -> CausalTraceResults:
        """
        Run causal tracing experiment.

        Args:
            query_text: The factual query (e.g., "The capital of France is")
            subject: The subject to corrupt (e.g., "France")
            target_token: Expected answer token (e.g., "Paris"), auto-detect if None
            config: Tracing configuration
            progress_callback: Optional callback(current, total) for progress updates
        """
        config = config or CausalTraceConfig()
        results = CausalTraceResults()
        results.query_text = query_text
        results.subject_text = subject

        # Tokenize
        tokens = self._tokenize(query_text)
        x = mx.array([tokens])

        # Detect subject positions
        if config.subject_positions:
            subject_positions = config.subject_positions
        else:
            subject_positions = self.detect_subject_positions(query_text, subject)

        if not subject_positions:
            # Fallback: corrupt middle positions
            mid = len(tokens) // 2
            subject_positions = list(range(max(0, mid-2), min(len(tokens), mid+2)))

        results.subject_positions = subject_positions

        # Get clean embeddings
        if self.embedding is None:
            return results

        clean_embeddings = self.embedding(x)
        mx.eval(clean_embeddings)

        # Create corrupted embeddings (add noise to subject positions)
        embedding_std = self._get_embedding_std()
        noise_scale = config.noise_multiplier * embedding_std

        # Generate noise for subject positions only (more efficient than full tensor)
        noise_shape = clean_embeddings.shape
        noise = mx.random.normal(noise_shape) * noise_scale
        mx.eval(noise)

        # Create a mask for subject positions to apply noise efficiently
        # Instead of repeated concatenation, use a single masked addition
        corrupted_embeddings = clean_embeddings.astype(mx.float32)
        seq_len = corrupted_embeddings.shape[1]

        # Create mask: 1 for subject positions, 0 elsewhere
        mask = mx.zeros((1, seq_len, 1))
        for pos in subject_positions:
            if pos < seq_len:
                # Build mask with 1s at subject positions
                mask = mask.at[:, pos:pos+1, :].add(mx.ones((1, 1, 1)))
        mx.eval(mask)

        # Apply noise only where mask is 1
        corrupted_embeddings = corrupted_embeddings + noise * mask
        mx.eval(corrupted_embeddings)

        # Step 1: Clean run - get P(target)
        clean_logits = self._forward_with_intervention(
            clean_embeddings, corrupted_embeddings,
            restore_layer=None, restore_positions=None
        )

        # Determine target token
        if config.target_token_id is not None:
            target_token_id = config.target_token_id
        elif target_token:
            # Try both with and without leading space - use whichever has higher prob
            target_tokens_no_space = self._tokenize(target_token)
            target_tokens_with_space = self._tokenize(" " + target_token)

            candidates = []
            if target_tokens_no_space:
                tid = target_tokens_no_space[0]
                prob = self._get_target_prob(clean_logits, tid)
                candidates.append((tid, prob))
            if target_tokens_with_space:
                tid = target_tokens_with_space[0]
                prob = self._get_target_prob(clean_logits, tid)
                candidates.append((tid, prob))

            # Pick the one with higher probability
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                target_token_id = candidates[0][0]
            else:
                target_token_id = 0
        else:
            # Auto-detect: use argmax of clean logits
            target_token_id = int(mx.argmax(clean_logits[0, -1, :]))

        results.target_token_id = target_token_id
        results.target_token = self._decode(target_token_id)
        results.clean_prob = self._get_target_prob(clean_logits, target_token_id)

        # Step 2: Corrupted run - P(target) should drop
        corrupted_logits = self._forward_with_intervention(
            corrupted_embeddings, corrupted_embeddings,
            restore_layer=None, restore_positions=None
        )
        results.corrupted_prob = self._get_target_prob(corrupted_logits, target_token_id)

        # Step 3: Restore runs - for each (layer, position), restore and measure recovery
        seq_len = len(tokens)

        # Determine which layers and positions to test
        if config.sample_layers and self.num_layers > 10:
            layers_to_test = list(range(0, self.num_layers, config.layer_sample_rate))
            if self.num_layers - 1 not in layers_to_test:
                layers_to_test.append(self.num_layers - 1)
        else:
            layers_to_test = list(range(self.num_layers))

        if config.sample_positions and seq_len > 10:
            positions_to_test = list(range(0, seq_len, config.position_sample_rate))
            # Always include subject positions
            for pos in subject_positions:
                if pos not in positions_to_test:
                    positions_to_test.append(pos)
            positions_to_test = sorted(positions_to_test)
            # Always include last position
            if seq_len - 1 not in positions_to_test:
                positions_to_test.append(seq_len - 1)
        else:
            positions_to_test = list(range(seq_len))

        results.layers_tested = layers_to_test
        results.positions_tested = positions_to_test

        # Initialize indirect effects matrix
        indirect_effects = np.zeros((len(layers_to_test), len(positions_to_test)))

        total_passes = len(layers_to_test) * len(positions_to_test)
        pass_count = 0

        for li, layer_idx in enumerate(layers_to_test):
            for pi, pos in enumerate(positions_to_test):
                # Restore clean activation at this (layer, position)
                restored_logits = self._forward_with_intervention(
                    clean_embeddings, corrupted_embeddings,
                    restore_layer=layer_idx, restore_positions=[pos]
                )
                restored_prob = self._get_target_prob(restored_logits, target_token_id)

                # Indirect effect = recovery relative to corruption damage
                # IE = (restored - corrupted) / (clean - corrupted)
                damage = results.clean_prob - results.corrupted_prob
                if abs(damage) > 1e-6:
                    recovery = restored_prob - results.corrupted_prob
                    ie = recovery / damage
                else:
                    ie = 0.0

                indirect_effects[li, pi] = ie

                pass_count += 1
                if progress_callback:
                    progress_callback(pass_count, total_passes)

        results.indirect_effects = indirect_effects
        # Note: num_forward_passes counts logical passes, but actual compute is higher
        # because _forward_with_intervention recomputes clean activations up to restore_layer.
        # Actual cost scales as O(L^2 * positions) rather than O(L * positions).
        results.num_forward_passes = 2 + total_passes  # clean + corrupted + restore runs

        # Find critical site (max recovery)
        max_idx = np.unravel_index(np.argmax(indirect_effects), indirect_effects.shape)
        results.critical_layer = layers_to_test[max_idx[0]]
        results.critical_position = positions_to_test[max_idx[1]]
        results.max_recovery = float(indirect_effects[max_idx])

        return results


# =============================================================================
# AI Interpretation Functions
# =============================================================================

def generate_ai_interpretation(
    model,
    tokenizer,
    context: str,
    question: str,
    interpreter: str = "claude",
    max_tokens: int = 200
) -> str:
    """
    Use AI to interpret probe results.

    Args:
        model: The loaded MLX model (used if interpreter="local")
        tokenizer: The tokenizer (used if interpreter="local")
        context: Data/statistics to interpret
        question: What to explain
        interpreter: "claude" for Claude Code, "local" for loaded model
        max_tokens: Max response length

    Returns:
        The AI's interpretation as a string
    """
    # Use Claude Code if selected
    if interpreter == "claude":
        return generate_claude_interpretation(context, question, max_tokens)

    # Otherwise use local model
    try:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        system_msg = "You are an expert in neural network interpretability. Provide clear, concise explanations of model behavior based on probe data."
        user_msg = f"""Based on this probe data:

{context}

{question}

Provide a concise interpretation (2-3 sentences)."""

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                prompt = f"{system_msg}\n\nUser: {user_msg}\n\nAssistant:"
        else:
            prompt = f"{system_msg}\n\nUser: {user_msg}\n\nAssistant:"

        sampler = make_sampler(temp=0.7)

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler
        )

        # Clean special tokens from reasoning models
        cleaned = clean_special_tokens(response)

        # Also extract just the answer if there's reasoning
        reasoning, answer = parse_reasoning_output(cleaned)
        if answer and not answer.startswith("(Response"):
            return answer.strip()

        return cleaned.strip()

    except Exception as e:
        return f"(AI interpretation unavailable: {str(e)[:50]})"


def get_layer_activation_interpretation(results: ProbeResults, model=None, tokenizer=None, use_ai: bool = False, interpreter: str = "claude") -> str:
    """Generate interpretation for layer activation patterns."""
    if not results.layer_outputs:
        return "No layer data available."

    norms = []
    for idx in sorted(results.layer_outputs.keys()):
        arr = results.layer_outputs[idx]
        norm = float(np.linalg.norm(arr, axis=-1).mean())
        norms.append(norm)

    if len(norms) < 2:
        return "Insufficient layer data for analysis."

    trend = "increasing" if norms[-1] > norms[0] else "decreasing"
    max_layer = np.argmax(norms)
    min_layer = np.argmin(norms)
    variance = np.var(norms)

    # Build context for AI interpretation
    layer_indices = sorted(results.layer_outputs.keys())
    norm_strs = [f"L{layer_indices[i]}: {norms[i]:.2f}" for i in range(min(len(norms), 10))]
    if len(norms) > 10:
        norm_strs.append("...")

    context = f"""Layer Activation Norms (L2):
{', '.join(norm_strs)}
Trend: {trend} (first: {norms[0]:.2f}, last: {norms[-1]:.2f})
Peak: Layer {layer_indices[max_layer]} ({norms[max_layer]:.2f})
Lowest: Layer {layer_indices[min_layer]} ({norms[min_layer]:.2f})
Variance: {variance:.2f}"""

    if use_ai:
        return generate_ai_interpretation(
            model, tokenizer, context,
            "What do these activation norm patterns tell us about how the model processes this input?",
            interpreter=interpreter
        )

    # Fallback static interpretation
    interpretation = f"""**Layer Activation Analysis:**

- **Trend:** Activation norms are {trend} through the network (first: {norms[0]:.2f}, last: {norms[-1]:.2f})
- **Peak activity:** Layer {layer_indices[max_layer]} (norm: {norms[max_layer]:.2f})
- **Lowest activity:** Layer {layer_indices[min_layer]} (norm: {norms[min_layer]:.2f})
- **Variance:** {variance:.2f} - {"high variation suggests distinct processing phases" if variance > 1 else "relatively uniform processing across layers"}

This pattern indicates the model {"builds up representations progressively" if trend == "increasing" else "compresses information as it processes"}."""

    return interpretation


def get_token_prob_interpretation(results: ProbeResults, model=None, tokenizer=None, use_ai: bool = False, interpreter: str = "claude") -> str:
    """Generate interpretation for token probability distribution."""
    if not results.top_k_tokens:
        return "No token probabilities available."

    top_token = results.top_k_tokens[0]
    top5_mass = sum(t[1] for t in results.top_k_tokens[:5])

    if results.token_probs is not None:
        entropy = float(-np.sum(results.token_probs * np.log(results.token_probs + 1e-10)))
    else:
        entropy = 0

    confidence = "very confident" if top_token[1] > 0.5 else "confident" if top_token[1] > 0.2 else "uncertain"

    # Build context for AI
    def safe_token_text(t):
        return t[2] if t[2] and not t[2].isspace() else f"<{t[0]}>"
    top_tokens_str = ", ".join([f'"{safe_token_text(t)}": {t[1]:.1%}' for t in results.top_k_tokens[:5]])
    context = f"""Token Prediction Analysis:
Input: "{results.input_text[:100]}..."
Top predictions: {top_tokens_str}
Top-5 probability mass: {top5_mass:.1%}
Entropy: {entropy:.2f} nats
Confidence level: {confidence}"""

    if use_ai:
        return generate_ai_interpretation(
            model, tokenizer, context,
            "What does this probability distribution reveal about the model's confidence and reasoning?",
            interpreter=interpreter
        )

    # Fallback static interpretation
    interpretation = f"""**Token Prediction Analysis:**

- **Top prediction:** "{top_token[2]}" with {top_token[1]:.1%} probability
- **Model confidence:** {confidence}
- **Top-5 probability mass:** {top5_mass:.1%}
- **Entropy:** {entropy:.2f} nats {"(low - focused prediction)" if entropy < 2 else "(high - multiple viable options)"}

The model is {confidence} about the next token. {"The probability is concentrated on a single prediction." if top_token[1] > 0.5 else "Multiple tokens are being considered as possibilities."}"""

    return interpretation


def get_ffn_interpretation(results: ProbeResults, model=None, tokenizer=None, use_ai: bool = False, interpreter: str = "claude") -> str:
    """Generate interpretation for FFN activation patterns."""
    if not results.ffn_activations:
        return "No FFN data available."

    sparsities = []
    mean_activations = []
    for idx in sorted(results.ffn_activations.keys()):
        arr = results.ffn_activations[idx]
        sparsity = float((np.abs(arr) < 0.1).mean())
        mean_act = float(np.abs(arr).mean())
        sparsities.append(sparsity)
        mean_activations.append(mean_act)

    avg_sparsity = np.mean(sparsities)
    sparsity_trend = "increasing" if sparsities[-1] > sparsities[0] else "decreasing"

    # Build context for AI
    context = f"""FFN (Feed-Forward Network) Analysis:
Average gate sparsity: {avg_sparsity:.1%} (fraction of near-zero activations)
Sparsity trend: {sparsity_trend} through layers
Mean activation range: {min(mean_activations):.2f} to {max(mean_activations):.2f}
Number of layers: {len(sparsities)}"""

    if use_ai:
        return generate_ai_interpretation(
            model, tokenizer, context,
            "What does this FFN sparsity pattern reveal about how the model routes and processes information?",
            interpreter=interpreter
        )

    # Fallback static interpretation
    interpretation = f"""**FFN Analysis:**

- **Average gate sparsity:** {avg_sparsity:.1%} of activations near zero
- **Sparsity trend:** {sparsity_trend} through layers
- **Mean activation range:** {min(mean_activations):.2f} to {max(mean_activations):.2f}

{"High sparsity indicates selective, efficient processing - the model has learned to activate only relevant pathways." if avg_sparsity > 0.5 else "Lower sparsity suggests more distributed processing across FFN pathways."}"""

    return interpretation


def get_embedding_interpretation(results: ProbeResults, model=None, tokenizer=None, use_ai: bool = False, interpreter: str = "claude") -> str:
    """Generate interpretation for embedding statistics."""
    if results.embeddings is None:
        return "No embedding data available."

    emb = results.embeddings
    if len(emb.shape) == 3:
        emb = emb[0]

    per_token_norms = np.linalg.norm(emb, axis=-1)
    mean_norm = float(np.mean(per_token_norms))
    std_norm = float(np.std(per_token_norms))

    # Build context for AI
    context = f"""Token Embedding Analysis:
Shape: {emb.shape}
Mean token norm: {mean_norm:.2f}
Norm std deviation: {std_norm:.2f}
Min/Max norm: {per_token_norms.min():.2f} / {per_token_norms.max():.2f}
Number of tokens: {len(results.input_tokens)}"""

    if use_ai:
        return generate_ai_interpretation(
            model, tokenizer, context,
            "What do these embedding statistics reveal about how the model represents this input?",
            interpreter=interpreter
        )

    # Fallback static interpretation
    interpretation = f"""**Embedding Analysis:**

- **Mean token norm:** {mean_norm:.2f}
- **Norm std deviation:** {std_norm:.2f}
- **Embedding shape:** {emb.shape}

{"Tokens have similar embedding magnitudes, suggesting uniform representation." if std_norm < 0.5 else "Significant variation in token norms - some tokens may carry more 'weight' in the representation."}"""

    return interpretation


def get_layer_similarity_interpretation(results: ProbeResults, model=None, tokenizer=None, use_ai: bool = False, interpreter: str = "claude") -> str:
    """Generate interpretation for layer similarity patterns."""
    if len(results.layer_outputs) < 2:
        return "Need at least 2 layers for similarity analysis."

    layer_indices = sorted(results.layer_outputs.keys())
    adjacent_sims = []

    for i in range(len(layer_indices) - 1):
        v1 = results.layer_outputs[layer_indices[i]].flatten()
        v2 = results.layer_outputs[layer_indices[i + 1]].flatten()
        v1 = v1 / (np.linalg.norm(v1) + 1e-8)
        v2 = v2 / (np.linalg.norm(v2) + 1e-8)
        sim = float(np.dot(v1, v2))
        adjacent_sims.append(sim)

    avg_sim = np.mean(adjacent_sims)
    min_sim_idx = np.argmin(adjacent_sims)

    # Build context for AI
    context = f"""Layer Similarity Analysis (Cosine Similarity):
Average adjacent layer similarity: {avg_sim:.3f}
Biggest transformation: Between layers {layer_indices[min_sim_idx]} and {layer_indices[min_sim_idx + 1]} (similarity: {adjacent_sims[min_sim_idx]:.3f})
Number of layers compared: {len(adjacent_sims)}
Similarity range: {min(adjacent_sims):.3f} to {max(adjacent_sims):.3f}"""

    if use_ai:
        return generate_ai_interpretation(
            model, tokenizer, context,
            "What do these layer similarity patterns reveal about how the model transforms representations through its layers?",
            interpreter=interpreter
        )

    # Fallback static interpretation
    interpretation = f"""**Layer Similarity Analysis:**

- **Average adjacent layer similarity:** {avg_sim:.3f}
- **Biggest transformation:** Between layers {layer_indices[min_sim_idx]} and {layer_indices[min_sim_idx + 1]} (similarity: {adjacent_sims[min_sim_idx]:.3f})

{"High similarity between adjacent layers suggests incremental refinement." if avg_sim > 0.9 else "Moderate similarity indicates meaningful transformations at each layer." if avg_sim > 0.7 else "Low similarity suggests significant changes in representation at each layer."}"""

    return interpretation


def get_residual_stream_interpretation(results: ProbeResults, model=None, tokenizer=None, use_ai: bool = False, interpreter: str = "claude") -> str:
    """Generate interpretation for residual stream patterns."""
    if not results.residual_stream_norms:
        return "No residual stream data available."

    norms = [x[1] for x in results.residual_stream_norms]
    deltas = [x[1] for x in results.residual_stream_deltas] if results.residual_stream_deltas else []

    norm_trend = "growing" if norms[-1] > norms[0] else "shrinking"

    if deltas:
        max_delta_idx = np.argmax(deltas)
        max_delta_layer = results.residual_stream_deltas[max_delta_idx][0]
        max_delta_val = deltas[max_delta_idx]
    else:
        max_delta_layer = "N/A"
        max_delta_val = 0

    # Build context for AI
    context = f"""Residual Stream Analysis:
Stream magnitude trend: {norm_trend} from {norms[0]:.2f} to {norms[-1]:.2f}
Largest layer contribution: {max_delta_layer} (delta: {max_delta_val:.2f})
Total positions tracked: {len(norms)}
Average norm: {np.mean(norms):.2f}"""

    if use_ai:
        return generate_ai_interpretation(
            model, tokenizer, context,
            "What does this residual stream pattern reveal about how information flows and transforms through the model?",
            interpreter=interpreter
        )

    # Fallback static interpretation
    interpretation = f"""**Residual Stream Analysis:**

- **Stream magnitude:** {norm_trend} from {norms[0]:.2f} to {norms[-1]:.2f}
- **Largest contribution:** {max_delta_layer} (delta: {max_delta_val:.2f})
- **Total layers tracked:** {len(norms)}

The residual stream is {norm_trend}, indicating {"information accumulation" if norm_trend == "growing" else "compression/filtering"}. Layer {max_delta_layer} makes the largest modification to the stream."""

    return interpretation


def get_logits_interpretation(results: ProbeResults, model=None, tokenizer=None, use_ai: bool = False, interpreter: str = "claude") -> str:
    """Generate interpretation for logits distribution."""
    if results.logits is None:
        return "No logits data available."

    logits = np.array(results.logits)
    mean_logit = float(logits.mean())
    std_logit = float(logits.std())
    max_logit = float(logits.max())
    min_logit = float(logits.min())

    # Get softmax distribution info
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    top_prob = float(probs.max())

    # Build context for AI
    context = f"""Logits Distribution Analysis:
Mean logit: {mean_logit:.4f}
Std deviation: {std_logit:.4f}
Range: [{min_logit:.4f}, {max_logit:.4f}]
Vocabulary size: {len(logits)}
Entropy (after softmax): {entropy:.4f}
Top token probability: {top_prob:.2%}"""

    if use_ai:
        return generate_ai_interpretation(
            model, tokenizer, context,
            "What does this logits distribution reveal about model confidence and the sharpness of the output distribution?",
            interpreter=interpreter
        )

    # Fallback static interpretation
    confidence = "high" if top_prob > 0.5 else "medium" if top_prob > 0.1 else "low"
    distribution = "sharp/confident" if std_logit > 5 else "diffuse/uncertain"

    interpretation = f"""**Logits Distribution Analysis:**

- **Statistics:** Mean={mean_logit:.2f}, Std={std_logit:.2f}
- **Range:** [{min_logit:.2f}, {max_logit:.2f}]
- **Entropy:** {entropy:.2f} ({"low uncertainty" if entropy < 2 else "high uncertainty"})
- **Top probability:** {top_prob:.1%} ({confidence} confidence)

The distribution is {distribution}, suggesting the model is {"confident in its prediction" if confidence == "high" else "considering multiple possibilities"}."""

    return interpretation


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
        go.Scatter(x=layers, y=sparsities, mode='lines+markers', name='Sparsity',
                   line=dict(color='#636EFA')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=layers, y=mean_acts, mode='lines+markers', name='Mean |Act|',
                   line=dict(color='#EF553B')),
        row=1, col=2
    )

    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="Layer", row=1, col=1)
    fig.update_xaxes(title_text="Layer", row=1, col=2)
    fig.update_yaxes(title_text="Sparsity (fraction near 0)", row=1, col=1)
    fig.update_yaxes(title_text="Mean |Activation|", row=1, col=2)

    return fig


def plot_embeddings_pca(results: ProbeResults, tokenizer) -> go.Figure:
    """Plot PCA of token embeddings with section-based coloring."""
    if results.embeddings is None:
        return go.Figure()

    emb = results.embeddings[0]

    if emb.shape[0] < 2:
        return go.Figure()

    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(emb)

    # Combine input and generated tokens
    all_tokens = list(results.input_tokens) if results.input_tokens else []
    if hasattr(results, 'generated_tokens') and results.generated_tokens:
        all_tokens = all_tokens + list(results.generated_tokens)

    # Detect sections for coloring
    sections = detect_prompt_sections(results, tokenizer)

    # Build labels and section assignments
    labels = []
    section_ids = []
    section_names = []

    # Define section colors (all possible section names from detect_prompt_sections)
    section_colors = {
        "📋 System Prompt": "#636EFA",      # Blue
        "👤 User Prompt": "#00CC96",         # Green
        "📥 Input Sequence": "#00CC96",      # Green (fallback)
        "📥 Input": "#00CC96",               # Green (fallback)
        "🤖 Response": "#AB63FA",            # Purple
        "🤖 Response (Pre-Reasoning)": "#AB63FA",  # Purple
        "🧠 Reasoning": "#FFA15A",           # Orange
        "🧠 Reasoning (In Progress)": "#FFA15A",   # Orange
        "✅ Final Response": "#EF553B",      # Red
        "📄 All Tokens": "#888888",          # Gray
        "Unknown": "#888888"                 # Gray
    }

    for i in range(emb.shape[0]):
        # Get token text
        if i < len(all_tokens):
            try:
                text = tokenizer.decode([all_tokens[i]]) if hasattr(tokenizer, 'decode') else str(all_tokens[i])
                text = text.replace('\n', '\\n')[:15]
                labels.append(f"{i}: {text}")
            except:
                labels.append(f"{i}: [{all_tokens[i]}]")
        else:
            labels.append(f"{i}")

        # Determine section
        section_name = "Unknown"
        for sec_name, (start, end) in sections.items():
            if start <= i < end:
                section_name = sec_name
                break
        section_ids.append(section_name)
        if section_name not in section_names:
            section_names.append(section_name)

    fig = go.Figure()

    # Add traces by section for legend
    for sec_name in section_names:
        indices = [i for i, s in enumerate(section_ids) if s == sec_name]
        if indices:
            fig.add_trace(go.Scatter(
                x=[emb_2d[i, 0] for i in indices],
                y=[emb_2d[i, 1] for i in indices],
                mode='markers',
                name=sec_name,
                text=[labels[i] for i in indices],
                hovertemplate='%{text}<extra></extra>',
                marker=dict(size=10, color=section_colors.get(sec_name, "#888888"))
            ))

    fig.update_layout(
        title=f"Token Embeddings PCA (explained var: {pca.explained_variance_ratio_.sum():.1%})",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_logits_distribution(results: ProbeResults, tokenizer=None) -> go.Figure:
    """Plot logits distribution with token labels."""
    if results.logits is None:
        return go.Figure()

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Logits Histogram", "Log Probability", "Top Token Logits")
    )

    logits = results.logits

    # 1. Raw logits histogram
    fig.add_trace(
        go.Histogram(x=logits, nbinsx=100, name='Logits', marker_color='#636EFA'),
        row=1, col=1
    )

    # 2. Log probability histogram
    if results.token_probs is not None:
        log_probs = np.log10(results.token_probs + 1e-15)
        fig.add_trace(
            go.Histogram(x=log_probs, nbinsx=100, name='Log Prob', marker_color='#00CC96'),
            row=1, col=2
        )

    # 3. Top tokens bar chart - show LOGIT VALUES (not probabilities) for better visibility
    if results.top_k_tokens and results.logits is not None:
        top_k = results.top_k_tokens[:10]
        # Handle empty/whitespace tokens - show token ID as fallback
        labels = []
        token_logits = []
        for t in top_k:
            token_text = t[2][:15] if t[2] and not t[2].isspace() else f"<{t[0]}>"
            labels.append(token_text)
            # Get the actual logit value for this token
            token_id = t[0]
            if 0 <= token_id < len(results.logits):
                token_logits.append(float(results.logits[token_id]))
            else:
                token_logits.append(0.0)

        fig.add_trace(
            go.Bar(x=labels, y=token_logits, name='Top Logits', marker_color='#EF553B'),
            row=1, col=3
        )

    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="Logit Value", row=1, col=1)
    fig.update_xaxes(title_text="Log₁₀(Probability)", row=1, col=2)
    fig.update_xaxes(title_text="Token", tickangle=45, row=1, col=3)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Logit Value", row=1, col=3)

    return fig


def plot_token_probabilities(results: ProbeResults) -> go.Figure:
    """Plot horizontal bar chart of top token probabilities."""
    if not results.top_k_tokens:
        return go.Figure()

    top_k = results.top_k_tokens[:15]

    # Check if distribution is extremely peaked (top prob > 99%)
    top_prob = top_k[0][1] if top_k else 0
    is_peaked = top_prob > 0.99

    # Handle empty/whitespace tokens - show token ID as fallback
    labels = []
    for t in reversed(top_k):
        token_text = t[2][:20] if t[2] and not t[2].isspace() else f"<{t[0]}>"
        labels.append(token_text)

    if is_peaked and results.logits is not None:
        # When distribution is peaked, show logit differences from max (more informative)
        max_logit = float(results.logits.max())
        logit_diffs = []
        for t in reversed(top_k):
            token_id = t[0]
            if 0 <= token_id < len(results.logits):
                logit_diffs.append(float(results.logits[token_id]) - max_logit)
            else:
                logit_diffs.append(-100.0)

        fig = go.Figure(go.Bar(
            x=logit_diffs,
            y=labels,
            orientation='h',
            marker_color='#636EFA',
            text=[f"{d:.1f}" for d in logit_diffs],
            textposition='auto'
        ))

        fig.update_layout(
            title="Top Token Predictions (Logit Difference from Max)",
            xaxis_title="Logit Difference (0 = highest)",
            yaxis_title="Token",
            height=max(400, len(labels) * 25)
        )
    else:
        # Normal distribution - show probabilities
        probs = [t[1] * 100 for t in reversed(top_k)]

        fig = go.Figure(go.Bar(
            x=probs,
            y=labels,
            orientation='h',
            marker_color='#636EFA',
            text=[f"{p:.1f}%" for p in probs],
            textposition='auto'
        ))

        fig.update_layout(
            title="Top Token Predictions",
            xaxis_title="Probability (%)",
            yaxis_title="Token",
            height=max(400, len(labels) * 25)
        )

    return fig


def plot_layer_similarity(results: ProbeResults) -> go.Figure:
    """Plot cosine similarity between layers."""
    layer_indices = sorted(results.layer_outputs.keys())
    n = len(layer_indices)

    if n < 2:
        return go.Figure()

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


def plot_causal_trace_heatmap(
    trace_results: CausalTraceResults,
    tokenizer=None,
    tokens: Optional[List[int]] = None
) -> go.Figure:
    """
    Plot causal trace indirect effects heatmap.

    Shows recovery of target probability when restoring clean activations
    at each (layer, position) combination.

    Args:
        trace_results: Results from CausalTracer.run_causal_trace()
        tokenizer: Optional tokenizer for position labels
        tokens: Optional token list for position labels
    """
    if trace_results.indirect_effects is None:
        fig = go.Figure()
        fig.add_annotation(text="No causal trace data available", showarrow=False)
        return fig

    ie = trace_results.indirect_effects
    layers_tested = trace_results.layers_tested
    positions_tested = trace_results.positions_tested

    # Create position labels
    if tokenizer and tokens:
        pos_labels = []
        for pos in positions_tested:
            if pos < len(tokens):
                try:
                    tok_text = tokenizer.decode([tokens[pos]])[:8]
                    # Mark subject positions
                    if pos in trace_results.subject_positions:
                        pos_labels.append(f"{pos}:*{tok_text}*")
                    else:
                        pos_labels.append(f"{pos}:{tok_text}")
                except:
                    pos_labels.append(f"Pos {pos}")
            else:
                pos_labels.append(f"Pos {pos}")
    else:
        pos_labels = [f"Pos {p}" for p in positions_tested]

    layer_labels = [f"L{l}" for l in layers_tested]

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=ie,
        x=pos_labels,
        y=layer_labels,
        colorscale='RdYlBu_r',  # Red=high recovery, Blue=low
        zmid=0.5,
        zmin=0,
        zmax=1,
        colorbar=dict(title="Recovery"),
        hovertemplate='Layer %{y}, %{x}<br>Recovery: %{z:.2%}<extra></extra>'
    ))

    # Add marker for critical site
    crit_layer_idx = layers_tested.index(trace_results.critical_layer) if trace_results.critical_layer in layers_tested else 0
    crit_pos_idx = positions_tested.index(trace_results.critical_position) if trace_results.critical_position in positions_tested else 0

    fig.add_trace(go.Scatter(
        x=[pos_labels[crit_pos_idx]],
        y=[layer_labels[crit_layer_idx]],
        mode='markers',
        marker=dict(
            size=20,
            color='lime',
            symbol='star',
            line=dict(color='black', width=2)
        ),
        name=f'Critical Site ({trace_results.max_recovery:.1%})',
        showlegend=True
    ))

    fig.update_layout(
        title=f"Causal Trace: '{trace_results.subject_text}' → '{trace_results.target_token}'",
        xaxis_title="Token Position (* = subject)",
        yaxis_title="Layer",
        height=max(400, len(layers_tested) * 25),
        yaxis=dict(autorange='reversed')  # Layer 0 at top
    )

    return fig


def plot_causal_trace_summary(trace_results: CausalTraceResults) -> go.Figure:
    """
    Plot summary statistics for causal trace results.

    Shows clean vs corrupted probabilities and layer-wise max recovery.
    """
    if trace_results.indirect_effects is None:
        fig = go.Figure()
        fig.add_annotation(text="No causal trace data available", showarrow=False)
        return fig

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Probability Comparison",
            "Max Recovery by Layer"
        ),
        column_widths=[0.3, 0.7]
    )

    # Left: Bar chart of clean vs corrupted probability
    fig.add_trace(
        go.Bar(
            x=['Clean', 'Corrupted'],
            y=[trace_results.clean_prob * 100, trace_results.corrupted_prob * 100],
            marker_color=['#2E86AB', '#E94F37'],
            text=[f"{trace_results.clean_prob:.1%}", f"{trace_results.corrupted_prob:.1%}"],
            textposition='auto'
        ),
        row=1, col=1
    )

    # Right: Line plot of max recovery per layer
    ie = trace_results.indirect_effects
    layers_tested = trace_results.layers_tested
    max_recovery_per_layer = np.max(ie, axis=1)  # Max across positions

    fig.add_trace(
        go.Scatter(
            x=[f"L{l}" for l in layers_tested],
            y=max_recovery_per_layer * 100,
            mode='lines+markers',
            line=dict(color='#28A745'),
            marker=dict(size=8),
            name='Max Recovery'
        ),
        row=1, col=2
    )

    # Mark critical layer
    crit_layer_idx = layers_tested.index(trace_results.critical_layer) if trace_results.critical_layer in layers_tested else 0
    fig.add_trace(
        go.Scatter(
            x=[f"L{trace_results.critical_layer}"],
            y=[trace_results.max_recovery * 100],
            mode='markers',
            marker=dict(size=15, color='gold', symbol='star', line=dict(color='black', width=2)),
            name=f'Critical Layer'
        ),
        row=1, col=2
    )

    fig.update_yaxes(title_text="Probability (%)", row=1, col=1)
    fig.update_yaxes(title_text="Recovery (%)", row=1, col=2)
    fig.update_xaxes(title_text="Layer", row=1, col=2)

    fig.update_layout(
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
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

    labels = [x[0] for x in results.residual_stream_norms]
    norms = [x[1] for x in results.residual_stream_norms]

    fig.add_trace(
        go.Scatter(x=labels, y=norms, mode='lines+markers', name='Norm',
                   line=dict(color='#636EFA')),
        row=1, col=1
    )

    if results.residual_stream_deltas:
        delta_labels = [x[0] for x in results.residual_stream_deltas]
        deltas = [x[1] for x in results.residual_stream_deltas]

        fig.add_trace(
            go.Bar(x=delta_labels, y=deltas, name='Delta', marker_color='#EF553B'),
            row=2, col=1
        )

    fig.update_layout(height=600, showlegend=False)
    fig.update_xaxes(title_text="Position", row=1, col=1)
    fig.update_xaxes(title_text="Layer", row=2, col=1)
    fig.update_yaxes(title_text="L2 Norm", row=1, col=1)
    fig.update_yaxes(title_text="Delta", row=2, col=1)

    return fig


def plot_moe_expert_load(results: ProbeResults, layer_idx: Optional[int] = None) -> go.Figure:
    """Plot MoE expert load distribution."""
    if not results.moe_expert_load:
        return go.Figure().add_annotation(text="No MoE data available", showarrow=False)

    if layer_idx is not None and layer_idx in results.moe_expert_load:
        # Single layer view
        load = results.moe_expert_load[layer_idx]
        experts = list(range(len(load)))
        counts = [load.get(e, 0) for e in experts]

        fig = go.Figure(data=[
            go.Bar(x=[f"E{e}" for e in experts], y=counts, marker_color='#636EFA')
        ])
        fig.update_layout(
            title=f"Expert Load - Layer {layer_idx}",
            xaxis_title="Expert",
            yaxis_title="Token Count",
            height=400
        )
    else:
        # All layers heatmap
        layers = sorted(results.moe_expert_load.keys())
        if not layers:
            return go.Figure()

        # Safely get num_experts
        num_experts = results.num_experts
        if not num_experts and results.moe_expert_load:
            # Try to infer from the data
            try:
                num_experts = max(
                    max(load.keys()) + 1 for load in results.moe_expert_load.values() if load
                )
            except ValueError:
                num_experts = 0
        if not num_experts:
            return go.Figure()

        # Build heatmap data
        z = []
        for layer in layers:
            load = results.moe_expert_load[layer]
            row = [load.get(e, 0) for e in range(num_experts)]
            z.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=[f"E{e}" for e in range(num_experts)],
            y=[f"L{l}" for l in layers],
            colorscale='Blues',
            colorbar=dict(title="Tokens")
        ))
        fig.update_layout(
            title="Expert Load Across Layers",
            xaxis_title="Expert",
            yaxis_title="Layer",
            height=max(400, len(layers) * 25)
        )

    return fig


def plot_moe_router_probs(results: ProbeResults, layer_idx: int, tokenizer=None) -> go.Figure:
    """Plot MoE router probabilities for a specific layer."""
    if layer_idx not in results.moe_router_outputs:
        return go.Figure().add_annotation(text="No router data for this layer", showarrow=False)

    router_data = results.moe_router_outputs[layer_idx]
    probs = router_data["probs"]  # Shape: (batch, seq, num_experts)
    selected = router_data["selected"]  # Shape: (batch, seq, top_k)

    # Take first batch item
    probs = probs[0]  # (seq, num_experts)
    selected = selected[0]  # (seq, top_k)

    seq_len, num_experts = probs.shape

    # Token labels - include both input and generated tokens
    all_tokens = list(results.input_tokens) if results.input_tokens else []
    if hasattr(results, 'generated_tokens') and results.generated_tokens:
        all_tokens = all_tokens + list(results.generated_tokens)

    if tokenizer and all_tokens:
        token_labels = []
        for i in range(seq_len):
            if i < len(all_tokens):
                try:
                    text = tokenizer.decode([all_tokens[i]])[:10]
                    token_labels.append(f"{i}: {text}")
                except:
                    token_labels.append(f"{i}")
            else:
                token_labels.append(f"{i}")
    else:
        token_labels = [f"Pos {i}" for i in range(seq_len)]

    fig = go.Figure(data=go.Heatmap(
        z=probs,
        x=[f"E{e}" for e in range(num_experts)],
        y=token_labels,
        colorscale='Viridis',
        colorbar=dict(title="Probability")
    ))

    fig.update_layout(
        title=f"Router Probabilities - Layer {layer_idx}",
        xaxis_title="Expert",
        yaxis_title="Token",
        height=max(400, seq_len * 20)
    )

    return fig


def plot_moe_expert_selection(results: ProbeResults, layer_idx: int, tokenizer=None) -> go.Figure:
    """Plot which experts were selected for each token."""
    if layer_idx not in results.moe_router_outputs:
        return go.Figure().add_annotation(text="No router data for this layer", showarrow=False)

    router_data = results.moe_router_outputs[layer_idx]
    selected = router_data["selected"][0]  # (seq, top_k)
    probs = router_data["probs"][0]  # (seq, num_experts)

    seq_len, top_k = selected.shape

    # Token labels - include both input and generated tokens
    all_tokens = list(results.input_tokens) if results.input_tokens else []
    if hasattr(results, 'generated_tokens') and results.generated_tokens:
        all_tokens = all_tokens + list(results.generated_tokens)

    if tokenizer and all_tokens:
        token_labels = []
        for i in range(seq_len):
            if i < len(all_tokens):
                try:
                    text = tokenizer.decode([all_tokens[i]])[:10]
                    token_labels.append(f"{i}: {text}")
                except:
                    token_labels.append(f"{i}")
            else:
                token_labels.append(f"{i}")
    else:
        token_labels = [f"Pos {i}" for i in range(seq_len)]

    # Create traces for each top-k position
    fig = go.Figure()

    # Use same distinct colors as other MoE charts
    rank_colors = [
        '#FFD700',  # Top-1: Gold
        '#FF00FF',  # Top-2: Magenta
        '#00FFFF',  # Top-3: Cyan
        '#FF6600',  # Top-4: Orange
        '#00FF00',  # Top-5: Lime
        '#9900FF',  # Top-6: Purple
        '#FF3333',  # Top-7: Red
        '#00CC99',  # Top-8: Teal
    ]

    # Note: argsort returns ascending order, so selected[:, -1] is highest prob (Top-1)
    for rank in range(1, top_k + 1):
        sel_idx = top_k - rank  # Top-1 → last index, Top-4 → index 0
        expert_ids = selected[:, sel_idx]
        # Get the probability for each selected expert
        expert_probs = [probs[i, expert_ids[i]] for i in range(seq_len)]

        fig.add_trace(go.Scatter(
            x=list(range(seq_len)),
            y=expert_ids,
            mode='markers',
            marker=dict(
                size=[p * 30 + 5 for p in expert_probs],  # Size by probability
                color=rank_colors[(rank - 1) % len(rank_colors)],
                opacity=0.7
            ),
            name=f"Top-{rank}",
            text=[f"Token: {token_labels[i]}<br>Expert: E{expert_ids[i]}<br>Prob: {expert_probs[i]:.2%}<br>Rank: Top-{rank}"
                  for i in range(seq_len)],
            hoverinfo='text'
        ))

    fig.update_layout(
        title=f"Expert Selection Pattern - Layer {layer_idx}",
        xaxis_title="Token Position",
        yaxis_title="Expert ID",
        yaxis=dict(dtick=1),
        height=400,
        showlegend=True
    )

    return fig


def plot_moe_topk_weights(results: ProbeResults, layer_idx: int, tokenizer=None, max_tokens: int = 30) -> go.Figure:
    """
    Plot top-k expert weights per token as a stacked horizontal bar chart.
    Shows exactly which experts were selected and their router probabilities.
    """
    if layer_idx not in results.moe_router_outputs:
        return go.Figure().add_annotation(text="No router data for this layer", showarrow=False)

    router_data = results.moe_router_outputs[layer_idx]
    selected = router_data["selected"][0]  # (seq, top_k)
    probs = router_data["probs"][0]  # (seq, num_experts)

    seq_len, top_k = selected.shape
    num_experts = probs.shape[1]

    # Limit tokens displayed for readability
    display_len = min(seq_len, max_tokens)

    # Combine input and generated tokens
    all_tokens = list(results.input_tokens) if results.input_tokens else []
    if hasattr(results, 'generated_tokens') and results.generated_tokens:
        all_tokens = all_tokens + list(results.generated_tokens)

    # Token labels
    token_labels = []
    for i in range(display_len):
        if tokenizer and i < len(all_tokens):
            try:
                text = tokenizer.decode([all_tokens[i]])
                # Clean up the text
                text = text.replace('\n', '\\n').replace('\t', '\\t')[:12]
                if not text or text.isspace():
                    text = f"<{all_tokens[i]}>"
                token_labels.append(f"{i}: {text}")
            except:
                token_labels.append(f"Pos {i}")
        else:
            token_labels.append(f"Pos {i}")

    # Create stacked bar chart - each selection rank gets a distinct color
    fig = go.Figure()

    # Distinct colors for selection rank (Top-1 through Top-8)
    # Using very distinct colors: Gold, Magenta, Cyan, Orange, Lime, Purple, Red, Teal
    rank_colors = [
        '#FFD700',  # Top-1: Gold/Yellow
        '#FF00FF',  # Top-2: Magenta/Pink
        '#00FFFF',  # Top-3: Cyan
        '#FF6600',  # Top-4: Orange
        '#00FF00',  # Top-5: Lime Green
        '#9900FF',  # Top-6: Purple
        '#FF3333',  # Top-7: Red
        '#00CC99',  # Top-8: Teal
    ]

    # Build data for each selection rank
    # Note: argsort returns indices in ascending order, so selected[i, -1] is highest prob (Top-1)
    # We iterate from Top-k to Top-1 so Top-1 appears on top of the stack (rightmost)
    for rank in range(top_k, 0, -1):  # rank 4, 3, 2, 1
        x_vals = []
        y_vals = []
        hover_texts = []

        # selected array index: rank 1 (Top-1) → index top_k-1, rank 4 (Top-4) → index 0
        sel_idx = top_k - rank

        for i in range(display_len):
            expert_id = int(selected[i, sel_idx])
            expert_prob = float(probs[i, expert_id])

            x_vals.append(expert_prob * 100)  # Convert to percentage
            y_vals.append(token_labels[i])
            hover_texts.append(f"Token: {token_labels[i]}<br>Expert E{expert_id}<br>Weight: {expert_prob:.2%}<br>Rank: Top-{rank}")

        # Use rank-based color (rank 1 = index 0 = Gold, rank 4 = index 3 = Orange)
        rank_color = rank_colors[(rank - 1) % len(rank_colors)]

        fig.add_trace(go.Bar(
            x=x_vals,
            y=y_vals,
            orientation='h',
            name=f"Top-{rank}",
            marker_color=rank_color,
            text=[f"E{int(selected[i, sel_idx])}" for i in range(display_len)],
            textposition='inside',
            textfont=dict(size=10, color='black' if rank == 1 else 'white'),
            hovertext=hover_texts,
            hoverinfo='text'
        ))

    fig.update_layout(
        title=f"Top-{top_k} Expert Selection per Token - Layer {layer_idx}",
        xaxis_title="Router Weight (%)",
        yaxis_title="Token",
        barmode='stack',
        height=max(400, display_len * 25),
        showlegend=True,
        legend_title="Selection Rank",
        yaxis=dict(autorange="reversed")  # First token at top
    )

    # Add explanation annotation at bottom
    fig.add_annotation(
        text=f"Each bar shows which {top_k} experts the router selected. Bar length = weight assigned. Labels show expert ID (E0-E{num_experts-1}).",
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=11, color="gray"),
        align="center"
    )

    # Add annotation if truncated
    if seq_len > max_tokens:
        fig.add_annotation(
            text=f"Showing first {max_tokens} of {seq_len} tokens",
            xref="paper", yref="paper",
            x=1, y=1.02,
            showarrow=False,
            font=dict(size=10, color="gray")
        )

    return fig


def plot_moe_expert_path(results: ProbeResults, token_position: int, tokenizer=None) -> go.Figure:
    """
    Plot the expert activation path for a specific token across all MoE layers.
    Shows which experts are activated at each layer, revealing the token's "pathway" through the network.
    """
    if not results.moe_router_outputs:
        return go.Figure().add_annotation(text="No MoE routing data available", showarrow=False)

    moe_layers = sorted(results.moe_router_outputs.keys())
    if not moe_layers:
        return go.Figure().add_annotation(text="No MoE layers found", showarrow=False)

    # Get token text if available - try multiple sources
    token_text = ""

    # Try results.all_tokens first
    if tokenizer and hasattr(results, 'all_tokens') and results.all_tokens:
        if token_position < len(results.all_tokens):
            try:
                token_text = tokenizer.decode([results.all_tokens[token_position]])
            except:
                pass

    # Try generation_timeline if available
    if not token_text and hasattr(results, 'generation_timeline') and results.generation_timeline:
        timeline = results.generation_timeline
        if token_position < len(timeline.input_tokens):
            try:
                token_text = tokenizer.decode([timeline.input_tokens[token_position]])
            except:
                pass
        elif timeline.steps:
            gen_idx = token_position - len(timeline.input_tokens)
            if 0 <= gen_idx < len(timeline.steps):
                token_text = timeline.steps[gen_idx].token_text

    # Clean up for display
    if token_text:
        token_text = token_text.replace('\n', '↵').replace('\t', '→')
        if len(token_text) > 30:
            token_text = token_text[:27] + "..."

    # Collect expert activations for this token at each layer
    num_experts = results.num_experts or 64
    top_k = results.num_experts_per_tok or 2

    # Build heatmap data: layers x experts
    activation_matrix = np.zeros((len(moe_layers), num_experts))
    layer_labels = []
    top_experts_per_layer = []

    for li, layer_idx in enumerate(moe_layers):
        router_data = results.moe_router_outputs[layer_idx]
        probs = router_data["probs"]

        # Handle shape
        if len(probs.shape) == 3:
            probs = probs[0]  # Remove batch dim

        if token_position < probs.shape[0]:
            token_probs = probs[token_position]  # (num_experts,)
            if hasattr(token_probs, 'tolist'):
                token_probs = np.array(token_probs.tolist())

            # Store in matrix
            activation_matrix[li, :len(token_probs)] = token_probs

            # Get top-K experts for this layer
            top_indices = np.argsort(token_probs)[-top_k:][::-1]
            top_experts_per_layer.append([(int(idx), float(token_probs[idx])) for idx in top_indices])
        else:
            top_experts_per_layer.append([])

        layer_labels.append(f"L{layer_idx}")

    # Create figure with two subplots: heatmap and path diagram
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.55, 0.45],
        subplot_titles=["Expert Activation Heatmap", "Top-K Expert Path"],
        horizontal_spacing=0.15
    )

    # Left: Heatmap of all expert activations
    fig.add_trace(
        go.Heatmap(
            z=activation_matrix,
            x=[f"E{e}" for e in range(num_experts)],
            y=layer_labels,
            colorscale="YlOrRd",
            showscale=True,
            colorbar_x=0.52,
            colorbar_len=0.8,
            colorbar_thickness=15,
            colorbar_title_text="Weight",
            hovertemplate="Layer %{y}<br>Expert %{x}<br>Weight: %{z:.3f}<extra></extra>"
        ),
        row=1, col=1
    )

    # Add bounding boxes around selected experts (top-K) with contrasting colors
    # Colors chosen to stand out against warm heatmap (yellow/orange/red)
    rank_colors = ['#00ff00', '#00bfff', '#000000', '#ff0000']  # Top-1: lime green, Top-2: deep sky blue, Top-3: black, Top-4: red
    rank_widths = [3.5, 3, 2.5, 2]  # Thicker border for higher rank

    for li, layer_idx in enumerate(moe_layers):
        if li < len(top_experts_per_layer) and top_experts_per_layer[li]:
            for rank, (expert_id, weight) in enumerate(top_experts_per_layer[li]):
                if rank < len(rank_colors):  # Only show top-4
                    # Add rectangle shape around the heatmap cell
                    # x coordinates are expert indices, y coordinates are layer indices
                    fig.add_shape(
                        type="rect",
                        x0=expert_id - 0.5,
                        x1=expert_id + 0.5,
                        y0=li - 0.5,
                        y1=li + 0.5,
                        line=dict(color=rank_colors[rank], width=rank_widths[rank]),
                        fillcolor="rgba(0,0,0,0)",  # Transparent fill
                        row=1, col=1
                    )

    # Right: Path diagram showing top-K experts at each layer
    # Draw connections between layers
    for li in range(len(moe_layers)):
        if li < len(top_experts_per_layer) and top_experts_per_layer[li]:
            for rank, (expert_id, weight) in enumerate(top_experts_per_layer[li]):
                # Position: x = expert_id normalized, y = layer index
                x_pos = expert_id / num_experts
                y_pos = li

                # Color based on rank (top-1 = red, top-2 = orange, etc.)
                colors = ['#ff4444', '#ff8844', '#ffaa44', '#ffcc44', '#ffee44']
                color = colors[min(rank, len(colors)-1)]

                # Size based on weight
                size = 10 + weight * 30

                fig.add_trace(
                    go.Scatter(
                        x=[x_pos],
                        y=[y_pos],
                        mode='markers+text',
                        marker=dict(size=size, color=color, line=dict(width=1, color='white')),
                        text=[f"E{expert_id}"],
                        textposition="middle center",
                        textfont=dict(size=8, color='white'),
                        hovertemplate=f"Layer {moe_layers[li]}<br>Expert {expert_id}<br>Weight: {weight:.3f}<br>Rank: {rank+1}<extra></extra>",
                        showlegend=False
                    ),
                    row=1, col=2
                )

                # Draw line to previous layer's top expert if exists
                if li > 0 and top_experts_per_layer[li-1]:
                    prev_expert, prev_weight = top_experts_per_layer[li-1][0]  # Connect to top-1
                    prev_x = prev_expert / num_experts
                    # Thicker lines that are visible on both dark and light themes
                    # Use cyan/teal color that contrasts well with warm markers
                    line_width = 2 + weight * 4  # Base width 2, scales with weight
                    fig.add_trace(
                        go.Scatter(
                            x=[prev_x, x_pos],
                            y=[li-1, li],
                            mode='lines',
                            line=dict(width=line_width, color='rgba(100,180,220,0.6)'),
                            hoverinfo='skip',
                            showlegend=False
                        ),
                        row=1, col=2
                    )

    # Update layout
    title_text = f"Expert Activation Path for Token {token_position}"
    if token_text:
        title_text += f": '{token_text}'"

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14)),
        height=400 + len(moe_layers) * 20,
        showlegend=False
    )

    # Update axes for heatmap
    fig.update_xaxes(title_text="Expert ID", row=1, col=1, tickangle=45)
    fig.update_yaxes(title_text="MoE Layer", row=1, col=1, autorange="reversed")

    # Update axes for path diagram
    fig.update_xaxes(
        title_text=f"Expert Position (0-{num_experts-1})",
        row=1, col=2,
        range=[-0.05, 1.05],
        tickvals=[0, 0.25, 0.5, 0.75, 1.0],
        ticktext=['E0', f'E{num_experts//4}', f'E{num_experts//2}', f'E{3*num_experts//4}', f'E{num_experts-1}']
    )
    fig.update_yaxes(
        title_text="MoE Layer",
        row=1, col=2,
        tickvals=list(range(len(moe_layers))),
        ticktext=layer_labels,
        autorange="reversed"
    )

    # Add legend annotation for border colors at top of chart (prominent position)
    fig.add_annotation(
        text="<b>Selected Expert Borders:</b>  🟢 Top-1  🔵 Top-2  ⬛ Top-3  🔴 Top-4",
        xref="paper", yref="paper",
        x=0.5, y=1.12,
        showarrow=False,
        font=dict(size=12),
        align="center"
    )

    return fig


def get_expert_path_summary(results: ProbeResults, token_position: int) -> Dict[str, Any]:
    """
    Get summary statistics about a token's expert activation path.
    """
    if not results.moe_router_outputs:
        return {}

    moe_layers = sorted(results.moe_router_outputs.keys())
    num_experts = results.num_experts or 64
    top_k = results.num_experts_per_tok or 2

    # Track which experts are used
    all_top1_experts = []
    all_topk_experts = set()
    total_weight_per_expert = {e: 0.0 for e in range(num_experts)}

    for layer_idx in moe_layers:
        router_data = results.moe_router_outputs[layer_idx]
        probs = router_data["probs"]

        if len(probs.shape) == 3:
            probs = probs[0]

        if token_position < probs.shape[0]:
            token_probs = probs[token_position]
            if hasattr(token_probs, 'tolist'):
                token_probs = np.array(token_probs.tolist())

            top_indices = np.argsort(token_probs)[-top_k:][::-1]
            all_top1_experts.append(int(top_indices[0]))
            for idx in top_indices:
                all_topk_experts.add(int(idx))
                total_weight_per_expert[int(idx)] += float(token_probs[idx])

    # Calculate statistics
    unique_top1 = len(set(all_top1_experts))
    unique_topk = len(all_topk_experts)

    # Most used experts
    sorted_experts = sorted(total_weight_per_expert.items(), key=lambda x: x[1], reverse=True)
    top_5_experts = sorted_experts[:5]

    # Path consistency (do same experts appear across layers?)
    from collections import Counter
    top1_counts = Counter(all_top1_experts)
    most_common_top1 = top1_counts.most_common(3)

    return {
        'num_moe_layers': len(moe_layers),
        'unique_top1_experts': unique_top1,
        'unique_topk_experts': unique_topk,
        'top_5_by_total_weight': top_5_experts,
        'most_common_top1': most_common_top1,
        'path_concentration': unique_top1 / len(moe_layers) if moe_layers else 0,  # Lower = more concentrated
    }


def plot_moe_expert_weights_table(results: ProbeResults, layer_idx: int, tokenizer=None) -> pd.DataFrame:
    """
    Create a DataFrame showing the top-k expert selections and weights per token.
    """
    if layer_idx not in results.moe_router_outputs:
        return pd.DataFrame()

    router_data = results.moe_router_outputs[layer_idx]
    selected = router_data["selected"][0]  # (seq, top_k)
    probs = router_data["probs"][0]  # (seq, num_experts)

    seq_len, top_k = selected.shape

    # Combine input and generated tokens
    all_tokens = list(results.input_tokens) if results.input_tokens else []
    if hasattr(results, 'generated_tokens') and results.generated_tokens:
        all_tokens = all_tokens + list(results.generated_tokens)

    rows = []
    for i in range(seq_len):
        # Get token text
        if tokenizer and i < len(all_tokens):
            try:
                text = tokenizer.decode([all_tokens[i]])
                text = text.replace('\n', '\\n').replace('\t', '\\t')[:15]
                if not text or text.isspace():
                    text = f"<{all_tokens[i]}>"
            except:
                text = f"<pos {i}>"
        else:
            text = f"<pos {i}>"

        row = {"Position": i, "Token": text}

        # Add each top-k expert and its weight
        # Note: argsort returns ascending order, so selected[i, -1] is highest prob (Top-1)
        for rank in range(1, top_k + 1):
            sel_idx = top_k - rank  # Top-1 → last index
            expert_id = int(selected[i, sel_idx])
            expert_prob = float(probs[i, expert_id])
            row[f"Top-{rank}"] = f"E{expert_id} ({expert_prob:.1%})"

        rows.append(row)

    return pd.DataFrame(rows)


def detect_prompt_sections(results: ProbeResults, tokenizer) -> Dict[str, Tuple[int, int]]:
    """
    Detect the boundaries of system prompt, user prompt, reasoning, and response sections.
    Returns dict with section names and (start_idx, end_idx) tuples.

    Supports common chat templates:
    - ChatML: <|im_start|>system, <|im_start|>user, <|im_start|>assistant
    - Llama: <|begin_of_text|>, <|start_header_id|>, [INST], [/INST]
    - Generic: system, user, assistant markers
    - Reasoning: <think>, </think>, <|thinking|>, <|/thinking|>, etc.
    """
    # Combine input and generated tokens for full sequence analysis
    all_tokens = list(results.input_tokens) if results.input_tokens else []
    input_len = len(all_tokens)
    if hasattr(results, 'generated_tokens') and results.generated_tokens:
        all_tokens = all_tokens + list(results.generated_tokens)

    total_tokens = len(all_tokens)

    if not tokenizer or total_tokens == 0:
        return {"📄 All Tokens": (0, total_tokens)}

    # Decode all tokens to find markers
    token_texts = []
    for tok_id in all_tokens:
        try:
            text = tokenizer.decode([tok_id])
            token_texts.append(text)
        except:
            token_texts.append("")

    # Scan through tokens to find section boundaries
    combined_text = ""
    token_positions = []  # Maps character position to token index

    for i, text in enumerate(token_texts):
        token_positions.append(len(combined_text))
        combined_text += text

    combined_lower = combined_text.lower()

    def find_token_at_position(char_pos: int) -> int:
        """Find which token index corresponds to a character position."""
        for i, pos in enumerate(token_positions):
            if pos >= char_pos:
                return max(0, i - 1)
        return len(token_positions) - 1

    # Common patterns to look for
    system_patterns = ['<|im_start|>system', '<|system|>', '<|start_header_id|>system', '[system]', '<|begin_of_text|>']
    user_patterns = ['<|im_start|>user', '<|user|>', '<|start_header_id|>user', '[inst]', '[user]', '<|eot_id|>']
    assistant_patterns = ['<|im_start|>assistant', '<|assistant|>', '<|start_header_id|>assistant', '[/inst]']

    # Reasoning patterns (start and end) - expanded list for various models
    reasoning_start_patterns = [
        # Standard reasoning tags
        '<think>', '<|thinking|>', '<|think|>', '<reasoning>', '<|reason|>',
        '<|startofthought|>', '<thought>', '<|thought|>', '\n<think>',
        # DeepSeek R1 style
        '```thinking', '<|begin_of_thought|>',
        # Other formats
        '**thinking**', '**thought**', '<internal_thought>',
        # QwQ style
        '<|box_start|>reasoning'
    ]
    reasoning_end_patterns = [
        # Standard reasoning tags
        '</think>', '<|/thinking|>', '<|/think|>', '</reasoning>', '<|/reason|>',
        '<|endofthought|>', '</thought>', '<|/thought|>',
        # DeepSeek R1 style
        '```\n', '<|end_of_thought|>',
        # Other formats
        '**end_thinking**', '</internal_thought>',
        # QwQ style
        '<|box_end|>'
    ]

    # Find section positions
    system_start = None
    user_start = None
    assistant_start = None
    reasoning_start = None
    reasoning_end = None

    # Find system section
    for pattern in system_patterns:
        pos = combined_lower.find(pattern.lower())
        if pos != -1:
            system_start = find_token_at_position(pos)
            break

    # Find user section
    for pattern in user_patterns:
        pos = combined_lower.find(pattern.lower())
        if pos != -1:
            user_start = find_token_at_position(pos)
            break

    # Find assistant section
    for pattern in assistant_patterns:
        pos = combined_lower.find(pattern.lower())
        if pos != -1:
            assistant_start = find_token_at_position(pos)
            break

    # Find reasoning section (within response)
    for pattern in reasoning_start_patterns:
        pos = combined_lower.find(pattern.lower())
        if pos != -1:
            reasoning_start = find_token_at_position(pos)
            break

    for pattern in reasoning_end_patterns:
        pos = combined_lower.find(pattern.lower())
        if pos != -1:
            # Find the end of the closing tag
            reasoning_end = find_token_at_position(pos + len(pattern))
            break

    # Build sections based on what we found
    sections = {}
    # total_tokens already calculated at top

    # Determine section boundaries
    if system_start is not None and user_start is not None and assistant_start is not None:
        # All three main sections found
        sections["📋 System Prompt"] = (system_start, user_start)
        sections["👤 User Prompt"] = (user_start, assistant_start)

        # Check for reasoning within response
        if reasoning_start is not None and reasoning_end is not None and reasoning_start >= assistant_start:
            if reasoning_start > assistant_start:
                sections["🤖 Response (Pre-Reasoning)"] = (assistant_start, reasoning_start)
            sections["🧠 Reasoning"] = (reasoning_start, reasoning_end)
            if reasoning_end < total_tokens:
                sections["✅ Final Response"] = (reasoning_end, total_tokens)
        elif reasoning_start is not None and reasoning_start >= assistant_start:
            # Reasoning started but didn't end (still thinking)
            if reasoning_start > assistant_start:
                sections["🤖 Response (Pre-Reasoning)"] = (assistant_start, reasoning_start)
            sections["🧠 Reasoning (In Progress)"] = (reasoning_start, total_tokens)
        else:
            sections["🤖 Response"] = (assistant_start, total_tokens)

    elif user_start is not None and assistant_start is not None:
        # No system prompt
        sections["👤 User Prompt"] = (0, assistant_start)

        # Check for reasoning within response
        if reasoning_start is not None and reasoning_end is not None and reasoning_start >= assistant_start:
            if reasoning_start > assistant_start:
                sections["🤖 Response (Pre-Reasoning)"] = (assistant_start, reasoning_start)
            sections["🧠 Reasoning"] = (reasoning_start, reasoning_end)
            if reasoning_end < total_tokens:
                sections["✅ Final Response"] = (reasoning_end, total_tokens)
        elif reasoning_start is not None and reasoning_start >= assistant_start:
            if reasoning_start > assistant_start:
                sections["🤖 Response (Pre-Reasoning)"] = (assistant_start, reasoning_start)
            sections["🧠 Reasoning (In Progress)"] = (reasoning_start, total_tokens)
        else:
            sections["🤖 Response"] = (assistant_start, total_tokens)

    elif assistant_start is not None:
        # Just input and response
        sections["📥 Input"] = (0, assistant_start)

        # Check for reasoning
        if reasoning_start is not None and reasoning_end is not None and reasoning_start >= assistant_start:
            if reasoning_start > assistant_start:
                sections["🤖 Response (Pre-Reasoning)"] = (assistant_start, reasoning_start)
            sections["🧠 Reasoning"] = (reasoning_start, reasoning_end)
            if reasoning_end < total_tokens:
                sections["✅ Final Response"] = (reasoning_end, total_tokens)
        elif reasoning_start is not None and reasoning_start >= assistant_start:
            if reasoning_start > assistant_start:
                sections["🤖 Response (Pre-Reasoning)"] = (assistant_start, reasoning_start)
            sections["🧠 Reasoning (In Progress)"] = (reasoning_start, total_tokens)
        else:
            sections["🤖 Response"] = (assistant_start, total_tokens)

    else:
        # Can't detect sections by chat template markers
        # Use input_len (calculated at top) to split input from generated
        gen_start = input_len  # Where generated tokens begin

        if hasattr(results, 'generated_tokens') and results.generated_tokens and gen_start < total_tokens:
            # We have generated tokens - create hierarchical sections

            # Try to split Input into System/User by looking for common patterns
            # Look for "You are" or system-like content at the start
            system_end = None
            you_are_pos = combined_lower.find("you are")
            if you_are_pos != -1 and you_are_pos < len(combined_text) // 3:
                # Found "you are" in first third - likely system prompt
                # Find where user content likely starts (after system message)
                newline_after = combined_lower.find("\n\n", you_are_pos)
                if newline_after != -1 and newline_after < gen_start:
                    system_end = find_token_at_position(newline_after)

            if system_end is not None and system_end > 0 and system_end < gen_start:
                sections["📋 System Prompt"] = (0, system_end)
                sections["👤 User Prompt"] = (system_end, gen_start)
            else:
                # Can't split input - show as single section
                sections["📥 Input Sequence"] = (0, gen_start)

            # Check for reasoning in generated portion
            if reasoning_start is not None and reasoning_end is not None and reasoning_start >= gen_start:
                if reasoning_start > gen_start:
                    sections["🤖 Response (Pre-Reasoning)"] = (gen_start, reasoning_start)
                sections["🧠 Reasoning"] = (reasoning_start, reasoning_end)
                if reasoning_end < total_tokens:
                    sections["✅ Final Response"] = (reasoning_end, total_tokens)
            elif reasoning_start is not None and reasoning_start >= gen_start:
                if reasoning_start > gen_start:
                    sections["🤖 Response (Pre-Reasoning)"] = (gen_start, reasoning_start)
                sections["🧠 Reasoning (In Progress)"] = (reasoning_start, total_tokens)
            else:
                # No reasoning detected - show as single Response section
                sections["🤖 Response"] = (gen_start, total_tokens)
        else:
            # No generated tokens - check for reasoning markers at all
            if reasoning_start is not None and reasoning_end is not None:
                if reasoning_start > 0:
                    sections["📥 Input"] = (0, reasoning_start)
                sections["🧠 Reasoning"] = (reasoning_start, reasoning_end)
                if reasoning_end < total_tokens:
                    sections["✅ Final Response"] = (reasoning_end, total_tokens)
            elif reasoning_start is not None:
                if reasoning_start > 0:
                    sections["📥 Input"] = (0, reasoning_start)
                sections["🧠 Reasoning (In Progress)"] = (reasoning_start, total_tokens)
            else:
                sections["📄 All Tokens"] = (0, total_tokens)

    # Remove any empty sections
    sections = {k: v for k, v in sections.items() if v[1] > v[0]}

    return sections


def render_moe_topk_by_section(results: ProbeResults, layer_idx: int, tokenizer, section_key: str):
    """
    Render the top-k expert weights for a specific section with show more functionality.
    Uses Streamlit widgets directly.
    """
    if layer_idx not in results.moe_router_outputs:
        st.info("No router data for this layer")
        return

    sections = detect_prompt_sections(results, tokenizer)
    if section_key not in sections:
        st.info(f"Section '{section_key}' not found")
        return

    start_idx, end_idx = sections[section_key]
    section_len = end_idx - start_idx

    if section_len == 0:
        st.info(f"No tokens in {section_key}")
        return

    router_data = results.moe_router_outputs[layer_idx]
    selected = router_data["selected"][0]  # (seq, top_k)
    probs = router_data["probs"][0]  # (seq, num_experts)

    seq_len, top_k = selected.shape
    num_experts = probs.shape[1]

    # Determine display range
    max_initial = 30
    show_all_key = f"show_all_{section_key}_{layer_idx}"

    if show_all_key not in st.session_state:
        st.session_state[show_all_key] = False

    if section_len > max_initial and not st.session_state[show_all_key]:
        display_end = start_idx + max_initial
        show_more = True
    else:
        display_end = end_idx
        show_more = False

    display_range = range(start_idx, min(display_end, seq_len))
    display_len = len(display_range)

    if display_len == 0:
        st.info(f"No tokens to display in {section_key}")
        return

    # Combine input and generated tokens for labels
    all_tokens = list(results.input_tokens) if results.input_tokens else []
    if hasattr(results, 'generated_tokens') and results.generated_tokens:
        all_tokens = all_tokens + list(results.generated_tokens)

    # Token labels
    token_labels = []
    for i in display_range:
        if tokenizer and i < len(all_tokens):
            try:
                text = tokenizer.decode([all_tokens[i]])
                text = text.replace('\n', '\\n').replace('\t', '\\t')[:12]
                if not text or text.isspace():
                    text = f"<{all_tokens[i]}>"
                token_labels.append(f"{i}: {text}")
            except:
                token_labels.append(f"Pos {i}")
        else:
            token_labels.append(f"Pos {i}")

    # Distinct colors for selection rank (Top-1 through Top-8)
    rank_colors = [
        '#FFD700',  # Top-1: Gold/Yellow
        '#FF00FF',  # Top-2: Magenta/Pink
        '#00FFFF',  # Top-3: Cyan
        '#FF6600',  # Top-4: Orange
        '#00FF00',  # Top-5: Lime Green
        '#9900FF',  # Top-6: Purple
        '#FF3333',  # Top-7: Red
        '#00CC99',  # Top-8: Teal
    ]

    # Create figure
    fig = go.Figure()

    # Note: argsort returns indices in ascending order, so selected[i, -1] is highest prob (Top-1)
    # We iterate from Top-k to Top-1 so Top-1 appears on top of the stack (rightmost)
    for rank in range(top_k, 0, -1):  # rank 4, 3, 2, 1
        x_vals = []
        y_vals = []
        hover_texts = []
        text_labels = []

        # selected array index: rank 1 (Top-1) → index top_k-1, rank 4 (Top-4) → index 0
        sel_idx = top_k - rank

        for idx, i in enumerate(display_range):
            if i < seq_len:
                expert_id = int(selected[i, sel_idx])
                expert_prob = float(probs[i, expert_id])

                x_vals.append(expert_prob * 100)
                y_vals.append(token_labels[idx])
                hover_texts.append(f"Token: {token_labels[idx]}<br>Expert E{expert_id}<br>Weight: {expert_prob:.2%}<br>Rank: Top-{rank}")
                text_labels.append(f"E{expert_id}")

        if x_vals:
            rank_color = rank_colors[(rank - 1) % len(rank_colors)]
            fig.add_trace(go.Bar(
                x=x_vals,
                y=y_vals,
                orientation='h',
                name=f"Top-{rank}",
                marker_color=rank_color,
                text=text_labels,
                textposition='inside',
                textfont=dict(size=10, color='black' if rank == 1 else 'white'),
                hovertext=hover_texts,
                hoverinfo='text'
            ))

    fig.update_layout(
        xaxis_title="Router Weight (%)",
        yaxis_title="Token",
        barmode='stack',
        height=max(300, display_len * 22),
        showlegend=True,
        legend_title="Selection Rank",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=10, b=50)
    )

    # Add brief explanation
    fig.add_annotation(
        text="Longer bars = higher router confidence. Labels show expert ID.",
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=10, color="gray"),
        align="center"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show more button
    if show_more:
        remaining = section_len - max_initial
        if st.button(f"📋 Show {remaining} more tokens...", key=f"more_{section_key}_{layer_idx}"):
            st.session_state[show_all_key] = True
            st.rerun()
    elif section_len > max_initial:
        if st.button(f"📋 Show less", key=f"less_{section_key}_{layer_idx}"):
            st.session_state[show_all_key] = False
            st.rerun()


def render_moe_topk_sections(results: ProbeResults, layer_idx: int, tokenizer):
    """
    Render the top-k expert weights in hierarchical groups:
    - Input Sequence: System Prompt, User Prompt
    - Response: Reasoning Loop, Final Response
    """
    sections = detect_prompt_sections(results, tokenizer)

    # Group sections into Input and Response categories
    input_sections = {}
    response_sections = {}

    for name, bounds in sections.items():
        if any(k in name for k in ["System", "User", "Input"]):
            input_sections[name] = bounds
        elif any(k in name for k in ["Response", "Reasoning", "Final", "🤖", "🧠", "✅"]):
            response_sections[name] = bounds
        else:
            # Fallback - put in response if after input
            input_len = len(results.input_tokens) if results.input_tokens else 0
            if bounds[0] >= input_len:
                response_sections[name] = bounds
            else:
                input_sections[name] = bounds

    # Calculate totals
    input_total = sum(e - s for s, e in [b for b in input_sections.values()]) if input_sections else 0
    response_total = sum(e - s for s, e in [b for b in response_sections.values()]) if response_sections else 0

    # Display summary
    has_system = any("System" in s for s in input_sections.keys())
    has_user = any("User" in s for s in input_sections.keys())
    has_reasoning = any("Reasoning" in s for s in response_sections.keys())
    has_final = any("Final" in s for s in response_sections.keys())

    input_desc = []
    if has_system:
        input_desc.append("System")
    if has_user:
        input_desc.append("User")
    if not input_desc and input_sections:
        input_desc.append("Input")

    response_desc = []
    if has_reasoning:
        response_desc.append("Reasoning")
    if has_final:
        response_desc.append("Final")
    if not response_desc and response_sections:
        response_desc.append("Response")

    all_desc = input_desc + response_desc
    st.caption(f"Tokens grouped by: {', '.join(all_desc) if all_desc else 'sequence position'}.")

    # Render Input Sequence group
    if input_sections:
        with st.expander(f"📥 **Input Sequence** ({input_total} tokens)", expanded=False):
            if len(input_sections) > 1:
                # Multiple sub-sections - show each
                for sub_name, (start, end) in input_sections.items():
                    token_count = end - start
                    st.markdown(f"**{sub_name}** ({token_count} tokens)")
                    render_moe_topk_by_section(results, layer_idx, tokenizer, sub_name)
                    st.markdown("---")
            else:
                # Single section - render directly
                for sub_name in input_sections.keys():
                    render_moe_topk_by_section(results, layer_idx, tokenizer, sub_name)

    # Render Response group
    if response_sections:
        with st.expander(f"🤖 **Response** ({response_total} tokens)", expanded=True):
            if len(response_sections) > 1:
                # Multiple sub-sections - show each with headers
                for sub_name, (start, end) in response_sections.items():
                    token_count = end - start
                    # Use appropriate emoji based on sub-section type
                    if "Reasoning" in sub_name:
                        st.markdown(f"🧠 **Reasoning Loop** ({token_count} tokens)")
                    elif "Final" in sub_name:
                        st.markdown(f"✅ **Final Response** ({token_count} tokens)")
                    else:
                        st.markdown(f"**{sub_name}** ({token_count} tokens)")
                    render_moe_topk_by_section(results, layer_idx, tokenizer, sub_name)
                    st.markdown("---")
            else:
                # Single section - render directly
                for sub_name in response_sections.keys():
                    render_moe_topk_by_section(results, layer_idx, tokenizer, sub_name)


@cache_with_hash
def compute_moe_influence_stats(results: ProbeResults) -> Dict[str, Any]:
    """
    Compute comprehensive MoE expert influence statistics using AI domain metrics.
    Results are cached based on probe results hash.

    Returns dict with:
    - expert_activation_frequency: How often each expert is selected (normalized)
    - expert_influence_ranking: Experts ranked by total activation count
    - router_entropy: Shannon entropy of routing distribution (diversity measure)
    - load_balance_score: 1 - normalized std dev (higher = more balanced)
    - gini_coefficient: Inequality measure (0 = perfect equality, 1 = max inequality)
    - auxiliary_loss_proxy: Approximation of load balancing auxiliary loss
    - dead_experts: Experts with zero activations
    - dominant_experts: Experts handling >2x average load
    - expert_specialization: Per-expert metrics
    """
    stats = {
        "expert_activation_frequency": {},
        "expert_influence_ranking": [],
        "router_entropy": 0.0,
        "load_balance_score": 0.0,
        "gini_coefficient": 0.0,
        "auxiliary_loss_proxy": 0.0,
        "dead_experts": [],
        "dominant_experts": [],
        "underutilized_experts": [],
        "expert_specialization": {},
        "total_routing_decisions": 0,
        "effective_expert_count": 0.0,
    }

    if not results.moe_expert_load:
        return stats

    num_experts = results.num_experts or 0
    if num_experts == 0:
        return stats

    # Aggregate expert load across all layers
    global_expert_counts = {e: 0 for e in range(num_experts)}
    layer_expert_counts = {}  # For per-layer analysis

    for layer_idx, load in results.moe_expert_load.items():
        layer_expert_counts[layer_idx] = load
        for expert_id, count in load.items():
            if expert_id in global_expert_counts:
                global_expert_counts[expert_id] += count

    total_activations = sum(global_expert_counts.values())
    stats["total_routing_decisions"] = total_activations

    if total_activations == 0:
        return stats

    # Expert Activation Frequency (normalized)
    activation_freq = {e: c / total_activations for e, c in global_expert_counts.items()}
    stats["expert_activation_frequency"] = activation_freq

    # Expert Influence Ranking (sorted by activation count, descending)
    ranking = sorted(global_expert_counts.items(), key=lambda x: x[1], reverse=True)
    stats["expert_influence_ranking"] = [
        {
            "rank": i + 1,
            "expert_id": expert_id,
            "activations": count,
            "frequency": count / total_activations,
            "percentile": (num_experts - i) / num_experts * 100
        }
        for i, (expert_id, count) in enumerate(ranking)
    ]

    # Router Entropy (Shannon entropy - measures routing diversity)
    # Higher entropy = more uniform distribution = better load balance
    probs = np.array(list(activation_freq.values()))
    probs = probs[probs > 0]  # Avoid log(0)
    if len(probs) > 0:
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(num_experts)  # Maximum possible entropy
        stats["router_entropy"] = float(entropy)
        stats["normalized_entropy"] = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    # Effective Expert Count (exponential of entropy)
    # Represents the "equivalent number of equally-used experts"
    if stats["router_entropy"] > 0:
        stats["effective_expert_count"] = float(2 ** stats["router_entropy"])

    # Load Balance Score (1 - coefficient of variation)
    counts = np.array(list(global_expert_counts.values()))
    mean_load = np.mean(counts)
    std_load = np.std(counts)
    if mean_load > 0:
        cv = std_load / mean_load
        stats["load_balance_score"] = float(max(0, 1 - cv))
        stats["coefficient_of_variation"] = float(cv)

    # Gini Coefficient (measure of inequality)
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    cumulative = np.cumsum(sorted_counts)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_counts))) / (n * np.sum(sorted_counts)) - (n + 1) / n
    stats["gini_coefficient"] = float(max(0, gini))

    # Auxiliary Loss Proxy (approximates load balancing loss used in training)
    # Based on Switch Transformer's auxiliary loss: sum(f_i * P_i)
    # Where f_i is fraction of tokens to expert i, P_i is avg router prob for expert i
    if results.moe_router_outputs:
        total_router_prob = {e: 0.0 for e in range(num_experts)}
        prob_count = 0
        for layer_data in results.moe_router_outputs.values():
            probs_arr = layer_data.get("probs", None)
            if probs_arr is not None:
                # Average router probability across all tokens
                avg_probs = np.mean(probs_arr, axis=(0, 1))
                for e in range(min(len(avg_probs), num_experts)):
                    total_router_prob[e] += avg_probs[e]
                prob_count += 1

        if prob_count > 0:
            avg_router_probs = {e: p / prob_count for e, p in total_router_prob.items()}
            # Auxiliary loss = num_experts * sum(f_i * P_i)
            aux_loss = num_experts * sum(
                activation_freq.get(e, 0) * avg_router_probs.get(e, 0)
                for e in range(num_experts)
            )
            stats["auxiliary_loss_proxy"] = float(aux_loss)

    # Identify problematic experts
    avg_activations = total_activations / num_experts

    # Dead experts (0 activations)
    stats["dead_experts"] = [e for e, c in global_expert_counts.items() if c == 0]

    # Dominant experts (>2x average load) - potential capacity bottleneck
    stats["dominant_experts"] = [
        {"expert_id": e, "activations": c, "ratio": c / avg_activations}
        for e, c in global_expert_counts.items()
        if c > 2 * avg_activations
    ]

    # Underutilized experts (<0.25x average load, but not dead)
    stats["underutilized_experts"] = [
        {"expert_id": e, "activations": c, "ratio": c / avg_activations}
        for e, c in global_expert_counts.items()
        if 0 < c < 0.25 * avg_activations
    ]

    # Per-expert specialization metrics
    for expert_id in range(num_experts):
        layer_presence = []
        for layer_idx, load in layer_expert_counts.items():
            layer_total = sum(load.values())
            if layer_total > 0:
                expert_share = load.get(expert_id, 0) / layer_total
                layer_presence.append(expert_share)

        if layer_presence:
            stats["expert_specialization"][expert_id] = {
                "mean_layer_share": float(np.mean(layer_presence)),
                "std_layer_share": float(np.std(layer_presence)),
                "consistency": float(1 - np.std(layer_presence)) if np.std(layer_presence) < 1 else 0.0,
                "layer_variance": float(np.var(layer_presence)),
            }

    return stats


def format_moe_influence_report(stats: Dict[str, Any], num_experts: int) -> str:
    """Format MoE influence stats as a readable markdown report."""
    lines = []

    if stats["total_routing_decisions"] == 0:
        return "No routing data available for analysis."

    # Header metrics
    lines.append("### Global Expert Influence Metrics\n")

    # Key metrics table
    lines.append("| Metric | Value | Interpretation |")
    lines.append("|--------|-------|----------------|")

    # Router Entropy
    entropy = stats.get("router_entropy", 0)
    norm_entropy = stats.get("normalized_entropy", 0)
    entropy_interp = "Excellent diversity" if norm_entropy > 0.9 else "Good diversity" if norm_entropy > 0.7 else "Moderate concentration" if norm_entropy > 0.5 else "High concentration"
    lines.append(f"| **Router Entropy** | {entropy:.3f} ({norm_entropy*100:.1f}% of max) | {entropy_interp} |")

    # Effective Expert Count
    eff_count = stats.get("effective_expert_count", 0)
    eff_pct = (eff_count / num_experts * 100) if num_experts > 0 else 0
    lines.append(f"| **Effective Expert Count** | {eff_count:.1f} / {num_experts} ({eff_pct:.1f}%) | Equivalent uniform experts |")

    # Load Balance Score
    lb_score = stats.get("load_balance_score", 0)
    lb_interp = "Well balanced" if lb_score > 0.8 else "Moderately balanced" if lb_score > 0.5 else "Imbalanced"
    lines.append(f"| **Load Balance Score** | {lb_score:.3f} | {lb_interp} |")

    # Gini Coefficient
    gini = stats.get("gini_coefficient", 0)
    gini_interp = "Very equal" if gini < 0.2 else "Moderately equal" if gini < 0.4 else "Unequal" if gini < 0.6 else "Highly unequal"
    lines.append(f"| **Gini Coefficient** | {gini:.3f} | {gini_interp} |")

    # Auxiliary Loss Proxy
    aux_loss = stats.get("auxiliary_loss_proxy", 0)
    # Ideal aux loss is ~1.0 (uniform), higher means imbalance
    aux_interp = "Optimal range" if aux_loss < 1.5 else "Acceptable" if aux_loss < 2.0 else "High - consider rebalancing"
    lines.append(f"| **Aux Loss Proxy** | {aux_loss:.3f} | {aux_interp} |")

    lines.append("")

    # Expert Influence Ranking
    lines.append("### Expert Influence Ranking\n")
    lines.append("*Ranked by total activation count across all MoE layers*\n")

    ranking = stats.get("expert_influence_ranking", [])
    if ranking:
        # Show top 10 and bottom 5
        lines.append("**Top Influential Experts:**")
        lines.append("| Rank | Expert | Activations | Share | Status |")
        lines.append("|------|--------|-------------|-------|--------|")

        avg_share = 1.0 / num_experts if num_experts > 0 else 0

        for item in ranking[:min(10, len(ranking))]:
            status = ""
            if item["frequency"] > 2 * avg_share:
                status = "🔥 Dominant"
            elif item["frequency"] > 1.5 * avg_share:
                status = "⬆️ High"
            elif item["frequency"] < 0.25 * avg_share:
                status = "⬇️ Underutilized"
            elif item["frequency"] < 0.5 * avg_share:
                status = "📉 Low"
            else:
                status = "✓ Normal"

            lines.append(f"| #{item['rank']} | Expert {item['expert_id']} | {item['activations']:,} | {item['frequency']*100:.2f}% | {status} |")

        # Show bottom experts if there are many
        if len(ranking) > 15:
            lines.append("")
            lines.append("**Least Active Experts:**")
            lines.append("| Rank | Expert | Activations | Share | Status |")
            lines.append("|------|--------|-------------|-------|--------|")
            for item in ranking[-5:]:
                status = "💀 Dead" if item["activations"] == 0 else "⬇️ Underutilized" if item["frequency"] < 0.25 * avg_share else "📉 Low"
                lines.append(f"| #{item['rank']} | Expert {item['expert_id']} | {item['activations']:,} | {item['frequency']*100:.2f}% | {status} |")

    lines.append("")

    # Warnings and insights
    dead = stats.get("dead_experts", [])
    dominant = stats.get("dominant_experts", [])
    underutil = stats.get("underutilized_experts", [])

    if dead or dominant or underutil:
        lines.append("### Routing Health Diagnostics\n")

        if dead:
            lines.append(f"⚠️ **Dead Experts ({len(dead)})**: Experts {dead[:10]}{'...' if len(dead) > 10 else ''} received no tokens.")
            lines.append("   - *Possible causes*: Router collapse, poor initialization, or training instability")
            lines.append("   - *Impact*: Reduced model capacity, wasted parameters")
            lines.append("")

        if dominant:
            dom_ids = [d["expert_id"] for d in dominant[:5]]
            lines.append(f"🔥 **Dominant Experts ({len(dominant)})**: Experts {dom_ids} handling >2x average load.")
            lines.append("   - *Possible causes*: Expert specialization or router bias")
            lines.append("   - *Impact*: Potential capacity bottleneck, reduced diversity")
            lines.append("")

        if underutil:
            under_ids = [u["expert_id"] for u in underutil[:5]]
            lines.append(f"📉 **Underutilized Experts ({len(underutil)})**: Experts {under_ids} handling <25% of average load.")
            lines.append("   - *Possible causes*: Weak expert representations or router preference bias")
            lines.append("   - *Impact*: Inefficient parameter usage")
            lines.append("")

    return "\n".join(lines)


@cache_with_hash
def get_expert_token_specialization(results: ProbeResults, tokenizer, top_n: int = 10) -> Dict[int, Dict]:
    """
    Analyze which tokens each expert specializes in processing.
    Results are cached based on probe results hash.

    Returns: {expert_id: {
        "tokens": [(token_text, token_id, count, avg_prob), ...],  # Most common tokens
        "total_activations": int,
        "unique_tokens": int,
        "specialization_score": float  # How focused vs diverse the expert is
    }}
    """
    expert_analysis = {}

    if not results.moe_expert_tokens:
        return expert_analysis

    for expert_id, token_data in results.moe_expert_tokens.items():
        if not token_data:
            expert_analysis[expert_id] = {
                "tokens": [],
                "total_activations": 0,
                "unique_tokens": 0,
                "specialization_score": 0.0
            }
            continue

        # Count token occurrences and average probabilities
        token_counts = {}  # token_id -> [count, sum_prob]
        for token_id, pos, layer_idx, prob in token_data:
            if token_id not in token_counts:
                token_counts[token_id] = [0, 0.0]
            token_counts[token_id][0] += 1
            token_counts[token_id][1] += prob

        # Sort by count (most frequent first)
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1][0], reverse=True)

        # Decode tokens and build result
        token_list = []
        for token_id, (count, sum_prob) in sorted_tokens[:top_n]:
            try:
                if tokenizer and token_id >= 0:
                    token_text = tokenizer.decode([token_id])
                    # Clean up for display
                    token_text = repr(token_text)[1:-1]  # Show escape sequences
                    if len(token_text) > 20:
                        token_text = token_text[:17] + "..."
                else:
                    token_text = f"<id:{token_id}>"
            except:
                token_text = f"<id:{token_id}>"

            avg_prob = sum_prob / count if count > 0 else 0
            token_list.append((token_text, token_id, count, avg_prob))

        total_activations = len(token_data)
        unique_tokens = len(token_counts)

        # Specialization score: how concentrated the expert's focus is
        # High score = expert focuses on few token types; Low = diverse
        if total_activations > 0 and unique_tokens > 0:
            # Use normalized entropy (inverse) as specialization score
            counts = np.array([c[0] for c in token_counts.values()])
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            max_entropy = np.log2(unique_tokens) if unique_tokens > 1 else 1
            specialization = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
        else:
            specialization = 0.0

        expert_analysis[expert_id] = {
            "tokens": token_list,
            "total_activations": total_activations,
            "unique_tokens": unique_tokens,
            "specialization_score": float(specialization)
        }

    return expert_analysis


# =============================================================================
# PDF Report Generation
# =============================================================================

def sanitize_text(text: str) -> str:
    """Sanitize text for PDF output."""
    replacements = {
        '\u2014': '--', '\u2013': '-', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2026': '...', '\u2022': '*',
        '\u2192': '->', '\u2190': '<-', '\u2248': '~=', '\u2260': '!=',
        '\u2264': '<=', '\u2265': '>=', '\u00d7': 'x', '\u00f7': '/',
    }
    for unicode_char, ascii_equiv in replacements.items():
        text = text.replace(unicode_char, ascii_equiv)
    text = text.encode('ascii', 'replace').decode('ascii')
    return text


def generate_pdf_report(
    results: ProbeResults,
    model_path: str,
    probe_config: ProbeConfig,
    interpretations: Optional[Dict[str, str]] = None
) -> bytes:
    """Generate a PDF report of the probe analysis."""
    if not PDF_AVAILABLE:
        raise ImportError("fpdf2 not installed. Run: pip install fpdf2")

    class ProbeReportPDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 10)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, 'MLXLMProbe Analysis Report', align='C')
            self.ln(15)

        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f'Page {self.page_no()}', align='C')

        def section_title(self, title: str):
            self.set_font('Helvetica', 'B', 14)
            self.set_text_color(31, 73, 125)
            self.cell(0, 10, sanitize_text(title), ln=True)
            self.ln(2)

        def subsection_title(self, title: str):
            self.set_font('Helvetica', 'B', 11)
            self.set_text_color(60, 60, 60)
            self.cell(0, 8, sanitize_text(title), ln=True)

        def body_text(self, text: str):
            self.set_font('Helvetica', '', 10)
            self.set_text_color(0, 0, 0)
            self.multi_cell(0, 5, sanitize_text(text))
            self.ln(2)

        def stat_line(self, label: str, value: str):
            self.set_font('Helvetica', 'B', 10)
            self.cell(60, 6, f"{sanitize_text(label)}:", ln=False)
            self.set_font('Helvetica', '', 10)
            self.cell(0, 6, sanitize_text(value), ln=True)

    pdf = ProbeReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title page
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(31, 73, 125)
    pdf.cell(0, 40, '', ln=True)
    pdf.cell(0, 15, 'MLXLMProbe Report', align='C', ln=True)
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', align='C', ln=True)
    pdf.cell(0, 8, sanitize_text(f'Model: {model_path}'), align='C', ln=True)
    pdf.ln(20)

    # Summary box
    pdf.set_fill_color(240, 248, 255)
    pdf.set_draw_color(31, 73, 125)
    pdf.rect(20, pdf.get_y(), 170, 40, style='DF')
    pdf.set_xy(25, pdf.get_y() + 5)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'Summary', ln=True)
    pdf.set_x(25)
    pdf.set_font('Helvetica', '', 10)
    summary = f"Input: {len(results.input_tokens)} tokens | Output: {len(results.generated_tokens)} tokens | Layers: {results.num_layers}"
    if results.top_k_tokens:
        top = results.top_k_tokens[0]
        top_text = top[2] if top[2] and not top[2].isspace() else f"<{top[0]}>"
        summary += f" | Top: '{top_text[:15]}' ({top[1]:.1%})"
    pdf.multi_cell(160, 5, sanitize_text(summary))

    # Input/Output
    pdf.add_page()
    pdf.section_title('1. Input & Output')
    pdf.subsection_title('Input Text')
    pdf.body_text(results.input_text[:500] + ('...' if len(results.input_text) > 500 else ''))
    pdf.stat_line("Input Tokens", str(len(results.input_tokens)))

    pdf.ln(5)
    pdf.subsection_title('Generated Output')
    if results.reasoning_text:
        pdf.body_text("Reasoning/Analysis:")
        pdf.body_text(results.reasoning_text[:500] + ('...' if len(results.reasoning_text) > 500 else ''))
        pdf.ln(3)
        pdf.body_text("Response:")
        pdf.body_text(results.answer_text if results.answer_text else "(No response)")
    else:
        pdf.body_text(results.generated_text if results.generated_text else "(No output generated)")
    pdf.stat_line("Output Tokens", str(len(results.generated_tokens)))

    # Token Predictions
    pdf.section_title('2. Token Predictions')
    if results.top_k_tokens:
        for i, (tok_id, prob, text) in enumerate(results.top_k_tokens[:10]):
            display_text = text if text and not text.isspace() else f"<{tok_id}>"
            pdf.stat_line(f"#{i+1}", f"'{display_text[:30]}' - {prob:.2%}")

    if interpretations and 'token_prob' in interpretations:
        pdf.ln(5)
        pdf.subsection_title('AI Interpretation')
        pdf.body_text(interpretations['token_prob'])

    # Layer Analysis
    pdf.add_page()
    pdf.section_title('3. Layer Analysis')
    if results.layer_outputs:
        pdf.subsection_title('Activation Norms')
        stats = []
        for idx in sorted(results.layer_outputs.keys())[:15]:
            arr = results.layer_outputs[idx]
            norm = float(np.linalg.norm(arr, axis=-1).mean())
            stats.append(f"L{idx}: {norm:.2f}")
        pdf.body_text("Norms: " + ", ".join(stats))

    if interpretations and 'layers' in interpretations:
        pdf.ln(5)
        pdf.subsection_title('AI Interpretation')
        pdf.body_text(interpretations['layers'])

    # FFN Analysis
    pdf.section_title('4. FFN Analysis')
    if results.ffn_activations:
        stats = []
        for idx in sorted(results.ffn_activations.keys())[:10]:
            arr = results.ffn_activations[idx]
            sparsity = float((np.abs(arr) < 0.1).mean())
            stats.append(f"L{idx}: {sparsity:.1%}")
        pdf.body_text("Sparsity: " + ", ".join(stats))

    if interpretations and 'ffn' in interpretations:
        pdf.ln(5)
        pdf.subsection_title('AI Interpretation')
        pdf.body_text(interpretations['ffn'])

    # Embeddings
    pdf.add_page()
    pdf.section_title('5. Embeddings')
    if results.embeddings is not None:
        emb = results.embeddings[0] if len(results.embeddings.shape) == 3 else results.embeddings
        pdf.stat_line("Shape", str(emb.shape))
        pdf.stat_line("Mean", f"{emb.mean():.4f}")
        pdf.stat_line("Std", f"{emb.std():.4f}")

    # Configuration
    pdf.add_page()
    pdf.section_title('6. Configuration')
    pdf.stat_line("Model", model_path)
    pdf.stat_line("Layers Probed", str(len(results.layer_outputs)))
    pdf.stat_line("Capture Embeddings", str(probe_config.capture_embeddings))
    pdf.stat_line("Capture Residual Stream", str(probe_config.capture_residual_stream))

    return bytes(pdf.output())


# =============================================================================
# HTML Report Generation
# =============================================================================

def generate_html_report(
    results: ProbeResults,
    model_path: str,
    probe_config: ProbeConfig,
    tokenizer=None,
    interpretations: Optional[Dict[str, str]] = None
) -> str:
    """Generate a standalone HTML report with interactive charts."""

    # Generate chart JSONs
    charts = {}

    if results.layer_outputs:
        charts['layers'] = plot_layer_norms(results).to_json()

    if results.ffn_activations:
        charts['ffn'] = plot_ffn_analysis(results).to_json()

    if results.logits is not None:
        charts['logits'] = plot_logits_distribution(results, tokenizer).to_json()

    if results.top_k_tokens:
        charts['token_probs'] = plot_token_probabilities(results).to_json()

    if results.embeddings is not None and tokenizer:
        charts['embeddings'] = plot_embeddings_pca(results, tokenizer).to_json()

    if len(results.layer_outputs) >= 2:
        charts['similarity'] = plot_layer_similarity(results).to_json()

    if results.residual_stream_norms:
        charts['residual'] = plot_residual_stream(results).to_json()

    # Prepare display values (handle empty tokens)
    if results.top_k_tokens:
        top_pred_text = results.top_k_tokens[0][2]
        if not top_pred_text or top_pred_text.isspace():
            top_pred_text = f"<{results.top_k_tokens[0][0]}>"
        top_pred_text = top_pred_text[:15]
    else:
        top_pred_text = "N/A"

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLXLMProbe Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0e1117; color: #fafafa; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 20px; text-align: center; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ opacity: 0.9; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .tabs {{ display: flex; flex-wrap: wrap; gap: 5px; margin: 20px 0; border-bottom: 2px solid #333; padding-bottom: 10px; }}
        .tab-btn {{ background: #1e1e1e; border: none; color: #fafafa; padding: 10px 20px; cursor: pointer; border-radius: 5px 5px 0 0; }}
        .tab-btn:hover {{ background: #333; }}
        .tab-btn.active {{ background: #667eea; }}
        .tab-content {{ display: none; padding: 20px 0; }}
        .tab-content.active {{ display: block; }}
        .card {{ background: #1e1e1e; border-radius: 10px; padding: 20px; margin: 20px 0; }}
        .card h3 {{ color: #667eea; margin-bottom: 15px; }}
        .stat {{ display: inline-block; background: #2d2d2d; padding: 10px 20px; border-radius: 5px; margin: 5px; }}
        .stat-label {{ font-size: 0.8em; color: #888; }}
        .stat-value {{ font-size: 1.2em; font-weight: bold; }}
        .interpretation {{ background: #1a1a2e; border-left: 4px solid #667eea; padding: 15px; margin: 15px 0; border-radius: 0 5px 5px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ background: #2d2d2d; }}
        .chart {{ width: 100%; min-height: 400px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>MLXLMProbe Report</h1>
        <p>Model: {model_path} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="container">
        <div class="tabs">
            <button class="tab-btn active" onclick="openTab(event, 'overview')">Overview</button>
            <button class="tab-btn" onclick="openTab(event, 'tokens')">Tokens</button>
            <button class="tab-btn" onclick="openTab(event, 'layers')">Layers</button>
            <button class="tab-btn" onclick="openTab(event, 'ffn')">FFN</button>
            <button class="tab-btn" onclick="openTab(event, 'logits')">Logits</button>
            <button class="tab-btn" onclick="openTab(event, 'embeddings')">Embeddings</button>
            <button class="tab-btn" onclick="openTab(event, 'similarity')">Similarity</button>
            <button class="tab-btn" onclick="openTab(event, 'residual')">Residual</button>
        </div>

        <div id="overview" class="tab-content active">
            <div class="card">
                <h3>Summary</h3>
                <div class="stat"><span class="stat-label">Input Tokens</span><br><span class="stat-value">{len(results.input_tokens)}</span></div>
                <div class="stat"><span class="stat-label">Output Tokens</span><br><span class="stat-value">{len(results.generated_tokens)}</span></div>
                <div class="stat"><span class="stat-label">Layers</span><br><span class="stat-value">{results.num_layers}</span></div>
                <div class="stat"><span class="stat-label">Top Prediction</span><br><span class="stat-value">{top_pred_text}</span></div>
            </div>
            <div class="card">
                <h3>Input</h3>
                <p style="background: #2d2d2d; padding: 15px; border-radius: 5px; white-space: pre-wrap;">{results.input_text[:1000]}</p>
            </div>
            <div class="card">
                <h3>Generated Output</h3>
                {f'''<details style="margin-bottom: 15px;">
                    <summary style="cursor: pointer; color: #667eea; font-weight: bold;">🧠 Reasoning / Analysis (click to expand)</summary>
                    <p style="background: #1a1a2e; padding: 15px; border-radius: 5px; white-space: pre-wrap; margin-top: 10px; border-left: 4px solid #667eea;">{results.reasoning_text}</p>
                </details>
                <h4 style="color: #4ade80; margin-bottom: 10px;">Response</h4>''' if results.reasoning_text else ''}
                <p style="background: #1a3a1a; padding: 15px; border-radius: 5px; white-space: pre-wrap; border-left: 4px solid #4ade80;">{results.answer_text if results.reasoning_text else (results.generated_text if results.generated_text else '(No output generated)')}</p>
                <small style="color: #888;">Generated {len(results.generated_tokens)} tokens</small>
            </div>
        </div>

        <div id="tokens" class="tab-content">
            <div class="card">
                <h3>Top Predicted Tokens</h3>
                {'<div class="chart" id="chart-token-probs"></div>' if 'token_probs' in charts else ''}
                <table>
                    <tr><th>Rank</th><th>Token</th><th>Probability</th></tr>
                    {''.join(f'<tr><td>{i+1}</td><td>{(t[2] if t[2] and not t[2].isspace() else f"&lt;{t[0]}&gt;")[:30]}</td><td>{t[1]:.2%}</td></tr>' for i, t in enumerate(results.top_k_tokens[:10]))}
                </table>
                {f'<div class="interpretation">{interpretations.get("token_prob", "")}</div>' if interpretations and interpretations.get("token_prob") else ''}
            </div>
        </div>

        <div id="layers" class="tab-content">
            <div class="card">
                <h3>Layer Activation Norms</h3>
                {'<div class="chart" id="chart-layers"></div>' if 'layers' in charts else '<p>No layer data</p>'}
                {f'<div class="interpretation">{interpretations.get("layers", "")}</div>' if interpretations and interpretations.get("layers") else ''}
            </div>
        </div>

        <div id="ffn" class="tab-content">
            <div class="card">
                <h3>FFN Analysis</h3>
                {'<div class="chart" id="chart-ffn"></div>' if 'ffn' in charts else '<p>No FFN data</p>'}
                {f'<div class="interpretation">{interpretations.get("ffn", "")}</div>' if interpretations and interpretations.get("ffn") else ''}
            </div>
        </div>

        <div id="logits" class="tab-content">
            <div class="card">
                <h3>Logits Distribution</h3>
                {'<div class="chart" id="chart-logits"></div>' if 'logits' in charts else '<p>No logits data</p>'}
                {f'<div class="interpretation">{interpretations.get("logits", "")}</div>' if interpretations and interpretations.get("logits") else ''}
            </div>
        </div>

        <div id="embeddings" class="tab-content">
            <div class="card">
                <h3>Token Embeddings (PCA)</h3>
                {'<div class="chart" id="chart-embeddings"></div>' if 'embeddings' in charts else '<p>No embedding data</p>'}
                {f'<div class="interpretation">{interpretations.get("embeddings", "")}</div>' if interpretations and interpretations.get("embeddings") else ''}
            </div>
        </div>

        <div id="similarity" class="tab-content">
            <div class="card">
                <h3>Layer Similarity</h3>
                {'<div class="chart" id="chart-similarity"></div>' if 'similarity' in charts else '<p>Need 2+ layers</p>'}
                {f'<div class="interpretation">{interpretations.get("similarity", "")}</div>' if interpretations and interpretations.get("similarity") else ''}
            </div>
        </div>

        <div id="residual" class="tab-content">
            <div class="card">
                <h3>Residual Stream</h3>
                {'<div class="chart" id="chart-residual"></div>' if 'residual' in charts else '<p>No residual data</p>'}
                {f'<div class="interpretation">{interpretations.get("residual", "")}</div>' if interpretations and interpretations.get("residual") else ''}
            </div>
        </div>
    </div>

    <script>
        function openTab(evt, tabName) {{
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
            evt.currentTarget.classList.add('active');
            window.dispatchEvent(new Event('resize'));
        }}

        // Render charts
        const charts = {json.dumps(charts)};
        for (const [key, data] of Object.entries(charts)) {{
            const el = document.getElementById('chart-' + key.replace('_', '-'));
            if (el && data) {{
                Plotly.newPlot(el, JSON.parse(data).data, JSON.parse(data).layout, {{responsive: true}});
            }}
        }}
    </script>
</body>
</html>"""

    return html


# =============================================================================
# Metric Definitions
# =============================================================================

METRIC_DEFINITIONS = {
    "activation_norm": {
        "name": "Activation Norm (L2)",
        "short": "Magnitude of neural activity at each layer",
        "full": """**Activation Norm (L2 Norm)** measures the Euclidean length of the activation vector.

**What it means:** Higher norms = stronger/more intense neural activity. The model is encoding more information or giving stronger "signals" at that layer.

---

**📊 Chart Axes:**
- **X-axis (Layer):** Layer index from 0 to N-1
- **Y-axis (L2 Norm):** The Euclidean magnitude of the activation vector: sqrt(sum(x²)). Higher = stronger signal.

---

**Why it matters:**
- Increasing norms through layers often indicate the model is building up representations
- Decreasing norms might indicate information compression
- Sudden changes can indicate important processing steps
- Very high norms can indicate instability; very low norms might mean vanishing gradients"""
    },
    "gate_sparsity": {
        "name": "FFN Gate Analysis",
        "short": "How the FFN filters and routes information",
        "full": """**FFN (Feed-Forward Network)** in each transformer layer uses SwiGLU activation:
```
output = SiLU(gate) × up_projection
```
The gate controls which pathways are "open" (active) or "closed" (blocked).

---

**📊 Left Chart: Gate Sparsity** (Y-axis: fraction from 0.0 to 1.0)

Measures what fraction of gate values are near zero (|x| < 0.1).

| Value | Meaning |
|-------|---------|
| 0.0 (0%) | All pathways open - dense processing |
| 0.5 (50%) | Half pathways closed - selective |
| 1.0 (100%) | All pathways closed - no information flows |

**Interpretation:** Higher sparsity = more selective/efficient processing. The model has learned to "turn off" irrelevant pathways.

---

**📊 Right Chart: Mean Gate Activation** (Y-axis: average magnitude)

Measures the average absolute value of activations: mean(|activation|)

| Value | Meaning |
|-------|---------|
| ~0.1-0.3 | Low activity - subtle processing |
| ~0.5-1.0 | Moderate activity - typical range |
| >1.5 | High activity - strong signals |

**Interpretation:** Higher values = stronger neural activity at that layer. Very high or very low across all layers may indicate issues.

---

**Why this matters:**
- Sparsity increasing in later layers = model becoming more selective/specialized
- Consistent mean activation = stable information flow
- Large variations may indicate where key processing happens"""
    },
    "token_probability": {
        "name": "Token Probability",
        "short": "Model's confidence for each possible next word",
        "full": """**Token Probability** is the softmax output showing the model's confidence that each vocabulary token should come next.

**What it means:**
- High probability on one token (>50%) = model is confident
- Spread across many tokens = model is uncertain or multiple valid continuations exist

---

**📊 Bar Chart Axes:**
- **X-axis (Token):** The vocabulary token (word/subword) as text
- **Y-axis (Probability):** The probability from 0.0 to 1.0 (0% to 100%). All probabilities sum to 1.0 across the full vocabulary.

---

**Key metrics:**
- **Top-K tokens:** The K highest probability candidates
- **Probability mass:** Sum of top-K probabilities. Higher = more confident
- **Entropy:** Measures uncertainty. Low (~0-2) = confident; High (>4) = uncertain"""
    },
    "embedding": {
        "name": "Token Embeddings",
        "short": "Dense vector representations of input words",
        "full": """**Token Embeddings** are learned vector representations that convert discrete tokens into continuous vectors the model can process.

**What it means:** Each token from the vocabulary is mapped to a high-dimensional vector. Similar concepts have similar vectors.

---

**📊 PCA Scatter Plot Axes:**
- **X-axis (PC1):** First principal component - captures the dimension of maximum variance in the embedding space
- **Y-axis (PC2):** Second principal component - captures the next most significant variation
- **Points:** Each point is a token; nearby points have similar embeddings

**📊 Heatmap Axes:**
- **X-axis (Dimension):** Embedding dimension index
- **Y-axis (Token):** Input tokens from the prompt
- **Color:** Activation value (blue=negative, white=zero, red=positive)

---

**Key statistics:**
- **Mean ≈ 0:** Centered embeddings (common after normalization)
- **Std:** Higher = more expressive/varied representations
- **Per-token norm:** Tokens with higher norms often represent more "important" or distinctive concepts"""
    },
    "logits": {
        "name": "Logits",
        "short": "Raw model outputs before probability conversion",
        "full": """**Logits** are the raw, unnormalized scores the model outputs for each vocabulary token before softmax converts them to probabilities.

**What it means:**
- Higher logit = model prefers that token
- Logits can be any real number (positive or negative)
- Softmax(logits) → probabilities

---

**📊 Histogram Axes (Raw Logits):**
- **X-axis (Logit Value):** The raw score value, can be negative or positive
- **Y-axis (Count):** Number of vocabulary tokens with that logit value

**📊 Histogram Axes (Log Probability):**
- **X-axis (Log₁₀ Probability):** Log-scaled probability (-15 to 0, where 0 = 100%)
- **Y-axis (Count):** Number of tokens at that probability level

**📊 Top Tokens Bar Chart:**
- **X-axis (Token):** The vocabulary token text
- **Y-axis (Logit Value):** The raw logit score for that token

---

**Why look at logits:**
- See the full distribution, not just top tokens
- Understand model confidence (tight vs spread distribution)
- Detect potential issues (all similar logits = confused model)"""
    },
    "layer_similarity": {
        "name": "Layer Similarity",
        "short": "How similar are representations across layers",
        "full": """**Layer Similarity** uses cosine similarity to measure how similar the representations are between different layers.

**What it means:**
- Similarity = 1.0: Identical representations
- Similarity ≈ 0: Orthogonal/unrelated
- Adjacent layers often have high similarity (incremental changes)

---

**📊 Heatmap Axes:**
- **X-axis (Layer j):** Layer index
- **Y-axis (Layer i):** Layer index
- **Color (Similarity):** Cosine similarity from 0.0 (blue, orthogonal) to 1.0 (red, identical)
- **Diagonal:** Always 1.0 (layer compared to itself)

**Reading the heatmap:** Each cell (i, j) shows how similar layer i's output is to layer j's output. Bright red = very similar, dark blue = very different.

---

**Why it matters:**
- Large drops indicate significant transformations
- High similarity across many layers might indicate redundancy
- Helps identify which layers do the "heavy lifting" """
    },
    "residual_stream": {
        "name": "Residual Stream",
        "short": "The main information highway through the transformer",
        "full": """**Residual Stream** is the central pathway through which information flows in a transformer.

**How it works:**
- Starts with token embeddings
- Each layer ADDS its contribution: `stream = stream + layer_output`
- This "residual connection" helps gradients flow and prevents degradation

---

**📊 Top Chart: Stream Magnitude** (Y-axis: L2 Norm)

Shows the "strength" of the signal at each point in the network.

| Pattern | Meaning |
|---------|---------|
| Increasing | Information accumulating, representations getting richer |
| Decreasing | Information compression or filtering |
| Stable | Balanced processing, no major changes |
| Spike | A layer added significant new information |

---

**📊 Bottom Chart: Layer Delta** (Y-axis: change magnitude)

Shows how much each layer changes the stream.

| Delta Value | Meaning |
|-------------|---------|
| High (>0.5) | Layer makes major transformation |
| Medium (0.1-0.5) | Normal processing |
| Low (<0.1) | Layer makes minimal changes |

**Why this matters:**
- High-delta layers are doing the "heavy lifting"
- Consistently low deltas might indicate redundant layers
- The pattern reveals where key transformations happen"""
    },
    "moe_routing": {
        "name": "MoE Expert Routing",
        "short": "How tokens are distributed across experts in Mixture of Experts models",
        "full": """**Mixture of Experts (MoE)** models use a router/gate to select which expert networks process each token.

**How it works:**
- Each layer has multiple expert networks (FFN modules)
- A learned router computes scores for each expert
- Top-k experts (usually 2) are selected per token
- Outputs are combined weighted by router probabilities

---

**📊 Expert Load Chart** (per layer)

Shows how many tokens each expert processes.

| Pattern | Meaning |
|---------|---------|
| Balanced | Experts share load evenly - good utilization |
| Imbalanced | Some experts overloaded - may indicate training issues |
| Dead experts | Zero load - expert not being used |

---

**📊 Router Probability Heatmap** (per token)

Shows the router's confidence for each expert.

- **X-axis:** Expert index
- **Y-axis:** Token position
- **Color:** Selection probability (darker = higher)

---

**📊 Expert Selection Pattern**

Shows which experts were selected for each token position.

---

**🎯 Expert Influence Analysis**

Advanced metrics using AI domain terminology:

| Metric | Description |
|--------|-------------|
| **Effective Expert Count** | Equivalent number of equally-used experts (2^entropy) |
| **Router Entropy** | Shannon entropy of routing - measures diversity (higher = more uniform) |
| **Load Balance Score** | 1 - coefficient of variation (higher = more balanced) |
| **Gini Coefficient** | Inequality measure (0 = perfect equality, 1 = max inequality) |
| **Auxiliary Loss Proxy** | Approximates Switch Transformer's load balancing loss |

**Expert Categories:**
- 🔥 **Dominant**: Handling >2x average load (capacity bottleneck risk)
- ✓ **Normal**: Within expected range
- 📉 **Underutilized**: <25% of average load (wasted capacity)
- 💀 **Dead**: Zero activations (collapsed routing)

---

**🔤 Expert Token Specialization**

Shows which actual tokens each expert prefers to process:

| Metric | Meaning |
|--------|---------|
| **Most Frequent Tokens** | Tokens this expert processes most often |
| **Specialization Score** | 0-1 measure of focus (1 = highly specialized, 0 = generalist) |
| **Avg Router Prob** | How confident the router is when selecting this expert |

**Specialization Patterns:**
- 🎯 **Specialized Expert** (score >0.7): Focuses on specific token types (e.g., punctuation, numbers, code keywords)
- 📊 **Moderate** (score 0.4-0.7): Some preferences but handles variety
- 🌐 **Generalist** (score <0.4): Processes diverse tokens without strong preferences

**Why this matters:**
- Reveals what each expert "learned" to specialize in
- Shows semantic clustering (similar tokens → same expert)
- Helps identify load balancing issues
- Can show if experts are being utilized effectively
- Indicates potential training instabilities or router collapse"""
    },
    "attention_patterns": {
        "name": "Attention Patterns",
        "short": "Visualize which tokens attend to which other tokens",
        "full": """**Attention patterns** show how each token "attends to" (looks at) other tokens when processing.

**The Heatmap:**
- **Rows (Y-axis):** Query tokens - the tokens that are "looking"
- **Columns (X-axis):** Key tokens - the tokens being "looked at"
- **Color intensity:** Attention weight (brighter = stronger attention)

**Common Patterns:**
| Pattern | Meaning |
|---------|---------|
| **Diagonal** | Tokens attending to themselves |
| **Vertical stripes** | Important tokens that many others attend to |
| **Triangular shape** | Causal mask - tokens can only see past tokens |
| **Horizontal bands** | Some tokens attend broadly to everything |

**Multi-Head Attention:**
Different attention heads learn different patterns:
- **Positional heads:** Attend to nearby tokens (local context)
- **Syntactic heads:** Track grammar (subject-verb agreement)
- **Semantic heads:** Connect related concepts
- **Induction heads:** Pattern matching and copying

**Statistics:**
| Metric | Meaning |
|--------|---------|
| **Entropy** | How spread vs focused attention is (low = focused, high = diffuse) |
| **Self-attention** | % of attention on the token itself |
| **First token attention** | Often high for BOS/system tokens |

**What to Look For:**
- Which tokens are "important" (receive most attention)?
- Do different heads specialize differently?
- How does attention change across layers?
- Are there clear syntactic or semantic patterns?"""
    },
    "knowledge_localization": {
        "name": "Knowledge Localization",
        "short": "Find where facts are stored using causal tracing",
        "full": """**Knowledge Localization** uses causal tracing to identify where factual knowledge is stored in the model.

**The Method (ROME paper):**
1. **Clean run**: Get P(target) for a factual query (e.g., P("Paris") for "The capital of France is")
2. **Corrupted run**: Add noise to subject tokens ("France"), probability drops
3. **Restore runs**: For each (layer, position), restore clean activations and measure recovery

---

**📊 Indirect Effect Heatmap**
- **X-axis (Position):** Token positions in the sequence (* marks subject tokens)
- **Y-axis (Layer):** Model layer index (0 at top)
- **Color:** Recovery percentage (blue=low, yellow/red=high)
- **Star:** Critical site with maximum recovery

---

**Interpreting Results:**
| Pattern | Meaning |
|---------|---------|
| **High recovery at subject position** | Facts stored where subject appears (typical) |
| **High recovery in middle layers** | MLPs store factual associations |
| **Distributed recovery** | Knowledge spread across multiple sites |
| **Low max recovery** | Fact may be compositional or not stored here |

---

**Key Metrics:**
- **Clean P(target):** Model's confidence before corruption
- **Corrupted P(target):** Confidence with subject corrupted (should drop)
- **Max Recovery:** Best recovery achieved by restoring a single site
- **Critical Site:** Layer and position with maximum effect

---

**Why This Matters:**
- Understand where specific facts are "stored"
- Essential for model editing (ROME, MEMIT)
- Reveals how knowledge is encoded
- Helps diagnose factual errors"""
    },
    "generation_replay": {
        "name": "Generation Replay",
        "short": "Step through token generation with playback controls",
        "full": """**Generation Replay** lets you step through token-by-token generation like a video player.

**Features:**
- **Timeline Scrubber:** Navigate to any point in generation
- **Playback Controls:** Step forward/backward through tokens
- **Alternatives View:** See what other tokens the model considered
- **Branching:** Select an alternative and see what would have happened

---

**📊 Alternatives Bar Chart**
- **X-axis (Token):** Top candidate tokens at this step
- **Y-axis (Probability):** Model's confidence for each token
- **Green:** Selected token
- **Gray:** Alternative tokens

---

**Controls:**
| Button | Function |
|--------|----------|
| ⏮ | Jump to first step |
| ⏪ | Previous step |
| ▶ | Play (manual stepping) |
| ⏩ | Next step |
| ⏭ | Jump to last step |
| 🌿 | Create branch from this step |

---

**Branching:**
1. Navigate to a step where you want to explore alternatives
2. Click "Branch" button
3. Select an alternative token from the dropdown
4. Click "Generate Branch" to see what would have happened

---

**Use Cases:**
- Debug unexpected outputs (where did it go wrong?)
- Explore alternative continuations
- Understand model confidence at each step
- Educational tool for understanding LLM generation"""
    }
}


def metric_tooltip(key: str) -> str:
    """Get tooltip text for a metric."""
    if key in METRIC_DEFINITIONS:
        return METRIC_DEFINITIONS[key]["short"]
    return ""


def metric_help(key: str) -> str:
    """Get full help text for a metric."""
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

    # Sidebar - Model Selection
    st.sidebar.header("Model")

    input_mode = st.sidebar.radio(
        "Select model by",
        ["Browse mlx-community", "Enter path manually"],
        horizontal=True,
        label_visibility="collapsed"
    )

    selected_model = None

    if input_mode == "Browse mlx-community":
        if HF_HUB_AVAILABLE:
            with st.sidebar.container():
                st.caption("Browse [mlx-community](https://huggingface.co/mlx-community) models")

                with st.spinner("Loading model list..."):
                    models = fetch_mlx_community_models(limit=300)

                if models:
                    search = st.text_input(
                        "🔍 Filter models",
                        placeholder="e.g., llama, mistral, 4bit...",
                        key="model_search"
                    )

                    if search:
                        search_lower = search.lower()
                        filtered = [m for m in models if search_lower in m['name'].lower()]
                    else:
                        filtered = models

                    if filtered:
                        options = {format_model_option(m): m['id'] for m in filtered[:100]}

                        selected_display = st.selectbox(
                            "Select model",
                            options=list(options.keys()),
                            key="model_select"
                        )

                        if selected_display:
                            selected_model = options[selected_display]
                            st.caption(f"`{selected_model}`")
                    else:
                        st.warning("No models match your search.")
                else:
                    st.warning("Could not load model list.")
        else:
            st.sidebar.warning("Install `huggingface-hub` to browse models")
            input_mode = "Enter path manually"

    if input_mode == "Enter path manually":
        selected_model = st.sidebar.text_input(
            "Model path or HuggingFace ID",
            value="mlx-community/Llama-3.2-1B-Instruct-4bit",
            help="Path to local MLX model or HuggingFace model ID"
        )

    # Load model button
    if selected_model:
        if st.sidebar.button("Load Model", type="primary", use_container_width=True):
            try:
                from mlx_lm import load

                is_hf_model = '/' in selected_model and not selected_model.startswith('/')

                if is_hf_model and HF_HUB_AVAILABLE:
                    progress_container = st.sidebar.container()
                    with progress_container:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        status_text.text("Preparing download...")
                        local_path = download_model_with_progress(
                            selected_model,
                            progress_bar,
                            status_text
                        )

                        status_text.text("Loading model into memory...")
                        model, tokenizer = load(local_path)

                        progress_bar.empty()
                        status_text.empty()
                else:
                    with st.spinner(f"Loading {selected_model}..."):
                        model, tokenizer = load(selected_model)

                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.model_path = selected_model

                # Extract and cache model topology
                st.session_state.topology = get_model_topology(model, tokenizer, selected_model)

                st.sidebar.success("Model loaded!")
                st.rerun()

            except Exception as e:
                st.sidebar.error(f"Error loading model: {e}")

    # Check if model is loaded
    if 'model' not in st.session_state:
        st.info("👈 Browse or enter a model in the sidebar, then click 'Load Model' to get started.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 🔍 Browse Models
            Use the sidebar to browse models from **mlx-community** on HuggingFace.
            - Filter by name (e.g., "llama", "4bit")
            - Sorted by popularity (downloads)
            - Only MLX-compatible models shown
            """)

        with col2:
            st.markdown("""
            ### 📁 Manual Entry
            Or enter a model path directly:
            - `mlx-community/Llama-3.2-1B-Instruct-4bit`
            - `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
            - Local path: `/path/to/mlx/model`
            """)

        st.markdown("---")
        st.markdown("""
        **Supported architectures:** Llama, Mistral, Mixtral, Phi, Qwen, Gemma, StarCoder, Falcon, GPT-2, and more.
        """)
        return

    model = st.session_state.model
    tokenizer = st.session_state.tokenizer

    st.sidebar.success(f"✓ {st.session_state.model_path}")

    # Model Topology Expander
    with st.expander("🏗️ Model Topology", expanded=False):
        if 'topology' in st.session_state:
            topology = st.session_state.topology
            st.markdown(format_topology_display(topology))
        else:
            # Generate topology if not cached (for models loaded before this feature)
            model_path = st.session_state.get('model_path', None)
            topology = get_model_topology(model, tokenizer, model_path)
            st.session_state.topology = topology
            st.markdown(format_topology_display(topology))

    # Probe Settings
    st.sidebar.header("🔬 Probe Settings")

    capture_embeddings = st.sidebar.checkbox("Capture Embeddings", value=True)
    capture_layers = st.sidebar.checkbox("Capture Layer Outputs", value=True)
    capture_ffn = st.sidebar.checkbox("Capture FFN Activations", value=True)
    capture_logits = st.sidebar.checkbox("Capture Logits", value=True)
    capture_attention = st.sidebar.checkbox("Capture Attention Patterns", value=True)
    capture_token_probs = st.sidebar.checkbox("Capture Token Probabilities", value=True)
    capture_residual = st.sidebar.checkbox("Capture Residual Stream", value=True)

    # Max sequence positions
    max_seq_positions = st.sidebar.slider("Max Sequence Positions", 32, 512, 256,
        help="Limit positions captured to reduce memory usage")

    # Layer selection
    st.sidebar.markdown("**Layers to Capture**")
    layer_mode = st.sidebar.radio("Layer capture mode", ["All", "Sample", "Custom"], horizontal=True, label_visibility="collapsed")

    probe_config = ProbeConfig(
        capture_embeddings=capture_embeddings,
        capture_layer_outputs=capture_layers,
        capture_ffn_activations=capture_ffn,
        capture_logits=capture_logits,
        capture_token_probs=capture_token_probs,
        capture_residual_stream=capture_residual,
        capture_attention=capture_attention,
        max_sequence_positions=max_seq_positions
    )

    if layer_mode == "Sample":
        sample_every = st.sidebar.slider("Sample every N layers", 1, 10, 4)
        probe_config.layer_indices = list(range(0, 100, sample_every))
    elif layer_mode == "Custom":
        custom_layers = st.sidebar.text_input("Layer indices (comma-separated)", "0,5,10,15,20")
        try:
            probe_config.layer_indices = [int(x.strip()) for x in custom_layers.split(",")]
        except:
            probe_config.layer_indices = None

    # Generation Settings
    st.sidebar.header("🚀 Generation Settings")
    gen_max_tokens = st.sidebar.slider("Max Tokens", 10, 2000, 1000,
        help="Maximum tokens to generate")
    gen_temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1,
        help="0.0 = greedy (deterministic), higher = more random")

    # AI Intelligence
    st.sidebar.header("🤖 AI Intelligence")
    use_ai_interpretation = st.sidebar.checkbox(
        "Enable AI Interpretation",
        value=True,
        help="Generate AI-powered analysis of probe results"
    )

    # Check for Claude Code availability
    claude_available = detect_claude_code()
    interpreter_choice = "local"  # Default to local LLM

    if use_ai_interpretation:
        if claude_available:
            st.sidebar.success("✓ Claude Code detected")
            # Get previous selection from session state
            prev_choice = st.session_state.get('interpreter_choice', 'local')
            default_idx = 1 if prev_choice == "claude" else 0

            interpreter_choice = st.sidebar.radio(
                "Interpreter",
                ["Local LLM", "Claude (via Claude Code)"],
                index=default_idx,
                help="Choose which AI to use for interpretation"
            )
            interpreter_choice = "claude" if "Claude" in interpreter_choice else "local"
            # Update session state immediately so tabs can use it
            st.session_state.interpreter_choice = interpreter_choice

            # Show warning when Claude is selected
            if interpreter_choice == "claude":
                st.sidebar.warning("⚠️ **Claude API Usage**: Generating interpretations will consume tokens. Rates may apply based on your subscription.")
        else:
            st.sidebar.info("Claude Code not detected - using local LLM")
            interpreter_choice = "local"
            st.session_state.interpreter_choice = interpreter_choice

    # Main Input
    st.header("💬 Input")

    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful assistant.",
        height=80,
        help="System instruction for the model (used when formatting the full prompt)"
    )

    # User prompt (changed default to match AFM7)
    user_prompt = st.text_area(
        "User Message",
        value="Explain the concept of attention in neural networks in one paragraph.",
        height=100,
        help="The text you want to probe"
    )

    # Format the prompt using chat template if available
    def format_prompt_for_model(system: str, user: str, tok) -> str:
        """Format prompt using chat template or fallback."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

        if hasattr(tok, 'apply_chat_template'):
            try:
                return tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                pass

        # Fallback: simple format
        return f"{system}\n\nUser: {user}\n\nAssistant:"

    # Show formatted prompt preview
    with st.expander("View formatted prompt"):
        formatted = format_prompt_for_model(system_prompt, user_prompt, tokenizer)
        st.code(formatted, language=None)

    run_probe = st.button("🚀 Run Inference with Probing", type="primary")

    # Run probe
    if run_probe:
        # Clear previous results and reset all counters - start fresh like a new chat
        keys_to_delete = []
        for key in st.session_state.keys():
            # Keep model, tokenizer, topology (expensive to reload)
            # Delete everything else related to previous probe
            if key not in ['model', 'tokenizer', 'model_path', 'topology']:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del st.session_state[key]

        with st.spinner("Running inference and probing..."):
            try:
                # Format prompt using chat template
                prompt = format_prompt_for_model(system_prompt, user_prompt, tokenizer)

                # Get topology for MoE info
                topology = st.session_state.get('topology', {})
                prober = ModelProber(model, tokenizer, probe_config, topology)
                # Use probe_with_generation_replay to capture timeline for replay
                results = prober.probe_with_generation_replay(
                    prompt,
                    max_tokens=gen_max_tokens,
                    temperature=gen_temperature,
                    capture_config=ReplayCaptureConfig(mode="minimal")
                )
                st.session_state.results = results
                st.session_state.prober = prober
                st.session_state.probe_config = probe_config
                st.session_state.use_ai_interpretation = use_ai_interpretation

                # Store settings for on-demand interpretation generation
                st.session_state.use_ai_interpretation = use_ai_interpretation
                st.session_state.interpreter_choice = interpreter_choice
                st.session_state.inference_complete = True

            except Exception as e:
                st.error(f"Error during probing: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Display results
    if 'results' not in st.session_state:
        return

    results = st.session_state.results
    prober = st.session_state.prober

    # Inference complete message
    if st.session_state.get('inference_complete'):
        st.success("✅ Inference complete!")

    # Generated Output section
    st.header("📝 Generated Output")
    if results.generated_text:
        # Check if there's reasoning to display separately
        if results.reasoning_text:
            # Show reasoning in a collapsible section
            with st.expander("🧠 Reasoning / Analysis", expanded=True):
                st.text_area(
                    "Model's internal reasoning",
                    value=results.reasoning_text,
                    height=150,
                    disabled=True,
                    label_visibility="collapsed"
                )

            # Show the final answer
            st.subheader("Response")
            if results.answer_text and not results.answer_text.startswith("(Response generation"):
                st.text_area(
                    "Final answer",
                    value=results.answer_text,
                    height=150,
                    disabled=True,
                    label_visibility="collapsed"
                )
            else:
                st.warning("⚠️ Response generation incomplete - try increasing 'Max tokens to generate' in the sidebar")
        else:
            # No reasoning detected, show raw output
            st.text_area("Response", value=results.generated_text, height=150, disabled=True)

        st.caption(f"Generated {len(results.generated_tokens)} tokens")
    else:
        st.info("No text generated yet.")

    # Quick stats
    st.markdown("---")
    st.header("📊 Visualizations")
    col1, col2, col3, col4 = st.columns(4)
    total_tokens = len(results.input_tokens) + len(results.generated_tokens) if results.generated_tokens else len(results.input_tokens)
    col1.metric("Total Tokens", total_tokens, delta=f"+{len(results.generated_tokens)} generated" if results.generated_tokens else None)
    col2.metric("Layers Probed", len(results.layer_outputs))
    if results.top_k_tokens:
        top_token_text = results.top_k_tokens[0][2]
        if not top_token_text or top_token_text.isspace():
            top_token_text = f"<{results.top_k_tokens[0][0]}>"
        col3.metric("Top Prediction", top_token_text[:15])

        # Show confidence or logit depending on distribution
        top_prob = results.top_k_tokens[0][1]
        if top_prob > 0.99 and results.logits is not None:
            # Distribution is peaked - show logit value instead (more informative)
            top_token_id = results.top_k_tokens[0][0]
            if 0 <= top_token_id < len(results.logits):
                top_logit = float(results.logits[top_token_id])
                col4.metric("Top Logit", f"{top_logit:.1f}")
            else:
                col4.metric("Confidence", ">99%")
        else:
            col4.metric("Confidence", f"{top_prob:.1%}")

    # Tabs - matching AFM7 structure
    # Build tab list - add MoE tab if model is MoE
    tab_names = [
        "Layer Activations",
        "FFN Analysis",
        "Token Probabilities",
        "Tokens",
        "Embeddings",
        "Logits",
        "Layer Similarity",
        "Residual Stream",
        "Attention"
    ]

    if results.is_moe:
        tab_names.append("MoE Routing")

    # Always add new feature tabs
    tab_names.append("Knowledge Localization")
    tab_names.append("Generation Replay")

    tabs = st.tabs(tab_names)

    # Unpack tabs (handle both MoE and non-MoE cases)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab_attn = tabs[:9]
    tab_idx = 9
    tab_moe = tabs[tab_idx] if results.is_moe else None
    if results.is_moe:
        tab_idx += 1
    tab_knowledge = tabs[tab_idx]
    tab_idx += 1
    tab_replay = tabs[tab_idx]

    # Get interpretation settings from session state
    ai_interpret = st.session_state.get('use_ai_interpretation', False)
    interp_choice = st.session_state.get('interpreter_choice', 'local')

    # Tab 1: Layer Activations
    with tab1:
        st.subheader("Layer-wise Activation Analysis")

        with st.expander("ℹ️ What is this? (click to learn)", expanded=False):
            st.markdown(metric_help("activation_norm"))

        if results.layer_outputs:
            fig = plot_layer_norms(results)
            st.plotly_chart(fig, use_container_width=True)

            # Individual layer histogram
            available_layers = sorted(results.layer_outputs.keys())
            if available_layers:
                selected_layer = st.selectbox(
                    "Select layer for distribution",
                    available_layers
                )
                # Plot distribution for selected layer
                layer_data = results.layer_outputs[selected_layer]
                flat_data = np.array(layer_data).flatten()
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=flat_data,
                    nbinsx=50,
                    name=f"Layer {selected_layer}"
                ))
                fig_dist.update_layout(
                    title=f"Activation Distribution - Layer {selected_layer}",
                    xaxis_title="Activation Value",
                    yaxis_title="Count",
                    height=300
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            # AI Interpretation (cached)
            if ai_interpret:
                interp_label = "Claude" if interp_choice == "claude" else "Local LLM"
                with st.expander("🧠 AI Interpretation", expanded=True):
                    render_cached_interpretation(
                        f"layer_activation_{interp_choice}",
                        results,
                        lambda: get_layer_activation_interpretation(
                            results, model, tokenizer, use_ai=True, interpreter=interp_choice
                        ),
                        interp_label
                    )
        else:
            st.info("Enable 'Capture Layer Outputs' to see this visualization.")

    # Tab 2: FFN Analysis
    with tab2:
        st.subheader("FFN Gate Pattern Analysis")

        with st.expander("ℹ️ What is this? (click to learn)", expanded=False):
            st.markdown(metric_help("gate_sparsity"))

        if results.ffn_activations:
            fig = plot_ffn_analysis(results)
            st.plotly_chart(fig, use_container_width=True)

            # AI Interpretation
            if ai_interpret:
                interp_label = "Claude" if interp_choice == "claude" else "Local LLM"
                with st.expander("🧠 AI Interpretation", expanded=True):
                    render_cached_interpretation(
                        f"ffn_analysis_{interp_choice}",
                        results,
                        lambda: get_ffn_interpretation(
                            results, model, tokenizer, use_ai=True, interpreter=interp_choice
                        ),
                        interp_label
                    )
        else:
            st.info("Enable 'Capture FFN Activations' to see this visualization.")

    # Tab 3: Token Probabilities
    with tab3:
        st.subheader("Token Probabilities by Position")

        with st.expander("ℹ️ What is this? (click to learn)", expanded=False):
            st.markdown("""
**Token Probabilities** shows what the model predicted at each position:

- **Input tokens**: The tokens you provided (no predictions - these were given)
- **Generated tokens**: For each position, shows:
  - The **chosen token** and its probability
  - **Top alternatives** the model considered
  - Whether the choice was **confident** (high probability) or **uncertain** (close alternatives)

**How to read:**
- 🟢 High confidence (>80%): Model was very sure
- 🟡 Medium confidence (30-80%): Reasonable certainty
- 🔴 Low confidence (<30%): Close call between alternatives
            """)

        # Detect sections for grouping
        sections = detect_prompt_sections(results, tokenizer)
        input_len = len(results.input_tokens) if results.input_tokens else 0
        gen_len = len(results.generated_tokens) if results.generated_tokens else 0

        st.caption(f"Total: {input_len} input tokens + {gen_len} generated tokens = {input_len + gen_len} total")

        # Group sections into Input and Response categories
        input_sections = {}
        response_sections = {}

        for name, bounds in sections.items():
            if any(k in name for k in ["System", "User", "Input"]):
                input_sections[name] = bounds
            elif any(k in name for k in ["Response", "Reasoning", "Final", "🤖", "🧠", "✅"]):
                response_sections[name] = bounds
            else:
                if bounds[0] >= input_len:
                    response_sections[name] = bounds
                else:
                    input_sections[name] = bounds

        # Calculate totals
        input_total = sum(e - s for s, e in input_sections.values()) if input_sections else 0
        response_total = sum(e - s for s, e in response_sections.values()) if response_sections else 0

        # === INPUT SEQUENCE ===
        if input_sections:
            with st.expander(f"📥 **Input Sequence** ({input_total} tokens) - no predictions", expanded=False):
                st.caption("Input tokens are provided by you, not predicted by the model.")

                for sub_name, (start, end) in input_sections.items():
                    token_count = end - start
                    st.markdown(f"**{sub_name}** ({token_count} tokens)")

                    # Show tokens in this section
                    section_data = []
                    for pos in range(start, min(end, input_len)):
                        if pos < len(results.input_tokens):
                            tok_id = results.input_tokens[pos]
                            try:
                                tok_text = tokenizer.decode([tok_id])
                                display_text = repr(tok_text)[1:-1]
                            except:
                                display_text = f"[{tok_id}]"
                            section_data.append({
                                "Pos": pos,
                                "Token": display_text,
                                "ID": tok_id
                            })

                    if section_data:
                        # Limit display with show more button
                        max_show = 30
                        show_all_key = f"show_all_input_{sub_name}"
                        show_all = st.session_state.get(show_all_key, False)

                        if len(section_data) > max_show and not show_all:
                            st.dataframe(pd.DataFrame(section_data[:max_show]), hide_index=True, use_container_width=True)
                            remaining = len(section_data) - max_show
                            if st.button(f"📋 Load {remaining} more tokens", key=f"load_more_input_{sub_name}"):
                                st.session_state[show_all_key] = True
                                st.rerun()
                        elif len(section_data) > max_show and show_all:
                            st.dataframe(pd.DataFrame(section_data), hide_index=True, use_container_width=True)
                            if st.button(f"📋 Show less", key=f"show_less_input_{sub_name}"):
                                st.session_state[show_all_key] = False
                                st.rerun()
                        else:
                            st.dataframe(pd.DataFrame(section_data), hide_index=True, use_container_width=True)
                    st.markdown("---")

        # === RESPONSE (Generated tokens with alternatives) ===
        if response_sections and results.per_token_alternatives:
            with st.expander(f"🤖 **Response** ({response_total} tokens) - with prediction alternatives", expanded=True):
                st.caption("For each generated token: chosen token, probability, and what alternatives were considered.")

                # Helper to format probability
                def format_prob(p):
                    if p >= 0.8:
                        return f"🟢 {p:.1%}"
                    elif p >= 0.3:
                        return f"🟡 {p:.1%}"
                    else:
                        return f"🔴 {p:.1%}"

                for sub_name, (start, end) in response_sections.items():
                    # Only process positions in the generated range
                    gen_start_pos = max(start, input_len)
                    gen_end_pos = min(end, input_len + gen_len)

                    if gen_end_pos <= gen_start_pos:
                        continue

                    token_count = gen_end_pos - gen_start_pos
                    if "Reasoning" in sub_name:
                        st.markdown(f"🧠 **Reasoning Loop** ({token_count} tokens)")
                    elif "Final" in sub_name:
                        st.markdown(f"✅ **Final Response** ({token_count} tokens)")
                    else:
                        st.markdown(f"**{sub_name}** ({token_count} tokens)")

                    # Show tokens with alternatives
                    section_data = []
                    for pos in range(gen_start_pos, gen_end_pos):
                        gen_idx = pos - input_len  # Index into generated_tokens and per_token_alternatives

                        if gen_idx < 0 or gen_idx >= len(results.generated_tokens):
                            continue
                        if gen_idx >= len(results.per_token_alternatives):
                            continue

                        # Get the chosen token
                        chosen_id = results.generated_tokens[gen_idx]
                        try:
                            chosen_text = tokenizer.decode([chosen_id])
                            chosen_display = repr(chosen_text)[1:-1]
                        except:
                            chosen_display = f"[{chosen_id}]"

                        # Get alternatives (already sorted descending by probability from argsort[-10:][::-1])
                        # alternatives[0] = highest probability, alternatives[-1] = lowest of top-10
                        alternatives = results.per_token_alternatives[gen_idx]

                        # Find the chosen token's probability
                        chosen_prob = 0.0
                        chosen_rank = "?"
                        for rank_idx, (alt_id, alt_prob, alt_text) in enumerate(alternatives):
                            if alt_id == chosen_id:
                                chosen_prob = alt_prob
                                chosen_rank = rank_idx + 1  # 1-indexed rank
                                break

                        # Get top-3 alternatives for display
                        # alternatives is already sorted: [0]=highest, [1]=second, [2]=third
                        top3_str = ", ".join([
                            f"{alt[2][:8]}({alt[1]:.0%})"
                            for alt in alternatives[:3]
                        ])

                        section_data.append({
                            "Pos": pos,
                            "Chosen": chosen_display[:15],
                            "Prob": format_prob(chosen_prob),
                            "Rank": f"#{chosen_rank}" if isinstance(chosen_rank, int) else chosen_rank,
                            "Top 3 Alternatives": top3_str
                        })

                    if section_data:
                        max_show = 30
                        show_key = f"show_all_probs_{sub_name}"
                        if show_key not in st.session_state:
                            st.session_state[show_key] = False

                        if len(section_data) > max_show and not st.session_state[show_key]:
                            st.dataframe(pd.DataFrame(section_data[:max_show]), hide_index=True, use_container_width=True)
                            if st.button(f"Show all {len(section_data)} tokens...", key=f"btn_{show_key}"):
                                st.session_state[show_key] = True
                                st.rerun()
                        else:
                            st.dataframe(pd.DataFrame(section_data), hide_index=True, use_container_width=True)
                            if len(section_data) > max_show:
                                if st.button("Show less", key=f"btn_less_{show_key}"):
                                    st.session_state[show_key] = False
                                    st.rerun()
                    st.markdown("---")

        elif not results.per_token_alternatives:
            st.info("No per-token alternatives captured. Run generation to see token-by-token predictions.")

        # === Final Prediction (what comes next) ===
        if results.top_k_tokens:
            with st.expander("🔮 **Next Token Prediction** (what would come after the response)", expanded=False):
                st.caption("If generation continued, what token would the model predict next?")

                # Check if distribution is peaked
                top_prob = results.top_k_tokens[0][1]
                if top_prob > 0.99:
                    st.info("⚠️ **Peaked Distribution**: Model is >99% confident. Showing logit differences instead of probabilities.")

                fig = plot_token_probabilities(results)
                st.plotly_chart(fig, use_container_width=True)

                # Token probability table
                if top_prob > 0.99 and results.logits is not None:
                    table_data = []
                    for t in results.top_k_tokens:
                        token_id = t[0]
                        token_text = t[2]
                        if 0 <= token_id < len(results.logits):
                            logit_val = float(results.logits[token_id])
                        else:
                            logit_val = 0.0
                        table_data.append((token_id, logit_val, token_text))
                    df = pd.DataFrame(table_data, columns=["Token ID", "Logit", "Token"])
                    df["Logit"] = df["Logit"].apply(lambda x: f"{x:.2f}")
                else:
                    df = pd.DataFrame(results.top_k_tokens, columns=["Token ID", "Probability", "Token"])
                    def format_prob_table(x):
                        if x >= 0.0001:
                            return f"{x:.4f} ({x*100:.2f}%)"
                        elif x > 0:
                            return f"{x:.2e}"
                        else:
                            return "~0"
                    df["Probability"] = df["Probability"].apply(format_prob_table)
                st.dataframe(df, hide_index=True)

        # Verification panel
        if results.per_token_alternatives:
            with st.expander("🔍 Verify Token Probability Data (Debug)", expanded=False):
                st.caption("Sanity check: verify alternatives are sorted correctly (highest probability first)")

                # Run sanity check
                violations = []
                for gen_idx, alternatives in enumerate(results.per_token_alternatives):
                    if len(alternatives) < 2:
                        continue
                    # Check that probabilities are in descending order
                    # alternatives[0] should have highest prob, alternatives[1] second highest, etc.
                    for i in range(len(alternatives) - 1):
                        if alternatives[i][1] < alternatives[i + 1][1]:
                            violations.append((gen_idx, i, alternatives[i], alternatives[i + 1]))

                if violations:
                    st.error(f"⚠️ Found {len(violations)} ordering violations!")
                    for v in violations[:5]:
                        st.write(f"Gen token {v[0]}: alt[{v[1]}] ({v[2][2]}, {v[2][1]:.2%}) < alt[{v[1]+1}] ({v[3][2]}, {v[3][1]:.2%})")
                else:
                    st.success(f"✅ All {len(results.per_token_alternatives)} positions pass: alternatives sorted correctly (highest prob first)")

                # Inspect single position
                st.markdown("---")
                st.markdown("**Inspect single generated token:**")

                if results.generated_tokens:
                    inspect_idx = st.slider("Generated token index", 0, len(results.generated_tokens) - 1, 0, key="inspect_gen_idx")

                    # Show chosen token
                    chosen_id = results.generated_tokens[inspect_idx]
                    try:
                        chosen_text = tokenizer.decode([chosen_id])
                    except:
                        chosen_text = f"[{chosen_id}]"

                    st.write(f"**Position {input_len + inspect_idx}** (gen index {inspect_idx})")
                    st.write(f"**Chosen token:** `{chosen_text}` (ID: {chosen_id})")

                    if inspect_idx < len(results.per_token_alternatives):
                        alternatives = results.per_token_alternatives[inspect_idx]
                        st.write(f"**Top {len(alternatives)} alternatives** (should be descending by probability):")

                        alt_df = pd.DataFrame([
                            {
                                "Rank": i + 1,
                                "Token": alt[2][:20],
                                "ID": alt[0],
                                "Prob": f"{alt[1]:.4%}",
                                "Chosen": "✅" if alt[0] == chosen_id else ""
                            }
                            for i, alt in enumerate(alternatives)
                        ])
                        st.dataframe(alt_df, hide_index=True, use_container_width=True)

                        # Verify chosen is in alternatives
                        chosen_in_alts = any(alt[0] == chosen_id for alt in alternatives)
                        if chosen_in_alts:
                            st.success("✅ Chosen token found in alternatives list")
                        else:
                            st.warning("⚠️ Chosen token not in top-10 alternatives (may happen with sampling)")

        # AI Interpretation
        if ai_interpret and results.top_k_tokens:
            interp_label = "Claude" if interp_choice == "claude" else "Local LLM"
            with st.expander("🧠 AI Interpretation", expanded=False):
                render_cached_interpretation(
                    f"token_prob_{interp_choice}",
                    results,
                    lambda: get_token_prob_interpretation(
                        results, model, tokenizer, use_ai=True, interpreter=interp_choice
                    ),
                    interp_label
                )

    # Tab 4: Tokens
    with tab4:
        st.subheader("Token Analysis")

        # Input tokens with decoded text
        st.markdown("**Input Tokens**")
        if results.input_tokens:
            input_token_data = []
            for i, tok_id in enumerate(results.input_tokens):
                tok_text = tokenizer.decode([tok_id])
                # Handle special characters for display
                display_text = repr(tok_text)[1:-1]  # Remove outer quotes
                input_token_data.append({
                    "Position": i,
                    "Token ID": tok_id,
                    "Text": tok_text,
                    "Display": display_text
                })
            input_df = pd.DataFrame(input_token_data)
            st.dataframe(input_df, hide_index=True, use_container_width=True)

            # MoE Expert Routing Inspector for input tokens
            if results.is_moe and results.moe_router_outputs:
                with st.expander("🔀 MoE Expert Routing Inspector (Input Tokens)", expanded=False):
                    st.caption("Select a token position to see which experts processed it")

                    # Position selector
                    max_pos = len(results.input_tokens) - 1
                    selected_pos = st.slider("Token Position", 0, max_pos, 0, key="input_token_moe_pos")

                    # Show token info
                    tok_id = results.input_tokens[selected_pos]
                    tok_text = tokenizer.decode([tok_id])
                    st.markdown(f"**Position {selected_pos}:** `{tok_text}` (ID: {tok_id})")

                    # Show routing for each MoE layer
                    moe_layers = sorted(results.moe_router_outputs.keys())

                    # Layer selector
                    if len(moe_layers) > 1:
                        selected_layer = st.selectbox(
                            "MoE Layer",
                            moe_layers,
                            format_func=lambda x: f"Layer {x}",
                            key="input_token_moe_layer"
                        )
                    else:
                        selected_layer = moe_layers[0]

                    router_data = results.moe_router_outputs[selected_layer]
                    probs = router_data["probs"][0]  # (seq, num_experts)
                    selected_experts = router_data["selected"][0]  # (seq, top_k)

                    if selected_pos < len(probs):
                        expert_probs = probs[selected_pos]
                        expert_selected = selected_experts[selected_pos]
                        top_k = len(expert_selected)
                        num_experts = len(expert_probs)

                        # Create expert routing visualization
                        col_info, col_chart = st.columns([1, 2])

                        with col_info:
                            st.markdown(f"**Top-{top_k} Experts Selected:**")
                            expert_data = []
                            # Note: argsort returns ascending order, so expert_selected[-1] is highest prob (Top-1)
                            for rank in range(1, top_k + 1):
                                sel_idx = top_k - rank  # Top-1 → last index
                                exp_id = int(expert_selected[sel_idx])
                                exp_prob = float(expert_probs[exp_id])
                                expert_data.append({
                                    "Rank": f"Top-{rank}",
                                    "Expert": f"E{exp_id}",
                                    "Weight": f"{exp_prob:.2%}",
                                })
                            expert_df = pd.DataFrame(expert_data)
                            st.dataframe(expert_df, hide_index=True)

                            # Routing summary - Top-1 is at index -1 (last)
                            dominant = int(expert_selected[-1])
                            dominant_w = float(expert_probs[dominant])
                            if dominant_w > 0.5:
                                st.info(f"⚡ **Dominant routing** to Expert {dominant}")
                            elif dominant_w > 0.3:
                                st.info(f"🔄 **Balanced routing** across experts")
                            else:
                                st.info(f"🌐 **Distributed routing** - no clear dominant")

                        with col_chart:
                            # Bar chart of all expert weights
                            fig = go.Figure(go.Bar(
                                x=list(range(num_experts)),
                                y=[float(expert_probs[e]) * 100 for e in range(num_experts)],
                                marker_color=['#EF553B' if e in expert_selected else '#636EFA'
                                             for e in range(num_experts)],
                                text=[f"E{e}" if e in expert_selected else "" for e in range(num_experts)],
                                textposition='outside'
                            ))
                            fig.update_layout(
                                title=f"Router Weights - All {num_experts} Experts",
                                xaxis_title="Expert ID",
                                yaxis_title="Weight (%)",
                                height=250,
                                margin=dict(l=40, r=20, t=40, b=40)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("🔴 Red = Selected experts | 🔵 Blue = Not selected")
                    else:
                        st.warning(f"Position {selected_pos} not captured in router outputs")
        else:
            st.info("No input tokens captured.")

        st.markdown("---")

        # Generated tokens section - matching input tokens style
        st.markdown("**Generated Tokens**")
        if results.generated_tokens:
            # Check if we have per-token alternatives
            has_alternatives = (hasattr(results, 'per_token_alternatives') and
                                results.per_token_alternatives and
                                len(results.per_token_alternatives) > 0)

            # Create table matching input tokens style
            gen_token_data = []
            for i, tok_id in enumerate(results.generated_tokens):
                tok_text = tokenizer.decode([tok_id])
                display_text = repr(tok_text)[1:-1]

                # Get probability of selected token
                selected_prob = "N/A"
                if has_alternatives and i < len(results.per_token_alternatives):
                    alts = results.per_token_alternatives[i]
                    for alt_id, alt_prob, _ in alts:
                        if alt_id == tok_id:
                            selected_prob = f"{alt_prob:.2%}"
                            break

                gen_token_data.append({
                    "Position": i,
                    "Seq Pos": len(results.input_tokens) + i,  # Position in full sequence
                    "Token ID": tok_id,
                    "Text": tok_text,
                    "Display": display_text,
                    "Prob": selected_prob
                })
            gen_df = pd.DataFrame(gen_token_data)
            st.dataframe(gen_df, hide_index=True, use_container_width=True)

            # MoE Expert Routing Inspector for generated tokens
            if results.is_moe and results.moe_router_outputs:
                with st.expander("🔀 MoE Expert Routing Inspector (Generated Tokens)", expanded=False):
                    st.caption("Select a generated token position to see which experts processed it")

                    # Position selector
                    max_gen_pos = len(results.generated_tokens) - 1
                    selected_gen_pos = st.slider("Generated Token Position", 0, max_gen_pos, 0, key="gen_token_moe_pos")

                    # Show token info
                    tok_id = results.generated_tokens[selected_gen_pos]
                    tok_text = tokenizer.decode([tok_id])
                    seq_pos = len(results.input_tokens) + selected_gen_pos
                    st.markdown(f"**Generated Position {selected_gen_pos}** (Sequence Position {seq_pos}): `{tok_text}` (ID: {tok_id})")

                    # Show routing for each MoE layer
                    moe_layers = sorted(results.moe_router_outputs.keys())

                    # Layer selector
                    if len(moe_layers) > 1:
                        selected_layer = st.selectbox(
                            "MoE Layer",
                            moe_layers,
                            format_func=lambda x: f"Layer {x}",
                            key="gen_token_moe_layer"
                        )
                    else:
                        selected_layer = moe_layers[0]

                    router_data = results.moe_router_outputs[selected_layer]
                    probs = router_data["probs"][0]  # (seq, num_experts)
                    selected_experts = router_data["selected"][0]  # (seq, top_k)

                    if seq_pos < len(probs):
                        expert_probs = probs[seq_pos]
                        expert_selected = selected_experts[seq_pos]
                        top_k = len(expert_selected)
                        num_experts = len(expert_probs)

                        # Create expert routing visualization
                        col_info, col_chart = st.columns([1, 2])

                        with col_info:
                            st.markdown(f"**Top-{top_k} Experts Selected:**")
                            expert_data = []
                            # Note: argsort returns ascending order, so expert_selected[-1] is highest prob (Top-1)
                            for rank in range(1, top_k + 1):
                                sel_idx = top_k - rank  # Top-1 → last index
                                exp_id = int(expert_selected[sel_idx])
                                exp_prob = float(expert_probs[exp_id])
                                expert_data.append({
                                    "Rank": f"Top-{rank}",
                                    "Expert": f"E{exp_id}",
                                    "Weight": f"{exp_prob:.2%}",
                                })
                            expert_df = pd.DataFrame(expert_data)
                            st.dataframe(expert_df, hide_index=True)

                            # Routing summary - Top-1 is at index -1 (last)
                            dominant = int(expert_selected[-1])
                            dominant_w = float(expert_probs[dominant])
                            if dominant_w > 0.5:
                                st.info(f"⚡ **Dominant routing** to Expert {dominant}")
                            elif dominant_w > 0.3:
                                st.info(f"🔄 **Balanced routing** across experts")
                            else:
                                st.info(f"🌐 **Distributed routing** - no clear dominant")

                        with col_chart:
                            # Bar chart of all expert weights
                            fig = go.Figure(go.Bar(
                                x=list(range(num_experts)),
                                y=[float(expert_probs[e]) * 100 for e in range(num_experts)],
                                marker_color=['#EF553B' if e in expert_selected else '#636EFA'
                                             for e in range(num_experts)],
                                text=[f"E{e}" if e in expert_selected else "" for e in range(num_experts)],
                                textposition='outside'
                            ))
                            fig.update_layout(
                                title=f"Router Weights - All {num_experts} Experts",
                                xaxis_title="Expert ID",
                                yaxis_title="Weight (%)",
                                height=250,
                                margin=dict(l=40, r=20, t=40, b=40)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("🔴 Red = Selected experts | 🔵 Blue = Not selected")
                    else:
                        st.warning(f"Position {seq_pos} not captured in router outputs (captured up to position {len(probs)-1})")

            # Token alternatives inspector
            if has_alternatives:
                with st.expander("🎯 Token Alternatives Inspector", expanded=False):
                    st.caption("See what other tokens the model considered at each generation step")

                    alt_pos = st.slider("Generated Token Position", 0, max_gen_pos, 0, key="gen_token_alt_pos")

                    tok_id = results.generated_tokens[alt_pos]
                    tok_text = tokenizer.decode([tok_id])
                    st.markdown(f"**Position {alt_pos}:** Selected `{tok_text}` (ID: {tok_id})")

                    if alt_pos < len(results.per_token_alternatives):
                        alts = results.per_token_alternatives[alt_pos]
                        alt_data = []
                        for rank, (alt_id, alt_prob, alt_text) in enumerate(alts):
                            is_selected = "✓" if alt_id == tok_id else ""
                            # Confidence indicator
                            if alt_prob > 0.5:
                                conf = "🟢"
                            elif alt_prob > 0.1:
                                conf = "🟡"
                            else:
                                conf = "🔴"
                            alt_data.append({
                                "Rank": rank + 1,
                                "Sel": is_selected,
                                "Conf": conf,
                                "Token ID": alt_id,
                                "Token": alt_text,
                                "Probability": f"{alt_prob:.4f}",
                                "Percent": f"{alt_prob:.2%}"
                            })
                        alt_df = pd.DataFrame(alt_data)
                        st.dataframe(alt_df, hide_index=True, use_container_width=True)

                        # Certainty analysis
                        top_prob = alts[0][1] if alts else 0
                        if top_prob > 0.9:
                            st.success(f"🎯 **High certainty** - model was {top_prob:.0%} confident")
                        elif top_prob > 0.5:
                            st.info(f"✓ **Moderate certainty** - {top_prob:.0%} confident in top choice")
                        elif top_prob > 0.2:
                            st.warning(f"⚖️ **Uncertain** - only {top_prob:.0%} confident, multiple alternatives")
                        else:
                            st.error(f"❓ **Very uncertain** - {top_prob:.0%} confidence, high entropy")

            # Detailed per-token output with alternatives (original format)
            st.markdown("---")
            st.markdown("**Generated Tokens Detail** *(with top-10 alternatives at each step)*")

            for i, tok_id in enumerate(results.generated_tokens):
                tok_text = tokenizer.decode([tok_id])
                display_text = repr(tok_text)[1:-1]

                # Get probability of selected token
                selected_prob = "N/A"
                if has_alternatives and i < len(results.per_token_alternatives):
                    alts = results.per_token_alternatives[i]
                    for alt_id, alt_prob, _ in alts:
                        if alt_id == tok_id:
                            selected_prob = f"{alt_prob:.2%}"
                            break

                # Create row for each token
                col1, col2, col3, col4 = st.columns([1, 2, 3, 2])
                with col1:
                    st.markdown(f"**{i}**")
                with col2:
                    st.markdown(f"`{tok_id}`")
                with col3:
                    st.markdown(f"**{tok_text}**")
                with col4:
                    st.markdown(f"*{selected_prob}*")

                # Show alternatives in expander
                if has_alternatives and i < len(results.per_token_alternatives):
                    alts = results.per_token_alternatives[i]
                    with st.expander(f"🎯 Top 10 alternatives at position {i}", expanded=False):
                        alt_data = []
                        for rank, (alt_id, alt_prob, alt_text) in enumerate(alts):
                            is_selected = "✓" if alt_id == tok_id else ""
                            alt_data.append({
                                "Rank": rank + 1,
                                "Selected": is_selected,
                                "Token ID": alt_id,
                                "Token": alt_text,
                                "Probability": f"{alt_prob:.4f}",
                                "Percent": f"{alt_prob:.2%}"
                            })
                        alt_df = pd.DataFrame(alt_data)
                        st.dataframe(alt_df, hide_index=True, use_container_width=True)

                        # MoE Expert Routing Info for this generated token
                        if results.is_moe and results.moe_router_outputs:
                            moe_layers = sorted(results.moe_router_outputs.keys())
                            first_layer = moe_layers[0]
                            router_data = results.moe_router_outputs[first_layer]
                            captured_len = len(router_data["probs"][0])

                            # Position for this generated token
                            seq_pos = len(results.input_tokens) + i

                            if seq_pos < captured_len:
                                # We have actual captured data for this generation step!
                                st.markdown("---")
                                st.markdown(f"**🔀 MoE Expert Routing** *(position {seq_pos})*")

                                # Show first and last MoE layers
                                if len(moe_layers) > 4:
                                    layer_options = [moe_layers[0], moe_layers[-1]]
                                    layer_labels = {moe_layers[0]: f"Layer {moe_layers[0]} (First)",
                                                   moe_layers[-1]: f"Layer {moe_layers[-1]} (Last)"}
                                else:
                                    layer_options = moe_layers
                                    layer_labels = {l: f"Layer {l}" for l in moe_layers}

                                for layer_idx in layer_options:
                                    router_data = results.moe_router_outputs[layer_idx]
                                    probs = router_data["probs"][0]
                                    selected = router_data["selected"][0]

                                    if seq_pos < len(probs):
                                        expert_probs = probs[seq_pos]
                                        expert_selected = selected[seq_pos]
                                        top_k = len(expert_selected)

                                        # Create expert routing table
                                        expert_data = []
                                        # Find max weight for relative scaling (Top-1 is at index -1)
                                        max_weight = float(expert_probs[int(expert_selected[-1])])
                                        # Note: argsort returns ascending order, so expert_selected[-1] is highest prob (Top-1)
                                        for rank in range(1, top_k + 1):
                                            sel_idx = top_k - rank  # Top-1 → last index
                                            exp_id = int(expert_selected[sel_idx])
                                            exp_prob = float(expert_probs[exp_id])
                                            # Scale bars relative to max (max=10 bars) with minimum of 1
                                            bar_count = max(1, int((exp_prob / max_weight) * 10)) if max_weight > 0 else 1
                                            expert_data.append({
                                                "Rank": f"Top-{rank}",
                                                "Expert": f"E{exp_id}",
                                                "Weight": f"{exp_prob:.1%}",
                                                "Bar": "█" * bar_count
                                            })

                                        st.markdown(f"**{layer_labels[layer_idx]}:**")
                                        expert_df = pd.DataFrame(expert_data)
                                        st.dataframe(expert_df, hide_index=True, use_container_width=True)

                                        # Routing summary - Top-1 is at index -1 (last)
                                        dominant = int(expert_selected[-1])
                                        dominant_w = float(expert_probs[dominant])
                                        if dominant_w > 0.5:
                                            st.caption(f"⚡ Dominant: E{dominant} ({dominant_w:.0%})")
                                        elif dominant_w > 0.3:
                                            st.caption(f"🔄 Balanced routing")
                                        else:
                                            st.caption(f"🌐 Distributed routing")
                            else:
                                # Fallback to context display
                                st.markdown("---")
                                st.caption(f"💡 Router data not captured for position {seq_pos}")

            st.markdown("---")
            st.markdown(f"**Total generated:** {len(results.generated_tokens)} tokens")
        else:
            st.info("No generated tokens yet.")

    # Tab 5: Embeddings
    with tab5:
        st.subheader("Token Embedding Analysis")

        with st.expander("ℹ️ What is this? (click to learn)", expanded=False):
            st.markdown(metric_help("embedding"))

        if results.embeddings is not None:
            try:
                fig = plot_embeddings_pca(results, tokenizer)
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.warning("Install scikit-learn for PCA visualization: pip install scikit-learn")
            except Exception as e:
                st.error(f"PCA visualization failed: {e}")

            # Embedding statistics
            emb = results.embeddings
            if len(emb.shape) == 3:
                emb = emb[0]
            st.markdown(f"""
            **Embedding Statistics:**
            - Shape: {emb.shape}
            - Mean: {emb.mean():.4f}
            - Std: {emb.std():.4f}
            - Min: {emb.min():.4f}
            - Max: {emb.max():.4f}
            """)

            # AI Interpretation
            if ai_interpret:
                interp_label = "Claude" if interp_choice == "claude" else "Local LLM"
                with st.expander("🧠 AI Interpretation", expanded=True):
                    render_cached_interpretation(
                        f"embeddings_{interp_choice}",
                        results,
                        lambda: get_embedding_interpretation(
                            results, model, tokenizer, use_ai=True, interpreter=interp_choice
                        ),
                        interp_label
                    )
        else:
            st.info("Enable 'Capture Embeddings' to see this visualization.")

    # Tab 6: Logits
    with tab6:
        st.subheader("Logits Distribution")

        with st.expander("ℹ️ What is this? (click to learn)", expanded=False):
            st.markdown(metric_help("logits"))

        if results.logits is not None:
            fig = plot_logits_distribution(results, tokenizer)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            **Logits Statistics:**
            - Vocabulary Size: {len(results.logits):,}
            - Mean: {results.logits.mean():.4f}
            - Std: {results.logits.std():.4f}
            - Min: {results.logits.min():.4f}
            - Max: {results.logits.max():.4f}
            """)

            # AI Interpretation
            if ai_interpret:
                interp_label = "Claude" if interp_choice == "claude" else "Local LLM"
                with st.expander("🧠 AI Interpretation", expanded=True):
                    render_cached_interpretation(
                        f"logits_{interp_choice}",
                        results,
                        lambda: get_logits_interpretation(
                            results, model, tokenizer, use_ai=True, interpreter=interp_choice
                        ),
                        interp_label
                    )
        else:
            st.info("Enable 'Capture Logits' to see this visualization.")

    # Tab 7: Layer Similarity
    with tab7:
        st.subheader("Layer Output Similarity")

        with st.expander("ℹ️ What is this? (click to learn)", expanded=False):
            st.markdown(metric_help("layer_similarity"))

        if len(results.layer_outputs) >= 2:
            fig = plot_layer_similarity(results)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Quick Reference:**
            - Diagonal: Self-similarity (always 1.0)
            - Off-diagonal: Similarity between different layers
            - Adjacent layers often have high similarity
            - Large differences may indicate feature transformations
            """)

            # AI Interpretation
            if ai_interpret:
                interp_label = "Claude" if interp_choice == "claude" else "Local LLM"
                with st.expander("🧠 AI Interpretation", expanded=True):
                    render_cached_interpretation(
                        f"layer_similarity_{interp_choice}",
                        results,
                        lambda: get_layer_similarity_interpretation(
                            results, model, tokenizer, use_ai=True, interpreter=interp_choice
                        ),
                        interp_label
                    )
        else:
            st.info("Enable 'Capture Layer Outputs' with multiple layers to see this visualization.")

    # Tab 8: Residual Stream
    with tab8:
        st.subheader("Residual Stream Analysis")

        with st.expander("ℹ️ What is this? (click to learn)", expanded=False):
            st.markdown(metric_help("residual_stream"))

        if results.residual_stream_norms:
            fig = plot_residual_stream(results)
            st.plotly_chart(fig, use_container_width=True)

            # Summary statistics
            if results.residual_stream_deltas:
                deltas = [d[1] for d in results.residual_stream_deltas]
                max_delta_idx = np.argmax(deltas)
                max_delta_layer = results.residual_stream_deltas[max_delta_idx][0]
                max_delta_val = deltas[max_delta_idx]

                st.markdown(f"""
                **Quick Statistics:**
                - Total layers tracked: {len(results.residual_stream_norms)}
                - Largest change at: **{max_delta_layer}** (delta = {max_delta_val:.4f})
                - Average delta: {np.mean(deltas):.4f}
                - Final stream magnitude: {results.residual_stream_norms[-1][1]:.4f}
                """)

            # AI Interpretation
            if ai_interpret:
                interp_label = "Claude" if interp_choice == "claude" else "Local LLM"
                with st.expander("🧠 AI Interpretation", expanded=True):
                    render_cached_interpretation(
                        f"residual_stream_{interp_choice}",
                        results,
                        lambda: get_residual_stream_interpretation(
                            results, model, tokenizer, use_ai=True, interpreter=interp_choice
                        ),
                        interp_label
                    )
        else:
            st.info("Enable 'Capture Residual Stream' to see this visualization.")

    # Tab 9: Attention Patterns
    with tab_attn:
        st.subheader("Attention Pattern Analysis")

        with st.expander("ℹ️ What is this? (click to learn)", expanded=False):
            st.markdown("""
**Attention patterns** show how each token "attends to" (looks at) other tokens when processing.

- **Heatmap**: Rows = query tokens (the one looking), Columns = key tokens (being looked at)
- **Bright cells** = high attention weight (strong influence)
- **Dark cells** = low attention weight (weak/no influence)
- **Diagonal pattern**: Tokens attending to themselves
- **Vertical stripes**: Important tokens that many others attend to (often punctuation, key words)
- **Triangular shape**: Causal mask - tokens can only see previous tokens, not future ones

**Multi-head attention**: Different heads learn different patterns:
- Some heads track syntax (subject-verb agreement)
- Some track position (nearby tokens)
- Some track semantics (related concepts)
- Some are "induction heads" (pattern matching)

**What to look for**:
- 🔍 Which tokens are "important" (vertical stripes)?
- 🔍 Are there clear syntactic patterns?
- 🔍 Do different heads specialize differently?
- 🔍 How does attention change across layers?
            """)

        if results.attention_patterns:
            # Layer selector
            available_layers = sorted(results.attention_patterns.keys())
            n_heads = results.n_attention_heads or results.attention_patterns[available_layers[0]].shape[0]

            col1, col2 = st.columns([1, 1])
            with col1:
                selected_layer = st.selectbox(
                    "Layer",
                    available_layers,
                    format_func=lambda x: f"Layer {x}",
                    key="attn_layer_select"
                )
            with col2:
                head_options = ["Average (all heads)"] + [f"Head {i}" for i in range(n_heads)]
                selected_head = st.selectbox(
                    "Attention Head",
                    head_options,
                    key="attn_head_select"
                )

            # Get attention weights for selected layer
            attn_weights = results.attention_patterns[selected_layer]  # (n_heads, seq, seq)
            seq_len = attn_weights.shape[1]

            # Select head or average
            if selected_head == "Average (all heads)":
                attn_matrix = attn_weights.mean(axis=0)  # (seq, seq)
                head_label = "Average"
            else:
                head_idx = int(selected_head.split()[-1])
                attn_matrix = attn_weights[head_idx]  # (seq, seq)
                head_label = f"Head {head_idx}"

            # Create token labels
            all_tokens = list(results.input_tokens) + results.generated_tokens
            token_labels = []
            for i, tok_id in enumerate(all_tokens[:seq_len]):
                tok_text = tokenizer.decode([tok_id])
                # Truncate long tokens
                if len(tok_text) > 10:
                    tok_text = tok_text[:8] + ".."
                # Escape special chars
                tok_text = tok_text.replace("\n", "\\n").replace("\t", "\\t")
                token_labels.append(f"{i}:{tok_text}")

            # Attention heatmap - apply sqrt transform to expand low-value contrast
            # Most attention values are low, so sqrt spreads them out better
            attn_display = np.sqrt(attn_matrix.copy())

            # Set true zeros (causal mask) to NaN so they render as transparent/background
            attn_display[attn_matrix == 0] = np.nan

            # Custom colorscale optimized for attention patterns
            # More color variation in the low range (0-0.5) where most values cluster
            attention_colorscale = [
                [0.0, '#000000'],   # Black at 0
                [0.08, '#0d0887'],  # Dark blue
                [0.16, '#3b0f8c'],  # Blue-purple
                [0.24, '#5c179e'],  # Purple
                [0.32, '#7c22a4'],  # Magenta-purple
                [0.40, '#9b2a9f'],  # Magenta
                [0.48, '#b93389'],  # Pink-magenta
                [0.56, '#d44467'],  # Pink-red
                [0.64, '#e8614c'],  # Red-orange
                [0.72, '#f58438'],  # Orange
                [0.80, '#fca736'],  # Yellow-orange
                [0.90, '#f6d644'],  # Yellow
                [1.0, '#f0f921'],   # Bright yellow
            ]

            fig = go.Figure(data=go.Heatmap(
                z=attn_display,
                x=token_labels,
                y=token_labels,
                colorscale=attention_colorscale,
                zmin=0,
                zmax=1,
                colorbar=dict(
                    title="Weight",
                    tickvals=[0, 0.316, 0.447, 0.548, 0.707, 1.0],  # sqrt of 0, 0.1, 0.2, 0.3, 0.5, 1.0
                    ticktext=["0", "0.1", "0.2", "0.3", "0.5", "1.0"]  # Original values
                ),
                customdata=attn_matrix,  # Store original values for hover
                hovertemplate="<b>FROM:</b> %{y}<br><b>TO:</b> %{x}<br><b>Weight:</b> %{customdata:.4f}<extra></extra>",
                hoverongaps=False  # Don't show hover for NaN/masked cells
            ))

            fig.update_layout(
                title=f"Attention Pattern - Layer {selected_layer}, {head_label}",
                xaxis_title="Key (attending TO)",
                yaxis_title="Query (attending FROM)",
                height=600,
                xaxis=dict(tickangle=45, tickfont=dict(size=8)),
                yaxis=dict(tickfont=dict(size=8), autorange="reversed")
            )

            st.plotly_chart(fig, use_container_width=True)

            # Attention statistics
            st.markdown("---")
            st.markdown("### Attention Statistics")

            col1, col2, col3 = st.columns(3)

            # Entropy of attention (how spread out vs focused)
            entropy = -np.sum(attn_matrix * np.log(attn_matrix + 1e-10), axis=-1).mean()
            max_entropy = np.log(seq_len)
            normalized_entropy = entropy / max_entropy

            with col1:
                st.metric("Attention Entropy", f"{entropy:.2f}")
                if normalized_entropy > 0.7:
                    st.caption("🌐 Diffuse attention (looks at many tokens)")
                elif normalized_entropy > 0.3:
                    st.caption("⚖️ Balanced attention")
                else:
                    st.caption("🎯 Focused attention (few key tokens)")

            # Self-attention strength (diagonal)
            diag_attn = np.diag(attn_matrix).mean()
            with col2:
                st.metric("Self-Attention", f"{diag_attn:.2%}")
                st.caption("How much tokens attend to themselves")

            # Attention to first token (often special/important)
            first_token_attn = attn_matrix[:, 0].mean()
            with col3:
                st.metric("First Token Attention", f"{first_token_attn:.2%}")
                st.caption("Often high for BOS/system tokens")

            # Top attended positions
            st.markdown("---")
            st.markdown("### Most Attended Tokens")
            st.caption("Which tokens receive the most attention overall")

            # Filter options
            n_input = len(results.input_tokens)
            filter_col1, filter_col2 = st.columns([1, 3])
            with filter_col1:
                token_filter = st.radio(
                    "Show tokens from:",
                    ["All", "Input Sequence", "Generated"],
                    horizontal=True,
                    key="attn_token_filter",
                    label_visibility="collapsed"
                )

            # Sum attention received by each position (column sums)
            attn_received = attn_matrix.sum(axis=0)

            # Filter positions based on selection
            if token_filter == "Input Sequence":
                valid_positions = [i for i in range(min(n_input, len(attn_received)))]
            elif token_filter == "Generated":
                valid_positions = [i for i in range(n_input, len(attn_received))]
            else:  # All
                valid_positions = list(range(len(attn_received)))

            # Sort by attention received within filtered positions
            if valid_positions:
                filtered_attn = [(pos, attn_received[pos]) for pos in valid_positions]
                filtered_attn.sort(key=lambda x: x[1], reverse=True)
                sorted_positions = [pos for pos, _ in filtered_attn]
            else:
                sorted_positions = []

            # Load more state
            attn_show_key = "attn_tokens_show_count"
            if attn_show_key not in st.session_state:
                st.session_state[attn_show_key] = 10

            show_count = st.session_state[attn_show_key]
            top_attended = sorted_positions[:show_count]

            attended_data = []
            for pos in top_attended:
                if pos < len(all_tokens):
                    tok_id = all_tokens[pos]
                    tok_text = tokenizer.decode([tok_id])
                    section = "Input" if pos < n_input else "Generated"
                    attended_data.append({
                        "Position": int(pos),
                        "Section": section,
                        "Token": tok_text,
                        "Attention Received": f"{attn_received[pos]:.3f}",
                        "% of Total": f"{attn_received[pos] / attn_received.sum() * 100:.1f}%"
                    })
            attended_df = pd.DataFrame(attended_data)
            st.dataframe(attended_df, hide_index=True, use_container_width=True)

            # Load more button
            remaining = len(sorted_positions) - show_count
            if remaining > 0:
                if st.button(f"Load more ({remaining} remaining)", key="attn_load_more"):
                    st.session_state[attn_show_key] = min(show_count + 20, len(sorted_positions))
                    st.rerun()

            # Head comparison (if showing average)
            if selected_head == "Average (all heads)" and n_heads > 1:
                st.markdown("---")
                st.markdown("### Head Specialization")
                st.caption("How different attention heads behave")

                # Load more state for heads
                head_show_key = "attn_heads_show_count"
                if head_show_key not in st.session_state:
                    st.session_state[head_show_key] = 16

                heads_to_show = min(st.session_state[head_show_key], n_heads)

                head_stats = []
                for h in range(heads_to_show):
                    h_attn = attn_weights[h]
                    h_entropy = -np.sum(h_attn * np.log(h_attn + 1e-10), axis=-1).mean()
                    h_diag = np.diag(h_attn).mean()
                    h_first = h_attn[:, 0].mean()

                    head_stats.append({
                        "Head": h,
                        "Entropy": f"{h_entropy:.2f}",
                        "Self-Attn": f"{h_diag:.1%}",
                        "First-Tok": f"{h_first:.1%}",
                        "Pattern": "🎯 Focused" if h_entropy < 2 else ("🌐 Diffuse" if h_entropy > 3 else "⚖️ Mixed")
                    })

                head_df = pd.DataFrame(head_stats)
                st.dataframe(head_df, hide_index=True, use_container_width=True)

                # Load more button for heads
                remaining_heads = n_heads - heads_to_show
                if remaining_heads > 0:
                    if st.button(f"Load more heads ({remaining_heads} remaining)", key="head_load_more"):
                        st.session_state[head_show_key] = min(heads_to_show + 16, n_heads)
                        st.rerun()

            # Attention vs Distance Analysis (RoPE effect)
            st.markdown("---")
            st.markdown("### Attention vs Distance (RoPE Effect)")
            st.caption("How attention weight changes with token distance - reveals positional encoding effects")

            # Compute attention vs distance for each head
            seq_len = attn_weights.shape[1]
            max_distance = min(seq_len - 1, 100)  # Cap at 100 for readability

            # Create distance matrix
            distances = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :])

            # Compute average attention at each distance for each head
            distance_attn_per_head = []
            for h in range(n_heads):
                h_attn = attn_weights[h]
                distance_attn = []
                for d in range(max_distance + 1):
                    # Get all attention values at this distance (only from lower triangle - causal)
                    mask = (distances == d) & (np.triu(np.ones((seq_len, seq_len)), k=1) == 0)
                    if mask.sum() > 0:
                        avg_attn = h_attn[mask].mean()
                        distance_attn.append(avg_attn)
                    else:
                        distance_attn.append(0)
                distance_attn_per_head.append(distance_attn)

            distance_attn_per_head = np.array(distance_attn_per_head)

            # Average across all heads
            avg_distance_attn = distance_attn_per_head.mean(axis=0)

            # Plot distance-attention curve
            fig_dist = go.Figure()

            # Add average line (thick)
            fig_dist.add_trace(go.Scatter(
                x=list(range(max_distance + 1)),
                y=avg_distance_attn,
                mode='lines',
                name='Average (all heads)',
                line=dict(color='#00ff00', width=3)
            ))

            # Add individual head lines (thin, semi-transparent)
            # Let user choose how many heads to show
            head_plot_options = ["Average only", "8 heads", "16 heads", "32 heads", "All heads"]
            head_plot_choice = st.radio(
                "Show individual heads:",
                head_plot_options,
                index=1,  # Default to 8 heads
                horizontal=True,
                key="rope_heads_choice"
            )

            if head_plot_choice == "Average only":
                heads_to_plot = 0
            elif head_plot_choice == "All heads":
                heads_to_plot = n_heads
            else:
                heads_to_plot = min(n_heads, int(head_plot_choice.split()[0]))

            # Extended color palette for many heads
            colors = [
                '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dfe6e9', '#fd79a8', '#a29bfe',
                '#ff9f43', '#10ac84', '#5f27cd', '#ee5a24', '#0abde3', '#f368e0', '#1dd1a1', '#ff6b81',
                '#7bed9f', '#70a1ff', '#eccc68', '#ff7f50', '#00d2d3', '#ff9ff3', '#54a0ff', '#5f27cd',
                '#c8d6e5', '#8395a7', '#576574', '#222f3e', '#f5cd79', '#78e08f', '#e77f67', '#cf6a87'
            ]

            for h in range(heads_to_plot):
                fig_dist.add_trace(go.Scatter(
                    x=list(range(max_distance + 1)),
                    y=distance_attn_per_head[h],
                    mode='lines',
                    name=f'Head {h}',
                    line=dict(color=colors[h % len(colors)], width=1),
                    opacity=0.6 if heads_to_plot <= 16 else 0.3
                ))

            fig_dist.update_layout(
                title="Attention Weight vs Token Distance",
                xaxis_title="Distance (tokens back)",
                yaxis_title="Average Attention Weight",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode='x unified'
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # Head locality classification
            st.markdown("#### Head Locality Analysis")
            st.caption("Classifies heads by their effective attention range")

            # Compute metrics for each head
            locality_data = []
            for h in range(n_heads):
                h_dist_attn = distance_attn_per_head[h]

                # Compute effective range (distance at which cumulative attention reaches 50% and 90%)
                cumsum = np.cumsum(h_dist_attn)
                if cumsum[-1] > 0:
                    cumsum_norm = cumsum / cumsum[-1]
                    range_50 = np.searchsorted(cumsum_norm, 0.5)
                    range_90 = np.searchsorted(cumsum_norm, 0.9)
                else:
                    range_50 = range_90 = 0

                # Compute locality score (higher = more local)
                # Weight attention by distance, normalize
                if h_dist_attn.sum() > 0:
                    weighted_dist = np.sum(h_dist_attn * np.arange(len(h_dist_attn))) / h_dist_attn.sum()
                    locality_score = 1.0 / (1.0 + weighted_dist / 10)  # Normalize to 0-1ish
                else:
                    weighted_dist = 0
                    locality_score = 0

                # Classify
                if range_50 <= 3:
                    pattern = "🎯 Very Local"
                elif range_50 <= 10:
                    pattern = "📍 Local"
                elif range_50 <= 30:
                    pattern = "⚖️ Mixed"
                else:
                    pattern = "🌐 Global"

                locality_data.append({
                    "Head": h,
                    "50% Range": range_50,
                    "90% Range": range_90,
                    "Avg Dist": f"{weighted_dist:.1f}",
                    "Pattern": pattern
                })

            # Show first 16 heads (with load more)
            locality_show_key = "locality_show_count"
            if locality_show_key not in st.session_state:
                st.session_state[locality_show_key] = 16

            locality_show = min(st.session_state[locality_show_key], n_heads)
            locality_df = pd.DataFrame(locality_data[:locality_show])
            st.dataframe(locality_df, hide_index=True, use_container_width=True)

            if locality_show < n_heads:
                if st.button(f"Load more ({n_heads - locality_show} remaining)", key="locality_load_more"):
                    st.session_state[locality_show_key] = min(locality_show + 16, n_heads)
                    st.rerun()

            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            local_count = sum(1 for d in locality_data if "Local" in d["Pattern"])
            global_count = sum(1 for d in locality_data if "Global" in d["Pattern"])
            mixed_count = sum(1 for d in locality_data if "Mixed" in d["Pattern"])

            with col1:
                st.metric("Very Local Heads", sum(1 for d in locality_data if "Very Local" in d["Pattern"]))
            with col2:
                st.metric("Local Heads", sum(1 for d in locality_data if d["Pattern"] == "📍 Local"))
            with col3:
                st.metric("Mixed Heads", mixed_count)
            with col4:
                st.metric("Global Heads", global_count)

        else:
            st.info("No attention patterns captured. This may happen if attention capture is disabled or the model architecture is not supported.")

            # Debug info
            with st.expander("Debug Info"):
                saved_config = st.session_state.get('probe_config')
                st.write(f"capture_attention config: {saved_config.capture_attention if saved_config else 'N/A'}")
                st.write(f"n_attention_heads: {results.n_attention_heads}")
                st.write(f"n_kv_heads: {results.n_kv_heads}")
                st.write(f"Layer count: {results.num_layers}")

                # Show error if any
                if hasattr(results, '_attention_error'):
                    st.error(f"Attention capture error: {results._attention_error}")

                # Check layer structure
                if 'prober' in st.session_state:
                    prober = st.session_state.prober
                    if prober.layers:
                        layer = prober.layers[0]
                        st.write(f"Layer type: {type(layer).__name__}")
                        for attr in ['self_attn', 'attention', 'attn']:
                            if hasattr(layer, attr):
                                attn = getattr(layer, attr)
                                st.write(f"Found {attr}: {type(attn).__name__}")
                                st.write(f"Has q_proj: {hasattr(attn, 'q_proj')}")
                                st.write(f"Has k_proj: {hasattr(attn, 'k_proj')}")
                                st.write(f"n_heads attr: {getattr(attn, 'n_heads', 'N/A')}")
                                st.write(f"n_kv_heads attr: {getattr(attn, 'n_kv_heads', 'N/A')}")
                                break
                        else:
                            st.write(f"No self_attn/attention/attn found")
                            st.write(f"Layer keys: {list(layer.keys()) if hasattr(layer, 'keys') else 'N/A'}")

    # MoE Routing tab (only if model is MoE)
    if tab_moe is not None:
        with tab_moe:
            st.subheader("Mixture of Experts Routing Analysis")

            with st.expander("ℹ️ What is this? (click to learn)", expanded=False):
                st.markdown(metric_help("moe_routing"))

            if results.is_moe:
                # MoE summary info
                col1, col2, col3 = st.columns(3)
                col1.metric("Experts", results.num_experts)
                col2.metric("Top-K Selection", results.num_experts_per_tok or "N/A")
                col3.metric("MoE Layers", len(results.moe_router_outputs) if results.moe_router_outputs else "N/A")

                if results.moe_expert_load:
                    st.markdown("---")
                    st.markdown("### Expert Load Distribution")
                    st.caption("How tokens are distributed across experts (all layers). Color: ⬜ light = low influence (rarely selected) → 🟦 dark blue = high influence (frequently selected, dominant expert)")

                    fig_load = plot_moe_expert_load(results)
                    st.plotly_chart(fig_load, use_container_width=True)

                    # Expert Path Analysis (cross-layer view)
                    st.markdown("---")
                    st.markdown("### Expert Activation Path")
                    st.caption("Track which experts are activated for a specific token across ALL MoE layers. This reveals the token's 'pathway' through the network.")

                    moe_layers = sorted(results.moe_router_outputs.keys())

                    if moe_layers and results.moe_router_outputs:
                        # Get sequence length from first layer's router data
                        first_layer_data = results.moe_router_outputs[moe_layers[0]]
                        first_probs = first_layer_data["probs"]
                        if len(first_probs.shape) == 3:
                            first_probs = first_probs[0]
                        seq_len = first_probs.shape[0]

                        # Token selector
                        col_path_sel, col_path_info = st.columns([2, 1])
                        with col_path_sel:
                            # Build token options - try multiple sources for token text
                            path_token_options = []
                            for i in range(seq_len):
                                tok_text = None

                                # Try results.all_tokens first
                                if tokenizer and hasattr(results, 'all_tokens') and results.all_tokens:
                                    if i < len(results.all_tokens):
                                        try:
                                            tok_text = tokenizer.decode([results.all_tokens[i]])
                                        except:
                                            pass

                                # Try generation_timeline if available
                                if tok_text is None and hasattr(results, 'generation_timeline') and results.generation_timeline:
                                    timeline = results.generation_timeline
                                    # Check input tokens
                                    if i < len(timeline.input_tokens):
                                        try:
                                            tok_text = tokenizer.decode([timeline.input_tokens[i]])
                                        except:
                                            pass
                                    # Check generated steps
                                    elif timeline.steps:
                                        gen_idx = i - len(timeline.input_tokens)
                                        if 0 <= gen_idx < len(timeline.steps):
                                            tok_text = timeline.steps[gen_idx].token_text

                                # Fallback
                                if tok_text is None:
                                    tok_text = f"[pos {i}]"
                                else:
                                    # Clean up token text for display
                                    tok_text = tok_text.replace('\n', '↵').replace('\t', '→')
                                    if len(tok_text) > 25:
                                        tok_text = tok_text[:22] + "..."

                                path_token_options.append(f"{i}: {tok_text}")

                            selected_path_token = st.selectbox(
                                "Select token to trace",
                                range(len(path_token_options)),
                                format_func=lambda i: path_token_options[i],
                                key="expert_path_token_select"
                            )

                        # Get path summary
                        path_summary = get_expert_path_summary(results, selected_path_token)

                        with col_path_info:
                            if path_summary:
                                st.metric(
                                    "Unique Top-1 Experts",
                                    f"{path_summary['unique_top1_experts']} / {path_summary['num_moe_layers']}",
                                    help="How many different experts were selected as Top-1 across layers"
                                )

                        # Show path visualization
                        fig_path = plot_moe_expert_path(results, selected_path_token, tokenizer)
                        st.plotly_chart(fig_path, use_container_width=True)

                        # Path summary statistics
                        if path_summary:
                            with st.expander("📊 Path Statistics", expanded=False):
                                col_stat1, col_stat2 = st.columns(2)

                                with col_stat1:
                                    st.markdown("**Most Frequently Selected (Top-1):**")
                                    for expert_id, count in path_summary.get('most_common_top1', []):
                                        pct = count / path_summary['num_moe_layers'] * 100
                                        st.text(f"  Expert {expert_id}: {count} layers ({pct:.0f}%)")

                                with col_stat2:
                                    st.markdown("**Highest Total Weight:**")
                                    for expert_id, weight in path_summary.get('top_5_by_total_weight', [])[:5]:
                                        if weight > 0.01:
                                            st.text(f"  Expert {expert_id}: {weight:.2f} total")

                                # Interpretation
                                concentration = path_summary.get('path_concentration', 1.0)
                                if concentration < 0.3:
                                    st.success("🎯 **Concentrated path**: This token consistently uses a few experts across layers")
                                elif concentration < 0.6:
                                    st.info("↔️ **Moderate path diversity**: Token uses different experts in different layers")
                                else:
                                    st.warning("🌐 **Highly distributed**: Token activates many different experts")

                    # Layer selector for detailed view
                    st.markdown("---")
                    st.markdown("### Per-Layer Analysis")
                    if moe_layers:
                        selected_moe_layer = st.selectbox(
                            "Select MoE layer to analyze",
                            moe_layers,
                            format_func=lambda x: f"Layer {x}"
                        )

                        # Show router probabilities heatmap
                        st.markdown("#### Router Probabilities")
                        st.caption("Router confidence for each expert per token. Color: 🟣 purple = low influence (expert ignored for this token) → 🟡 yellow = high influence (expert strongly activated for this token)")
                        fig_probs = plot_moe_router_probs(results, selected_moe_layer, tokenizer)
                        st.plotly_chart(fig_probs, use_container_width=True)

                        # Show expert selection pattern
                        st.markdown("#### Expert Selection Pattern")
                        st.caption("Marker size indicates router confidence")
                        fig_select = plot_moe_expert_selection(results, selected_moe_layer, tokenizer)
                        st.plotly_chart(fig_select, use_container_width=True)

                        # Top-K Expert Weights by Section (System, User, Response)
                        st.markdown("#### Top-K Expert Weights by Section")
                        top_k = results.num_experts_per_tok or 4
                        st.caption(f"""**How to read:** Each bar shows the {top_k} experts selected for that token.
Colors indicate selection rank (🟡 Top-1, 🟣 Top-2, 🔵 Top-3, 🟠 Top-4). Bar length = router weight.
Labels inside bars show expert ID (E0-E{results.num_experts-1 if results.num_experts else '?'}).
**Why {top_k} experts?** This model uses top-{top_k} routing—only {top_k} experts are activated per token (architecture setting, not a display limit).""")
                        render_moe_topk_sections(results, selected_moe_layer, tokenizer)

                        # Verification/Debug Panel
                        with st.expander("🔍 Verify Router Data (Debug)", expanded=False):
                            st.caption("Sanity check: verify Top-1 weight > Top-2 > Top-3 > Top-4 for selected tokens")

                            if selected_moe_layer in results.moe_router_outputs:
                                router_data = results.moe_router_outputs[selected_moe_layer]
                                selected_arr = router_data["selected"][0]  # (seq, top_k)
                                probs_arr = router_data["probs"][0]  # (seq, num_experts)

                                seq_len, top_k_actual = selected_arr.shape

                                # Run sanity check across all tokens
                                violations = []
                                for i in range(seq_len):
                                    weights = []
                                    for rank in range(1, top_k_actual + 1):
                                        sel_idx = top_k_actual - rank  # Top-1 → last index
                                        expert_id = int(selected_arr[i, sel_idx])
                                        weight = float(probs_arr[i, expert_id])
                                        weights.append((rank, expert_id, weight))

                                    # Check ordering: Top-1 should have highest weight
                                    for j in range(len(weights) - 1):
                                        if weights[j][2] < weights[j + 1][2]:
                                            violations.append((i, weights[j], weights[j + 1]))

                                if violations:
                                    st.error(f"⚠️ Found {len(violations)} ordering violations!")
                                    for v in violations[:5]:  # Show first 5
                                        st.write(f"Token {v[0]}: Top-{v[1][0]} (E{v[1][1]}, {v[1][2]:.2%}) < Top-{v[2][0]} (E{v[2][1]}, {v[2][2]:.2%})")
                                else:
                                    st.success(f"✅ All {seq_len} tokens pass ordering check: Top-1 ≥ Top-2 ≥ Top-3 ≥ Top-4")

                                # Show raw data for specific token
                                st.markdown("---")
                                st.markdown("**Inspect single token:**")

                                # Combine tokens for selection
                                all_tokens = list(results.input_tokens) if results.input_tokens else []
                                if hasattr(results, 'generated_tokens') and results.generated_tokens:
                                    all_tokens = all_tokens + list(results.generated_tokens)

                                inspect_pos = st.slider("Token position", 0, seq_len - 1, 0, key="inspect_token_pos")

                                # Get token text
                                if inspect_pos < len(all_tokens):
                                    try:
                                        tok_text = tokenizer.decode([all_tokens[inspect_pos]])
                                    except:
                                        tok_text = f"[{all_tokens[inspect_pos]}]"
                                else:
                                    tok_text = "?"

                                st.write(f"**Token {inspect_pos}:** `{tok_text}`")

                                # Show all expert probabilities sorted
                                all_probs = [(int(e), float(probs_arr[inspect_pos, e])) for e in range(probs_arr.shape[1])]
                                all_probs.sort(key=lambda x: x[1], reverse=True)

                                # Show top-k (selected) + 2 more (not selected) to see the drop-off
                                show_count = top_k_actual + 2
                                st.write(f"**All experts by probability** (top-{top_k_actual} are selected, rest ignored):")
                                top_df = pd.DataFrame([
                                    {
                                        "Rank": i+1,
                                        "Expert": f"E{e}",
                                        "Weight": f"{p:.4%}",
                                        "Status": "✅ Selected" if i < top_k_actual else "❌ Not used"
                                    }
                                    for i, (e, p) in enumerate(all_probs[:show_count])
                                ])
                                st.dataframe(top_df, hide_index=True, use_container_width=True)

                                # Show the drop-off
                                if len(all_probs) > top_k_actual:
                                    last_selected_weight = all_probs[top_k_actual - 1][1]
                                    first_rejected_weight = all_probs[top_k_actual][1]
                                    drop_pct = ((last_selected_weight - first_rejected_weight) / last_selected_weight * 100) if last_selected_weight > 0 else 0
                                    st.caption(f"Drop-off: Top-{top_k_actual} weight ({last_selected_weight:.2%}) → next expert ({first_rejected_weight:.2%}) = {drop_pct:.0f}% decrease")

                                # Show what the chart displays
                                st.write(f"**Chart shows (Top-{top_k_actual} selected):**")
                                chart_data = []
                                for rank in range(1, top_k_actual + 1):
                                    sel_idx = top_k_actual - rank
                                    expert_id = int(selected_arr[inspect_pos, sel_idx])
                                    weight = float(probs_arr[inspect_pos, expert_id])
                                    chart_data.append({"Rank": f"Top-{rank}", "Expert": f"E{expert_id}", "Weight": f"{weight:.4%}"})
                                st.dataframe(pd.DataFrame(chart_data), hide_index=True, use_container_width=True)
                            else:
                                st.info("No router data available for this layer")

                        # Expert Selection Table (full)
                        with st.expander("📋 Expert Selection Details (Full Table)", expanded=False):
                            st.caption("Complete table showing all tokens with their expert selections and weights")
                            df_experts = plot_moe_expert_weights_table(results, selected_moe_layer, tokenizer)
                            if not df_experts.empty:
                                st.dataframe(df_experts, hide_index=True, use_container_width=True)
                            else:
                                st.info("No expert selection data available")

                        # Expert load for this layer
                        st.markdown("#### Expert Load (This Layer)")
                        st.caption("Token count per expert in this layer. Taller bars = higher influence (expert processed more tokens)")
                        fig_layer_load = plot_moe_expert_load(results, selected_moe_layer)
                        st.plotly_chart(fig_layer_load, use_container_width=True)

                        # Load balance statistics
                        if selected_moe_layer in results.moe_expert_load:
                            load = results.moe_expert_load[selected_moe_layer]
                            counts = list(load.values())
                            if counts:
                                total = sum(counts)
                                max_load = max(counts)
                                min_load = min(counts)
                                avg_load = total / len(counts)
                                # Compute load imbalance (coefficient of variation)
                                if avg_load > 0:
                                    std_load = np.std(counts)
                                    imbalance = std_load / avg_load
                                else:
                                    imbalance = 0

                                st.markdown(f"""
                                **Load Statistics:**
                                - Total tokens routed: {total}
                                - Max expert load: {max_load} ({max_load/total*100:.1f}%)
                                - Min expert load: {min_load} ({min_load/total*100:.1f}%)
                                - Load imbalance (CV): {imbalance:.3f} (lower = more balanced)
                                """)

                                # Identify dead or overloaded experts
                                dead_experts = [e for e, c in load.items() if c == 0]
                                if dead_experts:
                                    st.warning(f"⚠️ Dead experts (no tokens): {dead_experts}")

                    # Global Expert Influence Analysis
                    st.markdown("---")
                    st.markdown("### 🎯 Expert Influence Analysis")
                    st.caption("Comprehensive routing statistics using AI domain metrics")

                    moe_stats = compute_moe_influence_stats(results)

                    if moe_stats["total_routing_decisions"] > 0:
                        # Summary metrics in columns
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

                        with col_m1:
                            eff_count = moe_stats.get("effective_expert_count", 0)
                            st.metric(
                                "Effective Experts",
                                f"{eff_count:.1f}",
                                delta=f"{eff_count/results.num_experts*100:.0f}% utilization" if results.num_experts else None,
                                delta_color="normal"
                            )

                        with col_m2:
                            norm_entropy = moe_stats.get("normalized_entropy", 0)
                            st.metric(
                                "Router Entropy",
                                f"{norm_entropy*100:.1f}%",
                                delta="High diversity" if norm_entropy > 0.8 else "Moderate" if norm_entropy > 0.5 else "Low diversity",
                                delta_color="normal" if norm_entropy > 0.5 else "inverse"
                            )

                        with col_m3:
                            lb_score = moe_stats.get("load_balance_score", 0)
                            st.metric(
                                "Load Balance",
                                f"{lb_score:.2f}",
                                delta="Balanced" if lb_score > 0.7 else "Moderate" if lb_score > 0.4 else "Imbalanced",
                                delta_color="normal" if lb_score > 0.5 else "inverse"
                            )

                        with col_m4:
                            gini = moe_stats.get("gini_coefficient", 0)
                            st.metric(
                                "Gini Coefficient",
                                f"{gini:.3f}",
                                delta="Equal" if gini < 0.3 else "Moderate" if gini < 0.5 else "Unequal",
                                delta_color="normal" if gini < 0.4 else "inverse"
                            )

                        # Detailed report
                        with st.expander("📊 Full Expert Influence Report", expanded=True):
                            report = format_moe_influence_report(moe_stats, results.num_experts)
                            st.markdown(report)

                        # Expert Token Specialization
                        if results.moe_expert_tokens:
                            with st.expander("🔤 Expert Token Specialization", expanded=False):
                                st.caption("Which tokens each expert prefers to process - reveals expert specialization patterns")

                                expert_token_analysis = get_expert_token_specialization(results, tokenizer, top_n=15)

                                # Let user select which expert to view
                                # Default to top influential expert
                                top_experts = [item["expert_id"] for item in moe_stats.get("expert_influence_ranking", [])[:10]]
                                if top_experts:
                                    selected_expert = st.selectbox(
                                        "Select Expert to Analyze",
                                        top_experts,
                                        format_func=lambda x: f"Expert {x} ({expert_token_analysis.get(x, {}).get('total_activations', 0):,} activations)"
                                    )

                                    if selected_expert in expert_token_analysis:
                                        analysis = expert_token_analysis[selected_expert]

                                        # Summary metrics
                                        col_e1, col_e2, col_e3 = st.columns(3)
                                        col_e1.metric("Total Activations", f"{analysis['total_activations']:,}")
                                        col_e2.metric("Unique Tokens", f"{analysis['unique_tokens']:,}")
                                        col_e3.metric(
                                            "Specialization",
                                            f"{analysis['specialization_score']:.2f}",
                                            delta="Focused" if analysis['specialization_score'] > 0.5 else "Diverse",
                                            delta_color="normal"
                                        )

                                        # Token table
                                        if analysis["tokens"]:
                                            st.markdown("**Most Frequently Processed Tokens:**")
                                            token_table = "| Rank | Token | Count | Avg Router Prob |\n"
                                            token_table += "|------|-------|-------|----------------|\n"
                                            for i, (token_text, token_id, count, avg_prob) in enumerate(analysis["tokens"], 1):
                                                # Escape pipe characters for markdown table
                                                safe_text = token_text.replace("|", "\\|")
                                                token_table += f"| {i} | `{safe_text}` | {count:,} | {avg_prob:.4f} |\n"
                                            st.markdown(token_table)

                                            # Quick interpretation
                                            if analysis['specialization_score'] > 0.7:
                                                st.info(f"🎯 **Highly Specialized Expert**: This expert strongly focuses on specific token types, suggesting learned specialization.")
                                            elif analysis['specialization_score'] > 0.4:
                                                st.info(f"📊 **Moderately Specialized Expert**: This expert shows some preference patterns but handles diverse tokens.")
                                            else:
                                                st.info(f"🌐 **Generalist Expert**: This expert processes a wide variety of tokens without strong preferences.")
                                        else:
                                            st.info("No token data available for this expert.")

                        # AI Interpretation for MoE
                        if ai_interpret:
                            interp_label = "Claude" if interp_choice == "claude" else "Local LLM"

                            # Build context for AI interpretation
                            moe_context = f"""
MoE Model Analysis:
- Total Experts: {results.num_experts}
- Top-K per Token: {results.num_experts_per_tok}
- Effective Expert Count: {moe_stats.get('effective_expert_count', 0):.1f} ({moe_stats.get('effective_expert_count', 0)/results.num_experts*100:.1f}% of total)
- Router Entropy: {moe_stats.get('normalized_entropy', 0)*100:.1f}% of maximum
- Load Balance Score: {moe_stats.get('load_balance_score', 0):.3f}
- Gini Coefficient: {moe_stats.get('gini_coefficient', 0):.3f}
- Dead Experts: {len(moe_stats.get('dead_experts', []))}
- Dominant Experts: {len(moe_stats.get('dominant_experts', []))}
- Auxiliary Loss Proxy: {moe_stats.get('auxiliary_loss_proxy', 0):.3f}

Top 5 Most Active Experts:
"""
                            for item in moe_stats.get('expert_influence_ranking', [])[:5]:
                                moe_context += f"  Expert {item['expert_id']}: {item['frequency']*100:.2f}% of tokens\n"

                            with st.expander("🧠 AI Interpretation", expanded=True):
                                render_cached_interpretation(
                                    f"moe_routing_{interp_choice}",
                                    results,
                                    lambda ctx=moe_context: generate_ai_interpretation(
                                        model,
                                        tokenizer,
                                        ctx,
                                        "Analyze these MoE routing patterns. What do the expert load distribution, router entropy, and specialization metrics reveal about how this model routes tokens to experts? Are there any concerns about load balancing or dead experts?",
                                        interpreter=interp_choice
                                    ),
                                    interp_label
                                )
                    else:
                        st.info("No routing statistics available. Run inference to capture MoE data.")
                else:
                    st.info("No MoE routing data captured. Enable 'Capture Layer Outputs' and ensure the model is MoE.")
            else:
                st.info("This model is not a Mixture of Experts (MoE) model.")

    # Knowledge Localization tab
    with tab_knowledge:
        st.subheader("Knowledge Localization (Causal Tracing)")

        with st.expander("ℹ️ What is this? (click to learn)", expanded=False):
            st.markdown("""
**Knowledge Localization** uses causal tracing to identify where facts are stored in the model.

**How it works:**
1. **Clean run**: Get probability of target answer (e.g., P("Paris") for "The capital of France is...")
2. **Corrupted run**: Add noise to subject tokens ("France"), probability drops
3. **Restore run**: For each (layer, position), restore clean activations and measure recovery

**Interpreting results:**
- **High recovery** (yellow/red) indicates information critical to the answer flows through that site
- **Critical site** (star) shows where restoring activations has maximum effect
- Facts are typically stored in middle-to-late MLP layers at the last subject token position

**Reference:** [ROME: Locating and Editing Factual Associations](https://rome.baulab.info/)
            """)

        # Initialize causal trace state
        if 'causal_trace_results' not in st.session_state:
            st.session_state.causal_trace_results = None
        if 'causal_tracer' not in st.session_state:
            st.session_state.causal_tracer = None

        # Query configuration
        st.markdown("### Query Configuration")

        col_query1, col_query2 = st.columns([2, 1])
        with col_query1:
            ct_query = st.text_input(
                "Query text",
                value="The capital of France is",
                help="The factual query to analyze",
                key="ct_query"
            )
        with col_query2:
            ct_subject = st.text_input(
                "Subject",
                value="France",
                help="The subject tokens to corrupt",
                key="ct_subject"
            )

        col_target, col_noise = st.columns(2)
        with col_target:
            ct_target = st.text_input(
                "Expected answer (optional)",
                value="Paris",
                help="Leave empty to auto-detect from model's prediction",
                key="ct_target"
            )
        with col_noise:
            ct_noise = st.slider(
                "Noise multiplier",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                help="How much noise to add (relative to embedding std)",
                key="ct_noise"
            )

        # Advanced options
        with st.expander("Advanced Options", expanded=False):
            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                ct_sample_layers = st.checkbox(
                    "Sample layers (faster)",
                    value=True,
                    help="Test every Nth layer instead of all",
                    key="ct_sample_layers"
                )
                ct_layer_rate = st.slider(
                    "Layer sample rate",
                    min_value=1,
                    max_value=4,
                    value=2,
                    disabled=not ct_sample_layers,
                    key="ct_layer_rate"
                )
            with col_adv2:
                ct_sample_pos = st.checkbox(
                    "Sample positions (faster)",
                    value=True,
                    help="Test every Nth position instead of all",
                    key="ct_sample_pos"
                )
                ct_pos_rate = st.slider(
                    "Position sample rate",
                    min_value=1,
                    max_value=4,
                    value=2,
                    disabled=not ct_sample_pos,
                    key="ct_pos_rate"
                )

        # Run button
        if st.button("Run Causal Trace", type="primary", key="run_causal_trace"):
            if not ct_query or not ct_subject:
                st.error("Please provide both query text and subject.")
            else:
                # Create tracer if needed
                if st.session_state.causal_tracer is None:
                    st.session_state.causal_tracer = CausalTracer(
                        model, tokenizer,
                        topology=st.session_state.get('model_topology', {})
                    )

                tracer = st.session_state.causal_tracer

                # Configure
                config = CausalTraceConfig(
                    noise_multiplier=ct_noise,
                    sample_layers=ct_sample_layers,
                    layer_sample_rate=ct_layer_rate,
                    sample_positions=ct_sample_pos,
                    position_sample_rate=ct_pos_rate
                )

                # Estimate forward passes
                num_layers = tracer.num_layers
                tokens = tracer._tokenize(ct_query)
                seq_len = len(tokens)

                layers_count = num_layers // ct_layer_rate if ct_sample_layers else num_layers
                pos_count = seq_len // ct_pos_rate if ct_sample_pos else seq_len
                estimated_passes = 2 + layers_count * pos_count

                # Run with progress
                progress_bar = st.progress(0, text="Running causal trace...")
                status_text = st.empty()

                def update_progress(current, total):
                    progress_bar.progress(current / total, text=f"Forward pass {current}/{total}")

                status_text.text(f"Estimated ~{estimated_passes} forward passes")

                try:
                    trace_results = tracer.run_causal_trace(
                        query_text=ct_query,
                        subject=ct_subject,
                        target_token=ct_target if ct_target else None,
                        config=config,
                        progress_callback=update_progress
                    )
                    st.session_state.causal_trace_results = trace_results
                    progress_bar.progress(1.0, text="Complete!")
                    status_text.text(f"Completed {trace_results.num_forward_passes} forward passes")

                except Exception as e:
                    st.error(f"Causal trace failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

        # Display results
        if st.session_state.causal_trace_results is not None:
            trace_results = st.session_state.causal_trace_results

            st.markdown("---")
            st.markdown("### Results")

            # Summary metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric(
                    "Clean P(target)",
                    f"{trace_results.clean_prob:.1%}",
                    help="Probability of target token without corruption"
                )
            with col_m2:
                st.metric(
                    "Corrupted P(target)",
                    f"{trace_results.corrupted_prob:.1%}",
                    delta=f"{trace_results.corrupted_prob - trace_results.clean_prob:.1%}",
                    delta_color="inverse"
                )
            with col_m3:
                st.metric(
                    "Max Recovery",
                    f"{trace_results.max_recovery:.1%}",
                    help="Maximum recovery when restoring clean activation"
                )
            with col_m4:
                st.metric(
                    "Critical Site",
                    f"L{trace_results.critical_layer}, Pos {trace_results.critical_position}",
                    help="Layer and position with maximum recovery"
                )

            # Target token info with diagnostics
            st.info(f"Target token: **{trace_results.target_token}** (ID: {trace_results.target_token_id})")

            # Show diagnostic if clean_prob is very low
            if trace_results.clean_prob < 0.01:
                with st.expander("⚠️ Low clean probability - Diagnostics", expanded=True):
                    st.warning("Clean probability is very low. The model may not predict this token.")

                    # Show what the model actually predicts
                    if 'causal_tracer' in st.session_state and st.session_state.causal_tracer:
                        tracer = st.session_state.causal_tracer
                        try:
                            # Run clean forward pass
                            tokens = tracer._tokenize(trace_results.query_text)
                            x = mx.array([tokens])
                            clean_emb = tracer.embedding(x)
                            mx.eval(clean_emb)
                            clean_logits = tracer._forward_with_intervention(clean_emb, clean_emb, None, None)

                            # Get top 5 predictions
                            probs = mx.softmax(clean_logits[0, -1, :], axis=-1)
                            mx.eval(probs)
                            top_indices = mx.argsort(probs)[-5:][::-1]
                            mx.eval(top_indices)

                            st.markdown("**Model's top 5 predictions:**")
                            for idx in top_indices.tolist():
                                tok_text = tracer._decode(idx)
                                tok_prob = float(probs[idx])
                                is_target = "← target" if idx == trace_results.target_token_id else ""
                                st.text(f"  {tok_prob:6.2%}  '{tok_text}' (ID: {idx}) {is_target}")

                            # Suggest alternatives
                            st.markdown("**Tip:** Try these alternatives for Expected answer:")
                            for idx in top_indices.tolist()[:3]:
                                tok_text = tracer._decode(idx)
                                st.text(f"  '{tok_text.strip()}'  or  '{tok_text}'")
                        except Exception as e:
                            st.error(f"Could not get model predictions: {e}")

            # Summary plot
            st.markdown("#### Probability & Layer Recovery")
            fig_summary = plot_causal_trace_summary(trace_results)
            st.plotly_chart(fig_summary, use_container_width=True)

            # Heatmap
            st.markdown("#### Indirect Effect Heatmap")
            st.caption("Shows recovery of target probability when restoring clean activations at each (layer, position). Star marks critical site.")

            # Get tokens for labels
            if st.session_state.causal_tracer:
                tokens = st.session_state.causal_tracer._tokenize(trace_results.query_text)
            else:
                tokens = None

            fig_heatmap = plot_causal_trace_heatmap(trace_results, tokenizer, tokens)
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # Interpretation
            st.markdown("#### Interpretation")

            # Check if MoE and critical layer is MoE
            if results.is_moe and trace_results.critical_layer in results.moe_router_outputs:
                st.markdown("**MoE Expert Analysis at Critical Site:**")

                router_data = results.moe_router_outputs[trace_results.critical_layer]
                probs_arr = router_data["probs"][0]  # (seq, num_experts)
                selected_arr = router_data["selected"][0]  # (seq, top_k)

                if trace_results.critical_position < probs_arr.shape[0]:
                    pos = trace_results.critical_position
                    selected_experts = selected_arr[pos]
                    expert_probs = probs_arr[pos]

                    st.write(f"At position {pos}, layer {trace_results.critical_layer}:")
                    for i, exp_id in enumerate(selected_experts[::-1]):  # Reverse for top-1 first
                        exp_id = int(exp_id)
                        prob = float(expert_probs[exp_id])
                        st.write(f"  - **Expert {exp_id}**: {prob:.1%} router weight (Top-{i+1})")

            # General interpretation based on critical position
            if trace_results.subject_positions:
                last_subject = max(trace_results.subject_positions)
                if trace_results.critical_position == last_subject:
                    st.success("The critical site is at the **last subject token position**, consistent with ROME findings that facts are localized at subject positions.")
                elif trace_results.critical_position in trace_results.subject_positions:
                    st.info("The critical site is within the **subject token span**.")
                else:
                    st.info(f"The critical site is at position {trace_results.critical_position} (subject positions: {trace_results.subject_positions}).")

            # Layer interpretation
            mid_layer = trace_results.layers_tested[len(trace_results.layers_tested) // 2] if trace_results.layers_tested else 0
            if trace_results.critical_layer >= mid_layer:
                st.info(f"The critical layer ({trace_results.critical_layer}) is in the **latter half** of the network, where factual associations are typically stored in MLP layers.")
            else:
                st.info(f"The critical layer ({trace_results.critical_layer}) is in the **earlier half** of the network, which may indicate information is propagated early.")

    # Generation Replay tab
    with tab_replay:
        st.subheader("Generation Replay")

        with st.expander("ℹ️ What is this? (click to learn)", expanded=False):
            st.markdown("""
**Generation Replay** lets you step through token generation with VCR-style controls.

**Features:**
- **Timeline scrubber**: Navigate to any point in generation
- **Playback controls**: Step forward/backward through tokens
- **Alternatives view**: See what other tokens the model considered at each step
- **Branching**: Select an alternative token and see what happens

**How to use:**
1. Generate text using "Run Inference" above (Generation Replay will capture the timeline)
2. Use the scrubber or controls to navigate
3. Click "Branch" to explore alternative continuations
            """)

        # Check if we have a timeline
        original_timeline = results.generation_timeline if hasattr(results, 'generation_timeline') else None

        if original_timeline is None or not original_timeline.steps:
            st.info("No generation timeline available. Run inference with generation to capture a timeline.")
            st.markdown("""
**To enable Generation Replay:**
1. Enter a prompt in the sidebar
2. Set **Max tokens** > 0
3. Click **Run Inference**

The timeline will be captured automatically during generation.
            """)
        else:
            # Initialize replay state
            if 'replay_branches' not in st.session_state:
                st.session_state.replay_branches = {}  # {branch_id: timeline}
            if 'viewing_branch_id' not in st.session_state:
                st.session_state.viewing_branch_id = None

            # Select which timeline to view (original or branch)
            viewing_branch_id = st.session_state.get('viewing_branch_id')
            if viewing_branch_id and viewing_branch_id in st.session_state.replay_branches:
                timeline = st.session_state.replay_branches[viewing_branch_id]
                is_viewing_branch = True
            else:
                timeline = original_timeline
                is_viewing_branch = False
                st.session_state.viewing_branch_id = None

            num_steps = len(timeline.steps)

            # Timeline header
            if is_viewing_branch:
                st.markdown(f"**🌿 Branch Timeline:** {num_steps} steps | ID: `{timeline.timeline_id}`")
                if st.button("← Back to Original", key="back_to_original"):
                    st.session_state.viewing_branch_id = None
                    st.session_state.replay_scrubber = 0
                    st.rerun()
            else:
                st.markdown(f"**Timeline:** {num_steps} steps | ID: `{timeline.timeline_id}`")

            # Show parent info if this is a branch
            if timeline.parent_timeline_id:
                bp = timeline.branch_point_step
                if bp is not None and bp < len(timeline.steps):
                    branch_tok = timeline.steps[bp].token_text
                    orig_tok = original_timeline.steps[bp].token_text if bp < len(original_timeline.steps) else "?"
                    st.caption(f"Branched at step {bp}: `{orig_tok}` → `{branch_tok}`")

            # Initialize slider state if needed
            if 'replay_scrubber' not in st.session_state:
                st.session_state.replay_scrubber = 0

            # Clamp to valid range
            st.session_state.replay_scrubber = min(st.session_state.replay_scrubber, num_steps - 1)

            # Playback controls (BEFORE slider so they can update state before slider renders)
            col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4, col_ctrl5, col_spacer, col_branch = st.columns([1, 1, 1, 1, 1, 2, 2])

            with col_ctrl1:
                if st.button("⏮", help="First step", key="replay_first"):
                    st.session_state.replay_scrubber = 0

            with col_ctrl2:
                if st.button("⏪", help="Previous step", key="replay_prev"):
                    if st.session_state.replay_scrubber > 0:
                        st.session_state.replay_scrubber -= 1

            with col_ctrl3:
                # Play/pause would require auto-refresh which Streamlit doesn't support natively
                st.button("▶", help="Play (manual stepping)", disabled=True, key="replay_play")

            with col_ctrl4:
                if st.button("⏩", help="Next step", key="replay_next"):
                    if st.session_state.replay_scrubber < num_steps - 1:
                        st.session_state.replay_scrubber += 1

            with col_ctrl5:
                if st.button("⏭", help="Last step", key="replay_last"):
                    st.session_state.replay_scrubber = num_steps - 1

            with col_branch:
                # Use session state to keep dialog open across reruns
                if 'show_branch_dialog' not in st.session_state:
                    st.session_state.show_branch_dialog = False

                if st.button("🌿 Branch", help="Create alternative timeline from this step", key="replay_branch"):
                    st.session_state.show_branch_dialog = True
                    # Reset any pending selections when opening dialog
                    st.session_state.branch_selection_confirmed = False
                    st.session_state.pending_branch_token_id = None
                    st.session_state.pending_branch_token_text = None
                    st.session_state.pending_branch_step = None
                    st.rerun()

            # Check for pending scrubber update (set by branch generation before rerun)
            if 'pending_scrubber_value' in st.session_state and st.session_state.pending_scrubber_value is not None:
                st.session_state.replay_scrubber = st.session_state.pending_scrubber_value
                st.session_state.pending_scrubber_value = None

            # Scrubber (after buttons so it reflects updated state)
            current_step = st.slider(
                "Step",
                min_value=0,
                max_value=num_steps - 1,
                key="replay_scrubber",
                help="Drag to navigate through generation"
            )

            st.markdown("---")

            # Current step display
            step = timeline.steps[current_step]

            col_step_info, col_response = st.columns([1, 2])

            with col_step_info:
                st.markdown(f"### Step {current_step + 1} of {num_steps}")
                st.markdown(f"**Token:** `{step.token_text}`")
                st.markdown(f"**Token ID:** {step.token_id}")
                st.markdown(f"**Probability:** {step.probability:.2%}")

                # FLOPs calculations
                topology = st.session_state.get('topology', {})
                if topology.get("config"):
                    input_len = len(timeline.input_tokens)
                    current_seq_len = input_len + current_step + 1

                    # FLOPs for this token (generation step)
                    this_token_flops = calculate_transformer_flops(
                        topology, seq_len=current_seq_len, is_prefill=False, num_new_tokens=1
                    )

                    # Cumulative FLOPs: prefill + all generation steps up to current
                    prefill_flops = calculate_transformer_flops(
                        topology, seq_len=input_len, is_prefill=True
                    )

                    # Sum generation FLOPs for each step (seq_len increases each step)
                    cumulative_gen_flops = 0
                    for s in range(current_step + 1):
                        step_seq_len = input_len + s + 1
                        step_flops = calculate_transformer_flops(
                            topology, seq_len=step_seq_len, is_prefill=False, num_new_tokens=1
                        )
                        cumulative_gen_flops += step_flops["total"]

                    total_flops = prefill_flops["total"] + cumulative_gen_flops

                    st.markdown("---")
                    st.markdown("**FLOPs:**")
                    st.caption(f"This token: {format_flops(this_token_flops['total'])}")
                    st.caption(f"Input ({input_len} tok): {format_flops(prefill_flops['total'])}")
                    st.caption(f"**Total: {format_flops(total_flops)}**")

                # Show alternatives as compact bar chart
                st.markdown("**Alternatives:**")
                if step.top_k_alternatives:
                    alt_df_data = []
                    for i, (tok_id, prob, tok_text) in enumerate(step.top_k_alternatives[:6]):
                        is_selected = tok_id == step.token_id
                        display_text = tok_text.replace('\n', '\\n')[:12]
                        alt_df_data.append({
                            'Token': display_text,
                            'Probability': prob * 100,
                            'Selected': 'Selected' if is_selected else 'Alternative',
                        })

                    alt_df = pd.DataFrame(alt_df_data)
                    fig_alts = go.Figure()
                    selected_df = alt_df[alt_df['Selected'] == 'Selected']
                    fig_alts.add_trace(go.Bar(
                        x=selected_df['Token'],
                        y=selected_df['Probability'],
                        name='Selected',
                        marker_color='#28A745',
                        text=[f"{p:.0f}%" for p in selected_df['Probability']],
                        textposition='auto'
                    ))
                    alt_only_df = alt_df[alt_df['Selected'] == 'Alternative']
                    fig_alts.add_trace(go.Bar(
                        x=alt_only_df['Token'],
                        y=alt_only_df['Probability'],
                        name='Alt',
                        marker_color='#6C757D',
                        text=[f"{p:.0f}%" for p in alt_only_df['Probability']],
                        textposition='auto'
                    ))
                    fig_alts.update_layout(
                        height=200,
                        margin=dict(l=0, r=0, t=10, b=0),
                        showlegend=False,
                        xaxis_title="",
                        yaxis_title=""
                    )
                    st.plotly_chart(fig_alts, use_container_width=True)
                else:
                    st.caption("No alternatives captured")

            with col_response:
                # Get reasoning and answer from results
                reasoning_text = results.reasoning_text if hasattr(results, 'reasoning_text') else ""
                answer_text = results.answer_text if hasattr(results, 'answer_text') else ""

                # Jump to token selector
                st.markdown("**Click token to jump:** *(or use selector)*")

                # Build token options for dropdown
                token_options = []
                for i, s in enumerate(timeline.steps):
                    # Truncate and clean token text for display
                    tok_display = s.token_text.replace('\n', '↵').replace('\t', '→')
                    if len(tok_display) > 20:
                        tok_display = tok_display[:17] + "..."
                    token_options.append(f"{i}: {tok_display}")

                # Token selector that jumps to position (uses callback to avoid state conflict)
                # Include timeline_id in key to force refresh when switching timelines
                selector_key = f"token_jump_selector_{timeline.timeline_id}"

                def make_on_token_select(key):
                    def on_token_select():
                        selected = st.session_state[key]
                        st.session_state.replay_scrubber = selected
                    return on_token_select

                st.selectbox(
                    "Jump to token",
                    range(len(token_options)),
                    index=current_step,
                    format_func=lambda i: token_options[i],
                    key=selector_key,
                    label_visibility="collapsed",
                    on_change=make_on_token_select(selector_key)
                )

                # Build clickable token display using columns of buttons
                st.markdown("**Response tokens:** *(click to jump)*")

                def render_clickable_tokens(timeline, current_step, tokens_per_row=10):
                    """Render tokens as clickable buttons in a flow layout."""
                    num_tokens = len(timeline.steps)

                    # Use HTML + CSS for a flow layout with visual styling
                    # Since we can't use JS clicks, we'll use the visual display
                    # and rely on the dropdown above for clicking

                    html_parts = []
                    html_parts.append("""
                    <style>
                    .token-flow { display: flex; flex-wrap: wrap; gap: 2px; padding: 8px; background: #0e1117; border-radius: 5px; max-height: 350px; overflow-y: auto; }
                    .tok { padding: 2px 6px; border-radius: 3px; font-family: monospace; font-size: 12px; cursor: pointer; display: inline-block; margin: 1px; }
                    .tok-past { background: #3d1515; color: #ff6b6b; }
                    .tok-current { background: #ff3333; color: white; font-weight: bold; box-shadow: 0 0 8px #ff3333; }
                    .tok-future { background: #1a1a2e; color: #666666; }
                    .tok:hover { opacity: 0.8; transform: scale(1.05); }
                    .tok-idx { font-size: 9px; color: #888; vertical-align: super; margin-left: 1px; }
                    </style>
                    <div class="token-flow">
                    """)

                    for i, s in enumerate(timeline.steps):
                        tok_text = s.token_text
                        # Escape and format
                        tok_text = tok_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        tok_text = tok_text.replace('\n', '↵').replace(' ', '·').replace('\t', '→')
                        if len(tok_text) > 15:
                            tok_text = tok_text[:12] + "…"

                        if i < current_step:
                            css_class = "tok tok-past"
                        elif i == current_step:
                            css_class = "tok tok-current"
                        else:
                            css_class = "tok tok-future"

                        # Add title for hover tooltip
                        full_text = s.token_text.replace('"', '&quot;').replace('\n', '↵')
                        html_parts.append(f'<span class="{css_class}" title="Step {i}: {full_text}">{tok_text}<span class="tok-idx">{i}</span></span>')

                    html_parts.append('</div>')
                    return ''.join(html_parts)

                # Render the clickable token display
                token_html = render_clickable_tokens(timeline, current_step)
                st.markdown(token_html, unsafe_allow_html=True)

                st.caption("🔴 Past tokens | ⬜ Current | ⚫ Future — Use dropdown above to jump")

                # Show content based on whether viewing original or branch
                if is_viewing_branch:
                    # For branches, show the generated text from the branch timeline
                    with st.expander("🌿 Branch Generated Text", expanded=True):
                        if timeline.steps:
                            last_step = timeline.steps[-1]
                            if last_step.cumulative_tokens and tokenizer:
                                branch_text = tokenizer.decode(last_step.cumulative_tokens)
                                # Try to extract just the generated part (after input)
                                if timeline.input_tokens:
                                    input_text = tokenizer.decode(timeline.input_tokens)
                                    if branch_text.startswith(input_text):
                                        branch_text = branch_text[len(input_text):]
                                st.text(branch_text[:1500] + ("..." if len(branch_text) > 1500 else ""))
                            else:
                                # Fallback: concatenate token texts
                                branch_text = "".join(s.token_text for s in timeline.steps)
                                st.text(branch_text[:1500] + ("..." if len(branch_text) > 1500 else ""))
                        else:
                            st.caption("No tokens generated")

                        # Show branch info
                        if timeline.branch_point_step is not None:
                            bp = timeline.branch_point_step
                            branch_tok = timeline.steps[bp].token_text if bp < len(timeline.steps) else "?"
                            orig_tok = original_timeline.steps[bp].token_text if bp < len(original_timeline.steps) else "?"
                            st.caption(f"Branched at step {bp}: `{orig_tok}` → `{branch_tok}`")
                else:
                    # Show original reasoning section if present
                    if reasoning_text:
                        with st.expander("🧠 Reasoning", expanded=False):
                            st.text(reasoning_text[:1000] + ("..." if len(reasoning_text) > 1000 else ""))

                    # Show original final answer section if present
                    if answer_text and answer_text != results.generated_text:
                        with st.expander("📝 Final Answer", expanded=True):
                            st.markdown(answer_text[:500] + ("..." if len(answer_text) > 500 else ""))

            # Show last branch debug info if available
            if 'branch_debug_log' in st.session_state and st.session_state.branch_debug_log:
                with st.expander("🔍 Last Branch Debug Log", expanded=True):
                    for line in st.session_state.branch_debug_log:
                        st.text(line)
                    if st.button("Clear Log", key="clear_branch_log"):
                        st.session_state.branch_debug_log = []
                        st.rerun()

            # Branch dialog - controlled by session state to persist across reruns
            if st.session_state.get('show_branch_dialog') and not is_viewing_branch:
                st.markdown("---")
                col_title, col_close = st.columns([4, 1])
                with col_title:
                    st.markdown("### Create Branch")
                with col_close:
                    if st.button("✕ Close", key="close_branch_dialog"):
                        st.session_state.show_branch_dialog = False
                        st.session_state.branch_selection_confirmed = False
                        st.session_state.pending_branch_token_id = None
                        st.session_state.pending_branch_token_text = None
                        st.session_state.pending_branch_step = None
                        st.rerun()

                # Show what we're replacing
                st.markdown(f"**Step {current_step}:** Original token = `{step.token_text}` (ID: {step.token_id})")

                # Initialize branch selection state if needed
                if 'branch_selection_confirmed' not in st.session_state:
                    st.session_state.branch_selection_confirmed = False

                # Two modes: select from alternatives OR search vocabulary
                branch_mode = st.radio(
                    "Select replacement token:",
                    ["From alternatives", "Search vocabulary"],
                    horizontal=True,
                    key="branch_mode"
                )

                selected_tok_id = None
                selected_tok_text = None

                if branch_mode == "From alternatives":
                    if step.top_k_alternatives:
                        branch_options = [(tok_id, prob, tok_text) for tok_id, prob, tok_text in step.top_k_alternatives if tok_id != step.token_id]

                        if branch_options:
                            option_labels = [f"`{tok_text}` (ID:{tok_id}, {prob:.1%})" for tok_id, prob, tok_text in branch_options[:8]]
                            selected_option = st.selectbox(
                                "Alternative token",
                                range(len(option_labels)),
                                format_func=lambda i: option_labels[i],
                                key="branch_token_select"
                            )
                            selected_tok_id, _, selected_tok_text = branch_options[selected_option]
                        else:
                            st.warning("No alternatives available")
                    else:
                        st.warning("No alternatives captured for this step")

                else:  # Search vocabulary
                    search_text = st.text_input(
                        "Enter token text to search:",
                        placeholder="e.g., 'hello', ' the', 'Paris'",
                        key="vocab_search"
                    )

                    if search_text:
                        # Try to encode the search text
                        try:
                            if hasattr(tokenizer, 'encode'):
                                found_tokens = tokenizer.encode(search_text)
                            else:
                                found_tokens = tokenizer(search_text)
                            if isinstance(found_tokens, dict):
                                found_tokens = found_tokens.get('input_ids', [])

                            if found_tokens:
                                # Show found tokens
                                st.write(f"Found {len(found_tokens)} token(s):")
                                token_choices = []
                                for tid in found_tokens[:10]:
                                    try:
                                        ttext = tokenizer.decode([tid])
                                        token_choices.append((tid, ttext))
                                        st.caption(f"  ID {tid}: `{ttext}`")
                                    except:
                                        token_choices.append((tid, f"[{tid}]"))

                                if token_choices:
                                    selected_vocab = st.selectbox(
                                        "Select token:",
                                        range(len(token_choices)),
                                        format_func=lambda i: f"ID {token_choices[i][0]}: `{token_choices[i][1]}`",
                                        key="vocab_token_select"
                                    )
                                    selected_tok_id, selected_tok_text = token_choices[selected_vocab]
                            else:
                                st.warning("No tokens found for that text")
                        except Exception as e:
                            st.error(f"Error encoding: {e}")

                # STEP 1: User must click "Confirm Selection" to lock in their choice
                st.markdown("---")
                if selected_tok_id is not None:
                    st.info(f"Preview: `{step.token_text}` → `{selected_tok_text}` (ID: {selected_tok_id})")

                    if st.button("✓ Confirm Selection", key="confirm_branch_selection"):
                        st.session_state.pending_branch_token_id = selected_tok_id
                        st.session_state.pending_branch_token_text = selected_tok_text
                        st.session_state.pending_branch_step = current_step
                        st.session_state.branch_selection_confirmed = True
                        st.rerun()

                # STEP 2: Show generate button only after selection is confirmed
                # Also verify pending_step matches current_step (in case user navigated)
                pending_step = st.session_state.get('pending_branch_step')
                if (st.session_state.get('branch_selection_confirmed') and
                    st.session_state.get('pending_branch_token_id') is not None and
                    pending_step == current_step):

                    pending_id = st.session_state.pending_branch_token_id
                    pending_text = st.session_state.pending_branch_token_text

                    st.success(f"**Confirmed:** Step {pending_step}, replace `{original_timeline.steps[pending_step].token_text}` with `{pending_text}` (ID: {pending_id})")

                    col_branch_btn, col_branch_cancel = st.columns(2)
                    with col_branch_btn:
                        generate_clicked = st.button("🌿 Generate Branch Now", type="primary", key="do_branch_final")

                    with col_branch_cancel:
                        if st.button("Cancel", key="cancel_branch"):
                            st.session_state.show_branch_dialog = False
                            st.session_state.pending_branch_token_id = None
                            st.session_state.pending_branch_token_text = None
                            st.session_state.pending_branch_step = None
                            st.session_state.branch_selection_confirmed = False
                            st.rerun()

                    # Only execute if button was actually clicked this run
                    if generate_clicked:
                        debug_log = []
                        debug_log.append(f"=== Branch Generation ===")
                        debug_log.append(f"Step: {pending_step}")
                        debug_log.append(f"Original: '{original_timeline.steps[pending_step].token_text}' (ID: {original_timeline.steps[pending_step].token_id})")
                        debug_log.append(f"Alternative: '{pending_text}' (ID: {pending_id})")

                        with st.spinner(f"Generating branch with `{pending_text}`..."):
                            try:
                                prober = st.session_state.get('prober')
                                if prober:
                                    # Use same max_tokens as the initial generation (from sidebar)
                                    branch_max_tokens = gen_max_tokens
                                    debug_log.append(f"Generating up to {branch_max_tokens} tokens (same as initial)")

                                    new_timeline = prober.generate_from_branch(
                                        timeline=original_timeline,
                                        branch_step=pending_step,
                                        alternative_token_id=pending_id,
                                        max_tokens=branch_max_tokens
                                    )

                                    debug_log.append(f"Branch created: {len(new_timeline.steps)} steps")

                                    # Add internal debug info from the branch generation
                                    if hasattr(new_timeline, 'debug_info') and new_timeline.debug_info:
                                        debug_log.append("--- Internal Branch Debug ---")
                                        debug_log.extend(new_timeline.debug_info)

                                    # Compare tokens
                                    branch_toks = []
                                    orig_toks = []
                                    if new_timeline.steps and pending_step < len(new_timeline.steps):
                                        branch_toks = [s.token_text for s in new_timeline.steps[pending_step:min(pending_step+10, len(new_timeline.steps))]]
                                        debug_log.append(f"Branch tokens: {branch_toks}")

                                    if pending_step < len(original_timeline.steps):
                                        orig_toks = [s.token_text for s in original_timeline.steps[pending_step:min(pending_step+10, len(original_timeline.steps))]]
                                        debug_log.append(f"Original tokens: {orig_toks}")

                                    # Check if different
                                    if branch_toks and orig_toks:
                                        if branch_toks == orig_toks:
                                            debug_log.append("⚠️ WARNING: Branch tokens IDENTICAL to original!")
                                        else:
                                            debug_log.append("✓ OK: Branch tokens differ from original")

                                    st.session_state.branch_debug_log = debug_log
                                    st.session_state.replay_branches[new_timeline.timeline_id] = new_timeline
                                    st.session_state.viewing_branch_id = new_timeline.timeline_id
                                    # Use pending value to update scrubber on next rerun (can't modify after widget rendered)
                                    st.session_state.pending_scrubber_value = pending_step

                                    # Clear pending, confirmation, and close dialog
                                    st.session_state.show_branch_dialog = False
                                    st.session_state.pending_branch_token_id = None
                                    st.session_state.pending_branch_token_text = None
                                    st.session_state.pending_branch_step = None
                                    st.session_state.branch_selection_confirmed = False

                                    st.rerun()
                                else:
                                    debug_log.append("ERROR: Prober not available")
                                    st.session_state.branch_debug_log = debug_log
                                    st.error("Prober not available. Please run inference first.")
                            except Exception as e:
                                import traceback
                                debug_log.append(f"ERROR: {str(e)}")
                                debug_log.append(traceback.format_exc())
                                st.session_state.branch_debug_log = debug_log
                                st.error(f"Branch generation failed: {str(e)}")

            # Timeline selector (original vs branches)
            st.markdown("---")

            # Check if we're viewing a branch
            viewing_branch = st.session_state.get('viewing_branch_id', None)

            if viewing_branch or st.session_state.replay_branches:
                st.markdown("### Timeline Selection")

                # Build timeline options
                timeline_options = {"original": "Original Timeline"}
                for bid, bt in st.session_state.replay_branches.items():
                    branch_token = bt.steps[bt.branch_point_step].token_text if bt.steps and bt.branch_point_step < len(bt.steps) else "?"
                    timeline_options[bid] = f"Branch {bid}: '{branch_token}' at step {bt.branch_point_step}"

                col_timeline, col_clear = st.columns([3, 1])

                with col_timeline:
                    selected_timeline = st.selectbox(
                        "View timeline",
                        list(timeline_options.keys()),
                        index=0 if viewing_branch is None else list(timeline_options.keys()).index(viewing_branch) if viewing_branch in timeline_options else 0,
                        format_func=lambda k: timeline_options[k],
                        key="timeline_selector"
                    )

                    # Switch timeline if changed
                    if selected_timeline == "original" and viewing_branch is not None:
                        st.session_state.viewing_branch_id = None
                        st.session_state.pending_scrubber_value = 0
                        st.rerun()
                    elif selected_timeline != "original" and selected_timeline != viewing_branch:
                        st.session_state.viewing_branch_id = selected_timeline
                        st.session_state.pending_scrubber_value = 0
                        st.rerun()

                with col_clear:
                    if st.session_state.replay_branches:
                        if st.button("🗑️ Clear branches", key="clear_branches"):
                            st.session_state.replay_branches = {}
                            st.session_state.viewing_branch_id = None
                            st.rerun()

            # Show branches info
            if st.session_state.replay_branches:
                with st.expander(f"📂 Branches ({len(st.session_state.replay_branches)})", expanded=False):
                    for branch_id, branch_timeline in st.session_state.replay_branches.items():
                        st.markdown(f"**Branch `{branch_id}`**")

                        # Show branch point info
                        if branch_timeline.steps and branch_timeline.branch_point_step is not None:
                            bp = branch_timeline.branch_point_step
                            branch_token = branch_timeline.steps[bp].token_text if bp < len(branch_timeline.steps) else "?"

                            # Find original token at same position
                            original_token = timeline.steps[bp].token_text if bp < len(timeline.steps) else "?"

                            st.caption(f"Step {bp}: `{original_token}` → `{branch_token}`")

                        # Show generated text
                        if branch_timeline.steps:
                            last_step = branch_timeline.steps[-1]
                            if last_step.cumulative_tokens:
                                branch_text = tokenizer.decode(last_step.cumulative_tokens) if tokenizer else f"[{len(last_step.cumulative_tokens)} tokens]"
                                st.text(branch_text[-200:] if len(branch_text) > 200 else branch_text)

                        col_view, col_del = st.columns(2)
                        with col_view:
                            if st.button(f"View", key=f"view_{branch_id}"):
                                st.session_state.viewing_branch_id = branch_id
                                st.session_state.pending_scrubber_value = 0
                                st.rerun()
                        with col_del:
                            if st.button(f"Delete", key=f"del_{branch_id}"):
                                del st.session_state.replay_branches[branch_id]
                                if st.session_state.get('viewing_branch_id') == branch_id:
                                    st.session_state.viewing_branch_id = None
                                st.rerun()

                        st.markdown("---")

    # Export section (not a tab, like AFM7)
    st.header("💾 Export")

    col_export1, col_export2, col_export3 = st.columns(3)

    with col_export1:
        if st.button("📄 Export JSON"):
            export_data = {
                'model': st.session_state.model_path,
                'input_text': results.input_text,
                'input_tokens': results.input_tokens,
                'generated_text': results.generated_text,
                'reasoning_text': results.reasoning_text,
                'answer_text': results.answer_text,
                'generated_tokens': results.generated_tokens,
                'num_layers': results.num_layers,
                'top_predictions': [
                    {'token': t[2], 'probability': t[1], 'id': t[0]}
                    for t in results.top_k_tokens[:10]
                ],
                'layer_norms': {
                    str(k): float(np.linalg.norm(v, axis=-1).mean())
                    for k, v in results.layer_outputs.items()
                }
            }
            st.download_button(
                "Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name="mlxlmprobe_results.json",
                mime="application/json"
            )

    with col_export2:
        if st.button("🌐 Export HTML"):
            with st.spinner("Generating HTML report..."):
                # Collect AI interpretations if enabled
                interpretations = {}
                if ai_interpret:
                    try:
                        interpretations["token_probs"] = get_token_prob_interpretation(
                            results, model, tokenizer, use_ai=True, interpreter=interp_choice)
                        interpretations["layer_activations"] = get_layer_activation_interpretation(
                            results, model, tokenizer, use_ai=True, interpreter=interp_choice)
                        interpretations["ffn"] = get_ffn_interpretation(
                            results, model, tokenizer, use_ai=True, interpreter=interp_choice)
                        interpretations["embeddings"] = get_embedding_interpretation(
                            results, model, tokenizer, use_ai=True, interpreter=interp_choice)
                        interpretations["logits"] = get_logits_interpretation(
                            results, model, tokenizer, use_ai=True, interpreter=interp_choice)
                        interpretations["layer_similarity"] = get_layer_similarity_interpretation(
                            results, model, tokenizer, use_ai=True, interpreter=interp_choice)
                    except Exception as e:
                        st.warning(f"Could not generate AI interpretations: {e}")

                try:
                    html_content = generate_html_report(
                        results,
                        st.session_state.model_path,
                        probe_config,
                        tokenizer,
                        interpretations if ai_interpret else None
                    )
                    st.download_button(
                        "📥 Download HTML",
                        data=html_content,
                        file_name="mlxlmprobe_report.html",
                        mime="text/html"
                    )
                    st.success("HTML report generated!")
                except Exception as e:
                    st.error(f"Failed to generate HTML: {e}")

    with col_export3:
        if PDF_AVAILABLE:
            if st.button("📑 Export PDF"):
                with st.spinner("Generating PDF report..."):
                    # Collect AI interpretations if enabled
                    interpretations = {}
                    if ai_interpret:
                        try:
                            interpretations["token_probs"] = get_token_prob_interpretation(
                                results, model, tokenizer, use_ai=True, interpreter=interp_choice)
                            interpretations["layer_activations"] = get_layer_activation_interpretation(
                                results, model, tokenizer, use_ai=True, interpreter=interp_choice)
                            interpretations["ffn"] = get_ffn_interpretation(
                                results, model, tokenizer, use_ai=True, interpreter=interp_choice)
                            interpretations["embeddings"] = get_embedding_interpretation(
                                results, model, tokenizer, use_ai=True, interpreter=interp_choice)
                            interpretations["logits"] = get_logits_interpretation(
                                results, model, tokenizer, use_ai=True, interpreter=interp_choice)
                            interpretations["layer_similarity"] = get_layer_similarity_interpretation(
                                results, model, tokenizer, use_ai=True, interpreter=interp_choice)
                        except Exception as e:
                            st.warning(f"Could not generate AI interpretations: {e}")

                    try:
                        pdf_bytes = generate_pdf_report(
                            results,
                            st.session_state.model_path,
                            probe_config,
                            interpretations if ai_interpret else None
                        )
                        st.download_button(
                            "📥 Download PDF",
                            data=pdf_bytes,
                            file_name="mlxlmprobe_report.pdf",
                            mime="application/pdf"
                        )
                        st.success("PDF report generated!")
                    except Exception as e:
                        st.error(f"Failed to generate PDF: {e}")
        else:
            st.warning("Install fpdf2 for PDF export: `pip install fpdf2`")


if __name__ == "__main__":
    main()
