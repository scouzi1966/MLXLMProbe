# MLXLMProbe - Project Context

## Mechanistic Interpretability Resources

- **MI Glossary**: https://www.neelnanda.io/mechanistic-interpretability/glossary
- **TransformerLens**: https://github.com/TransformerLensOrg/TransformerLens

## Experimental: NumPy → MLX GPU Acceleration (2026-01-28)

**Branch:** `experimental/numpy-to-mlx-gpu`
**Commit:** `d40b2a9`
**Goal:** Replace heavy NumPy computations with MLX for Apple Silicon GPU acceleration (also compatible with mlx-cuda on Linux)

### MLX Utility Functions Added

| Function | Purpose |
|----------|---------|
| `mlx_softmax()` | GPU-accelerated softmax |
| `mlx_norm()` | GPU-accelerated L2 norm |
| `mlx_entropy()` | GPU-accelerated entropy calculation |
| `mlx_cosine_similarity_matrix()` | Batch matrix multiply for pairwise similarity |
| `mlx_to_numpy()` / `numpy_to_mlx()` | Conversion helpers |

### High-Impact Operations Replaced

| Location | Operation | Change |
|----------|-----------|--------|
| Logit Lens (~line 1920) | Softmax | Compute on GPU before NumPy conversion |
| Token Probs (~line 2380) | Softmax | Compute on GPU before NumPy conversion |
| Full Sequence Logit Lens (~line 2430) | Softmax | Compute on GPU before NumPy conversion |
| Layer Similarity (~line 4630) | O(n²) cosine similarity | Single `mx.matmul()` call |

### Not Changed (data already NumPy)

- Entropy calculations in interpretation functions
- Norm calculations in visualization functions
- Statistical operations for plotting (Plotly requires NumPy)

### Assessment

- **Probability of success:** 85%
- **Biggest win:** Layer similarity matrix - eliminates O(n²) nested loop
- **Key principle:** Minimize MLX↔NumPy transfers by computing on GPU before conversion
