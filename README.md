# Gravity: A Physics-Inspired O(N) Neural Network Architecture

**Replace Transformer's O(N²) attention with O(N) field-equation processing. 672 bytes inference state per layer.**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Commercial License](https://img.shields.io/badge/commercial_license-available-orange.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Patent](https://img.shields.io/badge/patent-pending-red.svg)](PATENT.md)

---

## What is Gravity?

Gravity compresses input representations to **scalar density**, propagates density through **multi-scale causal field equations**, and extracts **physics-derived features** within a fixed local window. The entire pipeline is O(N) — no N×N matrix is ever constructed.

```
Input → Parameter Projection → Density Bottleneck → Multi-Scale Field Solve
      → Local Window + Physics Features → Projection + FFN → Output
```

The architecture draws inspiration from classical field theory. Each token generates a "mass density" that creates a field propagating to surrounding positions — analogous to how mass generates gravity. But the architecture operates independently of any physical theory.

**Native multi-dimensional**: Same mechanism for 1D (language), 2D (image), 3D (medical volume), 4D (video). Only the number of scan directions changes.

## Key Results

| Metric | Gravity | Transformer | Ratio |
|--------|---------|-------------|-------|
| Inference state (per layer) | 672 bytes | ~4 GB @128K | **6,000,000x** |
| Memory @32K context | 140 MB | 12,902 MB | **92x less** |
| Throughput @16K context | 2.5M tok/s | 336K tok/s | **7.5x faster** |
| Memory @65K context | 411 MB | OOM (A100 40GB) | **Gravity runs, TF can't** |
| Streaming 1M tokens | 71 MB (constant) | Not feasible | **O(1) state** |

**Tradeoff**: ~10% cross-entropy loss gap vs Transformer at current scale (100M-1B token probes). The gap decelerates with scale (+4.6pp then +1.1pp) and trends toward stabilization.

## Install

```bash
pip install gravity-nn
```

Or install from source:

```bash
git clone https://github.com/chiangjw90/gravity-nn.git
cd gravity-nn
pip install -e .
```

## Quick Start

### 1D — Language Model

```python
from gravity import GravityLM

model = GravityLM(
    vocab_size=50257,
    max_seq_len=2048,
    d_model=768,
    n_layers=12,
)
# 672 bytes inference state per layer — same at 1K or 1M tokens
logits = model(token_ids)  # [B, N, vocab_size]
```

### 2D — Image Classification

```python
from gravity import Gravity2D

model = Gravity2D(
    num_classes=10,        # e.g. CIFAR-10
    img_size=32,
    patch_size=4,          # 32/4 = 8×8 grid
    d_model=128,
    n_layers=4,
)
logits = model(images)  # images: [B, 3, 32, 32] → logits: [B, 10]
```

### 3D — Volume Classification

```python
from gravity import Gravity3D

model = Gravity3D(
    num_classes=11,
    vol_size=16,           # 16×16×16 volume
    patch_size=4,
    d_model=64,
    n_layers=2,
)
logits = model(volumes)  # volumes: [B, 1, 16, 16, 16] → logits: [B, 11]
```

### 3D — Diffusion (e.g. Medical Volume Generation)

```python
from gravity import Gravity3DDiffusion

model = Gravity3DDiffusion(
    in_channels=1,
    vol_size=16,
    patch_size=4,
    d_model=64,
    n_layers=2,
)
# noise_pred = model(noisy_vol, timestep)
```

### 4D — Video Classification

```python
from gravity import Gravity4D

model = Gravity4D(
    num_classes=10,
    n_frames=8,
    img_size=32,
    patch_size=4,
    d_model=64,
    n_layers=2,
)
logits = model(video)  # video: [B, 8, 3, 32, 32] → logits: [B, 10]
```

### Using Building Blocks Directly

```python
from gravity.layers import (
    GravityBlock1D,        # Full block: density → field → attention → features → FFN
    GravityBlock2D,
    MCFieldSolver,         # 1D field solver (parallel associative scan)
    FieldSolver2D,         # 2D: 4-direction scan decomposition
    FieldSolver3D,         # 3D: 6-direction scan decomposition
    FieldSolver4D,         # 4D: 5-direction scan (temporal causal + spatial symmetric)
    LocalFieldAttention1D, # O(N) windowed attention from field quantities
    DensityBottleneck,     # K-dim → scalar compression: ρ = softplus(W · c²)
    PhysicsFeatures1D,     # Composite feature vector from field quantities
)
```

## Architecture

### Core Pipeline (per layer)

1. **Parameter Projection**: `params = pi * tanh(W * LayerNorm(x))` → K=64 dimensions, bounded
2. **Density Bottleneck**: `rho = softplus(W) * params^2` → C=16 scalar density channels (4x compression). This is the key innovation — compress before propagate.
3. **Multi-Scale Field Solve**: `phi(t) = alpha * phi(t-1) + beta * rho(t)` at S=3 scales simultaneously via parallel associative scan
4. **Local Window + Physics Features**: Scores from field potential/gradient differences (not Q/K inner product). 227-dimensional composite feature vector including density, potentials, gradients, entropy, curvature.
5. **Projection + FFN**: Gated residual connection + feedforward network

### Multi-Dimensional Scan Decomposition

| Dim | Directions | Description |
|-----|-----------|-------------|
| 1D  | 1 scan    | Causal (left→right) for language |
| 2D  | 4 scans   | ±x, ±y for images |
| 3D  | 6 scans   | ±x, ±y, ±z for volumes |
| 4D  | 5 scans   | 4 spatial (bidirectional) + 1 temporal (causal) for video |

### Why Not Just Another SSM?

| | Transformer | Mamba / SSM | **Gravity** |
|---|---|---|---|
| **Propagates** | All token pairs (N×N) | High-dim state (d=16-256) | **Scalar density** (1 value) |
| **Interaction** | Q/K inner product | Direct SSM output | **Field-derived physics features** |
| **State** | O(N×d) KV-cache | O(d_state) per layer | **O(1) = 672 bytes** per layer |
| **Multi-dim** | Flatten to 1D | Flatten to 1D | **Native 1D/2D/3D/4D** |
| **Insight** | Attend to everything | Compress to hidden state | **Compress → Propagate → Extract** |

## Reproducing Paper Results

```bash
# Quick start: small model on dummy data
python examples/quick_start.py

# Train on your own data
python examples/train_custom.py --data my_data.txt

# Larger model, more epochs
python examples/train_custom.py --data my_data.txt --d_model 768 --n_layers 12 --epochs 10
```

Full reproduction scripts (OpenWebText 100M, SlimPajama 400M, benchmarks) will be released alongside the arXiv paper.

## Model Configurations

| Config | d_model | Layers | K | C | S | Params | Use case |
|--------|---------|--------|---|---|---|--------|----------|
| `tiny` | 128 | 1 | 64 | 16 | 3 | ~555K | Quick experiments |
| `small` | 256 | 4 | 64 | 16 | 3 | ~15M | Research prototyping |
| `base` | 768 | 12 | 64 | 16 | 3 | ~128M | Standard benchmarks |

## Project Structure

```
gravity-nn/
├── gravity/
│   ├── __init__.py              # Unified exports + backward compat aliases
│   ├── layers/                  # Core building blocks
│   │   ├── density.py           # DensityBottleneck: K→scalar compression
│   │   ├── field_solver.py      # MCFieldSolver: 1D parallel associative scan
│   │   ├── field_solver_nd.py   # FieldSolver2D/3D/4D: multi-dir decomposition
│   │   ├── local_field_attention.py  # O(N) windowed attention (1D/2D/3D/4D)
│   │   ├── physics_features.py  # Composite feature extraction (1D/2D/3D/4D)
│   │   └── gravity_block.py     # Full blocks (1D/2D/3D/4D)
│   ├── models/                  # Complete models
│   │   ├── gravity_lm.py        # 1D language model
│   │   ├── gravity_2d.py        # 2D image classifier
│   │   ├── gravity_3d.py        # 3D volume classifier + diffusion
│   │   ├── gravity_4d.py        # 4D video classifier
│   │   └── transformer_lm.py    # Transformer baseline for benchmarking
│   └── utils/                   # Utilities
│       ├── seed.py              # Reproducibility
│       └── metrics.py           # Parameter counting, memory estimation
├── tests/                       # Test suite
│   ├── test_shapes.py           # Forward pass shape checks (all models)
│   ├── test_causality.py        # Causal masking verification
│   └── test_o1_state.py         # O(1) state property verification
├── examples/                    # Runnable examples
│   ├── quick_start.py           # Train a small model in 2 minutes
│   ├── train_custom.py          # Train on your own text data
│   └── train_demo_model.py      # Train the demo checkpoint
├── configs/                     # YAML model configurations
│   ├── tiny.yaml
│   ├── small.yaml
│   └── base.yaml
├── pyproject.toml
├── CITATION.cff
├── NOTICE
├── PATENT.md
└── LICENSE
```

> For data profiling and curriculum training, see **[gravity-labeler](https://github.com/chiangjw90/gravity-labeler)**.

## Labeler: Architecture-Agnostic Data Intelligence

The Gravity Labeler is a **separate technical contribution** that works with any model:

- A 5M-parameter Gravity probe computes `density_cv` and `phi_var_ratio` per document
- Documents are classified by information dependency structure, not length
- A 10,000-token FAQ may only need local context → short sequence
- A 500-token legal clause may need full context → long sequence
- Length-based methods can't distinguish these; Gravity's physics metrics can

**Result**: 43.8% of documents get different bucket assignments. Transformer PPL improves 1.0-1.9% at 400M tokens.

Install separately: `pip install gravity-labeler` — see **[gravity-labeler](https://github.com/chiangjw90/gravity-labeler)**

## Tests

```bash
pytest tests/ -v
```

## Citation

```bibtex
@article{chiang2026gravity,
  title={Gravity: A Physics-Inspired O(N) Framework with O(1) State Across Dimensions},
  author={Chiang, Chia-Wei},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}

@article{chiang2026gravitycurriculum,
  title={Gravity Curriculum: Data-Centric Training Optimization via Density-Derived Information Locality Labels},
  author={Chiang, Chia-Wei},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

This software is offered under two licensing options:

1. **AGPL-3.0** — for open-source use that complies with all AGPL-3.0 obligations, including source disclosure for network services (Section 13). Academic and research use is freely permitted under this option.

2. **Commercial License** — for use cases where AGPL-3.0 obligations are not acceptable (proprietary products, closed-source SaaS, embedded/hardware implementations). Contact chiangjw90@gmail.com

**Patent pending**: U.S. Provisional Patent Application filed March 2026. See [PATENT.md](PATENT.md) for details.

---

*Gravity: Because information, like mass, should propagate through fields — not require pairwise comparison of every particle in the universe.*
