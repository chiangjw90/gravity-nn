# Gravity — Reproduction Notebooks

This package accompanies the paper:

> Chiang, C.-W. (2026). *Gravity: A Physics-Inspired O(N) Framework with O(1) Streaming State Across Dimensions*. Zenodo.

Three self-contained Jupyter notebooks reproduce the small-scale benchmarks reported
in **Tables 2, 10, and 11** of the paper — covering 1D sequences, 2D images, and 3D
volumes with the same underlying architecture.

---

## Contents

| Notebook | Paper Table | Domain | Target Metric | Params |
|---|---|---|---|---|
| `gravity_1d_wikitext2.ipynb` | Table 2 | 1D character-level language modelling | Test PPL ≈ 4.36 | ~486K |
| `gravity_2d_cifar10.ipynb` | Table 10 | 2D image classification | Test accuracy ≈ 82.3% | ~653K |
| `gravity_3d_organmnist.ipynb` | Table 11 | 3D medical volume classification | Test accuracy ≈ 89.7% | ~114K |

All three notebooks share the **same core architecture** — density bottleneck,
multi-scale field propagation, local window attention, and physics feature extraction.
Only the field solver changes:

- **1D**: 1 causal scan along the sequence
- **2D**: 4 directional scans (→ ← ↓ ↑) on the patch grid
- **3D**: 6 directional scans (±x, ±y, ±z) on the volumetric grid

This is the unified-framework claim demonstrated empirically.

---

## Quick start

### Option 1: Google Colab (no setup required)

1. Upload a notebook to https://colab.research.google.com
2. Runtime → Change runtime type → **GPU (T4 or A100)**
3. Runtime → Run all
4. Wait for the data download and training to complete

### Option 2: Local installation

```bash
pip install torch torchvision datasets medmnist numpy
jupyter notebook gravity_1d_wikitext2.ipynb
```

Then run cells top-to-bottom. Each notebook is self-contained and handles its own
data download.

---

## Expected runtime

| Notebook | NVIDIA A100 | Colab T4 (free) | CPU |
|---|---|---|---|
| 1D (20 epochs) | ~8 min | ~25 min | several hours |
| 2D (50 epochs) | ~60 min | ~3 hours | not recommended |
| 3D (50 epochs) | ~10 min | ~30 min | ~2 hours |

The 3D notebook is the fastest end-to-end and is recommended as a first sanity check
before running the longer 1D and 2D experiments.

---

## Reproduction expectations

These are **single-seed** reproductions (seed=42). Single-seed variance on small
datasets is typically ±0.2 PPL on WikiText-2 character-level and ±2–3% accuracy on
CIFAR-10 and OrganMNIST3D. The acceptable reproduction ranges are:

| Notebook | Paper Value | Acceptable Range |
|---|---|---|
| 1D (Test PPL) | 4.36 | 4.10 – 4.70 |
| 2D (Test Accuracy) | 82.3% | 80.0% – 84.0% |
| 3D (Test Accuracy) | 89.7% | 85.0% – 93.0% |

The paper's Table 2 reports single-seed PPL=4.36; Table 21 reports a 3-seed mean of
4.53. Results within these ranges are consistent with the paper. Results outside
these ranges may indicate hardware-specific numerical differences (e.g., MPS vs.
CUDA, fp32 vs. mixed precision) or seed variation rather than a fundamental
discrepancy.

---

## Datasets

All datasets are public and auto-downloaded by the notebooks:

| Notebook | Dataset | Source | License |
|---|---|---|---|
| 1D | WikiText-2 (raw character-level) | HuggingFace `datasets` or Salesforce raw | CC-BY-SA |
| 2D | CIFAR-10 | `torchvision.datasets` | MIT-like |
| 3D | OrganMNIST3D (MedMNIST v2) | `medmnist` pip package | CC-BY-4.0 |

Total data download: approximately 200 MB (most of which is CIFAR-10).

---

## What is being reproduced

These notebooks reproduce the **small-scale benchmarks** reported in the paper.
They do **not** reproduce:

- 100M-scale OpenWebText results (Section 4.4–4.6) — single training run takes days
- 4D Moving MNIST video classification (Section 4.10)
- 2D/3D diffusion experiments (Sections 4.11–4.12)
- C-MAPSS RUL prediction (Section 4.7)
- Streaming state and memory benchmarks (Section 4.13–4.17)

The reference implementation at the [main repository](https://github.com/chiangjw90/gravity-nn)
covers the architectural components used across these benchmarks.

---

## Architecture notes

The provided notebooks are **single-channel** implementations (C=1, the most
compressed configuration). The 100M-scale results in the paper use multi-channel
density (C=16, K=64) with a 4× compression ratio; the small-scale notebooks use
the maximum compression (C=1, K=8 or 15) which is sufficient for these benchmarks.

The exact parameter counts may differ from the paper by a few percent due to
implementation details of the feature projection layer. The reproduction priority
is the test metric (PPL or accuracy), not the parameter count to the byte.

---

## Causal correctness (1D)

The 1D notebook is fully causal. Empirical verification: perturbing input tokens
at position t+1 onwards produces **bit-identical** outputs at positions 0..t for
all tested positions (max diff = 0.00e+00). This matches the paper's claim in
Section 5.3 that the architecture has zero future-information leakage.

To verify locally:

```python
x1 = torch.randint(0, vocab_size, (1, seq_len))
x2 = x1.clone()
x2[0, 33:] = torch.randint(0, vocab_size, (seq_len - 33,))
with torch.no_grad():
    out1, out2 = model(x1), model(x2)
diff = (out1[0, :33] - out2[0, :33]).abs().max()
assert diff == 0.0, "Causal leak detected"
```

---

## File outputs

Each notebook produces a checkpoint file on completion:

- `gravity_1d_wikitext2_555k.pt`
- `gravity_2d_cifar10_653k.pt`
- `gravity_3d_organmnist_114k.pt`

Each checkpoint contains:
- `model_state_dict` — trained weights
- `config` — all hyperparameters
- `metrics` — final and best validation/test metrics
- `history` — full training trajectory (loss, accuracy per epoch)
- SHA-256 hash printed for reproducibility verification

---

## Citation

If you use these notebooks or the Gravity architecture in your research:

```bibtex
@article{chiang2026gravity,
  title={Gravity: A Physics-Inspired O(N) Framework with O(1) Streaming State Across Dimensions},
  author={Chiang, Chia-Wei},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.20273259}
}
```

For the OrganMNIST3D dataset used in the 3D notebook:

```bibtex
@article{yang2023medmnist,
  title={MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
  author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
  journal={Scientific Data},
  volume={10},
  number={41},
  year={2023}
}
```

---

## License and patent notice

The notebooks themselves are licensed under **AGPL-3.0**.

The Gravity architecture implements technology covered by a pending U.S. provisional
patent application filed March 2026. Use under AGPL-3.0 is freely permitted for
academic and non-commercial research; commercial licensing is available separately.

Commercial inquiries: **chiangjw90@gmail.com**

Full reference implementation: **https://github.com/chiangjw90/gravity-nn**

---

## Troubleshooting

**1D notebook: `datasets` package not available**
The notebook falls back to downloading raw text files automatically. No action needed.

**3D notebook: `medmnist` install fails on Colab**
Run `!pip install medmnist` in a separate cell before running the notebook.

**Out-of-memory on small GPU (< 8 GB)**
Reduce `BATCH_SIZE` in the relevant cell:
- 1D: from 8 to 4
- 2D: from 128 to 64 or 32
- 3D: from 64 to 32

**Slower-than-expected training**
The notebooks use mixed precision (AMP) when CUDA is available. On CPU or MPS,
training will be substantially slower; consider running the 3D notebook first
(fastest end-to-end).

**PPL / accuracy outside the reproduction range**
Check that the random seed is set (SEED=42 at the top of each notebook), and that
you are running on a GPU with fp32 or AMP enabled. Single-seed variance on small
datasets can be ±0.2 PPL or ±2-3% accuracy; if your result is well outside the
ranges above, please open an issue at https://github.com/chiangjw90/gravity-nn

---

*This README accompanies the Zenodo record for this paper. For source code, issue
tracking, and updates beyond this snapshot, see the GitHub repository.*
