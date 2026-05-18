# Patent Notice and Licensing

**STATUS:** The Gravity neural network architecture, the density bottleneck mechanism, the multi-scale exponential kernel field solver with parallel associative scan, the directional scan decomposition for multi-dimensional lattices, the physics feature extraction (potential / gradient / curvature / entropy / centroid / tidal), and related embodiments described in this repository and in the accompanying paper are subject to a United States provisional patent application filed by Chia-Wei Chiang.

**Patent pending.**

---

## What this means for users

### Permitted without separate license

- **Academic research:** reading the code, running it for research purposes, citing the paper in academic publications, reproducing the experiments reported in the paper
- **Personal evaluation:** running the code on personal machines to understand the architecture
- **Educational use:** teaching the architecture in coursework, presentations, blog posts, or tutorials (with appropriate citation)
- **Reproducing benchmarks:** running the included benchmarks scripts to verify reported numbers

These uses fall under the Apache License 2.0 grant in the `LICENSE` file.

### Requires separate license

The following uses are **not** authorized under the Apache 2.0 grant and require a separate written agreement with the rights holder:

- **Commercial deployment:** integrating Gravity into a product, service, API, or platform offered to customers (paid or free)
- **Commercial training:** training Gravity-based models on commercial datasets for the purpose of commercial use of the trained weights
- **Modified commercial works:** building modified versions of Gravity (including hybrids with other architectures) for commercial deployment
- **Licensing or sublicensing:** offering Gravity (in original or modified form) as part of a licensed software offering
- **Patent monetization:** asserting patents (your own or third parties') against Gravity or its users based on derivative concepts

### Specifically reserved

- The density bottleneck mechanism (compressing K-dimensional field parameters to scalar density via a learnable nonlinear projection before propagation)
- The directional scan decomposition into 2 × D one-dimensional scans for D-dimensional lattices
- The combination of scalar density propagation with extracted physics features as a unified inference pipeline
- The bounded streaming state architecture for autoregressive / online inference

---

## How to obtain a commercial license

Send a brief inquiry to: chiangjw90@gmail.com

Please include:
1. Your organization
2. Intended use case (one or two paragraphs)
3. Approximate scale of deployment (research / pilot / production)

Licensing terms are negotiable and depend on use case. Initial conversations are free and do not commit either party.

---

## Why this dual structure

This repository is published to maximize academic reproducibility, technical scrutiny, and community feedback. The patent ensures the same work cannot be appropriated and commercialized without the rights holder's participation. We believe both reproducibility and intellectual property protection are essential for sustainable research by independent inventors.

If you want to use Gravity in a commercial setting, please reach out—we are friendly and want to make it work.

---

## Citation

If you use Gravity in academic work, please cite:

```bibtex
@article{chiang2026gravity,
  title={Gravity: A Physics-Inspired O(N) Framework with O(1) Streaming State Across Dimensions},
  author={Chiang, Chia-Wei},
  year={2026},
  journal={arXiv preprint arXiv:XXXX.XXXXX}
}
```

---

## No warranty

This software is provided as-is, without warranty of any kind. Patent pending status does not warrant the validity, scope, or enforceability of any patent claim. Users should consult their own legal counsel for IP-related decisions affecting their work.

---

*Last updated: [date of repo public release]*
*Patent application status: provisional filed [filing date]; non-provisional conversion planned within 12 months.*
