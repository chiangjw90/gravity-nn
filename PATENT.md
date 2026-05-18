# Patent Notice and Licensing

**STATUS:** The Gravity neural network architecture, the density bottleneck mechanism, the multi-scale exponential kernel field solver with parallel associative scan, the directional scan decomposition for multi-dimensional lattices, the physics feature extraction (potential / gradient / curvature / entropy / centroid / tidal), and related embodiments described in this repository and in the accompanying paper are subject to a United States provisional patent application filed by Chia-Wei Chiang.

**Patent pending.**

---

## Dual-License Structure

This software is offered under two licensing options. Your choice of option determines the applicable patent terms.

### Option 1: AGPL-3.0

Open-source use under the GNU Affero General Public License version 3.0. Under this option, the patent license granted by AGPL-3.0 Section 11 applies to your use of this software, subject to the conditions and limitations specified therein.

The following uses are permitted under AGPL-3.0 without additional patent license:

- **Academic research:** reading the code, running it for research purposes, citing the paper in academic publications, reproducing the experiments reported in the paper
- **Personal evaluation:** running the code on personal machines to understand the architecture
- **Educational use:** teaching the architecture in coursework, presentations, blog posts, or tutorials (with appropriate citation)
- **Reproducing benchmarks:** running the included benchmark scripts to verify reported numbers
- **Open-source derivatives:** building and distributing modified versions, provided all AGPL-3.0 obligations are met (including source disclosure for network services per Section 13)

### Option 2: Commercial License

A separate commercial license is available for use cases where AGPL-3.0 obligations are not acceptable. Patent licensing terms are negotiated separately as part of the commercial license agreement and may extend beyond the scope of AGPL-3.0 Section 11.

The following uses require a commercial license:

- **Proprietary deployment:** integrating Gravity into a closed-source product, service, API, or platform
- **Commercial SaaS:** deploying Gravity as part of a network service without disclosing source code
- **Embedded/hardware:** implementing Gravity in SoC, ASIC, FPGA, or neuromorphic hardware
- **Commercial training:** training Gravity-based models on commercial datasets for commercial use of the trained weights
- **Licensing or sublicensing:** offering Gravity (in original or modified form) as part of a licensed software offering

### No License Granted Otherwise

No patent license is granted, expressly or by implication, for any use that does not comply with either Option 1 or Option 2 above.

---

## Specifically Reserved Patent Claims

- The density bottleneck mechanism (compressing K-dimensional field parameters to scalar density via a learnable nonlinear projection before propagation)
- The directional scan decomposition into 2 x D one-dimensional scans for D-dimensional lattices
- The combination of scalar density propagation with extracted physics features as a unified inference pipeline
- The bounded streaming state architecture for autoregressive / online inference

---

## How to Obtain a Commercial License

Send a brief inquiry to: **chiangjw90@gmail.com**

Please include:
1. Your organization
2. Intended use case (one or two paragraphs)
3. Approximate scale of deployment (research / pilot / production)

Licensing terms are negotiable and depend on use case. Initial conversations are free and do not commit either party.

---

## Why This Dual Structure

This repository is published to maximize academic reproducibility, technical scrutiny, and community feedback. The patent ensures the same work cannot be appropriated and commercialized without the rights holder's participation. We believe both reproducibility and intellectual property protection are essential for sustainable research by independent inventors.

If you want to use Gravity in a commercial setting, please reach out — we are friendly and want to make it work.

---

## Citation

If you use Gravity in academic work, please cite:

```bibtex
@article{chiang2026gravity,
  title={Gravity: A Physics-Inspired O(N) Framework with O(1) Streaming State Across Dimensions},
  author={Chiang, Chia-Wei},
  year={2026},
  doi={10.5281/zenodo.20273259},
  url={https://doi.org/10.5281/zenodo.20273259}
}
```

---

## No Warranty

This software is provided as-is, without warranty of any kind. Patent pending status does not warrant the validity, scope, or enforceability of any patent claim. Users should consult their own legal counsel for IP-related decisions affecting their work.

---

*Last updated: May 2026*
*Patent application status: provisional filed March 2026; non-provisional conversion planned within 12 months.*
