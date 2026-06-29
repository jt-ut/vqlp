# vqlp

**Tools for fitting vector quantizers under arbitrary L*p* metrics and computing recall quantities** — prototype assignments, quantization error, receptive fields, and manifold connectivity via the CONN matrix.

> ⚠️ **Status: early-stage development.** The API may change between commits.

---

## Overview

`vqlp` fits vector quantizers (VQ) to a data matrix and then *recalls* the data through the learned prototypes to extract a suite of analysis products:

- **Prototype fitting** under arbitrary L*p* metrics (FAISS k-means backend for the Euclidean *p = 2* case).
- **Best-matching-unit (BMU) assignment** — which observations map to which prototype, with support for multiple BMUs per observation.
- **Quantization error (QE)** — per-observation distortion against assigned prototypes.
- **Receptive fields (RF)** — the set and size of observations captured by each prototype.
- **CONN matrix** — a sparse prototype-to-prototype connectivity graph describing the learned manifold structure.
- **Reconstruction** — hard and soft reconstruction of data from the prototypes.

There is nothing domain-specific here: `vqlp` operates on any numeric data matrix.

---

## Installation

`vqlp` is not yet on PyPI. Install from source:

```bash
git clone https://github.com/cosmic-learner/vqlp.git
cd vqlp
pip install -e .
```

The editable install (`-e`) lets you pull updates with `git pull` without reinstalling.

See [`REQUIREMENTS.txt`](REQUIREMENTS.txt) for dependencies. <!-- TODO: confirm minimum supported Python version -->

---

## Quick start

The user supplies two things: a **data matrix** `X` of shape `(n_observations, n_features)` and a desired **number of prototypes** `M`.

```python
import numpy as np
from vqlp import VQFitter

# X: your data matrix, shape (N, d)
X = np.random.randn(500, 2)

# 1. Fit M prototypes (p=2 uses the FAISS k-means backend)
fitter = VQFitter(M=20, p=2, max_bmu=2, random_state=42)
fitter.fit(X, method="kmeans", niter=30, nredo=3)

W = fitter.W            # prototype matrix, shape (M, d)

# 2. Recall the data through the learned prototypes
fitter.recall()         # finalizes RF, CONN, etc. on the training data
recaller = fitter.recaller

# 3. Inspect the recall products
summary = fitter.get_summary()

bmu_ids = recaller.BMU[:, 0]      # 1st BMU per observation
qe       = recaller.QE            # quantization error, shape (N, max_bmu)
rf_sizes = recaller.RFSize        # receptive-field size per prototype
conn     = recaller.CONN          # sparse (M x M) connectivity matrix

# 4. Reconstruct data from prototypes
X_hard = recaller.reconstruct(W, X, method="hard")
X_soft = recaller.reconstruct(W, X, method="soft")
```

A complete worked example — synthetic data, fitting, full recall, and plotting the BMU assignments and CONN graph — is in [`examples/example_vqlp.py`](examples/example_vqlp.py).

---

## Recall products at a glance

| Attribute | Shape | Description |
|---|---|---|
| `recaller.BMU` | `(N, max_bmu)` | Best-matching prototype index/indices per observation |
| `recaller.QE` | `(N, max_bmu)` | Quantization error against assigned prototype(s) |
| `recaller.RF` | list of arrays | Observation indices in each prototype's receptive field |
| `recaller.RFSize` | `(M,)` | Number of observations per receptive field |
| `recaller.CONN` | `(M, M)` sparse | Prototype connectivity (manifold) matrix |
| `recaller.CONN_nhbs` | list | Neighbour prototype indices per prototype |
| `recaller.CONN_nhbs_size` | `(M,)` | Connectivity degree per prototype |

Use `fitter.get_summary()` for aggregate quantities (e.g. number of connectivity edges, mean connectivity degree).

---

## Project structure

```
vqlp/
├── src/                     # package source
├── examples/
│   └── example_vqlp.py      # worked end-to-end example
├── README.md
└── LICENSE
```

---

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

---

## Author

**Josh Taylor**
Cosmic AI · Oden Institute for Computational Engineering & Sciences · The University of Texas at Austin

---

## Citation

No associated paper yet. <!-- TODO: add citation / BibTeX once published -->
