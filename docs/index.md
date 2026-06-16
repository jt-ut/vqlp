# vqlp

Tools for fitting vector quantizers under arbitrary L*p* metrics and computing
recall quantities — prototype assignments, quantization error, receptive fields,
and manifold connectivity via the CONN matrix.

!!! warning "Early-stage development"
    The API may change between commits.

## Overview

`vqlp` fits vector quantizers (VQ) to a data matrix and then *recalls* the data
through the learned prototypes to extract a suite of analysis products:

- **Prototype fitting** under arbitrary L*p* metrics, with several algorithms
  (random, IRLS, gradient descent, PAM, and FAISS k-means for *p = 2*).
- **Best-matching-unit (BMU) assignment** — which observations map to which
  prototype.
- **Quantization error (QE)** — per-observation distortion.
- **Receptive fields (RF)** — observations captured by each prototype.
- **CONN matrix** — sparse prototype-to-prototype connectivity describing the
  learned manifold.
- **Reconstruction** — hard and soft reconstruction from the prototypes.

## Quick start

```python
import numpy as np
from vqlp import VQFitter

X = np.random.randn(500, 2)          # your data matrix, shape (N, d)

fitter = VQFitter(M=20, p=2, max_bmu=2, random_state=42)
fitter.fit(X, method="kmeans", niter=30, nredo=3)

fitter.recall()                      # finalize recall products
recaller = fitter.recaller

bmu_ids  = recaller.BMU[:, 0]        # 1st BMU per observation
qe       = recaller.QE               # quantization error
conn     = recaller.CONN             # sparse (M x M) connectivity matrix
```

See the [API Reference](api.md) for full class and method documentation.
