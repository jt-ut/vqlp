"""
vqlp.cadj_utils
===============
Utilities for the Cumulative Adjacency matrix of Prototypes (CADJ).

CADJ[i, j] counts the number of data points for which prototype i was
the first Best Matching Unit (BMU) and prototype j was the second, giving
a directed, density-weighted view of prototype neighbourhood topology.

Functions
---------
pad_CADJ
    Ensure every row of CADJ has at least ``min_nhbs`` nonzero entries by
    adding synthetic low-weight edges to the nearest unconnected prototypes.

CADJ_sigmas
    Compute the per-prototype local bandwidth sigma_i: the CADJ-weighted
    mean Euclidean distance from prototype i to its CADJ neighbours.

CADJ_self_tuning_kernel
    Full-pairwise self-tuning similarity kernel (Zelnik-Manor & Perona,
    2004) with bandwidths derived from CADJ neighbourhood structure.

References
----------
Zelnik-Manor, L. & Perona, P. (2004). Self-tuning spectral clustering.
    Advances in Neural Information Processing Systems, 17.
Tasdemir, K. & Merenyi, E. (2009). Exploiting data topology in
    visualization and clustering of self-organizing maps.
    IEEE Transactions on Neural Networks, 20(4), 549-562.
"""

from __future__ import annotations

__all__ = [
    "pad_CADJ",
    "CADJ_sigmas",
    "CADJ_self_tuning_kernel",
]

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform


# ---------------------------------------------------------------------------
# pad_CADJ
# ---------------------------------------------------------------------------

def pad_CADJ(
    CADJ: sp.spmatrix,
    CADJ_nhbs: list[list[int]],
    CADJ_nhbs_size: np.ndarray,
    W: np.ndarray,
    min_nhbs: int = 3,
    fill_val: int = 1,
) -> tuple[sp.csr_matrix, list[list[int]], np.ndarray]:
    """
    Pad a CADJ matrix so that every row has at least ``min_nhbs`` nonzero
    entries.

    Rows already satisfying the threshold are never touched.  For any
    prototype i with fewer than ``min_nhbs`` neighbours, synthetic directed
    edges with weight ``fill_val`` are added to the nearest (by Euclidean
    distance in W-space) unconnected prototypes.  Padding is one-directional:
    only row i is modified, preserving CADJ's asymmetric nature.

    Parameters
    ----------
    CADJ : scipy.sparse matrix, shape (M, M)
        Asymmetric co-adjacency matrix from VQRecaller.  A copy is made
        internally; the original is never mutated.
    CADJ_nhbs : list of list of int, length M
        Precomputed neighbour index lists (nonzero column indices per row).
    CADJ_nhbs_size : np.ndarray, shape (M,), dtype int
        Number of neighbours per prototype (``len(CADJ_nhbs[i])`` for each i).
    W : np.ndarray, shape (M, d)
        Prototype matrix.  Used to find geometrically nearest candidates for
        synthetic edges.

        .. note::
           For large M and d, ``np.linalg.norm(W - W[i], axis=1)`` inside the
           per-row loop allocates a full (M, d) array each iteration.  This is
           acceptable when few rows need padding (the common case).  If many
           rows are under-connected, consider precomputing the full distance
           matrix with ``scipy.spatial.distance.cdist`` before calling this
           function and passing row slices instead.

    min_nhbs : int, default 3
        Minimum number of nonzero entries required per row.
    fill_val : int, default 1
        Weight assigned to synthetic padding edges.  The default of 1
        is intentionally chosen to match CADJ's integer count semantics:
        a padding edge is treated as the weakest possible real connection
        (one co-mapping event), keeping PCADJ in integer dtype and
        avoiding implicit float promotion.

    Returns
    -------
    PCADJ : scipy.sparse.csr_matrix, shape (M, M), dtype int
        Padded CADJ matrix; every row has at least ``min_nhbs`` nonzero
        entries.  Dtype matches the input CADJ (integer counts) unless
        ``fill_val`` is changed to a float.
    PCADJ_nhbs : list of list of int, length M
        Neighbour index lists for each row of PCADJ.
    PCADJ_nhbs_size : np.ndarray, shape (M,), dtype int
        Number of neighbours per prototype in PCADJ.
    """
    PCADJ = CADJ.copy().tolil()
    PCADJ_nhbs = [list(nhbs) for nhbs in CADJ_nhbs]
    PCADJ_nhbs_size = CADJ_nhbs_size.copy()
    M = W.shape[0]

    for i in range(M):
        n_existing = PCADJ_nhbs_size[i]
        if n_existing >= min_nhbs:
            continue

        n_needed = min_nhbs - n_existing
        existing_cols = set(PCADJ_nhbs[i])

        # Distance from W[i] to all other prototypes
        dists = np.linalg.norm(W - W[i], axis=1)
        dists[i] = np.inf
        for j in existing_cols:
            dists[j] = np.inf

        # argpartition is O(M); avoids a full O(M log M) sort
        candidates = np.argpartition(dists, n_needed)[:n_needed]

        for j in candidates:
            if np.isfinite(dists[j]):
                PCADJ[i, j] = fill_val
                PCADJ_nhbs[i].append(int(j))
                PCADJ_nhbs_size[i] += 1

    return PCADJ.tocsr(), PCADJ_nhbs, PCADJ_nhbs_size


# ---------------------------------------------------------------------------
# CADJ_sigmas
# ---------------------------------------------------------------------------

def CADJ_sigmas(
    W: np.ndarray,
    CADJ: sp.spmatrix,
    CADJ_nhbs: list[list[int]] | None = None,
) -> np.ndarray:
    """
    Compute the per-prototype local bandwidth sigma_i from CADJ.

    For each prototype i, sigma_i is the CADJ-weighted mean Euclidean
    distance from W[i] to its CADJ neighbours::

        sigma_i = sum_j( CADJ[i,j] * dist(W[i], W[j]) )
                  / sum_j( CADJ[i,j] )

    This gives a locally adaptive scale that reflects the typical reach of
    prototype i's data-manifold neighbourhood.  Strongly populated CADJ
    edges (many data points co-mapped to the i-j pair) dominate the
    weighted mean, so sigma_i is driven by the dense core of the
    neighbourhood rather than by sparse boundary connections.

    Prototypes with no CADJ neighbours receive the median sigma of all
    well-connected prototypes (falling back to 1.0 if all rows are empty).
    If the caller has already padded CADJ with ``pad_CADJ``, no row will be
    empty and the fallback is never triggered.

    Parameters
    ----------
    W : np.ndarray, shape (M, d)
        Prototype matrix.
    CADJ : scipy.sparse matrix, shape (M, M)
        Asymmetric co-adjacency matrix.  Should be in CSR format for
        efficient row slicing; converted internally if not.
    CADJ_nhbs : list of list of int, length M, optional
        Precomputed neighbour index lists.  If ``None``, derived from the
        nonzero structure of CADJ (``CADJ.indices`` / ``CADJ.indptr``).
        Passing the precomputed lists avoids the CSR index extraction but
        is otherwise equivalent.

    Returns
    -------
    sigma : np.ndarray, shape (M,), dtype float64
        Per-prototype local bandwidth.  All values are strictly positive.
    """
    CADJ_csr = CADJ.tocsr()
    M = W.shape[0]

    # Derive neighbour lists from CSR structure if not supplied
    if CADJ_nhbs is None:
        CADJ_nhbs = [
            list(CADJ_csr.indices[CADJ_csr.indptr[i]:CADJ_csr.indptr[i + 1]])
            for i in range(M)
        ]

    sigma = np.empty(M, dtype=np.float64)

    for i in range(M):
        nhbs = CADJ_nhbs[i]
        if len(nhbs) == 0:
            sigma[i] = -1.0  # sentinel; replaced below
            continue

        dists = np.linalg.norm(W[nhbs] - W[i], axis=1)  # shape (k,)
        weights = np.asarray(
            CADJ_csr[i, nhbs].todense()
        ).ravel().astype(np.float64)
        wsum = weights.sum()

        sigma[i] = np.dot(weights, dists) / wsum if wsum > 0.0 else -1.0

    # Replace sentinels with median of valid sigmas
    valid = sigma > 0.0
    fallback = float(np.median(sigma[valid])) if valid.any() else 1.0
    sigma[~valid] = fallback

    return sigma


# ---------------------------------------------------------------------------
# CADJ_self_tuning_kernel
# ---------------------------------------------------------------------------

def CADJ_self_tuning_kernel(
    W: np.ndarray,
    CADJ: sp.spmatrix,
    CADJ_nhbs: list[list[int]] | None = None,
    min_similarity: float = 0.01,
    min_neighbors: int = 3,
) -> sp.csr_matrix:
    """
    Full-pairwise self-tuning similarity kernel with CADJ-derived bandwidths.

    Computes a symmetric M×M kernel matrix where the similarity between
    prototypes i and j is::

        K[i, j] = exp( -dist²(W[i], W[j]) / (sigma_i * sigma_j) )

    with sigma_i computed by :func:`CADJ_sigmas`.  Unlike a plain Gaussian
    kernel, the denominator adapts to local prototype density: tightly packed
    regions produce small sigma values (narrow kernel), while sparse regions
    produce large sigma values (broad kernel), making cluster boundaries
    visible without manual bandwidth selection.

    The kernel is evaluated for **all** (i, j) pairs regardless of whether
    CADJ[i, j] is nonzero.  CADJ enters only through sigma_i, not through
    the sparsity pattern.  This allows the Euclidean views (high-D and low-D)
    to introduce edges that CADJ did not encode, which is the intent of the
    multi-view fusion in ``mpec``.

    **Sparsification.**  Entries with K[i, j] < ``min_similarity`` are set to
    zero.  Because K[i, j] is already normalised by sigma_i * sigma_j in the
    exponent, a fixed cutoff on K values is effectively adaptive: K[i, j] = c
    means the same relative distance regardless of local scale.  Each row is
    guaranteed at least ``min_neighbors`` nonzero off-diagonal entries,
    overriding the threshold if necessary.

    **Symmetry.**  The kernel formula is symmetric by construction, but the
    per-row ``min_neighbors`` guarantee can break exact symmetry in edge cases.
    The output is explicitly symmetrised as ``(K + K.T) / 2``.

    **Diagonal.**  Returned as 1.0 (K[i, i] = exp(0) = 1).  The caller is
    responsible for zeroing or ignoring it if needed (e.g., igraph's
    ``graph_adjacency`` accepts a ``diag=False`` argument).

    .. note::
       Distance computation uses ``scipy.spatial.distance.pdist``, which
       computes only the M*(M-1)/2 unique pairs and is therefore both faster
       and more memory-efficient than a full ``cdist(W, W)`` call.  The result
       is expanded to a full (M, M) matrix via ``squareform`` for the
       subsequent kernel and masking steps.  For M ≳ 5 000 the squareform
       matrix itself (~200 MB at float64) may become a bottleneck; chunked or
       approximate-NN approaches can be substituted in a future version.

    Parameters
    ----------
    W : np.ndarray, shape (M, d)
        Prototype matrix.
    CADJ : scipy.sparse matrix, shape (M, M)
        Asymmetric co-adjacency matrix.  Used only to compute sigma_i via
        :func:`CADJ_sigmas`; does not gate which pairs receive a kernel value.
        Should be padded (via :func:`pad_CADJ`) before calling so that no row
        is empty.
    CADJ_nhbs : list of list of int, length M, optional
        Precomputed neighbour index lists.  Passed through to
        :func:`CADJ_sigmas`; derived from CADJ's CSR structure if ``None``.
    min_similarity : float, default 0.01
        Hard lower bound on kernel values retained after sparsification.
        Entries with K[i, j] < ``min_similarity`` are set to zero.
        Because K values are in (0, 1], this is equivalent to discarding
        pairs whose normalised squared distance exceeds
        ``-log(min_similarity) * sigma_i * sigma_j``.
    min_neighbors : int, default 3
        Minimum number of nonzero off-diagonal entries guaranteed per row
        after sparsification, regardless of ``min_similarity``.

    Returns
    -------
    K : scipy.sparse.csr_matrix, shape (M, M)
        Symmetric self-tuning kernel matrix.  Diagonal entries are 1.0.

    References
    ----------
    Zelnik-Manor, L. & Perona, P. (2004). Self-tuning spectral clustering.
        Advances in Neural Information Processing Systems, 17.
    """
    M = W.shape[0]

    # --- Step 1: per-prototype bandwidths ------------------------------------
    sigma = CADJ_sigmas(W=W, CADJ=CADJ, CADJ_nhbs=CADJ_nhbs)  # shape (M,)

    # --- Step 2: pairwise squared distances ----------------------------------
    # pdist computes only the M*(M-1)/2 unique pairs (upper triangle),
    # halving computation vs cdist(W, W).  squareform expands to (M, M).
    D2 = squareform(pdist(W, metric="sqeuclidean"))  # shape (M, M), float64

    # --- Step 3: self-tuning kernel ------------------------------------------
    # Sequential broadcast division avoids allocating a full (M, M) scales
    # matrix (np.outer(sigma, sigma)):
    #   D2 / sigma[:, None]  divides each row i by sigma_i
    #   / sigma[None, :]     divides each column j by sigma_j
    # giving  D2[i,j] / (sigma_i * sigma_j)  without an intermediate M×M array.
    K = np.exp(-D2 / sigma[:, None] / sigma[None, :])  # shape (M, M)

    # --- Step 4: sparsification ----------------------------------------------
    mask = K < min_similarity  # True where we want to zero out

    # Guarantee at least min_neighbors off-diagonal entries per row.
    # For rows where the threshold would leave too few entries, force-keep
    # the closest prototypes by squared Euclidean distance.
    n_kept = (~mask).sum(axis=1) - 1  # subtract 1 to exclude diagonal
    under = np.where(n_kept < min_neighbors)[0]
    for i in under:
        order = np.argsort(D2[i])      # ascending distance; index 0 is self
        force_keep = order[1: min_neighbors + 1]
        mask[i, force_keep] = False

    K[mask] = 0.0

    # --- Step 5: symmetrise --------------------------------------------------
    # Convert to sparse before symmetrisation to avoid holding two dense
    # M×M float64 arrays simultaneously.
    K_sparse = sp.csr_matrix(K)
    del K, D2, mask  # free dense buffers

    K_sym = (K_sparse + K_sparse.T) / 2.0

    return K_sym