"""
vqlp.cadj_utils
===============
Utilities for the Cumulative Adjacency matrix of Prototypes (CADJ).

CADJ[i, j] counts the number of data points for which prototype i was
the first Best Matching Unit (BMU) and prototype j was the second, giving
a directed, density-weighted view of prototype neighborhood topology.

Functions
---------
pad_CADJ
    Ensure every row of CADJ has at least ``min_nhbs`` nonzero entries by
    adding synthetic low-weight edges to the nearest unconnected prototypes.

CADJ_sigmas
    Compute the per-prototype local bandwidth sigma_i: the CADJ-weighted
    mean Euclidean distance from prototype i to its CADJ neighbors.

CADJ_self_tuning_kernel
    Full-pairwise self-tuning similarity kernel (Zelnik-Manor & Perona,
    2004) with bandwidths derived from CADJ neighborhood structure.

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
    prototype i with fewer than ``min_nhbs`` neighbors, synthetic directed
    edges with weight ``fill_val`` are added to the nearest (by Euclidean
    distance in W-space) unconnected prototypes.  Padding is one-directional:
    only row i is modified, preserving CADJ's asymmetric nature.

    Parameters
    ----------
    CADJ : scipy.sparse matrix, shape (M, M)
        Asymmetric co-adjacency matrix from VQRecaller.  A copy is made
        internally; the original is never mutated.
    CADJ_nhbs : list of list of int, length M
        Precomputed neighbor index lists (nonzero column indices per row).
    CADJ_nhbs_size : np.ndarray, shape (M,), dtype int
        Number of neighbors per prototype (``len(CADJ_nhbs[i])`` for each i).
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
        Neighbor index lists for each row of PCADJ.
    PCADJ_nhbs_size : np.ndarray, shape (M,), dtype int
        Number of neighbors per prototype in PCADJ.

    Notes
    -----
    If all rows already satisfy ``>= min_nhbs``, the function returns
    immediately with the original input objects (no copy made).  Otherwise
    an internal copy is made and the originals are never mutated.  Callers
    should not mutate the returned objects, since in the no-op case they
    are the same objects that were passed in.
    """
    # Fast path: nothing to do — return originals without copying
    if np.all(CADJ_nhbs_size >= min_nhbs):
        return CADJ, CADJ_nhbs, CADJ_nhbs_size

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
    distance from W[i] to its CADJ neighbors::

        sigma_i = sum_j( CADJ[i,j] * dist(W[i], W[j]) )
                  / sum_j( CADJ[i,j] )

    This gives a locally adaptive scale that reflects the typical reach of
    prototype i's data-manifold neighborhood.  Strongly populated CADJ
    edges (many data points co-mapped to the i-j pair) dominate the
    weighted mean, so sigma_i is driven by the dense core of the
    neighborhood rather than by sparse boundary connections.

    Prototypes with no CADJ neighbors receive the median sigma of all
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
        Precomputed neighbor index lists.  If ``None``, derived from the
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

    # Derive neighbor lists from CSR structure if not supplied
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
    CADJ_nhbs_size: np.ndarray | None = None,
    min_similarity: float = 0.01,
    min_nhbs: int = 3,
    support: str = "dense",
) -> sp.csr_matrix:
    """
    Self-tuning similarity kernel with CADJ-derived bandwidths.

    Computes a symmetric M×M kernel matrix where the similarity between
    prototypes i and j is::

        K[i, j] = exp( -dist²(W[i], W[j]) / (sigma_i * sigma_j) )

    with sigma_i computed by :func:`CADJ_sigmas`.  Unlike a plain Gaussian
    kernel, the denominator adapts to local prototype density: tightly packed
    regions produce small sigma values (narrow kernel), while sparse regions
    produce large sigma values (broad kernel), making cluster boundaries
    visible without manual bandwidth selection.

    The ``support`` parameter controls which (i, j) pairs are evaluated:

    - ``support="dense"`` (default): kernel evaluated for all M×M pairs.
      CADJ enters only through sigma_i, not through the sparsity pattern,
      allowing the kernel to introduce edges that CADJ did not encode.
    - ``support="CADJ"``: kernel evaluated only for pairs where CADJ[i, j] > 0
      (after padding via :func:`pad_CADJ` to ensure every row has at least
      ``min_nhbs`` entries).  Much cheaper for large M when CADJ is sparse.

    In both cases :func:`pad_CADJ` is called first (cheaply — a no-op if all
    rows already satisfy ``>= min_nhbs``) so that sigma computation never
    falls back to the median heuristic.

    **Sparsification.**  After kernel evaluation, entries with
    K[i, j] < ``min_similarity`` are set to zero.  Each row is guaranteed
    at least ``min_nhbs`` nonzero off-diagonal entries, overriding the
    threshold if necessary.

    **Symmetry.**  The kernel formula is symmetric by construction, but the
    per-row ``min_nhbs`` guarantee can break exact symmetry in edge cases.
    The output is explicitly symmetrized as ``(K + K.T) / 2``.

    **Diagonal.**  Returned as 1.0 (K[i, i] = exp(0) = 1).  The caller is
    responsible for zeroing or ignoring it if needed (e.g., igraph's
    ``graph_adjacency`` accepts a ``diag=False`` argument).

    .. note::
       ``support="dense"`` uses ``scipy.spatial.distance.pdist`` to compute
       only the M*(M-1)/2 unique pairs, then ``squareform`` to expand to
       (M, M).  For M ≳ 5 000 the squareform matrix (~200 MB at float64)
       may become a bottleneck; use ``support="CADJ"`` instead.

       ``support="CADJ"`` extracts nonzero (i, j) index pairs directly from
       the CSR structure and computes distances for those pairs in a single
       vectorized operation, avoiding any O(M²) dense intermediate.

       Both branches share a post-evaluation sparsification step that loops
       over M rows to apply the ``min_similarity`` threshold and enforce the
       ``min_nhbs`` per-row guarantee.  This loop is O(M) in the number of
       iterations (not O(M²) or O(nnz)), and operates on short per-row
       arrays extracted from the LIL sparse format.

    Parameters
    ----------
    W : np.ndarray, shape (M, d)
        Prototype matrix.
    CADJ : scipy.sparse matrix, shape (M, M)
        Asymmetric co-adjacency matrix.  Used to compute sigma_i via
        :func:`CADJ_sigmas` and, when ``support="CADJ"``, to determine
        which pairs receive a kernel value.
    CADJ_nhbs : list of list of int, length M, optional
        Precomputed neighbor index lists.  Passed through to
        :func:`CADJ_sigmas` and :func:`pad_CADJ`; derived from CADJ's CSR
        structure if ``None``.
    CADJ_nhbs_size : np.ndarray, shape (M,), dtype int, optional
        Number of neighbors per prototype.  Required by :func:`pad_CADJ`
        to determine which rows need padding.  Derived from CADJ's CSR
        structure if ``None``.
    min_similarity : float, default 0.01
        Hard lower bound on kernel values retained after sparsification.
        Entries with K[i, j] < ``min_similarity`` are set to zero.
        Because K values are in (0, 1], this is equivalent to discarding
        pairs whose normalized squared distance exceeds
        ``-log(min_similarity) * sigma_i * sigma_j``.
    min_nhbs : int, default 3
        Minimum number of nonzero off-diagonal entries guaranteed per row,
        both for padding (passed to :func:`pad_CADJ` as ``min_nhbs``) and
        for the post-sparsification guarantee.
    support : {"dense", "CADJ"}, default "dense"
        Which (i, j) pairs to evaluate the kernel for:

        - ``"dense"``: all M×M pairs (original behavior).
        - ``"CADJ"``: only pairs where CADJ[i, j] > 0 after padding.

    Returns
    -------
    K : scipy.sparse.csr_matrix, shape (M, M)
        Symmetric self-tuning kernel matrix.  Diagonal entries are 1.0.

    References
    ----------
    Zelnik-Manor, L. & Perona, P. (2004). Self-tuning spectral clustering.
        Advances in Neural Information Processing Systems, 17.
    """
    if support not in ("dense", "CADJ"):
        raise ValueError(f"support must be 'dense' or 'CADJ', got {support!r}")

    M = W.shape[0]

    # --- Step 1: pad CADJ if needed (no-op if all rows already >= min_nhbs) --
    CADJ_csr = CADJ.tocsr()
    if CADJ_nhbs is None:
        CADJ_nhbs = [
            CADJ_csr.indices[CADJ_csr.indptr[i]:CADJ_csr.indptr[i + 1]].tolist()
            for i in range(M)
        ]
    if CADJ_nhbs_size is None:
        CADJ_nhbs_size = np.diff(CADJ_csr.indptr).astype(int)

    PCADJ, PCADJ_nhbs, _ = pad_CADJ(
        CADJ=CADJ_csr,
        CADJ_nhbs=CADJ_nhbs,
        CADJ_nhbs_size=CADJ_nhbs_size,
        W=W,
        min_nhbs=min_nhbs,
    )

    # --- Step 2: per-prototype bandwidths ------------------------------------
    sigma = CADJ_sigmas(W=W, CADJ=PCADJ, CADJ_nhbs=PCADJ_nhbs)  # shape (M,)

    # --- Step 3: kernel evaluation -------------------------------------------
    if support == "dense":
        # Compute all M*(M-1)/2 unique pairs via pdist, expand to (M, M).
        D2 = squareform(pdist(W, metric="sqeuclidean"))  # shape (M, M)

        # Sequential broadcast avoids allocating np.outer(sigma, sigma):
        #   D2 / sigma[:, None]  divides each row i by sigma_i
        #   / sigma[None, :]     divides each column j by sigma_j
        K_dense = np.exp(-D2 / sigma[:, None] / sigma[None, :])  # (M, M)
        del D2

        K_sparse = sp.csr_matrix(K_dense)
        del K_dense

    else:  # support == "CADJ"
        # Extract nonzero (i, j) pairs from padded CADJ's CSR structure.
        row_idx, col_idx = PCADJ.nonzero()

        # Vectorized squared Euclidean distances for all nonzero pairs:
        #   diffs[k] = W[row_idx[k]] - W[col_idx[k]], shape (nnz, d)
        # einsum 'ij,ij->i' is a row-wise dot product — faster than
        # (diffs ** 2).sum(axis=1) and avoids a temporary (nnz, d) square.
        diffs = W[row_idx] - W[col_idx]
        sq_dists = np.einsum("ij,ij->i", diffs, diffs)
        del diffs

        # Kernel values for each nonzero pair, shape (nnz,)
        K_vals = np.exp(-sq_dists / (sigma[row_idx] * sigma[col_idx]))

        # Assemble into sparse matrix directly from COO data
        K_sparse = sp.csr_matrix((K_vals, (row_idx, col_idx)), shape=(M, M))

    # --- Step 4: sparsification (shared) -------------------------------------
    # Work in LIL format for efficient per-row threshold + min_nhbs guarantee.
    # Force-keep logic uses kernel values directly (higher K = closer prototype)
    # avoiding the need for a separate distance matrix in either branch.
    K_lil = K_sparse.tolil()
    del K_sparse

    for i in range(M):
        row_data = np.array(K_lil.data[i], dtype=float)
        row_cols = np.array(K_lil.rows[i], dtype=int)

        if len(row_data) == 0:
            continue

        # Off-diagonal mask
        off_diag = row_cols != i

        # Apply min_similarity threshold to off-diagonal entries
        keep_off = (row_data >= min_similarity) & off_diag
        n_kept = int(keep_off.sum())

        # If too few off-diagonal entries survive, force-keep the strongest
        # (highest K value = geometrically closest) among those available
        if n_kept < min_nhbs:
            off_diag_vals = row_data[off_diag]
            off_diag_cols = row_cols[off_diag]
            n_force = min(min_nhbs, len(off_diag_vals))
            top_k = np.argpartition(off_diag_vals, -n_force)[-n_force:]
            forced_cols = set(off_diag_cols[top_k].tolist())
            keep_off = keep_off | (off_diag & np.isin(row_cols, list(forced_cols)))

        # Diagonal entry (if present) is always kept; off-diagonal filtered
        keep = (~off_diag) | keep_off
        row_data[~keep] = 0.0
        K_lil.data[i] = row_data.tolist()

    K_sparse = K_lil.tocsr()
    K_sparse.eliminate_zeros()
    del K_lil

    # --- Step 5: symmetrize --------------------------------------------------
    K_sym = (K_sparse + K_sparse.T) / 2.0

    return K_sym