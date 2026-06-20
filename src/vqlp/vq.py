import numpy as np
from scipy.sparse import lil_matrix
from scipy.spatial.distance import cdist
from .utils import (
    exp_kernel_bw_perplexity,
    lp_norm_stable,
    lp_centroid_objective_p_root,
    lp_centroid_gradient_p_root,
)
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

try:
    import faiss
except ImportError:
    raise ImportError("VQRecaller and VQFitter require FAISS. Install with: pip install faiss-cpu")

__all__ = [
    "VQRecaller",
    "VQFitter",
]

class VQRecaller:
    """
    Recall analysis for Vector Quantizer results.
    
    Computes and stores analysis products from applying a set of prototypes
    to data, including Best Matching Units, connectivity matrices, receptive
    fields, and reconstruction capabilities.
    
    Nearest-prototype search is performed via FAISS, which supports arbitrary
    Lp metrics. Both exact (flat) and approximate (IVF) search are available
    via the index_type parameter.
    
    Note: FAISS operates in float32 internally. Input data is cast to float32
    for the search step regardless of input dtype.
    """
    
    def __init__(self, p=2, max_bmu=2, index_type="flat", nlist=None, nprobe=None):
        """
        Initialize VQRecaller.
        
        Parameters
        ----------
        p : float, default=2
            Order of the Lp distance metric. Any positive value is supported
            for exact search (index_type="flat"). Approximate search
            (index_type="ivf") is restricted to p in {1, 2}.
        max_bmu : int, default=2
            Number of Best Matching Units to track for each observation
        index_type : str, default="flat"
            FAISS index type for nearest-neighbor search:
            - "flat": Exact brute-force search via IndexFlat. Correct for any
              p and recommended for codebooks up to a few thousand prototypes,
              and for any iterative fitting loop.
            - "ivf": Approximate search via IndexIVFFlat. Useful for very
              large codebooks where exact search is a bottleneck. Not suitable
              for iterative fitting (non-deterministic assignments interfere
              with convergence) and restricted to p in {1, 2}.
        nlist : int, optional
            (IVF only) Number of Voronoi cells. Defaults to sqrt(M).
        nprobe : int, optional
            (IVF only) Number of cells searched at query time. Higher values
            give better recall at the cost of speed. Defaults to nlist // 10.
        """
        if index_type not in ("flat", "ivf"):
            raise ValueError(f"index_type must be 'flat' or 'ivf', got {index_type!r}")
        if index_type == "ivf" and p not in (1, 2):
            raise ValueError(
                f"Approximate (IVF) search supports only p in {{1, 2}}, got p={p}. "
                f"Use index_type='flat' for exotic Lp metrics."
            )

        # Model parameters
        self.p = p
        self.max_bmu = max_bmu
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        
        # Recall products (computed by recall())
        self.BMU = None          # Best Matching Unit indices (N, max_bmu)
        self.QE = None           # Quantization Errors (N, max_bmu)
        self.AFF = None          # Affinities (N, max_bmu): Neuralware-form soft weights from QE
        self.RF = None           # Receptive Field: list of observation indices per prototype
        self.RFSize = None       # Size of Receptive Field per prototype
        self.CONN = None         # Connectivity matrix (sparse)
        self.CONN_nhbs = None    # List of nonzero indices for each row of CONN
        self.CONN_nhbs_size = None  # Size of each CONN_nhbs[i] as numpy array

        # Label recall products (computed by recall_labels())
        self.WL = None           # Winning label per prototype, object array (M,); None for empty RFs
        self.WL_Dist = None      # Fuzzy label frequency table (M, n_unique_labels), row-normalized
        self.WL_Purity = None    # Hellinger-based purity score per prototype (M,)
        self.WL_unq = None       # Sorted unique label values; column j of WL_Dist corresponds to WL_unq[j]

        # Store shape info for validation (don't store X itself)
        self._M = None           # Number of prototypes
        self._N = None           # Number of observations
        self._d = None           # Dimensionality
    
    def recall(self, X=None, W=None, p=None, max_bmu=None, labels=None):
        """
        Perform recall analysis. Can operate in two modes:
        
        Mode 1 - Full analysis: Provide X and W to compute everything from scratch
        Mode 2 - Finalize: Provide no arguments to compute derived quantities 
                          from existing BMU/QE (assumes update_BMU() was called)
        
        Parameters
        ----------
        X : array-like, shape (N, d), optional
            Data matrix with N observations and d features
        W : array-like, shape (M, d), optional  
            Prototype matrix with M prototypes and d features
        p : float, optional
            Lp distance order. If None, uses self.p
        max_bmu : int, optional
            Number of BMUs to track. If None, uses self.max_bmu
        labels : array-like, shape (N,), optional
            Observation labels. Any hashable type is accepted (int, str, float,
            etc.). If provided, recall_labels() is called automatically after
            the main recall pipeline. Can also be called independently via
            recall_labels() on an already-recalled object.
            
        Returns
        -------
        self : VQRecaller
            Returns self for method chaining
        """
        # Update parameters if provided
        if p is not None:
            if self.index_type == "ivf" and p not in (1, 2):
                raise ValueError(
                    f"Approximate (IVF) search supports only p in {{1, 2}}, got p={p}."
                )
            self.p = p
        if max_bmu is not None:
            self.max_bmu = max_bmu
        
        # Mode 1: Full analysis from scratch
        if X is not None and W is not None:
            self.update_BMU(X, W)
            
        # Mode 2: Finalize existing BMU/QE
        elif X is None and W is None:
            # Validate that BMU and QE exist and are consistent
            if self.BMU is None or self.QE is None:
                raise RuntimeError("BMU and QE must be populated first. Call update_BMU() or provide X and W.")
            if self.BMU.shape != self.QE.shape:
                raise RuntimeError(f"BMU shape {self.BMU.shape} doesn't match QE shape {self.QE.shape}")
            if self._N is None or self._M is None:
                raise RuntimeError("Shape information missing. Call update_BMU() first.")
            
        # Invalid: partial arguments provided
        else:
            raise ValueError("Either provide both X and W, or neither (to finalize existing BMU/QE)")
        
        # Compute the derived recall products (cheap operations)
        self._compute_receptive_fields()
        self._compute_connectivity_matrix()
        
        if labels is not None:
            self.recall_labels(labels)
        
        print(f"Recall analysis complete. MQE: {np.mean(self.QE[:, 0]):.6f}")
        return self
    
    def update_BMU(self, X, W):
        """
        Update Best Matching Units (BMU) and Quantization Errors (QE) via FAISS.
        
        This is the computationally expensive part of recall analysis.
        Can be called repeatedly during iterative fitting algorithms.
        
        Parameters
        ----------
        X : array-like, shape (N, d)
            Data matrix with N observations and d features
        W : array-like, shape (M, d)
            Prototype matrix with M prototypes and d features
        """
        X = np.array(X)
        W = np.array(W)
        
        # Store shape info for validation
        self._N, self._d = X.shape
        self._M = W.shape[0]
        
        if W.shape[1] != self._d:
            raise ValueError(f"W has {W.shape[1]} features but X has {self._d} features")
        if self.max_bmu > self._M:
            raise ValueError(f"max_bmu={self.max_bmu} exceeds number of prototypes M={self._M}")

        # FAISS requires float32 and contiguous arrays
        X_f32 = np.ascontiguousarray(X, dtype=np.float32)
        W_f32 = np.ascontiguousarray(W, dtype=np.float32)

        index = self._build_index(W_f32)
        distances, indices = index.search(X_f32, self.max_bmu)

        self.BMU = indices.astype(int)
        self.QE = distances.astype(X.dtype)
        self._compute_affinity()

    def _build_index(self, W_f32):
        """Build and populate the FAISS index for prototype set W."""
        if self.index_type == "flat":
            index = faiss.IndexFlat(self._d, faiss.METRIC_Lp)
            index.metric_arg = float(self.p)
            index.add(W_f32)
            return index

        # IVF approximate search (p in {1, 2}, enforced at construction)
        nlist = self.nlist if self.nlist is not None else int(np.clip(int(np.sqrt(self._M)), 1, 65536))
        nlist = min(nlist, self._M)

        if self.p == 2:
            quantizer = faiss.IndexFlatL2(self._d)
            index = faiss.IndexIVFFlat(quantizer, self._d, nlist, faiss.METRIC_L2)
        else:  # p == 1
            quantizer = faiss.IndexFlat(self._d, faiss.METRIC_L1)
            index = faiss.IndexIVFFlat(quantizer, self._d, nlist, faiss.METRIC_L1)

        index.train(W_f32)
        index.add(W_f32)
        index.nprobe = self.nprobe if self.nprobe is not None else max(1, nlist // 10)
        return index

    def _compute_affinity(self):
        """
        Compute affinity matrix AFF from quantization errors QE.

        Uses the Neuralware form: the first BMU always gets affinity 1 before
        normalization, and each subsequent BMU k gets QE[:,0] / (QE[:,0] + QE[:,k]).
        Rows are then normalized to sum to 1. Division-by-zero entries (when both
        QE[:,0] and QE[:,k] are zero, i.e. the observation sits exactly on a
        prototype) are set to 0 before normalization, which concentrates all
        weight on the first BMU.

        This mirrors cpp_QE2Affinity() from AnnoyVQRecall.hpp.

        Result is stored as self.AFF with the same shape as self.QE.
        """
        AFF = np.ones_like(self.QE)  # column 0 = 1 for all rows
        q0 = self.QE[:, 0]
        for k in range(1, self.QE.shape[1]):
            denom = q0 + self.QE[:, k]
            # Where denom == 0, both prototypes coincide with the point;
            # set to 0 so row-normalization assigns all weight to BMU 0.
            with np.errstate(invalid="ignore", divide="ignore"):
                AFF[:, k] = np.where(denom > 0, q0 / denom, 0.0)
        # Row-normalize
        row_sums = AFF.sum(axis=1, keepdims=True)
        # row_sums should always be > 0 (col 0 is 1), but guard anyway
        AFF = np.where(row_sums > 0, AFF / row_sums, AFF)
        self.AFF = AFF

    def _compute_receptive_fields(self):
        """
        Compute receptive fields (RF) and their sizes (RFSize).
        """
        self.RF = [[] for _ in range(self._M)]
        
        # Assign each observation to the RF of its closest prototype
        for obs_idx, bmu_indices_row in enumerate(self.BMU):
            first_bmu_idx = bmu_indices_row[0]
            self.RF[first_bmu_idx].append(obs_idx)
        
        self.RFSize = np.array([len(rf_list) for rf_list in self.RF], dtype=int)
        print("Receptive fields computed.")
    
    def _compute_connectivity_matrix(self):
        """
        Compute the connectivity matrix (CONN) and neighbor lists (CONN_nhbs).
        """
        if self.max_bmu < 2:
            print("Warning: max_bmu < 2. Connectivity matrix will be empty.")
            self.CONN = None
            self.CONN_nhbs = None
            self.CONN_nhbs_size = None
            return
            
        # Initialize adjacency matrix as LIL for efficient updates
        cadj_matrix = lil_matrix((self._M, self._M), dtype=int)
        
        # Count co-occurrences of 1st and 2nd BMUs
        for i in range(self.BMU.shape[0]):
            bmu1_idx = self.BMU[i, 0]  # Closest prototype
            bmu2_idx = self.BMU[i, 1]  # Second closest prototype
            cadj_matrix[bmu2_idx, bmu1_idx] += 1
        
        # Convert to CSC and make symmetric
        cadj_csc = cadj_matrix.tocsc()
        self.CONN = cadj_csc + cadj_csc.transpose()
        
        # Compute neighbor lists for efficient access
        conn_csr = self.CONN.tocsr()
        self.CONN_nhbs = [[] for _ in range(self._M)]
        for i in range(self._M):
            self.CONN_nhbs[i] = conn_csr.indices[conn_csr.indptr[i]:conn_csr.indptr[i+1]].tolist()
        
        # Compute sizes of neighbor lists as numpy array
        self.CONN_nhbs_size = np.array([len(nhbs) for nhbs in self.CONN_nhbs], dtype=int)
        
        print("Connectivity matrix computed.")

    def recall_labels(self, labels):
        """
        Compute per-prototype label summaries from observation labels.

        Must be called after update_BMU() (or recall()) so that self.BMU,
        self.AFF, self.RF, and self.RFSize are available.

        For each observation i, its label receives fractional credit distributed
        across all max_bmu prototypes in proportion to their affinity weights
        (self.AFF[i, :]). This produces a fuzzy frequency table (WL_Dist) whose
        rows are then normalized to probability distributions. The winning label
        per prototype is the argmax of that distribution, and purity is derived
        from the Hellinger distance between the distribution and the ideal
        one-hot encoding of the winner.

        This mirrors VQRecallLabels_worker / cpp_RecallLabels() from
        AnnoyVQRecall.hpp.

        Parameters
        ----------
        labels : array-like, shape (N,)
            Observation labels. Any hashable type is accepted (int, str,
            float, etc.). Length must equal the number of observations used
            in the most recent update_BMU() / recall() call.

        Returns
        -------
        self : VQRecaller
            Returns self for method chaining.

        Attributes set
        --------------
        WL : np.ndarray of object, shape (M,)
            Winning label for each prototype. None for prototypes whose
            receptive field is empty (RFSize == 0).
        WL_Dist : np.ndarray of float, shape (M, n_unique_labels)
            Row-normalized fuzzy label frequency table. Column j corresponds
            to WL_unq[j]. Rows for empty RFs are all-zero.
        WL_Purity : np.ndarray of float, shape (M,)
            Hellinger-based purity score in [0, 1]. 1 means all affinity mass
            falls on a single label; 0 for empty RFs.
        WL_unq : np.ndarray, shape (n_unique_labels,)
            Sorted unique label values. Use to interpret WL_Dist columns.
        """
        if self.BMU is None or self.AFF is None:
            raise RuntimeError(
                "BMU and AFF must be populated first. Call update_BMU() or recall() before recall_labels()."
            )

        labels = np.asarray(labels)
        if labels.shape[0] != self._N:
            raise ValueError(
                f"labels has {labels.shape[0]} entries but recall used {self._N} observations."
            )

        # Discover label universe and build index map {label -> col index}
        WL_unq, label_indices = np.unique(labels, return_inverse=True)
        n_labels = len(WL_unq)

        # Accumulate fuzzy frequency table via scatter-add.
        # For each of the max_bmu columns k, every observation i contributes
        # AFF[i, k] to the cell (BMU[i, k], label_col_of_i).
        WL_Dist = np.zeros((self._M, n_labels), dtype=float)
        for k in range(self.max_bmu):
            proto_indices = self.BMU[:, k]          # shape (N,)
            aff_weights   = self.AFF[:, k]          # shape (N,)
            # Scatter-add: WL_Dist[proto_indices[i], label_indices[i]] += aff_weights[i]
            np.add.at(WL_Dist, (proto_indices, label_indices), aff_weights)

        # Row-normalize to get probability distributions; leave empty-RF rows as zeros
        row_sums = WL_Dist.sum(axis=1, keepdims=True)
        active = (row_sums > 0).ravel()             # boolean mask of non-empty RFs
        WL_Dist[active] /= row_sums[active]

        # Winning label and Hellinger purity
        WL = np.empty(self._M, dtype=object)        # object dtype accepts any label type
        WL[:] = None                                 # default: no label for empty RFs
        WL_Purity = np.zeros(self._M, dtype=float)

        winner_cols = np.argmax(WL_Dist, axis=1)    # shape (M,); 0 for empty rows (harmless)
        for i in np.where(active)[0]:
            WL[i] = WL_unq[winner_cols[i]]
            p_winner = WL_Dist[i, winner_cols[i]]
            # Hellinger distance to a perfect one-hot: sqrt(1 - sqrt(p_winner))
            # Purity = 1 - hell_dist (similarity, not distance)
            hell_dist = np.sqrt(max(0.0, 1.0 - np.sqrt(p_winner)))
            WL_Purity[i] = 1.0 - hell_dist

        self.WL      = WL
        self.WL_Dist = WL_Dist
        self.WL_Purity = WL_Purity
        self.WL_unq  = WL_unq

        n_labeled = int(active.sum())
        print(
            f"Label recall complete. {n_labeled}/{self._M} prototypes labeled "
            f"({n_labels} unique labels). Mean purity: {WL_Purity[active].mean():.4f}"
        )
        return self
    
    def reconstruct(self, W, X, method="hard"):
        """
        Reconstruct data using the quantizer.
        
        Parameters
        ----------
        W : array-like, shape (M, d)
            Prototype matrix with M prototypes and d features
        X : array-like, shape (N, d)
            Data matrix to reconstruct with N observations and d features
        method : str, default="hard"
            Reconstruction method:
            - "hard": Use closest prototype for each observation
            - "soft": Weighted average using UMAP-style local connectivity
            
        Returns
        -------
        np.ndarray : Reconstructed data with shape (N, d)
        """
        if self.BMU is None:
            raise RuntimeError("Must call recall() before reconstruct()")
        
        W = np.array(W)
        X = np.array(X)
        
        # Validate dimensions
        if W.shape[0] != self._M:
            raise ValueError(f"W has {W.shape[0]} prototypes but recall used {self._M}")
        if X.shape[0] != self._N:
            raise ValueError(f"X has {X.shape[0]} observations but recall used {self._N}")
        if X.shape[1] != self._d or W.shape[1] != self._d:
            raise ValueError(f"Dimension mismatch: expected d={self._d}")
        
        if method == "hard":
            return self._reconstruct_hard(X, W)
        elif method == "soft":
            return self._reconstruct_soft(X, W)
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")
    
    def _reconstruct_hard(self, X, W):
        """Hard reconstruction using closest prototypes."""
        first_bmu_indices = self.BMU[:, 0]
        return W[first_bmu_indices].astype(X.dtype)
    
    def _reconstruct_soft(self, X, W):
        """Soft reconstruction using UMAP-style weighted averaging."""
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        
        QX = np.zeros_like(X, dtype=np.float32)
        
        # Process each prototype's receptive field
        for i in range(self._M):
            if self.RFSize[i] == 0:
                continue
                
            obs_indices_in_rf = self.RF[i]
            nhbs_indices = np.array(list(set([i] + self.CONN_nhbs[i])), dtype=int)
            
            X_subset = X[obs_indices_in_rf, :]
            W_subset = W[nhbs_indices, :]
            
            # Compute distances from observations to local prototypes
            dist_matrix = cdist(X_subset, W_subset, metric='euclidean')
            
            # Reconstruct each observation in this receptive field
            for j, original_idx in enumerate(obs_indices_in_rf):
                dists = dist_matrix[j, :]
                
                # Calculate UMAP-style weights
                sigma = exp_kernel_bw_perplexity(dists)
                weights = np.exp(-(dists - np.min(dists)) / sigma)
                
                # Weighted average reconstruction
                sum_weights = np.sum(weights)
                if sum_weights == 0:
                    # Fallback to closest prototype
                    closest_idx = np.argmin(dists)
                    QX[original_idx, :] = W_subset[closest_idx, :]
                else:
                    QX[original_idx, :] = np.sum(weights[:, np.newaxis] * W_subset, axis=0) / sum_weights
        
        print("Soft reconstruction complete.")
        return QX
    
    def get_summary(self):
        """
        Get a summary of the recall analysis results.
        
        Returns
        -------
        dict : Summary statistics
        """
        if self.BMU is None:
            return {"status": "No recall analysis performed"}
        
        summary = {
            "M": self._M,                    # Number of prototypes
            "N": self._N,                    # Number of observations  
            "d": self._d,                    # Dimensionality
            "p_norm": self.p,
            "max_bmu": self.max_bmu,
            "index_type": self.index_type,
            "mean_quantization_error": np.mean(self.QE[:, 0]),
            "empty_prototypes": np.sum(self.RFSize == 0),
            "largest_rf_size": np.max(self.RFSize) if self.RFSize is not None else 0,
            "connectivity_edges": self.CONN.nnz // 2 if self.CONN is not None else 0,
            "mean_connectivity_degree": np.mean(self.CONN_nhbs_size) if self.CONN_nhbs_size is not None else 0,
            "max_connectivity_degree": np.max(self.CONN_nhbs_size) if self.CONN_nhbs_size is not None else 0,
        }

        if self.WL is not None:
            active = self.RFSize > 0
            summary.update({
                "n_unique_labels":       len(self.WL_unq),
                "labeled_prototypes":    int(active.sum()),
                "mean_label_purity":     float(self.WL_Purity[active].mean()) if active.any() else float("nan"),
                "min_label_purity":      float(self.WL_Purity[active].min())  if active.any() else float("nan"),
            })

        return summary

class VQFitter:
    """
    Vector Quantizer Fitter using arbitrary Lp distance metrics.
    
    Supports multiple fitting algorithms including random sampling,
    IRLS (Iteratively Reweighted Least Squares), gradient descent (GD),
    PAM (Partitioning Around Medoids), and FAISS k-means (p=2 only).
    """
    
    AVAILABLE_METHODS = ["random", "IRLS", "GD", "PAM", "kmeans"]
    
    def __init__(self, M, p=2, max_bmu=2, random_state=None):
        """
        Initialize the VQ Fitter.
        
        Parameters
        ----------
        M : int
            Number of prototypes to create
        p : float, default=2
            Order of the Lp distance metric (e.g., 1 for Manhattan, 2 for Euclidean)
        max_bmu : int, default=2
            Number of Best Matching Units (closest prototypes) to track
        random_state : int, optional
            Random seed for reproducible results
        """
        self.M = int(M)
        self.p = p
        self.max_bmu = max_bmu
        self.random_state = random_state
        self.W = None  # Prototype matrix (M, d)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # VQRecaller instance for BMU tracking during fitting
        self.recaller = VQRecaller(p=self.p, max_bmu=self.max_bmu)
    
    def fit(self, X, method=None, distX=None, **kwargs):
        """
        Fit the vector quantizer using the specified method.
        
        Defaults to "kmeans" when p=2, and "IRLS" otherwise.
        
        Parameters
        ----------
        X : array-like, shape (N, d)
            Training data matrix
        method : str, optional
            Fitting method. One of: ["random", "IRLS", "GD", "PAM", "kmeans"].
            Defaults to "kmeans" if p=2, "IRLS" otherwise.
        distX : array, shape (N, N), optional
            Precomputed distance matrix. If provided, internal distance
            calculation is skipped (PAM only).
        **kwargs : additional arguments
            Method-specific parameters (see individual _fit_* methods).
            
        Returns
        -------
        self : VQFitter
            Returns self for method chaining
        """
        X = np.array(X)
        N, d = X.shape
        
        if self.M > N:
            raise ValueError(f"Cannot create {self.M} prototypes from only {N} observations")
        
        # Choose default method based on p
        if method is None:
            method = "kmeans" if self.p == 2 else "IRLS"

        if method not in self.AVAILABLE_METHODS:
            raise ValueError(f"Unknown fitting method '{method}'. Available methods: {self.AVAILABLE_METHODS}")
        
        if method == "kmeans" and self.p != 2:
            raise ValueError(f"kmeans only supports p=2 (Euclidean distance), got p={self.p}")

        if method == "random":
            self._fit_random(X)
        elif method == "IRLS":
            self._fit_IRLS(X, **kwargs)
        elif method == "GD":
            self._fit_GD(X, **kwargs)
        elif method == "PAM":
            self._fit_PAM(X, distX=distX, **kwargs)
        elif method == "kmeans":
            self._fit_kmeans(X, **kwargs)
        
        print(f"Fitting complete using {method} method.")
        return self
            
    def _fit_random(self, X):
        """
        Fit by randomly sampling prototypes from X.
        
        Parameters
        ----------
        X : array, shape (N, d)
            Training data
        """
        N = X.shape[0]
        indices = np.random.choice(N, size=self.M, replace=False)
        self.W = X[indices].copy()
        self.recaller.update_BMU(X, self.W)
        print(f"Randomly sampled {self.M} prototypes from {N} observations")
    
    def _fit_IRLS(self, X, tol_RFstab=0.01, tol_MQE=1e-4, max_iter=100, init_method="random"):
        """
        Fit using Iteratively Reweighted Least Squares.
        
        Parameters
        ----------
        X : array, shape (N, d)
            Training data
        tol_RFstab : float, default=0.01
            Tolerance for receptive field stability (proportion of changed assignments)
        tol_MQE : float, default=1e-4
            Tolerance for relative change in Mean Quantization Error
        max_iter : int, default=100
            Maximum number of iterations
        init_method : str, default="random"
            Initialization method:
            - "random": sample M observations from X without replacement
            - "PAM": use FasterPAM medoids as starting prototypes
            - "current": use the existing self.W (must be set via set_prototypes
              or a prior fit call)
        """
        if init_method == "random":
            self._fit_random(X)
        elif init_method == "PAM":
            self._fit_PAM(X)
        elif init_method == "current":
            if self.W is None:
                raise RuntimeError(
                    "init_method='current' requires W to be set first via "
                    "set_prototypes() or a prior fit() call."
                )
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
        
        self.recaller.update_BMU(X, self.W)
        prevBMU = self.recaller.BMU[:, 0].copy()
        prevMQE = np.mean(self.recaller.QE[:, 0])
        
        print(f"Initial MQE: {prevMQE:.6f}")
        
        for iter_num in range(1, max_iter + 1):
            self._update_prototypes_IRLS(X)
            self.recaller.update_BMU(X, self.W)
            
            current_BMU = self.recaller.BMU[:, 0]
            RFstab = np.mean(current_BMU != prevBMU)
            MQE = np.mean(self.recaller.QE[:, 0])
            rel_MQE_change = (MQE - prevMQE) / prevMQE if prevMQE != 0 else 0
            
            print(f"Iter {iter_num:3d}: RFstab = {RFstab:.4f}, MQE = {MQE:.6f}, rel_MQE_change = {rel_MQE_change:.6f}")
            
            prevBMU = current_BMU.copy()
            prevMQE = MQE
            
            if RFstab < tol_RFstab:
                print(f"Converged after {iter_num} iterations (RFstab < {tol_RFstab})")
                break
            elif abs(rel_MQE_change) < tol_MQE:
                print(f"Converged after {iter_num} iterations (|rel_MQE_change| < {tol_MQE})")
                break
        else:
            print(f"Reached maximum iterations ({max_iter}) without convergence")

    def _update_prototypes_IRLS(self, X):
        """
        Update prototype vectors using IRLS for minimizing the sum of Lp norms.
        Uses lp_norm_stable for numerical stability at high p values.
        
        Parameters
        ----------
        X : array, shape (N, d)
            Training data matrix
        """
        for j in range(self.W.shape[0]):
            cluster_mask = (self.recaller.BMU[:, 0] == j)
            cluster_points = X[cluster_mask]
            
            if len(cluster_points) == 0:
                continue
            
            # Add tiny perturbation to avoid singularities
            W_perturbed = self.W[j] + np.random.normal(0, 1e-12, self.W[j].shape)
            diff = cluster_points - W_perturbed
            
            distances = lp_norm_stable(diff, self.p)
            distances[distances < 1e-12] = 1e-12

            weights = distances ** (1 - self.p)
            weights_reshaped = weights.reshape(-1, 1)

            numerator = np.sum(cluster_points * weights_reshaped, axis=0)
            denominator = np.sum(weights)
            
            if denominator == 0:
                continue
                
            self.W[j] = numerator / denominator

    def _fit_GD(self, X, tol_RFstab=0.01, tol_MQE=1e-4, max_iter=100, init_method="random"):
        """
        Fit using gradient descent on the Lp centroid objective.
        
        Parameters
        ----------
        X : array, shape (N, d)
            Training data
        tol_RFstab : float, default=0.01
            Tolerance for receptive field stability (proportion of changed assignments)
        tol_MQE : float, default=1e-4
            Tolerance for relative change in Mean Quantization Error
        max_iter : int, default=100
            Maximum number of iterations
        init_method : str, default="random"
            Initialization method:
            - "random": sample M observations from X without replacement
            - "PAM": use FasterPAM medoids as starting prototypes
            - "current": use the existing self.W (must be set via set_prototypes
              or a prior fit call)
        """
        if init_method == "random":
            self._fit_random(X)
        elif init_method == "PAM":
            self._fit_PAM(X)
        elif init_method == "current":
            if self.W is None:
                raise RuntimeError(
                    "init_method='current' requires W to be set first via "
                    "set_prototypes() or a prior fit() call."
                )
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
        
        self.recaller.update_BMU(X, self.W)
        prevBMU = self.recaller.BMU[:, 0].copy()
        prevMQE = np.mean(self.recaller.QE[:, 0])
        
        print(f"Initial MQE: {prevMQE:.6f}")
        
        for iter_num in range(1, max_iter + 1):
            self._update_prototypes_GD(X)
            self.recaller.update_BMU(X, self.W)
            
            current_BMU = self.recaller.BMU[:, 0]
            RFstab = np.mean(current_BMU != prevBMU)
            MQE = np.mean(self.recaller.QE[:, 0])
            rel_MQE_change = (MQE - prevMQE) / prevMQE if prevMQE != 0 else 0
            
            print(f"Iter {iter_num:3d}: RFstab = {RFstab:.4f}, MQE = {MQE:.6f}, rel_MQE_change = {rel_MQE_change:.6f}")
            
            prevBMU = current_BMU.copy()
            prevMQE = MQE
            
            if RFstab < tol_RFstab:
                print(f"Converged after {iter_num} iterations (RFstab < {tol_RFstab})")
                break
            elif abs(rel_MQE_change) < tol_MQE:
                print(f"Converged after {iter_num} iterations (|rel_MQE_change| < {tol_MQE})")
                break
        else:
            print(f"Reached maximum iterations ({max_iter}) without convergence")

    def _update_prototypes_GD(self, X):
        """
        Update prototype vectors using gradient descent for Lp centroids.
        Uses L-BFGS-B with the analytic gradient. For p=2 shortcuts to the mean.
        
        Parameters
        ----------
        X : array, shape (N, d)
            Training data matrix
        """
        for j in range(self.W.shape[0]):
            cluster_mask = (self.recaller.BMU[:, 0] == j)
            cluster_points = X[cluster_mask]
            
            if len(cluster_points) == 0:
                continue
            
            L2_centroid = np.mean(cluster_points, axis=0)

            if self.p == 2:
                self.W[j] = L2_centroid
            else:
                result = minimize(
                    x0=L2_centroid,
                    fun=lp_centroid_objective_p_root,
                    jac=lp_centroid_gradient_p_root,
                    args=(cluster_points, self.p),
                    method='L-BFGS-B'
                )
                self.W[j] = result.x

    def _fit_PAM(self, X, distX=None, **kwargs):
        """
        Fit using PAM (Partitioning Around Medoids) via FasterPAM.
        
        Parameters
        ----------
        X : array, shape (N, d)
            Training data
        distX : array, shape (N, N), optional
            Precomputed distance matrix. If provided, internal distance
            calculation is skipped.
        **kwargs : keyword arguments
            Passed to kmedoids.fasterpam (e.g. max_iter, init, n_cpu).
            random_state is always taken from the fitter and cannot be overridden.
        """
        try:
            from kmedoids import fasterpam
        except ImportError:
            raise ImportError("kmedoids package required for PAM. Install with: pip install kmedoids")
        
        N = X.shape[0]
        
        if distX is None:
            print(f"Computing {N}x{N} Lp distance matrix for PAM...")
            distX = squareform(pdist(X, metric='minkowski', p=self.p))
        
        print(f"Running FasterPAM with {self.M} medoids...")
        
        kwargs.pop('random_state', None)
        result = fasterpam(diss=distX, medoids=self.M, random_state=self.random_state, **kwargs)
        
        self.W = X[result.medoids].copy()
        self.recaller.update_BMU(X, self.W)
        
        print(f"PAM fitting complete. Final MQE: {np.mean(self.recaller.QE[:, 0]):.6f}")

    def _fit_kmeans(self, X, niter=20, nredo=1, init_method="faiss", **kwargs):
        """
        Fit using FAISS k-means (p=2 only).
        
        Parameters
        ----------
        X : array, shape (N, d)
            Training data
        niter : int, default=20
            Number of k-means iterations per run.
        nredo : int, default=1
            Number of independent runs; the best result (lowest objective) is
            kept. Increase to compensate for random initialization, at the cost
            of nredo x the compute time. Ignored when init_method="current".
        init_method : str, default="faiss"
            Initialization method:
            - "faiss": FAISS random initialization (controlled by seed/nredo)
            - "current": use the existing self.W as starting centroids (must be
              set via set_prototypes() or a prior fit() call). nredo is
              overridden to 1 since restarting from a fixed point is not useful.
        **kwargs : keyword arguments
            Additional parameters passed to faiss.Kmeans constructor
            (e.g. verbose, gpu, max_points_per_centroid).
        """
        if init_method not in ("faiss", "current"):
            raise ValueError(f"init_method must be 'faiss' or 'current', got {init_method!r}")
        if init_method == "current" and self.W is None:
            raise RuntimeError(
                "init_method='current' requires W to be set first via "
                "set_prototypes() or a prior fit() call."
            )

        d = X.shape[1]
        X_f32 = np.ascontiguousarray(X, dtype=np.float32)
        seed = self.random_state if self.random_state is not None else 1234

        # nredo is meaningless when starting from a fixed W
        effective_nredo = 1 if init_method == "current" else nredo

        kmeans = faiss.Kmeans(
            d, self.M,
            niter=niter,
            nredo=effective_nredo,
            seed=seed,
            **kwargs
        )

        if init_method == "current":
            init_centroids = np.ascontiguousarray(self.W, dtype=np.float32)
            print(f"Running FAISS k-means with {self.M} clusters (niter={niter}, init_method='current')...")
            kmeans.train(X_f32, init_centroids=init_centroids)
        else:
            print(f"Running FAISS k-means with {self.M} clusters (niter={niter}, nredo={effective_nredo})...")
            kmeans.train(X_f32)

        self.W = kmeans.centroids.copy().astype(X.dtype)
        self.recaller.update_BMU(X, self.W)

        print(f"K-means complete. Final MQE: {np.mean(self.recaller.QE[:, 0]):.6f}")

    def set_prototypes(self, W, X):
        """
        Set the prototype matrix W directly, bypassing fitting.

        Useful when prototypes are known in advance (e.g. from a previous run,
        an external algorithm, or domain-specific initialization) and you want
        to use the VQFitter machinery for recall analysis or as a warm start
        for further fitting.

        BMU and QE are computed immediately from X, keeping the object in a
        consistent state — the same guarantee provided by all _fit_* methods.

        Parameters
        ----------
        W : array-like, shape (M, d)
            Prototype matrix. Must have exactly self.M rows. If the data
            dimensionality d is already known (from a prior fit or recall),
            W must match it; otherwise d is inferred from W.
        X : array-like, shape (N, d)
            Data matrix used to compute BMU and QE.

        Returns
        -------
        self : VQFitter
            Returns self for method chaining
        """
        W = np.array(W)
        X = np.array(X)

        if W.shape[0] != self.M:
            raise ValueError(
                f"W has {W.shape[0]} prototypes but this fitter expects M={self.M}"
            )

        # Validate d if already known, otherwise infer from W
        known_d = self.W.shape[1] if self.W is not None else None
        if known_d is not None and W.shape[1] != known_d:
            raise ValueError(
                f"W has {W.shape[1]} features but prior data had d={known_d}"
            )
        if X.shape[1] != W.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features but W has {W.shape[1]} features"
            )

        self.W = W.copy()
        self.recaller.update_BMU(X, self.W)
        print(f"Prototypes set externally. MQE: {np.mean(self.recaller.QE[:, 0]):.6f}")
        return self

    def recall(self, X=None):
        """
        Perform recall analysis using the fitted prototypes.
        
        If X is provided, performs recall analysis on new data.
        If X is None, finalizes recall analysis on training data (assuming
        BMU has been computed during fitting).
        
        Parameters
        ----------
        X : array-like, shape (N, d), optional
            Data matrix for recall analysis. If None, finalizes training recall.
            
        Returns
        -------
        self : VQFitter
            Returns self for method chaining
        """
        if self.W is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if X is not None:
            X = np.array(X)
            self.recaller.recall(X, self.W)
        else:
            self.recaller.recall()
        
        return self
    
    def get_summary(self):
        """
        Get a summary of the fitted model and recall analysis.
        
        Returns
        -------
        dict : Summary statistics
        """
        summary = {
            "fitted": self.W is not None,
            "M": self.M,
            "d": self.W.shape[1] if self.W is not None else None,
            "p_norm": self.p,
            "max_bmu": self.max_bmu,
            "random_state": self.random_state
        }
        
        if self.recaller.BMU is not None:
            recall_summary = self.recaller.get_summary()
            summary.update(recall_summary)
        
        return summary

