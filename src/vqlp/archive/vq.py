import numpy as np
from scipy.sparse import lil_matrix
from scipy.spatial.distance import pdist, squareform
from .utils import softmax_sigma_umap
from scipy.optimize import minimize
from statx.distances import lp_norm_stable

__all__ = [
    "VQRecaller",
    "VQFitter"
]

class VQRecaller:
    """
    Recall analysis for Vector Quantizer results.
    
    Computes and stores analysis products from applying a set of prototypes
    to data, including Best Matching Units, connectivity matrices, receptive
    fields, and reconstruction capabilities.
    """
    
    def __init__(self, p=2, max_bmu=2):
        """
        Initialize VQRecaller.
        
        Parameters:
        -----------
        p : float, default=2
            Order of the Lp distance metric
        max_bmu : int, default=2
            Number of Best Matching Units to track for each observation
        """
        # Model parameters
        self.p = p
        self.max_bmu = max_bmu
        
        # Recall products (computed by recall())
        self.BMU = None          # Best Matching Unit indices (N, max_bmu)
        self.QE = None           # Quantization Errors (N, max_bmu)
        self.RF = None           # Receptive Field: list of observation indices per prototype
        self.RFSize = None       # Size of Receptive Field per prototype
        self.CONN = None         # Connectivity matrix (sparse)
        self.CONN_nhbs = None    # List of nonzero indices for each row of CONN
        self.CONN_nhbs_size = None  # Size of each CONN_nhbs[i] as numpy array
        
        # Store shape info for validation (don't store X itself)
        self._M = None           # Number of prototypes
        self._N = None           # Number of observations
        self._d = None           # Dimensionality
    
    def recall(self, X=None, W=None, p=None, max_bmu=None):
        """
        Perform recall analysis. Can operate in two modes:
        
        Mode 1 - Full analysis: Provide X and W to compute everything from scratch
        Mode 2 - Finalize: Provide no arguments to compute derived quantities 
                          from existing BMU/QE (assumes update_BMU() was called)
        
        Parameters:
        -----------
        X : array-like, shape (N, d), optional
            Data matrix with N observations and d features
        W : array-like, shape (M, d), optional  
            Prototype matrix with M prototypes and d features
        p : float, optional
            Lp distance order. If None, uses self.p
        max_bmu : int, optional
            Number of BMUs to track. If None, uses self.max_bmu
            
        Returns:
        --------
        self : VQRecaller
            Returns self for method chaining
        """
        # Update parameters if provided
        if p is not None:
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
        
        print(f"Recall analysis complete. MQE: {np.mean(self.QE[:, 0]):.6f}")
        return self
    
    def update_BMU(self, X, W, method="auto"):
        """
        Update Best Matching Units (BMU) and Quantization Errors (QE).
        
        This is the computationally expensive part of recall analysis.
        Can be called repeatedly during iterative fitting algorithms.
        
        Parameters:
        -----------
        X : array-like, shape (N, d)
            Data matrix with N observations and d features
        W : array-like, shape (M, d)
            Prototype matrix with M prototypes and d features
        method : str, default="auto"
            Method for nearest neighbor search:
            - "auto": Choose automatically based on problem size
            - "numpy": Use numpy implementation (always available)
            - "faiss": Use FAISS library (requires faiss-cpu or faiss-gpu)
            - "sklearn": Use scikit-learn (limited Lp support)
        """
        X = np.array(X)
        W = np.array(W)
        
        # Store shape info for validation
        self._N, self._d = X.shape
        self._M = W.shape[0]
        
        if W.shape[1] != self._d:
            raise ValueError(f"W has {W.shape[1]} features but X has {self._d} features")
        
        # Choose method automatically if requested
        if method == "auto":
            method = self._choose_method(self._N, self._M, self._d)
        
        # Dispatch to appropriate implementation
        if method == "faiss":
            self._update_BMU_faiss(X, W)
        elif method == "sklearn":
            self._update_BMU_sklearn(X, W)
        elif method == "numpy":
            self._update_BMU_numpy(X, W)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _choose_method(self, N, M, d):
        """Choose the best method based on problem size."""
        # Simple heuristic - can be refined
        total_ops = N * M * d
        
        # For small problems, numpy is fine and has no dependencies
        if total_ops < 1e6:
            return "numpy"
        
        # For larger problems, try FAISS first, fallback to numpy
        try:
            import faiss
            return "faiss"
        except ImportError:
            try:
                from sklearn.neighbors import NearestNeighbors
                # sklearn only supports certain p values well
                if self.p in [1, 2] or self.p == np.inf:
                    return "sklearn"
                else:
                    return "numpy"
            except ImportError:
                return "numpy"
    
    def _update_BMU_faiss(self, X, W):
        """FAISS-based BMU update with arbitrary Lp metric."""
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        # Ensure float32 (FAISS requirement)
        W_f32 = W.astype(np.float32)
        X_f32 = X.astype(np.float32)
        
        # Create the index and set the metric and p-value
        index = faiss.IndexFlat(self._d, faiss.METRIC_Lp)
        index.metric_arg = float(self.p)  # Ensure float type
        index.add(W_f32)
        
        # Search for k nearest neighbors
        distances, indices = index.search(X_f32, self.max_bmu)
        
        # Store results (convert back to original precision)
        self.BMU = indices.astype(int)
        self.QE = distances.astype(X.dtype)
    
    def _update_BMU_sklearn(self, X, W):
        """Scikit-learn based BMU update."""
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            raise ImportError("Scikit-learn not available. Install with: pip install scikit-learn")
        
        # Map p values to sklearn metric names
        if self.p == 1:
            metric = 'manhattan'
        elif self.p == 2:
            metric = 'euclidean'
        elif self.p == np.inf:
            metric = 'chebyshev'
        else:
            # Use minkowski with parameter p
            metric = 'minkowski'
        
        # Create and fit the nearest neighbors model
        if metric == 'minkowski':
            nbrs = NearestNeighbors(n_neighbors=self.max_bmu, metric=metric, p=self.p)
        else:
            nbrs = NearestNeighbors(n_neighbors=self.max_bmu, metric=metric)
        
        nbrs.fit(W)
        
        # Find nearest neighbors
        distances, indices = nbrs.kneighbors(X)
        
        # Store results
        self.BMU = indices.astype(int)
        self.QE = distances.astype(X.dtype)
    
    def _update_BMU_numpy(self, X, W):
        """Numpy-based BMU update (original implementation)."""
        # Compute all pairwise Lp distances between X and W
        # Shape: (N, M)
        # distances = np.linalg.norm(X[:, None, :] - W[None, :, :], ord=self.p, axis=2)
        
        N = X.shape[0]
        M = self.W.shape[0]
        # Pre-allocate a distances matrix
        distances = np.zeros((N, M))
        # Calculate distances one prototype at a time to avoid creating
        # a large intermediate tensor.
        for j in range(M):
            diff = X - self.W[j]
            distances[:, j:j+1] = lp_norm_stable(diff, self.p)

        
        # Find indices of the max_bmu closest prototypes for each observation
        self.BMU = np.argpartition(distances, self.max_bmu-1, axis=1)[:, :self.max_bmu]
        
        # Get the corresponding distances (quantization errors)
        self.QE = np.take_along_axis(distances, self.BMU, axis=1)
        
        # Sort BMU and QE by distance (closest first)
        sort_indices = np.argsort(self.QE, axis=1)
        self.BMU = np.take_along_axis(self.BMU, sort_indices, axis=1)
        self.QE = np.take_along_axis(self.QE, sort_indices, axis=1)

    
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
    
    def reconstruct(self, W, X, method="hard"):
        """
        Reconstruct data using the quantizer.
        
        Parameters:
        -----------
        W : array-like, shape (M, d)
            Prototype matrix with M prototypes and d features
        X : array-like, shape (N, d)
            Data matrix to reconstruct with N observations and d features
        method : str, default="hard"
            Reconstruction method:
            - "hard": Use closest prototype for each observation
            - "soft": Weighted average using UMAP-style local connectivity
            
        Returns:
        --------
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
        from scipy.spatial.distance import cdist
        
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
                sigma = softmax_sigma_umap(dists)
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
        
        Returns:
        --------
        dict : Summary statistics
        """
        if self.BMU is None:
            return {"status": "No recall analysis performed"}
        
        return {
            "M": self._M,                    # Number of prototypes
            "N": self._N,                    # Number of observations  
            "d": self._d,                    # Dimensionality
            "p_norm": self.p,
            "max_bmu": self.max_bmu,
            "mean_quantization_error": np.mean(self.QE[:, 0]),
            "empty_prototypes": np.sum(self.RFSize == 0),
            "largest_rf_size": np.max(self.RFSize) if self.RFSize is not None else 0,
            "connectivity_edges": self.CONN.nnz // 2 if self.CONN is not None else 0,
            "mean_connectivity_degree": np.mean(self.CONN_nhbs_size) if self.CONN_nhbs_size is not None else 0,
            "max_connectivity_degree": np.max(self.CONN_nhbs_size) if self.CONN_nhbs_size is not None else 0
        }


# class VQFitter:

#     """
#     Vector Quantizer Fitter using arbitrary Lp distance metrics.
    
#     Supports multiple fitting algorithms including random sampling,
#     IRLS (Iteratively Reweighted Least Squares), and PAM (Partitioning Around Medoids).
#     """
    
#     AVAILABLE_METHODS = ["random", "IRLS", "PAM", "kmeans"]
    
#     def __init__(self, M, p=2, max_bmu=2, random_state=None):
#         """
#         Initialize the VQ Fitter.
        
#         Parameters:
#         -----------
#         M : int
#             Number of prototypes to create
#         p : float, default=2
#             Order of the Lp distance metric (e.g., 1 for Manhattan, 2 for Euclidean)
#         max_bmu : int, default=2
#             Number of Best Matching Units (closest prototypes) to track
#         random_state : int, optional
#             Random seed for reproducible results
#         """
#         self.M = int(M)  # Ensure M is an integer
#         self.p = p
#         self.max_bmu = max_bmu
#         self.random_state = random_state
#         self.W = None  # Prototype matrix (M, d)
        
#         # Set random seed if provided
#         if self.random_state is not None:
#             np.random.seed(self.random_state)
        
#         # VQRecaller instance for analysis
#         self.recaller = VQRecaller(p=self.p, max_bmu=self.max_bmu)
    
#     def fit(self, X, method="IRLS", distX = None, **kwargs):
#         """
#         Fit the vector quantizer using the specified method.
        
#         Parameters:
#         -----------
#         X : array-like, shape (N, d)
#             Training data matrix
#         method : str, default="IRLS"
#             Fitting method to use. Available methods: {cls.AVAILABLE_METHODS}
#         distX: array, shape (N, N). Optional distance matrix of X. If provided, internal distance calculation is skipped. 
#         **kwargs : additional arguments
#             Method-specific parameters
            
#         Returns:
#         --------
#         self : VQFitter
#             Returns self for method chaining
#         """
#         X = np.array(X)
#         N, d = X.shape
        
#         if self.M > N:
#             raise ValueError(f"Cannot create {self.M} prototypes from only {N} observations")
        
#         # Validate method
#         if method not in self.AVAILABLE_METHODS:
#             raise ValueError(f"Unknown fitting method '{method}'. Available methods: {self.AVAILABLE_METHODS}")
        
#         # Dispatch to appropriate fitting method
#         if method == "random":
#             self._fit_random(X, **kwargs)
#         elif method == "IRLS":
#             self._fit_IRLS(X, **kwargs)
#         elif method == "PAM":
#             self._fit_PAM(X, distX=distX, **kwargs)
#         elif method == "kmeans":
#             self._fit_kmeans(X, **kwargs)
        
#         print(f"Fitting complete using {method} method.")
#         return self
        
#         print(f"Fitting complete using {method} method.")
#         return self
    
#     def _fit_random(self, X):
#         """
#         Fit by randomly sampling prototypes from X.
        
#         Parameters:
#         -----------
#         X : array, shape (N, d)
#             Training data
#         """
#         N = X.shape[0]
#         indices = np.random.choice(N, size=self.M, replace=False)
#         self.W = X[indices].copy()
        
#         # Update BMU assignments for consistency with other methods
#         self.recaller.update_BMU(X, self.W)
        
#         print(f"Randomly sampled {self.M} prototypes from {N} observations")
    
#     def _fit_IRLS(self, X, tol_RFstab=0.01, tol_MQE=1e-4, max_iter=100, init_method="random"):
#         """
#         Fit using Iteratively Reweighted Least Squares.
        
#         Parameters:
#         -----------
#         X : array, shape (N, d)
#             Training data
#         tol_RFstab : float, default=0.01
#             Tolerance for receptive field stability (proportion of changed assignments)
#         tol_MQE : float, default=1e-4
#             Tolerance for relative change in Mean Quantization Error
#         max_iter : int, default=100
#             Maximum number of iterations
#         init_method : str, default="random"
#             Initialization method for prototypes
#         """
#         # Initialize prototypes
#         if init_method == "random":
#             self._fit_random(X)
#         elif init_method == "PAM":
#             self._fit_PAM(X)
#         else:
#             raise ValueError(f"Unknown initialization method: {init_method}")
        
#         # Initial BMU assignment
#         self.recaller.update_BMU(X, self.W)
#         prevBMU = self.recaller.BMU[:, 0].copy()  # First column (closest prototype indices)
#         prevMQE = np.mean(self.recaller.QE[:, 0])  # Initial MQE
        
#         print(f"Initial MQE: {prevMQE:.6f}")
        
#         # Main IRLS loop
#         for iter_num in range(1, max_iter + 1):
#             # Update all prototypes using IRLS formula
#             self._update_prototypes_IRLS(X)
            
#             # Update BMU assignments (expensive operation)
#             self.recaller.update_BMU(X, self.W)
            
#             # Compute receptive field stability
#             current_BMU = self.recaller.BMU[:, 0]
#             RFstab = np.mean(current_BMU != prevBMU)
            
#             # Compute MQE and its relative change
#             MQE = np.mean(self.recaller.QE[:, 0])
#             rel_MQE_change = (MQE - prevMQE) / prevMQE if prevMQE != 0 else 0
            
#             # Print status
#             print(f"Iter {iter_num:3d}: RFstab = {RFstab:.4f}, MQE = {MQE:.6f}, rel_MQE_change = {rel_MQE_change:.6f}")
            
#             # Update previous values for next iteration
#             prevBMU = current_BMU.copy()
#             prevMQE = MQE
            
#             # Check convergence criteria
#             if RFstab < tol_RFstab:
#                 print(f"Converged after {iter_num} iterations (RFstab < {tol_RFstab})")
#                 break
#             elif abs(rel_MQE_change) < tol_MQE:
#                 print(f"Converged after {iter_num} iterations (|rel_MQE_change| < {tol_MQE})")
#                 break
#         else:
#             print(f"Reached maximum iterations ({max_iter}) without convergence")
    
#     def _update_prototypes_IRLS(self, X):
#         """
#         Update prototype vectors using IRLS formula for Lp distance.
        
#         Parameters:
#         -----------
#         X : array, shape (N, d)
#             Training data matrix
#         """
#         M = self.W.shape[0]
        
#         for j in range(M):
#             # Find all observations assigned to prototype j
#             cluster_mask = (self.recaller.BMU[:, 0] == j)
#             cluster_points = X[cluster_mask]
            
#             if len(cluster_points) == 0:
#                 # Dead prototype - skip update
#                 continue
            
#             if self.p == 2:
#                 # Standard K-means update (mean)
#                 self.W[j] = np.mean(cluster_points, axis=0)
#             else:
#                 # IRLS update for arbitrary Lp
#                 # Add tiny perturbation to avoid singularities
#                 W_perturbed = self.W[j] + np.random.normal(0, 1e-12, self.W[j].shape)
                
#                 # Compute weights: |x_ij - w_j|^(p-2)
#                 diff = np.abs(cluster_points - W_perturbed)
#                 weights = diff**(self.p - 2)
                
#                 # Weighted mean update
#                 numerator = np.sum(cluster_points * weights, axis=0)
#                 denominator = np.sum(weights, axis=0)
                
#                 # Avoid division by zero (shouldn't happen with perturbation, but be safe)
#                 denominator = np.where(denominator == 0, 1e-10, denominator)
                
#                 self.W[j] = numerator / denominator
    
#     def _fit_PAM(self, X, distX=None, **kwargs):
#         """
#         Fit using PAM (Partitioning Around Medoids) algorithm.
        
#         Parameters:
#         -----------
#         X : array, shape (N, d)
#             Training data
#         distX: array, shape (N, N). Optional distance matrix of X. If provided, internal distance calculation is skipped. 
#         **kwargs : keyword arguments
#             All kmedoids.fasterpam parameters (max_iter=100, init='random', random_state=None, n_cpu=-1)
#             The random_state from initialization will override any provided random_state
#         """
#         try:
#             from kmedoids import fasterpam
#         except ImportError:
#             raise ImportError("kmedoids package required for PAM. Install with: pip install kmedoids")
        
#         N = X.shape[0]
        
#         if distX is None:
#             print(f"Computing {N}x{N} Lp distance matrix for PAM...")
#             distX = squareform(pdist(X, metric='minkowski', p=self.p)) 
        
#         print(f"Running FasterPAM with {self.M} medoids...")
        
#         # Use stored random_state if available
#         # Remove random_state from kwargs if provided, we will use the one stored in the class 
#         kwargs.pop('random_state', None)  # Remove if present
#         random_state = self.random_state
#         result = fasterpam(diss=distX, medoids=self.M, random_state=random_state, **kwargs)
#         medoid_indices = result.medoids
        
#         # Set prototypes to the selected medoids
#         self.W = X[medoid_indices].copy()
        
#         # Update BMU assignments based on PAM result
#         self.recaller.update_BMU(X, self.W)
        
#         print(f"PAM fitting complete. Final MQE: {np.mean(self.recaller.QE[:, 0]):.6f}")
    
#     def _fit_kmeans(self, X, **kwargs):
#         """
#         Fit using scikit-learn's K-means algorithm.
        
#         Only works when p=2 (Euclidean distance). Accepts all standard
#         K-means parameters as keyword arguments.
        
#         Parameters:
#         -----------
#         X : array, shape (N, d)
#             Training data
#         **kwargs : keyword arguments
#             All K-means parameters (init, n_init, max_iter, tol, algorithm, etc.)
#             The random_state from initialization will override any provided random_state
#         """
#         if self.p != 2:
#             raise ValueError(f"K-means only supports p=2 (Euclidean distance), got p={self.p}")
        
#         try:
#             from sklearn.cluster import KMeans
#         except ImportError:
#             raise ImportError("Scikit-learn required for K-means. Install with: pip install scikit-learn")
        
#         # Extract random_state from kwargs if provided, but use stored one
#         kwargs.pop('random_state', None)  # Remove if present
#         kwargs.pop('n_clusters', None)    # Remove if present (we use self.M)
        
#         # Create K-means with our parameters
#         kmeans = KMeans(
#             n_clusters=self.M,
#             random_state=self.random_state,
#             **kwargs
#         )
        
#         print(f"Running K-means with {self.M} clusters...")
        
#         # Fit the model
#         kmeans.fit(X)
        
#         # Store the cluster centers as prototypes
#         self.W = kmeans.cluster_centers_.copy()
        
#         # Update BMU assignments for consistency with other methods
#         self.recaller.update_BMU(X, self.W)
        
#         print(f"K-means fitting complete. Final MQE: {np.mean(self.recaller.QE[:, 0]):.6f}")
#         print(f"K-means converged: {kmeans.n_iter_} iterations, inertia: {kmeans.inertia_:.6f}")
    
#     def recall(self, X=None):
#         """
#         Perform recall analysis using the fitted prototypes.
        
#         If X is provided, performs recall analysis on new data.
#         If X is None, finalizes recall analysis on training data (assuming
#         BMU has been computed during fitting).
        
#         Parameters:
#         -----------
#         X : array-like, shape (N, d), optional
#             Data matrix for recall analysis. If None, finalizes training recall.
            
#         Returns:
#         --------
#         self : VQFitter
#             Returns self for method chaining
#         """
#         if self.W is None:
#             raise RuntimeError("Model not fitted. Call fit() first.")
        
#         if X is not None:
#             # Recall analysis on new data
#             X = np.array(X)
#             self.recaller.recall(X, self.W)
#         else:
#             # Finalize recall on training data (BMU should already exist)
#             self.recaller.recall()
        
#         return self
    
#     def get_summary(self):
#         """
#         Get a summary of the fitted model and recall analysis.
        
#         Returns:
#         --------
#         dict : Summary statistics
#         """
#         summary = {
#             "fitted": self.W is not None,
#             "M": self.M,
#             "d": self.W.shape[1] if self.W is not None else None,
#             "p_norm": self.p,
#             "max_bmu": self.max_bmu,
#             "random_state": self.random_state
#         }
        
#         # Add recall summary if available
#         if self.recaller.BMU is not None:
#             recall_summary = self.recaller.get_summary()
#             summary.update(recall_summary)
        
#         return summary
    

# def lp_norm_stable(x, p):
    # """
    # Computes the Lp norm of a vector or matrix using a numerically stable method.
    
    # Args:
    #     x (numpy.ndarray): The vector or matrix of vectors.
    #     p (float): The p-norm value.
        
    # Returns:
    #     numpy.ndarray: The Lp norm.
    # """
    # if p < 1:
    #     p_root = 1 / (p + 1e-12)
    # else:
    #     p_root = 1 / p

    # abs_x = np.abs(x)
    
    # # Use a small epsilon to avoid log(0)
    # abs_x[abs_x < 1e-12] = 1e-12

    # # Compute the log of the p-th powers
    # log_p_powers = p * np.log(abs_x)
    
    # # Clip the log powers to prevent exp() from overflowing with a safe upper bound
    # log_p_powers = np.clip(log_p_powers, None, 700)

    # # Use logsumexp trick to sum the powers in a stable way
    # log_sum_powers = np.logaddexp.reduce(log_p_powers, axis=-1, keepdims=True)
    
    # # The final Lp norm is exp(log_sum_powers / p)
    # return np.exp(p_root * log_sum_powers)



def lp_centroid_objective(c, X, p):
    """
    Computes the Lp centroid objective function.
    
    Args:
        c (numpy.ndarray): The candidate centroid.
        X (numpy.ndarray): The dataset of shape (N, d).
        p (float): The p-norm value.
        
    Returns:
        float: The value of the objective function.
    """
    # Calculate the difference between each data point and the centroid
    diff = X - c
    
    # Compute the Lp norm for each difference vector
    lp_norm = np.linalg.norm(diff, ord=p, axis=1)
    
    # Return the sum of the p-th powers of the norms
    return np.sum(lp_norm**p)

def lp_centroid_gradient(c, X, p):

    """
    Computes the gradient of the Lp centroid objective function.
    
    Args:
        c (numpy.ndarray): The candidate centroid.
        X (numpy.ndarray): The dataset of shape (N, d).
        p (float): The p-norm value.
        
    Returns:
        numpy.ndarray: The gradient vector.
    """
    if p == 1:
        # Special case for L1, the derivative is the sign function
        return -np.sum(np.sign(X - c), axis=0)
    else:
        # General case for p > 1
        diff = X - c
        # The gradient is -p * sum((x_i - c) * |x_i - c|^(p-2))
        return -p * np.sum(diff * np.abs(diff)**(p - 2), axis=0)
    

def lp_centroid_objective_stable_exp(c, X, p):
    """
    Computes the Lp centroid objective function with numerical stability
    using the Log-Exp trick.
    
    Args:
        c (numpy.ndarray): The candidate centroid.
        X (numpy.ndarray): The dataset of shape (N, d).
        p (float): The p-norm value.
        
    Returns:
        float: The value of the objective function.
    """
    if p == 2:
        # Standard L2 norm calculation for efficiency
        return np.sum(np.linalg.norm(X - c, ord=2, axis=1)**2)

    diff = X - c
    abs_diff = np.abs(diff)
    
    # Use a small epsilon to avoid log(0)
    abs_diff[abs_diff < 1e-12] = 1e-12
    
    # Calculate log of the absolute differences
    log_abs_diff = np.log(abs_diff)
    
    # Find the maximum log for each data point
    max_log = np.max(log_abs_diff, axis=1, keepdims=True)
    
    # Calculate the exponent term for the log-sum-exp trick
    exponent_term = p * (log_abs_diff - max_log)
    
    # The inner sum is exp(max_log*p) * sum(exp(exponent_term))
    # This is equivalent to sum(|diff|^p)
    inner_sum = np.exp(p * max_log) * np.sum(np.exp(exponent_term), axis=1, keepdims=True)
    
    # The objective function is the sum of these inner sums
    return np.sum(inner_sum)

def lp_centroid_gradient_stable_exp(c, X, p):
    """
    Computes the gradient of the Lp centroid objective function using a more
    numerically stable approach for large p.
    
    Args:
        c (numpy.ndarray): The candidate centroid.
        X (numpy.ndarray): The dataset of shape (N, d).
        p (float): The p-norm value.
        
    Returns:
        numpy.ndarray: The gradient vector.
    """
    diff = X - c
    abs_diff = np.abs(diff)
    
    # Handle the p=1 case separately to avoid log(0) issues
    if p == 1:
        return -np.sum(np.sign(diff), axis=0)

    # Use a small epsilon to avoid log(0)
    abs_diff[abs_diff < 1e-12] = 1e-12
    
    # This is the problematic term: |x - c|^(p-2)
    # We compute it using exp and log for numerical stability
    log_power_term = (p - 2) * np.log(abs_diff)
    
    # Check for potential overflow before exponentiation
    if np.any(log_power_term > np.log(np.finfo(float).max)):
        print("Warning: Potential overflow in gradient calculation. This may lead to instability.")

    power_term = np.exp(log_power_term)
    
    # The gradient is -p * sum(diff * power_term)
    return -p * np.sum(diff * power_term, axis=0)


def lp_centroid_objective_p_root(c, X, p):
    """
    Computes the p-th root of the Lp centroid objective function with numerical stability.
    
    Args:
        c (numpy.ndarray): The candidate centroid.
        X (numpy.ndarray): The dataset of shape (N, d).
        p (float): The p-norm value.
        
    Returns:
        float: The value of the objective function.
    """
    diff = X - c
    
    # The objective is the sum of the Lp norms of the difference vectors
    return np.sum(lp_norm_stable(diff, p))

def lp_centroid_gradient_p_root(c, X, p):
    """
    Computes the gradient of the p-th root of the Lp centroid objective function.
    
    Args:
        c (numpy.ndarray): The candidate centroid.
        X (numpy.ndarray): The dataset of shape (N, d).
        p (float): The p-norm value.
        
    Returns:
        numpy.ndarray: The gradient vector.
    """
    if p == 1:
        # Special case for L1
        return -np.sum(np.sign(X - c), axis=0)

    diff = X - c
    abs_diff = np.abs(diff)
    abs_diff[abs_diff < 1e-12] = 1e-12
    
    # Get the stable Lp norm for each data point
    lp_norm_vector = lp_norm_stable(diff, p)
    
    # Add a small epsilon to prevent division by zero or NaN issues
    lp_norm_vector += 1e-12
    
    # The core term is |x_ij - c_j|^(p-2)
    power_term = np.exp((p - 2) * np.log(abs_diff))
    
    # The second part of the gradient expression is the diff * power_term
    weighted_diff = diff * power_term
    
    # The final gradient is -sum(weighted_diff / (p-norm)^(p-1))
    denominator = lp_norm_vector**(p - 1)
    
    # Reshape denominator for broadcasting
    denominator = denominator.reshape(-1, 1)

    return -np.sum(weighted_diff / denominator, axis=0)


class VQFitter:
    """
    Vector Quantizer Fitter using arbitrary Lp distance metrics.
    
    Supports multiple fitting algorithms including random sampling,
    IRLS (Iteratively Reweighted Least Squares), and PAM (Partitioning Around Medoids).
    """
    
    AVAILABLE_METHODS = ["random", "IRLS", "PAM", "kmeans", "GD"]
    
    def __init__(self, M, p=2, max_bmu=2, random_state=None):
        """
        Initialize the VQ Fitter.
        
        Parameters:
        -----------
        M : int
            Number of prototypes to create
        p : float, default=2
            Order of the Lp distance metric (e.g., 1 for Manhattan, 2 for Euclidean)
        max_bmu : int, default=2
            Number of Best Matching Units (closest prototypes) to track
        random_state : int, optional
            Random seed for reproducible results
        """
        self.M = int(M)  # Ensure M is an integer
        self.p = p
        self.max_bmu = max_bmu
        self.random_state = random_state
        self.W = None  # Prototype matrix (M, d)
        
        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # VQRecaller instance for analysis
        self.recaller = VQRecaller(p=self.p, max_bmu=self.max_bmu)
    
    def fit(self, X, method="IRLS", distX = None, **kwargs):
        """
        Fit the vector quantizer using the specified method.
        
        Parameters:
        -----------
        X : array-like, shape (N, d)
            Training data matrix
        method : str, default="IRLS"
            Fitting method to use. Available methods: {cls.AVAILABLE_METHODS}
        distX: array, shape (N, N). Optional distance matrix of X. If provided, internal distance calculation is skipped. 
        **kwargs : additional arguments
            Method-specific parameters
            
        Returns:
        --------
        self : VQFitter
            Returns self for method chaining
        """
        X = np.array(X)
        N, d = X.shape
        
        if self.M > N:
            raise ValueError(f"Cannot create {self.M} prototypes from only {N} observations")
        
        # Validate method
        if method not in self.AVAILABLE_METHODS:
            raise ValueError(f"Unknown fitting method '{method}'. Available methods: {self.AVAILABLE_METHODS}")
        
        # Dispatch to appropriate fitting method
        if method == "random":
            self._fit_random(X, **kwargs)
        elif method == "IRLS":
            self._fit_IRLS(X, **kwargs)
        elif method == "PAM":
            self._fit_PAM(X, distX=distX, **kwargs)
        elif method == "kmeans":
            self._fit_kmeans(X, **kwargs)
        elif method == "GD":
            self._fit_GD(X, **kwargs)
        
        print(f"Fitting complete using {method} method.")
        return self
            
    def _fit_random(self, X):
        """
        Fit by randomly sampling prototypes from X.
        
        Parameters:
        -----------
        X : array, shape (N, d)
            Training data
        """
        N = X.shape[0]
        indices = np.random.choice(N, size=self.M, replace=False)
        self.W = X[indices].copy()
        
        # Update BMU assignments for consistency with other methods
        self.recaller.update_BMU(X, self.W)
        
        print(f"Randomly sampled {self.M} prototypes from {N} observations")
    
    def _fit_IRLS(self, X, tol_RFstab=0.01, tol_MQE=1e-4, max_iter=100, init_method="random"):
        """
        Fit using Iteratively Reweighted Least Squares.
        
        Parameters:
        -----------
        X : array, shape (N, d)
            Training data
        tol_RFstab : float, default=0.01
            Tolerance for receptive field stability (proportion of changed assignments)
        tol_MQE : float, default=1e-4
            Tolerance for relative change in Mean Quantization Error
        max_iter : int, default=100
            Maximum number of iterations
        init_method : str, default="random"
            Initialization method for prototypes
        """
        # Initialize prototypes
        if init_method == "random":
            self._fit_random(X)
        elif init_method == "PAM":
            self._fit_PAM(X)
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
        
        # Initial BMU assignment
        self.recaller.update_BMU(X, self.W)
        prevBMU = self.recaller.BMU[:, 0].copy()  # First column (closest prototype indices)
        prevMQE = np.mean(self.recaller.QE[:, 0])  # Initial MQE
        
        print(f"Initial MQE: {prevMQE:.6f}")
        
        # Main IRLS loop
        for iter_num in range(1, max_iter + 1):
            # Update all prototypes using IRLS formula
            self._update_prototypes_IRLS_pth_root(X)
            
            # Update BMU assignments (expensive operation)
            self.recaller.update_BMU(X, self.W)
            
            # Compute receptive field stability
            current_BMU = self.recaller.BMU[:, 0]
            RFstab = np.mean(current_BMU != prevBMU)
            
            # Compute MQE and its relative change
            MQE = np.mean(self.recaller.QE[:, 0])
            rel_MQE_change = (MQE - prevMQE) / prevMQE if prevMQE != 0 else 0
            
            # Print status
            print(f"Iter {iter_num:3d}: RFstab = {RFstab:.4f}, MQE = {MQE:.6f}, rel_MQE_change = {rel_MQE_change:.6f}")
            
            # Update previous values for next iteration
            prevBMU = current_BMU.copy()
            prevMQE = MQE
            
            # Check convergence criteria
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
        Update prototype vectors using IRLS formula for Lp distance.
        (Numerically stable for high p values)
        
        Parameters:
        -----------
        X : array, shape (N, d)
            Training data matrix
        """
        M = self.W.shape[0]
        
        for j in range(M):
            # Find all observations assigned to prototype j
            cluster_mask = (self.recaller.BMU[:, 0] == j)
            cluster_points = X[cluster_mask]
            
            if len(cluster_points) == 0:
                # Dead prototype - skip update
                continue
            
            if self.p == 2:
                # Standard K-means update (mean)
                self.W[j] = np.mean(cluster_points, axis=0)
            else:
                # Add tiny perturbation to avoid singularities
                W_perturbed = self.W[j] + np.random.normal(0, 1e-12, self.W[j].shape)
                
                # Calculate component-wise differences
                diff = cluster_points - W_perturbed
                
                # --- 💥 This is the core stabilization change 💥 ---
                # Calculate weights using a numerically stable method
                abs_diff = np.abs(diff)
                abs_diff[abs_diff < 1e-12] = 1e-12
                
                log_weights = (self.p - 2) * np.log(abs_diff)
                np.clip(log_weights, None, np.log(np.finfo(np.float64).max), out=log_weights)
                weights = np.exp(log_weights)
                
                # Weighted mean update
                numerator = np.sum(cluster_points * weights, axis=0)
                denominator = np.sum(weights, axis=0)
                
                # Avoid division by zero
                denominator = np.where(denominator == 0, 1e-10, denominator)
                
                self.W[j] = numerator / denominator

    def _update_prototypes_IRLS_pth_root(self, X):
        """
        Update prototype vectors using Iteratively Reweighted Least Squares (IRLS)
        for minimizing the sum of Lp norms.
        (Numerically stable for high p values)
        
        Parameters:
        -----------
        X : array, shape (N, d)
            Training data matrix
        """
        M = self.W.shape[0]
        
        for j in range(M):
            # Find all observations assigned to prototype j
            cluster_mask = (self.recaller.BMU[:, 0] == j)
            cluster_points = X[cluster_mask]
            
            if len(cluster_points) == 0:
                continue
            
            # Add a tiny perturbation to avoid singularities
            W_perturbed = self.W[j] + np.random.normal(0, 1e-12, self.W[j].shape)
            
            # Calculate distances from cluster points to the current prototype
            diff = cluster_points - W_perturbed
            
            # Use a stable Lp norm calculation
            distances = lp_norm_stable(diff, self.p)
            
            # Add a small epsilon to avoid division by zero
            distances[distances < 1e-12] = 1e-12

            # The weight update is now (distance)**(1-p) 💥
            weights = distances**(1 - self.p)
            
            # Reshape weights for broadcasting
            weights_reshaped = weights.reshape(-1, 1)

            # Update the prototype using the new weights
            numerator = np.sum(cluster_points * weights_reshaped, axis=0)
            denominator = np.sum(weights)
            
            # Avoid division by zero
            if denominator == 0:
                continue
                
            self.W[j] = numerator / denominator
    
    def _fit_PAM(self, X, distX=None, **kwargs):
        """
        Fit using PAM (Partitioning Around Medoids) algorithm.
        
        Parameters:
        -----------
        X : array, shape (N, d)
            Training data
        distX: array, shape (N, N). Optional distance matrix of X. If provided, internal distance calculation is skipped. 
        **kwargs : keyword arguments
            All kmedoids.fasterpam parameters (max_iter=100, init='random', random_state=None, n_cpu=-1)
            The random_state from initialization will override any provided random_state
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
        
        # Use stored random_state if available
        # Remove random_state from kwargs if provided, we will use the one stored in the class 
        kwargs.pop('random_state', None)  # Remove if present
        random_state = self.random_state
        result = fasterpam(diss=distX, medoids=self.M, random_state=random_state, **kwargs)
        medoid_indices = result.medoids
        
        # Set prototypes to the selected medoids
        self.W = X[medoid_indices].copy()
        
        # Update BMU assignments based on PAM result
        self.recaller.update_BMU(X, self.W)
        
        print(f"PAM fitting complete. Final MQE: {np.mean(self.recaller.QE[:, 0]):.6f}")
    
    def _fit_kmeans(self, X, **kwargs):
        """
        Fit using scikit-learn's K-means algorithm.
        
        Only works when p=2 (Euclidean distance). Accepts all standard
        K-means parameters as keyword arguments.
        
        Parameters:
        -----------
        X : array, shape (N, d)
            Training data
        **kwargs : keyword arguments
            All K-means parameters (init, n_init, max_iter, tol, algorithm, etc.)
            The random_state from initialization will override any provided random_state
        """
        if self.p != 2:
            raise ValueError(f"K-means only supports p=2 (Euclidean distance), got p={self.p}")
        
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError("Scikit-learn required for K-means. Install with: pip install scikit-learn")
        
        # Extract random_state from kwargs if provided, but use stored one
        kwargs.pop('random_state', None)  # Remove if present
        kwargs.pop('n_clusters', None)    # Remove if present (we use self.M)
        
        # Create K-means with our parameters
        kmeans = KMeans(
            n_clusters=self.M,
            random_state=self.random_state,
            **kwargs
        )
        
        print(f"Running K-means with {self.M} clusters...")
        
        # Fit the model
        kmeans.fit(X)
        
        # Store the cluster centers as prototypes
        self.W = kmeans.cluster_centers_.copy()
        
        # Update BMU assignments for consistency with other methods
        self.recaller.update_BMU(X, self.W)
        
        print(f"K-means fitting complete. Final MQE: {np.mean(self.recaller.QE[:, 0]):.6f}")
        print(f"K-means converged: {kmeans.n_iter_} iterations, inertia: {kmeans.inertia_:.6f}")
    
    def _update_prototypes_GD(self, X):
        """
        Update prototype vectors using Gradient Descent for Lp centroids.
        
        Parameters:
        -----------
        X : array, shape (N, d)
            Training data matrix
        """
        M = self.W.shape[0]
        
        for j in range(M):
            # Find all observations assigned to prototype j
            cluster_mask = (self.recaller.BMU[:, 0] == j)
            cluster_points = X[cluster_mask]
            
            if len(cluster_points) == 0:
                # Dead prototype - skip update
                continue
            
            # Precompute the L2 mean, it will serve as a starting point 
            L2_centroid = np.mean(cluster_points, axis=0)

            if self.p == 2:
                # Standard K-means update (mean)
                self.W[j] = L2_centroid
            else:
                # Minimize the objective function
                result = minimize(x0=L2_centroid, 
                                  fun=lp_centroid_objective_p_root,
                                  jac=lp_centroid_gradient_p_root,  # Provide the gradient function
                                  args=(cluster_points, self.p),  # Pass data and p to the objective function
                                  method='L-BFGS-B'  # An efficient quasi-Newton method
                )
                # The optimal centroid is in the 'x' attribute of the result
                self.W[j] = result.x

    def _fit_GD(self, X, tol_RFstab=0.01, tol_MQE=1e-4, max_iter=100, init_method="random"):
        """
        Fit using Iteratively Reweighted Least Squares.
        
        Parameters:
        -----------
        X : array, shape (N, d)
            Training data
        tol_RFstab : float, default=0.01
            Tolerance for receptive field stability (proportion of changed assignments)
        tol_MQE : float, default=1e-4
            Tolerance for relative change in Mean Quantization Error
        max_iter : int, default=100
            Maximum number of iterations
        init_method : str, default="random"
            Initialization method for prototypes
        """
        # Initialize prototypes
        if init_method == "random":
            self._fit_random(X)
        elif init_method == "PAM":
            self._fit_PAM(X)
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
        
        # Initial BMU assignment
        self.recaller.update_BMU(X, self.W)
        prevBMU = self.recaller.BMU[:, 0].copy()  # First column (closest prototype indices)
        prevMQE = np.mean(self.recaller.QE[:, 0])  # Initial MQE
        
        print(f"Initial MQE: {prevMQE:.6f}")
        
        # Main GD loop
        for iter_num in range(1, max_iter + 1):
            # Update all prototypes using IRLS formula
            self._update_prototypes_GD(X)
            
            # Update BMU assignments (expensive operation)
            self.recaller.update_BMU(X, self.W)
            
            # Compute receptive field stability
            current_BMU = self.recaller.BMU[:, 0]
            RFstab = np.mean(current_BMU != prevBMU)
            
            # Compute MQE and its relative change
            MQE = np.mean(self.recaller.QE[:, 0])
            rel_MQE_change = (MQE - prevMQE) / prevMQE if prevMQE != 0 else 0
            
            # Print status
            print(f"Iter {iter_num:3d}: RFstab = {RFstab:.4f}, MQE = {MQE:.6f}, rel_MQE_change = {rel_MQE_change:.6f}")
            
            # Update previous values for next iteration
            prevBMU = current_BMU.copy()
            prevMQE = MQE
            
            # Check convergence criteria
            if RFstab < tol_RFstab:
                print(f"Converged after {iter_num} iterations (RFstab < {tol_RFstab})")
                break
            elif abs(rel_MQE_change) < tol_MQE:
                print(f"Converged after {iter_num} iterations (|rel_MQE_change| < {tol_MQE})")
                break
        else:
            print(f"Reached maximum iterations ({max_iter}) without convergence")
    

    def recall(self, X=None):
        """
        Perform recall analysis using the fitted prototypes.
        
        If X is provided, performs recall analysis on new data.
        If X is None, finalizes recall analysis on training data (assuming
        BMU has been computed during fitting).
        
        Parameters:
        -----------
        X : array-like, shape (N, d), optional
            Data matrix for recall analysis. If None, finalizes training recall.
            
        Returns:
        --------
        self : VQFitter
            Returns self for method chaining
        """
        if self.W is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if X is not None:
            # Recall analysis on new data
            X = np.array(X)
            self.recaller.recall(X, self.W)
        else:
            # Finalize recall on training data (BMU should already exist)
            self.recaller.recall()
        
        return self
    
    def get_summary(self):
        """
        Get a summary of the fitted model and recall analysis.
        
        Returns:
        --------
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
        
        # Add recall summary if available
        if self.recaller.BMU is not None:
            recall_summary = self.recaller.get_summary()
            summary.update(recall_summary)
        
        return summary
