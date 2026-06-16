"""
example_vqlp.py
---------------
A worked example of the vqlp package demonstrating:
  1. Synthetic data generation (sklearn make_blobs)
  2. Fitting prototypes via FAISS k-means (VQFitter)
  3. Full recall analysis on the learned prototypes (VQRecaller)
  4. Plotting data, prototypes, and the connectivity (CONN) graph
  5. Inspecting the recall products
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from vqlp import VQFitter

# =============================================================================
# 1. Generate synthetic data
# =============================================================================
N_SAMPLES = 500
N_BLOBS   = 6
N_FEATURES = 2      # 2D so we can plot it easily
M_PROTOTYPES = 20   # more prototypes than blobs to get interesting CONN structure

X, true_labels = make_blobs(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    centers=N_BLOBS,
    cluster_std=0.8,
    random_state=42
)
print(f"Data shape: {X.shape}")

# =============================================================================
# 2. Fit prototypes using FAISS k-means (default for p=2)
# =============================================================================
fitter = VQFitter(M=M_PROTOTYPES, p=2, max_bmu=2, random_state=42)
fitter.fit(X, method="kmeans", niter=30, nredo=3)

W = fitter.W   # prototype matrix, shape (M, d)
print(f"Prototype matrix shape: {W.shape}")

# =============================================================================
# 3. Full recall analysis
# =============================================================================
fitter.recall()        # finalizes RF, CONN etc. on training data
recaller = fitter.recaller

# =============================================================================
# 4. Plot: data coloured by BMU, prototypes, and CONN graph
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left panel: data coloured by BMU assignment --------------------------
ax = axes[0]
ax.set_title("Data coloured by BMU assignment")

bmu_ids = recaller.BMU[:, 0]
scatter = ax.scatter(X[:, 0], X[:, 1], c=bmu_ids, cmap="tab20",
                     s=15, alpha=0.6, zorder=1)
ax.scatter(W[:, 0], W[:, 1], c="black", marker="X", s=120,
           zorder=3, label="Prototypes")
ax.legend(loc="upper right")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
plt.colorbar(scatter, ax=ax, label="BMU index")

# --- Right panel: prototype connectivity (CONN) graph ---------------------
ax = axes[1]
ax.set_title("Prototype connectivity (CONN) graph")

# Draw edges weighted by CONN value
conn_csr = recaller.CONN.tocsr()
for i in range(recaller._M):
    for j in recaller.CONN_nhbs[i]:
        if j <= i:   # draw each edge once
            continue
        weight = conn_csr[i, j]
        ax.plot([W[i, 0], W[j, 0]],
                [W[i, 1], W[j, 1]],
                color="steelblue", linewidth=weight * 0.3, alpha=0.6, zorder=1)

# Prototypes sized by receptive field size
rf_sizes = recaller.RFSize
ax.scatter(W[:, 0], W[:, 1],
           s=rf_sizes * 4 + 20,
           c=np.arange(M_PROTOTYPES), cmap="tab20",
           edgecolors="black", linewidths=0.5, zorder=2)

for i, (x, y) in enumerate(W):
    ax.annotate(str(i), (x, y), fontsize=7, ha="center", va="center",
                color="white", fontweight="bold", zorder=3)

ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title("CONN graph\n(edge width ∝ connectivity, node size ∝ RF size)")

plt.tight_layout()
plt.savefig("vqlp_example.png", dpi=150)
plt.show()
print("Plot saved to vqlp_example.png")

# =============================================================================
# 5. Inspect recall products
# =============================================================================
print("\n--- get_summary() ---")
summary = fitter.get_summary()
for k, v in summary.items():
    print(f"  {k:35s}: {v}")

print("\n--- BMU & QE (first 8 observations) ---")
print(f"  BMU shape : {recaller.BMU.shape}   (N x max_bmu)")
print(f"  QE  shape : {recaller.QE.shape}   (N x max_bmu)")
for i in range(8):
    print(f"  obs {i:3d}  |  1st BMU: {recaller.BMU[i,0]:3d}  QE1: {recaller.QE[i,0]:.4f}"
          f"  |  2nd BMU: {recaller.BMU[i,1]:3d}  QE2: {recaller.QE[i,1]:.4f}")

print("\n--- Receptive fields (RF) ---")
print(f"  RFSize range: {recaller.RFSize.min()} – {recaller.RFSize.max()} observations")
print(f"  Empty prototypes: {int(np.sum(recaller.RFSize == 0))}")
for i in range(min(5, M_PROTOTYPES)):
    print(f"  RF[{i}]: {recaller.RFSize[i]:3d} obs  ->  indices {recaller.RF[i][:6]}{'...' if recaller.RFSize[i] > 6 else ''}")

print("\n--- Connectivity matrix (CONN) ---")
print(f"  CONN shape  : {recaller.CONN.shape}   (M x M, sparse)")
print(f"  Nonzero entries: {recaller.CONN.nnz}  ({summary['connectivity_edges']} unique edges)")
print(f"  Degree range: {recaller.CONN_nhbs_size.min()} – {recaller.CONN_nhbs_size.max()} neighbours")
print(f"  Mean degree : {summary['mean_connectivity_degree']:.2f}")
for i in range(min(5, M_PROTOTYPES)):
    print(f"  Prototype {i:2d}: {recaller.CONN_nhbs_size[i]} neighbours  ->  {recaller.CONN_nhbs[i]}")

print("\n--- Reconstruction ---")
X_hard = recaller.reconstruct(W, X, method="hard")
X_soft = recaller.reconstruct(W, X, method="soft")
hard_err = np.mean(np.linalg.norm(X - X_hard, axis=1))
soft_err = np.mean(np.linalg.norm(X - X_soft, axis=1))
print(f"  Hard reconstruction mean L2 error: {hard_err:.4f}")
print(f"  Soft reconstruction mean L2 error: {soft_err:.4f}")