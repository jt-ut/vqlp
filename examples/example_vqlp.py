"""
example_vqlp.py
---------------
A worked example of the vqlp package demonstrating:
1. Synthetic data generation (sklearn make_blobs)
2. Fitting prototypes via FAISS k-means (VQFitter)
3. Full recall analysis on the learned prototypes (VQRecaller), including labels
4. Plotting data, prototypes, and the connectivity (CONN) graph
5. Inspecting the recall products, including label summary products
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.datasets import make_blobs
from vqlp import VQFitter

# =============================================================================
# 1. Generate synthetic data
# =============================================================================

N_SAMPLES = 500
N_BLOBS = 6
N_FEATURES = 2  # 2D so we can plot it easily
M_PROTOTYPES = 20  # more prototypes than blobs to get interesting CONN structure

X, true_labels = make_blobs(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    centers=N_BLOBS,
    cluster_std=0.8,
    random_state=42
)

print(f"Data shape: {X.shape}")
print(f"Unique blob labels: {np.unique(true_labels)}")

# Build a fixed colormap for blob labels so colors are consistent across panels
N_LABEL_COLORS = N_BLOBS
label_cmap = plt.cm.get_cmap("tab10", N_LABEL_COLORS)
label_colors = {lbl: label_cmap(i) for i, lbl in enumerate(np.unique(true_labels))}

# =============================================================================
# 2. Fit prototypes using FAISS k-means (default for p=2)
# =============================================================================

fitter = VQFitter(M=M_PROTOTYPES, p=2, max_bmu=2, random_state=42)
fitter.fit(X, method="kmeans", niter=30, nredo=3)

W = fitter.W  # prototype matrix, shape (M, d)
print(f"Prototype matrix shape: {W.shape}")

# =============================================================================
# 3. Full recall analysis, passing true_labels so WL/WL_Dist/WL_Purity are computed
# =============================================================================

fitter.recall(labels=true_labels)  # finalizes RF, CONN, and label recall
recaller = fitter.recaller

# =============================================================================
# 4. Plot: data coloured by BMU, prototypes, and CONN graph
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left panel: data coloured by true blob label, prototypes by winning label --

ax = axes[0]
ax.set_title("Data coloured by true label\nPrototypes coloured by winning label (WL)")

# Scatter data points, coloured by true blob label
for lbl in np.unique(true_labels):
    mask = true_labels == lbl
    ax.scatter(X[mask, 0], X[mask, 1],
               color=label_colors[lbl], s=15, alpha=0.5, zorder=1,
               label=f"Blob {lbl}")

# Prototype face colour = winning label colour; grey if empty RF (WL is None)
proto_colors = []
for i in range(M_PROTOTYPES):
    wl = recaller.WL[i]
    proto_colors.append(label_colors[wl] if wl is not None else (0.7, 0.7, 0.7, 1.0))

ax.scatter(W[:, 0], W[:, 1],
           c=proto_colors, marker="X", s=160,
           edgecolors="black", linewidths=0.8, zorder=3,
           label="Prototypes (WL color)")

ax.legend(loc="upper right", fontsize=7, ncol=2)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")

# --- Right panel: prototype connectivity (CONN) graph, nodes coloured by WL -----

ax = axes[1]
ax.set_title("CONN graph\n(edge width ∝ connectivity, node size ∝ RF size, node color = WL)")

conn_csr = recaller.CONN.tocsr()
for i in range(recaller._M):
    for j in recaller.CONN_nhbs[i]:
        if j <= i:  # draw each edge once
            continue
        weight = conn_csr[i, j]
        ax.plot([W[i, 0], W[j, 0]],
                [W[i, 1], W[j, 1]],
                color="steelblue", linewidth=weight * 0.3, alpha=0.6, zorder=1)

rf_sizes = recaller.RFSize
ax.scatter(W[:, 0], W[:, 1],
           s=rf_sizes * 4 + 20,
           c=proto_colors,
           edgecolors="black", linewidths=0.5, zorder=2)

for i, (x, y) in enumerate(W):
    ax.annotate(str(i), (x, y), fontsize=7, ha="center", va="center",
                color="white", fontweight="bold", zorder=3)

ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")

plt.tight_layout()
plt.savefig("vqlp_example.png", dpi=150)
plt.show()
print("Plot saved to vqlp_example.png")

# =============================================================================
# 5. Inspect recall products, including label summary
# =============================================================================

print("\n--- get_summary() ---")
summary = fitter.get_summary()
for k, v in summary.items():
    print(f"  {k:35s}: {v}")

print("\n--- BMU & QE (first 8 observations) ---")
print(f"  BMU shape : {recaller.BMU.shape} (N x max_bmu)")
print(f"  QE shape  : {recaller.QE.shape} (N x max_bmu)")
print(f"  AFF shape : {recaller.AFF.shape} (N x max_bmu)")
for i in range(8):
    print(f"  obs {i:3d} | 1st BMU: {recaller.BMU[i,0]:3d} QE1: {recaller.QE[i,0]:.4f}"
          f" AFF1: {recaller.AFF[i,0]:.4f}"
          f" | 2nd BMU: {recaller.BMU[i,1]:3d} QE2: {recaller.QE[i,1]:.4f}"
          f" AFF2: {recaller.AFF[i,1]:.4f}")

print("\n--- Receptive fields (RF) ---")
print(f"  RFSize range: {recaller.RFSize.min()} – {recaller.RFSize.max()} observations")
print(f"  Empty prototypes: {int(np.sum(recaller.RFSize == 0))}")
for i in range(min(5, M_PROTOTYPES)):
    print(f"  RF[{i}]: {recaller.RFSize[i]:3d} obs -> indices {recaller.RF[i][:6]}"
          f"{'...' if recaller.RFSize[i] > 6 else ''}")

print("\n--- Connectivity matrix (CONN) ---")
print(f"  CONN shape      : {recaller.CONN.shape} (M x M, sparse)")
print(f"  Nonzero entries : {recaller.CONN.nnz} ({summary['connectivity_edges']} unique edges)")
print(f"  Degree range    : {recaller.CONN_nhbs_size.min()} – {recaller.CONN_nhbs_size.max()} neighbours")
print(f"  Mean degree     : {summary['mean_connectivity_degree']:.2f}")
for i in range(min(5, M_PROTOTYPES)):
    print(f"  Prototype {i:2d}: {recaller.CONN_nhbs_size[i]} neighbours -> {recaller.CONN_nhbs[i]}")

print("\n--- Label recall (WL / WL_Dist / WL_Purity) ---")
print(f"  Unique labels seen : {recaller.WL_unq}")
print(f"  WL shape           : {recaller.WL.shape}  (winning label per prototype)")
print(f"  WL_Dist shape      : {recaller.WL_Dist.shape}  (M x n_unique_labels, row-normalized)")
print(f"  WL_Purity shape    : {recaller.WL_Purity.shape}")
print()
print(f"  {'Proto':>6}  {'WL':>5}  {'Purity':>7}  {'RF size':>7}  {'WL_Dist (per label)'}")
print(f"  {'-----':>6}  {'--':>5}  {'------':>7}  {'-------':>7}  "
      + "  ".join(f"L{l}" for l in recaller.WL_unq))
for i in range(M_PROTOTYPES):
    wl_str = str(recaller.WL[i]) if recaller.WL[i] is not None else "None"
    dist_str = "  ".join(f"{recaller.WL_Dist[i, j]:.3f}" for j in range(len(recaller.WL_unq)))
    print(f"  {i:6d}  {wl_str:>5}  {recaller.WL_Purity[i]:7.4f}  {recaller.RFSize[i]:7d}  {dist_str}")

print("\n--- Reconstruction ---")
X_hard = recaller.reconstruct(W, X, method="hard")
X_soft = recaller.reconstruct(W, X, method="soft")
hard_err = np.mean(np.linalg.norm(X - X_hard, axis=1))
soft_err = np.mean(np.linalg.norm(X - X_soft, axis=1))
print(f"  Hard reconstruction mean L2 error: {hard_err:.4f}")
print(f"  Soft reconstruction mean L2 error: {soft_err:.4f}")