import numpy as np

__all__ = [
    "lp_norm_stable",
    "lp_centroid_objective_p_root",
    "lp_centroid_gradient_p_root",
    "exp_kernel_bw_perplexity",
]


def lp_norm_stable(x, p):
    """
    Computes the Lp norm of row vectors using a numerically stable log-sum-exp method.

    Parameters
    ----------
    x : np.ndarray
        Input array. Norms are computed along the last axis.
    p : float
        Order of the norm. Must be positive. The 1e-12 guard in the exponent
        prevents division by zero at p=0; for any realistic p > 0 the
        numerical impact is negligible.

    Returns
    -------
    np.ndarray
        Lp norm along the last axis, with that axis kept (shape (..., 1)).
    """
    # 1e-12 guards against p=0; for any realistic p > 0 the impact is negligible
    p_root = 1 / (p + 1e-12)

    # np.where avoids in-place mutation of the input array
    abs_x = np.where(np.abs(x) < 1e-12, 1e-12, np.abs(x))

    log_p_powers = p * np.log(abs_x)
    log_p_powers = np.clip(log_p_powers, None, 700)
    log_sum_powers = np.logaddexp.reduce(log_p_powers, axis=-1, keepdims=True)

    return np.exp(p_root * log_sum_powers)


def lp_centroid_objective_p_root(c, X, p):
    """
    Computes the Lp centroid objective: sum of Lp norms from each point in X
    to candidate centroid c. This is the objective minimized when finding the
    Lp geometric median of a cluster.

    Parameters
    ----------
    c : np.ndarray, shape (d,)
        Candidate centroid.
    X : np.ndarray, shape (N, d)
        Cluster data points.
    p : float
        Order of the norm.

    Returns
    -------
    float
        Objective value.
    """
    diff = X - c
    return float(np.sum(lp_norm_stable(diff, p)))


def lp_centroid_gradient_p_root(c, X, p):
    """
    Computes the gradient of lp_centroid_objective_p_root with respect to c.

    Parameters
    ----------
    c : np.ndarray, shape (d,)
        Candidate centroid.
    X : np.ndarray, shape (N, d)
        Cluster data points.
    p : float
        Order of the norm.

    Returns
    -------
    np.ndarray, shape (d,)
        Gradient vector.
    """
    if p == 1:
        return -np.sum(np.sign(X - c), axis=0)

    diff = X - c
    abs_diff = np.where(np.abs(diff) < 1e-12, 1e-12, np.abs(diff))

    lp_norm_vector = lp_norm_stable(diff, p)
    lp_norm_vector = np.where(lp_norm_vector < 1e-12, 1e-12, lp_norm_vector)

    power_term = np.exp((p - 2) * np.log(abs_diff))
    weighted_diff = diff * power_term
    denominator = lp_norm_vector ** (p - 1)

    return -np.sum(weighted_diff / denominator, axis=0)


def exp_kernel_bw_perplexity(dists):
    """
    Solves for the bandwidth sigma of an exponential kernel via perplexity
    matching, as used in UMAP's asymmetric edge-weight calibration.

    Finds sigma such that:
        sum(exp(-(d_i - rho) / sigma)) == log2(k)
    where rho is the distance to the nearest neighbor and k is the number
    of neighbors. This ensures the effective number of connected neighbors
    equals log2(k), adapting the kernel width to local density.

    Parameters
    ----------
    dists : np.ndarray
        1D array of distances from a single query point to its k nearest
        neighbors. Need not be sorted.

    Returns
    -------
    float
        Calibrated sigma. Returns 1.0 as a safe fallback for degenerate
        inputs (empty array, all-equal distances, or no root found).
    """
    from scipy.optimize import brentq

    k = len(dists)
    if k == 0:
        return 1.0

    rho = np.min(dists)
    target = np.log2(k)
    exp_vals = np.maximum(0.0, dists - rho)

    # All neighbors equidistant: limiting solution is sigma -> inf
    if np.all(exp_vals == 0):
        return 1.0

    def objective(sigma):
        if sigma <= 0:
            return np.inf
        return np.sum(np.exp(-exp_vals / sigma)) - target

    a = max(rho / 2, 1e-10)
    b = 1.0

    if objective(a) >= 0:
        return 1.0

    for _ in range(100):
        if objective(b) >= 0:
            break
        b *= 2.0
        if b > 1e10:
            return 1.0
    else:
        return 1.0

    try:
        return brentq(objective, a, b)
    except ValueError:
        return 1.0