import numpy as np

__all__ = [
    "softmax_sigma_umap"
]

def softmax_sigma_umap(dists: np.ndarray) -> float:
    """
    Calculates the sigma (local connectivity scale) for a given set
    of nearest neighbor distances, as used in UMAP's asymmetric edge weight calculation.
    Args:
        dists (np.ndarray): A 1D array of distances from a single query point to its
                            k nearest neighbors. These distances are not necessarily sorted.
    Returns:
        float: The calculated sigma value.
    """
    from scipy.optimize import brentq
    
    if len(dists) == 0:
        return 1.0 # Default or handle error for no distances
    rho = np.min(dists)
    k = len(dists)
    target_sum = np.log2(k)
    # Calculate the values that will be exponentiated
    exp_vals = np.maximum(0, dists - rho)
    # --- Special Case: All distances are identical (or effectively zero after subtracting rho) ---
    # If all exp_vals are 0, then sum(exp(-(0)/sigma)) will be k.
    # The objective function will be a constant: k - target_sum.
    # If this constant is not zero, brentq will never find a root.
    if np.all(exp_vals == 0):
        # In this case, all weights should ideally be 1.0. This corresponds to sigma -> infinity.
        # Returning 1.0 (or a very large number) is a practical default.
        return 1.0
    # --- Standard Case: Find sigma using root-finding ---
    def objective(sigma):
        # Ensure sigma is positive to avoid issues
        if sigma <= 0:
            return np.inf # Penalize non-positive sigma
        
        # Calculate the sum of exponential terms
        current_sum = np.sum(np.exp(-exp_vals / sigma))
        return current_sum - target_sum
    # The function `objective(sigma)` is monotonically increasing with sigma.
    # We need to find a bracket [a, b] where objective(a) < 0 and objective(b) > 0.
    # Initial search bounds for sigma
    # Set 'a' adaptively based on rho, ensuring it's always positive
    a = max(rho / 2, 1e-10) # Using rho/2 as a lower bound, ensuring it's not zero and adaptive
    b = 1.0  # Initial guess for upper bound
    # If objective(a) is already positive, it means target_sum is <= count(dists == rho).
    # Since sum(exp_terms) is always >= count(dists == rho),
    # objective(sigma) will always be >= 0. No root exists for positive sigma.
    if objective(a) >= 0:
        return 1.0 # Fallback: target sum is too low for any positive sigma
    # Now we know objective(a) is negative. We need to find 'b' where objective(b) is positive.
    max_b_expand_attempts = 100
    current_b_attempt = 0
    
    while objective(b) < 0 and current_b_attempt < max_b_expand_attempts:
        b *= 2.0 # Double the upper bound to search for a positive objective value
        current_b_attempt += 1
        if b > 1e10: # Prevent 'b' from becoming excessively large (numerical stability)
            print("Warning: Upper bound for sigma search became excessively large. Returning default.")
            return 1.0 # Fallback if range explodes
    if current_b_attempt == max_b_expand_attempts:
        print("Warning: Max iterations reached while expanding sigma upper bound. Returning default.")
        return 1.0 # Fallback if upper bound not found
    # Now we have a valid bracket [a, b] where objective(a) < 0 and objective(b) >= 0.
    try:
        sigma = brentq(objective, a, b)
    except ValueError:
        # This can still happen due to subtle numerical precision issues or if the function
        # is extremely flat within the bracket, even if a theoretical root exists.
        print("Warning: brentq failed to find root within bracket. Returning default.")
        sigma = 1.0 # Fallback
    
    return sigma

