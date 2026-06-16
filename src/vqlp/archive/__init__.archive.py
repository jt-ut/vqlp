"""
Vector Quantizer with arbitrary Lp distance metrics.

A Python package for vector quantization supporting arbitrary Lp norms,
multiple fitting algorithms (IRLS, PAM, K-means), and comprehensive
recall analysis with connectivity matrices and reconstruction capabilities.
"""

from .vq import VQFitter, VQRecaller
from .utils import softmax_sigma_umap

# Package metadata
__version__ = "0.1.0"
__author__ = "Josh Taylor"
__email__ = "joshtaylor@utexas.edu"

# Define what gets imported with "from vqlp import *"
__all__ = [
    "VQFitter",
    "VQRecaller", 
    "softmax_sigma_umap",
]

# Optional: Add convenience aliases or additional exports
# VQ = VQFitter  # Shorter alias if desired