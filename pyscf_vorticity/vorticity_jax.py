"""
JAX-accelerated vorticity calculation
"""

import jax.numpy as jnp
from jax import jit
from functools import partial

@partial(jit, static_argnums=(1, 2))
def compute_vorticity(rdm2, n_orb, svd_cut=0.95):
    """
    Compute vorticity from 2-RDM (JAX version)
    
    GPU-accelerated when available
    """
    # Reshape to matrix
    M = rdm2.reshape(n_orb**2, n_orb**2)
    
    # SVD (GPU accelerated)
    U, S, Vt = jnp.linalg.svd(M, full_matrices=False)
    
    # Dynamic k selection
    total_var = jnp.sum(S**2)
    cumvar = jnp.cumsum(S**2) / (total_var + 1e-14)
    k = jnp.searchsorted(cumvar, svd_cut) + 1
    k = jnp.maximum(k, 2)
    k = jnp.minimum(k, len(S))
    
    # Λ-space projection
    S_proj = U[:, :k]
    M_lambda = S_proj.T @ M @ S_proj
    
    # Gradient
    grad_M = jnp.zeros_like(M_lambda)
    grad_M = grad_M.at[:-1, :].set(M_lambda[1:, :] - M_lambda[:-1, :])
    
    # Current: J = M_λ @ ∇M_λ
    J_lambda = M_lambda @ grad_M
    
    # Vorticity: ||J - J^T||²
    curl_J = J_lambda - J_lambda.T
    V = jnp.sum(curl_J**2)
    
    return jnp.sqrt(V), k


# Batched version for multiple systems
@jit
def compute_vorticity_batch(rdm2_batch, n_orb):
    """Vectorized vorticity for multiple RDMs"""
    return jax.vmap(lambda x: compute_vorticity(x, n_orb))(rdm2_batch)
