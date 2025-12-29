"""
pyscf-vorticity: Geometric correlation analysis
"""

# Backend selection
try:
    import jax
    from .vorticity_jax import compute_vorticity
    BACKEND = 'jax'

    # GPU check
    try:
        devices = jax.devices('gpu')
        if devices:
            DEVICE = 'gpu'
        else:
            DEVICE = 'cpu'
    except Exception:
        DEVICE = 'cpu'

except ImportError:
    from .vorticity_numpy import compute_vorticity
    BACKEND = 'numpy'
    DEVICE = 'cpu'

print(f"pyscf-vorticity: backend={BACKEND}, device={DEVICE}")

# Public API
from .gamma import extract_gamma, compute_alpha  # noqa: E402

__version__ = '0.1.2'
__all__ = ['compute_vorticity', 'extract_gamma', 'compute_alpha', 'BACKEND', 'DEVICE']
