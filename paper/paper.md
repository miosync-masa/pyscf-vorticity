---
title: 'pyscf-vorticity: Geometric correlation analysis for density functional theory'
tags:
  - Python
  - quantum chemistry
  - density functional theory
  - exchange-correlation
  - vorticity
authors:
  - name: Masamichi Iizumi
    orcid: 0009-0007-0755-403X
    affiliation: 1
affiliations:
  - name: Miosync, Inc., Tokyo, Japan
    index: 1
date: 28 December 2025
bibliography: paper.bib
---

# Summary

Density functional theory (DFT) is the most widely used method in computational 
chemistry and materials science, yet the exchange-correlation functional 
$E_{xc}$ remains fundamentally approximate. Standard functionals contain 
empirical "magic numbers"—such as the 20% exact exchange in B3LYP—whose 
optimal values vary across chemical systems but lack first-principles 
determination.

`pyscf-vorticity` provides a geometric approach to this problem by computing 
an effective correlation dimension $\gamma$ from the two-particle reduced 
density matrix (2-RDM). This dimension characterizes the correlation structure 
of a quantum system and directly relates to the optimal exact exchange 
mixing ratio via $a \approx 1/(1+\gamma)$.

# Statement of Need

Researchers using DFT face a fundamental dilemma: which functional to choose? 
The answer often relies on empirical benchmarks or trial-and-error, leading to:

- Inconsistent functional choices across studies
- Poor performance for strongly correlated systems
- Lack of systematic guidance for new chemical systems

`pyscf-vorticity` addresses this need by providing:

1. **First-principles estimation** of optimal exact exchange mixing
2. **System-specific guidance** rather than universal parameters
3. **Physical insight** into correlation structure through $\gamma$

The package integrates seamlessly with PySCF [@sun2018pyscf], enabling 
immediate application to any system accessible via full configuration 
interaction (FCI) or multi-reference methods.

# Theoretical Background

The exchange-correlation energy scales with a vorticity measure $V$ computed 
from the 2-RDM:

$$V = \sqrt{\sum_{ij}(J_{ij} - J_{ji})^2}$$

where $J = M_\Lambda \cdot \nabla M_\Lambda$ is the correlation current in 
the projected $\Lambda$-space obtained via singular value decomposition of 
the reshaped 2-RDM.

The coupling constant $\alpha = |E_{xc}|/V$ exhibits system-size scaling:

$$\alpha \propto N^{-\gamma}$$

where $\gamma$ is the effective correlation dimension. This scaling allows 
extraction of $\gamma$ from calculations on small model systems, providing 
guidance for larger-scale DFT calculations.

# Implementation

`pyscf-vorticity` provides two computational backends:

- **NumPy**: CPU-based fallback for maximum compatibility
- **JAX**: GPU-accelerated computation for larger systems

The main function `compute_vorticity(rdm2, n_orb)` accepts a 2-RDM from any 
PySCF post-Hartree-Fock calculation:
```python
from pyscf import gto, scf, fci
from pyscf_vorticity import compute_vorticity

mol = gto.M(atom='He 0 0 0', basis='cc-pvdz')
mf = scf.RHF(mol).run()
cisolver = fci.FCI(mf)
E_fci, fcivec = cisolver.kernel()

rdm1, rdm2 = cisolver.make_rdm12(fcivec, mol.nao, (1, 1))
V, k = compute_vorticity(rdm2, mol.nao)
```

# Validation

The package has been validated against exact results and the companion 
theoretical paper [@iizumi2025geometric]:

## Atomic Systems

| System | $n_e$ | $V$ | $\alpha$ |
|--------|-------|-----|----------|
| He     | 2     | 0.361 | 0.090 |
| Be     | 4     | 15.64 | 0.003 |
| Ne     | 10    | 55.42 | 0.004 |

The 30-fold decrease in $\alpha$ from 2-electron to 4+ electron systems 
(the "$\alpha$-cliff") reflects the emergence of many-body correlation 
structure.

## H₂ Dissociation

| R (Å) | $\alpha$ | Correlation regime |
|-------|----------|-------------------|
| 0.74  | 0.062    | Weak (equilibrium) |
| 1.50  | 0.039    | Optimal efficiency |
| 5.00  | 0.097    | Strong (Mott limit) |

The U-shaped $\alpha(R)$ curve correctly tracks the transition from 
dynamic to static correlation during bond breaking.

## Isoelectronic Universality

Systems with equal electron count exhibit nearly identical vorticity 
regardless of nuclear configuration:

- Ne (atom): $V = 55.44$, $\alpha = 0.0035$
- HF (molecule): $V = 54.99$, $\alpha = 0.0038$

This confirms that $\gamma$ is fundamentally an electronic property.

# Practical Guidelines

Based on computed $\gamma$ values:

| $\gamma$ | Optimal $a$ | Recommended Functional |
|----------|-------------|------------------------|
| $\approx 0$ | $\approx 1.0$ | HF, high exact exchange |
| $\approx 1$ | $\approx 0.5$ | M06-2X, BH&HLYP |
| $\approx 2$ | $\approx 0.33$ | PBE0, B3LYP |
| $> 3$ | $< 0.25$ | TPSSh, pure GGA |

# Limitations

Users should be aware of the following limitations:

## Finite-Size Effects

The correlation dimension $\gamma$ is extracted from finite systems via 
FCI calculations. Results are most reliable when:

- Multiple system sizes are used for scaling analysis
- Shell effects are considered (open-shell extrapolation recommended)
- Basis set convergence is verified

## Effective Nature of $\gamma$

The computed $\gamma$ is an **effective** correlation dimension, not a 
rigorous topological invariant. It provides practical guidance but should 
not be over-interpreted as an exact physical quantity.

## Computational Cost

FCI calculations scale exponentially with system size. For systems beyond 
~14 electrons, consider:

- Active space methods (CASSCF)
- Smaller basis sets for $\gamma$ estimation
- Extrapolation from smaller model systems

## Strong Correlation Regime

While `pyscf-vorticity` correctly identifies strongly correlated systems 
($\gamma \to 0$), such systems may require methods beyond DFT (DMFT, 
quantum Monte Carlo) for quantitative accuracy.

# Availability

`pyscf-vorticity` is available via:

- PyPI: `pip install pyscf-vorticity`
- GitHub: https://github.com/miosync-masa/pyscf-vorticity

The package is released under the MIT license.

# Acknowledgements

AI language assistance (Claude, Anthropic) was used for code development 
and manuscript preparation. All scientific content and interpretations 
are solely the author's responsibility.

# References
