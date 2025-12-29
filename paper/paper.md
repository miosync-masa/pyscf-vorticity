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
date: 29 December 2025
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
4. **Two-variable diagnosis** for robust single-point functional selection

The package integrates seamlessly with PySCF [@sun2018pyscf], enabling 
immediate application to any system accessible via full configuration 
interaction (FCI) or multi-reference methods.

![H₂ dissociation U-shaped curve](fig1_h2_ushaped.png)

*Figure 1: The vorticity stiffness α = |E_corr|/V exhibits a U-shaped 
curve during H₂ bond dissociation, demonstrating the ambiguity of 
single-variable diagnosis. The minimum at R ≈ 1.5 Å marks optimal 
correlation efficiency.*

![Correlation dimension scaling](fig2_gamma_scaling.png)

*Figure 2: (a) System-size scaling of exchange-correlation energy at 
various U/t values. (b) The extracted correlation dimension γ shows 
linear dependence on interaction strength, ranging from γ ≈ 2 (metallic) 
to γ → 0 (Mott insulator).*

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

# Two-Variable Diagnosis

A key challenge in single-point functional selection is that the vorticity 
stiffness $\alpha$ exhibits a **U-shaped curve** during bond dissociation, 
making it ambiguous on its own. We address this by introducing a two-variable 
diagnostic:

| Variable | Physical Meaning | Behavior |
|----------|------------------|----------|
| $\alpha$ (Vorticity Stiffness) | Topological complexity | U-shaped |
| $|E_{corr}|/N$ (Energy Density) | Mean-field breakdown | Monotonic |

The correlation energy density serves as a "tie-breaker" that distinguishes 
between weak correlation (small $|E_{corr}|/N$) and strong static correlation 
(large $|E_{corr}|/N$, indicating RHF failure).

This approach correctly identifies the dissociation limit of H$_2$ as 
strongly correlated ($\gamma \to 0$), recommending high exact exchange 
functionals, while equilibrium H$_2$ is correctly classified as weakly 
correlated ($\gamma \approx 2$).

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

A functional recommender tool is included for practical use:
```python
# examples/04_functional_recommender.py
from pyscf_vorticity.recommender import diagnose_system
diagnose_system("H2", "H 0 0 0; H 0 0 0.74")
```

# Validation

The package has been validated against exact results for atomic and 
molecular systems.

## Atomic Systems

| System | $n_e$ | $V$ | $\alpha$ |
|--------|-------|-----|----------|
| He     | 2     | 0.361 | 0.090 |
| Be     | 4     | 15.64 | 0.003 |
| Ne     | 10    | 55.42 | 0.004 |

The 30-fold decrease in $\alpha$ from 2-electron to 4+ electron systems 
(the "$\alpha$-cliff") reflects the emergence of many-body correlation 
structure.

## Two-Variable Diagnosis Validation

| System | $\alpha$ | $|E_{corr}|/N$ | $\gamma$ | Recommendation |
|--------|----------|----------------|----------|----------------|
| H$_2$ (eq) | 0.062 | 0.47 eV | 1.80 | PBE0/B3LYP |
| H$_2$ (R=3Å) | 0.071 | 2.36 eV | 0.05 | M06-HF |
| LiH | 0.002 | 0.21 eV | 3.21 | PBE/TPSSh |
| He | 0.090 | 0.44 eV | 1.80 | PBE0/B3LYP |

Despite similar $\alpha$ values, H$_2$ at equilibrium and dissociation are 
correctly distinguished by the energy density diagnostic.

## Isoelectronic Universality

Systems with equal electron count exhibit nearly identical vorticity 
regardless of nuclear configuration:

- Ne (atom): $V = 55.42$, $\alpha = 0.0035$
- HF (molecule): $V = 54.97$, $\alpha = 0.0038$

This confirms that $\gamma$ is fundamentally an electronic property.

# Practical Guidelines

Based on computed $\gamma$ values:

| $\gamma$ | Optimal $a$ | Recommended Functional |
|----------|-------------|------------------------|
| $< 0.5$ | $> 0.67$ | HF, M06-HF |
| $0.5-1.2$ | $0.45-0.67$ | M06-2X, BH&HLYP |
| $1.2-2.0$ | $0.33-0.45$ | PBE0, B3LYP |
| $2.0-3.0$ | $0.25-0.33$ | B3LYP, TPSSh |
| $> 3.0$ | $< 0.25$ | TPSSh, pure GGA |

**Note:** This diagnostic is heuristic but physically motivated, intended 
to guide functional choice rather than guarantee optimality.

# Limitations

Users should be aware of the following limitations:

- **Finite-size effects**: $\gamma$ is extracted from finite FCI calculations
- **Heuristic nature**: The two-variable diagnostic provides guidance, not guarantees
- **Computational cost**: FCI scales exponentially; use CASSCF for larger systems
- **Strong correlation**: Systems with $\gamma \to 0$ may require beyond-DFT methods

# Availability

`pyscf-vorticity` is available via:

- PyPI: `pip install pyscf-vorticity`
- GitHub: https://github.com/miosync-masa/pyscf-vorticity
- DOI: 10.5281/zenodo.18085362

The package is released under the MIT license.

# Acknowledgements

The author thanks AI research assistants (Claude/Anthropic, Gemini/Google) 
for collaborative development. All scientific content and interpretations 
are solely the author's responsibility.

# References
