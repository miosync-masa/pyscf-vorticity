"""
Example 4: Functional Recommender (Two-Variable Diagnosis)
==========================================================
Diagnose a system using both Vorticity (α) and Correlation Energy Density
to distinguish between weak and strong (static) correlation regimes.

Key insight: α alone is U-shaped and ambiguous.
|E_corr|/N breaks the degeneracy!
"""

import numpy as np
from pyscf import gto, scf, fci
from pyscf_vorticity import compute_vorticity


def estimate_gamma_robust(alpha, n_elec, E_corr_eV):
    """
    Robust γ estimation using Two-Variable Diagnosis.

    Variables:
    1. alpha (α): Vorticity stiffness. (U-shaped, ambiguous on its own)
    2. E_dens (|E_corr|/N): Correlation energy density. (Monotonic with static correlation)

    This corresponds to the Two-Axis Phase Diagram (Fig.5):
    - Mott (d=0): γ_min, high E_dens
    - SC (d=1): α_min
    - Metal (d=2): γ_max, low E_dens
    """
    # 1. Base Gamma from System Size (The "Alpha Cliff" correction)
    #    N=2 systems naturally have high alpha (~0.06-0.1).
    #    N>=4 systems drop to ~0.002.
    if n_elec <= 2:
        base_gamma = 1.8  # Start assumes weak correlation for small systems
    else:
        # Scaling law correction: alpha scales as N^(-gamma)
        if alpha < 1e-4:
            alpha = 1e-4
        log_a = np.log10(alpha)
        # Empirical linear mapping based on LiH/Ne data
        # log(0.002) ~ -2.7 -> gamma ~ 3.0
        # log(0.1)   ~ -1.0 -> gamma ~ 1.5
        base_gamma = 1.0 - 0.8 * log_a

    # 2. Static Correlation Detection (The Tie-Breaker)
    #    Measure: Correlation Energy per Electron (eV)
    #    Weak corr (dynamic): ~0.2 - 0.5 eV/e
    #    Strong corr (static): > 1.0 eV/e (RHF breakdown)
    E_dens = abs(E_corr_eV) / n_elec

    static_corr_factor = 0.0

    if E_dens > 2.0:
        # Extreme static correlation (Dissociation limit)
        # Force gamma towards 0 (Mott/Localization)
        static_corr_factor = 2.5
    elif E_dens > 1.0:
        # Strong static correlation
        static_corr_factor = 1.5 * (E_dens - 1.0) + 0.5
    elif E_dens > 0.5:
        # Onset of static correlation
        static_corr_factor = 0.5 * (E_dens - 0.5)

    # Apply penalty
    gamma = base_gamma - static_corr_factor

    # 3. Physical Bounds
    gamma = max(0.05, min(4.0, gamma))

    return gamma, E_dens


def recommend_functional(gamma, E_dens):
    """Recommendation logic with regime identification"""
    optimal_a = 1.0 / (1.0 + gamma)

    print(f"\n{'='*55}")
    print(f"FUNCTIONAL RECOMMENDATION")
    print(f"{'='*55}")

    # Identify Regime
    if gamma < 0.8:
        regime = "STRONG STATIC CORRELATION (Mott/Dissociation)"
    elif gamma < 1.8:
        regime = "INTERMEDIATE / STRONG DYNAMIC"
    else:
        regime = "WEAK CORRELATION (Standard Chemistry)"

    print(f"  Diagnostics:")
    print(f"    γ (Correlation Dim) : {gamma:.2f}")
    print(f"    E_corr Density      : {E_dens:.2f} eV/e")
    print(f"  Regime: {regime}")
    print(f"  Optimal Exact Exchange: {optimal_a*100:.1f}%")
    print(f"{'-'*55}")

    if gamma < 0.5:
        rec = "M06-HF (100%) or Range-Separated (wB97X-V)"
        xc = "'M06HF' or 'WB97XV'"
        note = "RHF has failed. Use 100% exact exchange."
    elif 0.5 <= gamma < 1.2:
        rec = "M06-2X (54%) or BH&HLYP (50%)"
        xc = "'M062X' or 'BHHLYP'"
        note = "Significant localization. Standard hybrids underperform."
    elif 1.2 <= gamma < 2.0:
        rec = "PBE0 (25%) or B3LYP (20%)"
        xc = "'PBE0' or 'B3LYP'"
        note = "Standard DFT regime. Golden rule applies."
    elif 2.0 <= gamma < 3.0:
        rec = "B3LYP (20%) or TPSSh (10%)"
        xc = "'B3LYP' or 'TPSSH'"
        note = "Delocalized electrons. Screening is effective."
    else:
        rec = "Pure GGA (PBE) or TPSSh"
        xc = "'PBE' or 'TPSS'"
        note = "Metallic limit. Exact exchange may harm results."

    print(f"  Recommended: {rec}")
    print(f"  PySCF Code : mf.xc = {xc}")
    print(f"  Note       : {note}")
    print(f"{'='*55}")

    return optimal_a, xc


def diagnose_system(name, atom_string, basis='cc-pvdz'):
    """Run full diagnostic on a molecular system"""
    print(f"\n\n{'#'*60}")
    print(f"# {name}")
    print(f"# Geometry: {atom_string}")
    print(f"# Basis: {basis}")
    print(f"{'#'*60}")

    mol = gto.M(atom=atom_string, basis=basis, unit='angstrom')
    mf = scf.RHF(mol).run(verbose=0)

    n_orb = mol.nao
    n_elec = mol.nelectron

    print(f"\n[System Info]")
    print(f"  Electrons: {n_elec}")
    print(f"  Orbitals:  {n_orb}")

    # FCI
    cisolver = fci.FCI(mf)
    E_fci, fcivec = cisolver.kernel()
    E_corr = E_fci - mf.e_tot
    E_corr_eV = E_corr * 27.211

    print(f"\n[Energies]")
    print(f"  E(HF)   = {mf.e_tot:.6f} Ha")
    print(f"  E(FCI)  = {E_fci:.6f} Ha")
    print(f"  E_corr  = {E_corr:.6f} Ha = {E_corr_eV:.3f} eV")

    # Vorticity
    nelec_tuple = (n_elec // 2, n_elec // 2)
    rdm1, rdm2 = cisolver.make_rdm12(fcivec, n_orb, nelec_tuple)
    V, k = compute_vorticity(rdm2, n_orb)
    alpha = abs(E_corr) / V if V > 0 else 0

    print(f"\n[Vorticity Analysis]")
    print(f"  V = {V:.4f}")
    print(f"  k = {k} (SVD rank)")
    print(f"  α = {alpha:.4f}")

    # Two-Variable Diagnosis
    gamma_est, E_dens = estimate_gamma_robust(alpha, n_elec, E_corr_eV)

    print(f"\n[Two-Variable Diagnosis]")
    print(f"  α (Vorticity Stiffness) = {alpha:.4f}")
    print(f"  |E_corr|/e (Energy Density) = {E_dens:.2f} eV/electron")
    print(f"  → γ = {gamma_est:.2f}")

    # Recommend
    recommend_functional(gamma_est, E_dens)

    return {
        'name': name,
        'V': V,
        'alpha': alpha,
        'gamma': gamma_est,
        'E_dens': E_dens,
        'E_corr': E_corr,
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("pyscf-vorticity: Functional Recommender")
    print("Two-Variable Diagnosis (α + E_dens)")
    print("="*60)

    results = []

    # Case 1: H2 Equilibrium (Weak correlation)
    # Expected: PBE0/B3LYP
    results.append(diagnose_system(
        "H2 (Equilibrium)",
        'H 0 0 0; H 0 0 0.74'
    ))

    # Case 2: H2 Dissociation (Strong Static correlation)
    # Expected: M06-HF / High Exchange
    results.append(diagnose_system(
        "H2 (Stretched R=3.0A)",
        'H 0 0 0; H 0 0 3.0'
    ))

    # Case 3: LiH (Ionic/Moderate)
    # Expected: B3LYP/PBE0
    results.append(diagnose_system(
        "LiH",
        'Li 0 0 0; H 0 0 1.6'
    ))

    # Case 4: He atom (Reference)
    # Expected: PBE0/B3LYP
    results.append(diagnose_system(
        "He Atom",
        'He 0 0 0'
    ))

    # Summary Table
    print("\n\n" + "="*70)
    print("SUMMARY: Two-Variable Diagnosis Results")
    print("="*70)
    print(f"{'System':<25} {'α':<10} {'E_dens':<10} {'γ':<8} {'a_opt':<8}")
    print("-"*70)
    for r in results:
        a_opt = 1.0 / (1.0 + r['gamma'])
        print(f"{r['name']:<25} {r['alpha']:<10.4f} {r['E_dens']:<10.2f} "
              f"{r['gamma']:<8.2f} {a_opt*100:<8.1f}%")
    print("="*70)
