"""
Commutator of two BosonicGenerator objects in Â_n.

Computation steps:
  1. Expand each generator into complex normal-ordered monomials.
  2. Compute [A, B] = AB − BA in the monomial basis using the Weyl algebra product
     (bosonic CCR: a_i a†_j = a†_j a_i + δ_ij).
  3. Project the result back to the real {g+, g-} basis.
"""

from __future__ import annotations

from itertools import product as iterproduct
from math import comb, factorial

from bosonic_dla_finiteness.constants import ZERO_TOL as _ZERO_TOL
from bosonic_dla_finiteness.operators.monomial import GammaIndex
from bosonic_dla_finiteness.operators.operator import (
    BasisKey,
    BosonicGenerator,
    basis_key_from_complex_monomials,
)


def normal_order_product(
    gamma1: GammaIndex,
    gamma2: GammaIndex,
) -> dict[GammaIndex, complex]:
    """
    Compute a^γ1 · a^γ2 in normal order using the bosonic CCR [a_i, a†_j] = δ_ij.

    The annihilators β1_i from γ1 pass through the creators α2_i from γ2. For each mode:

        a_i^m (a†_i)^k = Σ_{j=0}^{min(m,k)} C(m,j)·C(k,j)·j! · (a†_i)^{k-j} · a_i^{m-j}

    Modes are independent, so the result is summed over all contraction patterns
    j = (j_0, ..., j_{n-1}) with 0 ≤ j_i ≤ min(β1_i, α2_i).

    Returns a sparse dict GammaIndex → complex coefficient.
    """
    alpha1, beta1 = gamma1
    alpha2, beta2 = gamma2
    n = len(alpha1)

    result: dict[GammaIndex, complex] = {}

    for j in iterproduct(
        *(range(min(b, a) + 1) for b, a in zip(beta1, alpha2))
    ):
        coeff: complex = 1.0 + 0j
        for i in range(n):
            coeff *= (
                comb(int(beta1[i]), j[i])
                * comb(int(alpha2[i]), j[i])
                * factorial(j[i])
            )

        if abs(coeff) < _ZERO_TOL:
            continue

        alpha_r = tuple(
            int(alpha1[i]) + int(alpha2[i]) - j[i] for i in range(n)
        )
        beta_r = tuple(int(beta1[i]) - j[i] + int(beta2[i]) for i in range(n))
        gamma_r: GammaIndex = (alpha_r, beta_r)

        result[gamma_r] = result.get(gamma_r, 0j) + coeff

    return {k: v for k, v in result.items() if abs(v) > _ZERO_TOL}


def commutator_coeffs(
    g1: BosonicGenerator,
    g2: BosonicGenerator,
) -> dict[BasisKey, float]:
    """
    Compute [g1, g2] = g1·g2 − g2·g1 as real coefficients in the {g+, g-} basis.
    """
    assert g1.n == g2.n, "generators must have the same number of modes"

    m1 = g1.to_complex_monomials()
    m2 = g2.to_complex_monomials()

    raw: dict[GammaIndex, complex] = {}

    for gam1, c1 in m1.items():
        for gam2, c2 in m2.items():
            for gam_r, coeff in normal_order_product(gam1, gam2).items():
                raw[gam_r] = raw.get(gam_r, 0j) + c1 * c2 * coeff
            for gam_r, coeff in normal_order_product(gam2, gam1).items():
                raw[gam_r] = raw.get(gam_r, 0j) - c2 * c1 * coeff

    raw = {k: v for k, v in raw.items() if abs(v) > _ZERO_TOL}
    return basis_key_from_complex_monomials(raw, g1.n)


def commutator_support(
    g1: BosonicGenerator,
    g2: BosonicGenerator,
) -> frozenset[BosonicGenerator]:
    """Return the set of basis elements that appear with nonzero coefficient in [g1, g2]."""
    return frozenset(
        BosonicGenerator(kind, gamma)
        for kind, gamma in commutator_coeffs(g1, g2)
    )
