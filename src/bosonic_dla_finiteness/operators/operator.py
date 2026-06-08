"""
BosonicGenerator: a single basis element g_σ^γ of the skew-Hermitian Weyl algebra Â_n.

No coefficient is stored — this object is a label for the basis element.
Coefficients belong to the drift Hamiltonian (the omegas vector).

Basis elements:
    g_+^(α,β) = i(a^(β,α) + a^(α,β))   for (α,β) ≥ (β,α) lexicographically
    g_-^(α,β) = a^(β,α) − a^(α,β)       for (α,β) > (β,α) strictly
"""

from __future__ import annotations

from bosonic_dla_finiteness.constants import ZERO_TOL
from bosonic_dla_finiteness.io.models import GeneratorKind
from bosonic_dla_finiteness.operators.monomial import (
    GammaIndex,
    MultiIndex,
    gamma_degree,
)

BasisKey = tuple[GeneratorKind, GammaIndex]


def _is_canonical(alpha: MultiIndex, beta: MultiIndex) -> bool:
    """
    True if (α,β) ≥ (β,α) as 2n-tuples under lexicographic ordering.

    Selects one canonical representative from each adjoint pair {(α,β), (β,α)}.
    Diagonal elements (α=β) are their own canonical representatives.

    Uses lexicographic order on the concatenated 2n-tuple (α‖β), not component-wise.
    """
    return alpha + beta >= beta + alpha


class BosonicGenerator:
    """
    Single basis element g_σ^γ of Â_n (skew-Hermitian Weyl algebra). No coefficient stored.

    Stores (kind, gamma) where:
        kind="+" → g_+^(α,β) = i(a^(β,α) + a^(α,β)),  (α,β) ≥ (β,α) lexicographically
        kind="-" → g_-^(α,β) = a^(β,α) − a^(α,β),      (α,β) > (β,α) strictly
    """

    def __init__(self, kind: GeneratorKind | str, gamma: GammaIndex) -> None:
        kind = GeneratorKind(kind)
        alpha, beta = gamma
        assert len(alpha) == len(beta), (
            "alpha and beta must have the same length"
        )
        assert all(a >= 0 for a in alpha) and all(b >= 0 for b in beta)

        if kind == GeneratorKind.plus:
            assert _is_canonical(alpha, beta), (
                f"g_+ requires (α,β) ≥ (β,α): got α={alpha}, β={beta}"
            )
        else:
            assert (
                _is_canonical(alpha, beta) and alpha + beta != beta + alpha
            ), f"g_- requires (α,β) > (β,α) strictly: got α={alpha}, β={beta}"

        self._kind = kind
        self._gamma = gamma

    @property
    def kind(self) -> GeneratorKind:
        return self._kind

    @property
    def gamma(self) -> GammaIndex:
        return self._gamma

    @property
    def alpha(self) -> MultiIndex:
        return self._gamma[0]

    @property
    def beta(self) -> MultiIndex:
        return self._gamma[1]

    @property
    def n(self) -> int:
        return len(self._gamma[0])

    @property
    def degree(self) -> int:
        return gamma_degree(self._gamma)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BosonicGenerator):
            return NotImplemented
        return self._kind == other.kind and self._gamma == other.gamma

    def __hash__(self) -> int:
        return hash((self._kind, self._gamma))

    def __repr__(self) -> str:
        return f"g_{self._kind.value}^{self._gamma}"

    @classmethod
    def g_plus(cls, gamma: GammaIndex) -> BosonicGenerator:
        """Construct g_+^γ. Requires (α,β) ≥ (β,α) lexicographically."""
        return cls(GeneratorKind.plus, gamma)

    @classmethod
    def g_minus(cls, gamma: GammaIndex) -> BosonicGenerator:
        """Construct g_-^γ. Requires (α,β) > (β,α) strictly (α≠β)."""
        return cls(GeneratorKind.minus, gamma)

    def to_complex_monomials(self) -> dict[GammaIndex, complex]:
        """
        Expand in the complex normal-ordered monomial basis a^γ.

            g_+^(α,β) → {(β,α): 1j, (α,β): 1j}
            g_-^(α,β) → {(β,α): 1+0j, (α,β): −1+0j}

        When α=β the two keys coincide: g_+^(α,α) → {(α,α): 2j}.
        """
        alpha, beta = self._gamma
        gamma_dag: GammaIndex = (beta, alpha)

        if self._kind == GeneratorKind.plus:
            if self._gamma == gamma_dag:
                return {self._gamma: 2j}
            return {gamma_dag: 1j, self._gamma: 1j}

        return {gamma_dag: 1.0 + 0j, self._gamma: -1.0 + 0j}


def basis_key_from_complex_monomials(
    terms: dict[GammaIndex, complex], n: int
) -> dict[BasisKey, float]:
    """
    Project a skew-Hermitian complex monomial expansion back to real {g+, g-} coefficients.

    For each adjoint pair (γ, γ† = (β,α)):

    Diagonal (α=β):
        g_+^(α,α) = 2i a^(α,α), so  c·a^(α,α) = (c/2i) g_+^(α,α)
        → coeff of g_+^(α,α) = Im(c) / 2

    Off-diagonal — let c_can = coeff of the canonical representative γ_c:
        g_+^γ_c = i(a^(γ_c†) + a^γ_c), so the coeff of a^γ_c in g_+^γ_c is i
        g_-^γ_c = a^(γ_c†) − a^γ_c,     so the coeff of a^γ_c in g_-^γ_c is −1
        → coeff of g_+^γ_c = Im(c_can)
        → coeff of g_-^γ_c = −Re(c_can)
    """
    result: dict[BasisKey, float] = {}
    visited: set[GammaIndex] = set()

    for gamma, c in terms.items():
        if gamma in visited:
            continue
        alpha, beta = gamma
        gamma_dag: GammaIndex = (beta, alpha)
        visited.add(gamma)
        visited.add(gamma_dag)

        if gamma == gamma_dag:
            coeff_plus = c.imag / 2.0
            if abs(coeff_plus) > ZERO_TOL:
                key: BasisKey = (GeneratorKind.plus, gamma)
                result[key] = result.get(key, 0.0) + coeff_plus
        else:
            if _is_canonical(alpha, beta):
                gamma_c, c_can = gamma, c
            else:
                gamma_c = gamma_dag
                c_can = terms.get(gamma_dag, 0j)

            coeff_plus = c_can.imag
            coeff_minus = -c_can.real

            if abs(coeff_plus) > ZERO_TOL:
                key = (GeneratorKind.plus, gamma_c)
                result[key] = result.get(key, 0.0) + coeff_plus
            if abs(coeff_minus) > ZERO_TOL:
                key = (GeneratorKind.minus, gamma_c)
                result[key] = result.get(key, 0.0) + coeff_minus

    return result
