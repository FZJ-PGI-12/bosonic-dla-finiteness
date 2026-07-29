"""
BosonicGenerator: a single basis element g_σ^γ of the skew-Hermitian Weyl algebra Â_n.

No coefficient is stored — this object is a label for the basis element.
Rescaling a generator does not change the Lie algebra it generates, so the
classification depends only on which basis elements are present, not on their
prefactors. (The omegas vector holds free-Hamiltonian coefficients, which are
a separate quantity: there the values do matter, via chi_F.)

Basis elements:
    g_+^(α,β) = i(a^(β,α) + a^(α,β))   for (α,β) ≥ (β,α) lexicographically
    g_-^(α,β) = a^(β,α) − a^(α,β)       for (α,β) > (β,α) strictly
"""

from __future__ import annotations

from bosonic_dla_finiteness.io.models import GeneratorKind
from bosonic_dla_finiteness.operators.monomial import (
    GammaIndex,
    MultiIndex,
    gamma_degree,
)


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
