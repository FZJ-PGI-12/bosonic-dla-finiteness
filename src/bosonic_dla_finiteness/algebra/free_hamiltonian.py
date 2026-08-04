"""
FreeHamiltonian: element of the free subspace Â^0_n.

Each free Hamiltonian has the form
    X = Σ_k x_k (i a†_k a_k)   with   x_k ∈ ℝ

stored as an immutable coefficient tuple x ∈ ℝ^n.  The relation to the g_+ basis is
    X = Σ_k (x_k / 2) g_+^{τ_k}.

The algorithm input F = {X^(ℓ) : ℓ ∈ L} is a finite set of free Hamiltonians.
compute_F_prime returns F' = a basis for span{F} in ℝ^n, found by Gaussian
elimination with pivoting. Note that entries below ZERO_TOL in absolute value
are treated as exact zeros, so near-dependent inputs are resolved by that
threshold rather than by a rank-revealing decomposition.
"""

from __future__ import annotations

import numpy as np

from bosonic_dla_finiteness.constants import ZERO_TOL as _ZERO_TOL
from bosonic_dla_finiteness.operators.monomial import GammaIndex


class FreeHamiltonian:
    """X = Σ_k x_k (i a†_k a_k) ∈ Â^0_n with coefficient vector x ∈ ℝ^n."""

    def __init__(self, coeffs: tuple[float, ...] | list[float]) -> None:
        self._x: tuple[float, ...] = tuple(float(c) for c in coeffs)

    @property
    def coeffs(self) -> tuple[float, ...]:
        """Coefficient vector x ∈ ℝ^n: x_k is the coefficient of i a†_k a_k."""
        return self._x

    @property
    def n(self) -> int:
        return len(self._x)

    def is_zero(self, tol: float = _ZERO_TOL) -> bool:
        """True if every coefficient vanishes, i.e. X = 0. Such elements carry
        no information and are dropped by compute_F_prime."""
        return all(abs(x) <= tol for x in self._x)

    def __repr__(self) -> str:
        return f"FreeHamiltonian({list(self._x)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FreeHamiltonian):
            return NotImplemented
        return len(self._x) == len(other._x) and all(
            abs(a - b) <= _ZERO_TOL for a, b in zip(self._x, other._x)
        )

    def __hash__(self) -> int:
        return hash(self._x)

    @classmethod
    def from_omegas(cls, omegas: list[float]) -> FreeHamiltonian:
        """Construct from drift frequencies: iH_d = Σ_k ω_k (i a†_k a_k), so x_k = ω_k."""
        return cls(omegas)


def compute_F_prime(F: list[FreeHamiltonian]) -> list[FreeHamiltonian]:
    """
    Compute F' = a basis for span{F} ⊆ ℝ^n, consisting only of free Hamiltonians.

    Selects a maximal linearly independent subset of F (in input order).
    The returned elements are unchanged vectors from the original F.
    Returns [] if F is empty or every element is zero.
    """
    non_zero_hamiltonians = [fh for fh in F if not fh.is_zero()]
    if not non_zero_hamiltonians:
        return []

    result: list[FreeHamiltonian] = []
    pivots: list[tuple[int, np.ndarray]] = []  # (pivot_col, normalized row)

    for fh in non_zero_hamiltonians:
        v = np.asarray(fh.coeffs, dtype=float)
        for col, p in pivots:
            v = v - v[col] * p
        nz = np.where(np.abs(v) > _ZERO_TOL)[0]
        if len(nz):
            pivots.append((int(nz[0]), v / v[nz[0]]))
            result.append(fh)

    return result


def compute_chi_F_gamma(
    F: list[FreeHamiltonian], gamma: GammaIndex
) -> tuple[float, ...]:
    """
    Compute χ_F(γ) = (χ_X(1)(γ), ..., χ_X(m)(γ)) ∈ ℝ^m.

    The zero vector indicates that a^γ commutes with all free Hamiltonians in F.
    """
    return tuple(compute_chi_freehamiltonian_gamma(fh, gamma) for fh in F)


def compute_chi_freehamiltonian_gamma(
    fh: FreeHamiltonian, gamma: GammaIndex
) -> float:
    """
    Compute χ_X(γ) = Σ_k x_k (α_k − β_k) for a single free Hamiltonian X.

    This is one scalar component of the vector-valued map χ_F: N^{2n} → R^m
    that encodes whether a generator a^γ commutes with all free Hamiltonians in F.
    """
    alpha, beta = gamma
    n = fh.n
    if len(alpha) != n or len(beta) != n:
        raise ValueError(
            f"gamma has α of length {len(alpha)} and β of length {len(beta)}, "
            f"but the free Hamiltonian has n={n} modes."
        )
    xi = sum(fh.coeffs[k] * (alpha[k] - beta[k]) for k in range(n))
    return xi
