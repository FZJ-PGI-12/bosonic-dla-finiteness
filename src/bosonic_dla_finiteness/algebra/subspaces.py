"""
Decomposition of Â_n into orthogonal subspaces:

    Â_n = G^0 ⊕ G^1 ⊕ G^2 ⊕ G^= ⊕ G^om ⊕ G^⊥

G^0   — constant 2i and number operators (degree=0 or degree-2 diagonal)
G^1   — degree-1 elements
G^2   — degree-2, off-diagonal
G^=   — diagonal (α=β), degree≥4
G^om  — degree≥3, unique mode k with α_k+β_k=1, all other modes diagonal
          ("om" stands for optomechanics)
G^⊥   — orthogonal complement of G^core = G^0⊕G^1⊕G^2⊕G^=⊕G^om
G^⊥   — orthogonal complement of G^core = G^0⊕G^1⊕G^2⊕G^=⊕G^om

Because the basis {g_σ^γ} is orthogonal under the scalar product ⟨·|·⟩, classification
is per basis element and decomposition is a single-pass partition of the generator list.
"""

from __future__ import annotations

from enum import Enum

from bosonic_dla_finiteness.operators.operator import BosonicGenerator


class Subspace(Enum):
    G0 = "G0"  # free: constant 2i + number operators
    G1 = "G1"  # degree 1
    G2 = "G2"  # degree 2, off-diagonal
    G2_F = "G2_F"  # degree 2, off-diagonal, commutes with given set of free Hamiltonians F
    G2_core = "G2_core"  # degree 2, off-diagonal, non-commuting with given set of free Hamiltonians F
    Geq = "G="  # diagonal, degree >= 4
    Geq_F = "G=F"  # diagonal, degree >= 4, commutes with given set of free Hamiltonians F
    Gom = "Gom"  # optomechanical
    Gperp = "G_perp"  # orthogonal complement of G^core
    Gperp_F = "G_perp_F"  # orthogonal complement of G^core commuting with the set of free Hamiltonians F


def determine_subspace(gen: BosonicGenerator) -> Subspace:
    """Classify a single BosonicGenerator into one of the six subspaces."""
    alpha, beta = gen.alpha, gen.beta
    n = gen.n
    deg = gen.degree

    # G^0: constant (degree 0) or number operator (degree-2 diagonal, α=β)
    if deg == 0:
        return Subspace.G0
    if deg == 2 and alpha == beta:
        return Subspace.G0

    # G^1: degree 1
    if deg == 1:
        return Subspace.G1

    # G^2: degree 2, off-diagonal (degree-2 diagonal caught above)
    if deg == 2:
        return Subspace.G2

    # G^=: diagonal (α=β), degree≥4
    # (degree-3 diagonal is impossible: 2|α|=3 has no integer solution)
    if alpha == beta:
        return Subspace.Geq

    # G^om: degree≥3, exactly one mode k with α_k+β_k=1, all other modes diagonal
    offdiag = [k for k in range(n) if alpha[k] + beta[k] == 1]
    if len(offdiag) == 1 and all(
        alpha[j] == beta[j] for j in range(n) if j != offdiag[0]
    ):
        return Subspace.Gom

    return Subspace.Gperp


def decompose_generators(
    generators: list[BosonicGenerator],
) -> dict[Subspace, set[BosonicGenerator]]:
    """Partition a list of generators by subspace (single pass)."""
    result: dict[Subspace, set[BosonicGenerator]] = {
        s: set() for s in Subspace
    }
    for gen in generators:
        result[determine_subspace(gen)].add(gen)
    return result
