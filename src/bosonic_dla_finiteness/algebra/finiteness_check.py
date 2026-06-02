from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from bosonic_dla_finiteness.algebra.free_hamiltonian import (
    FreeHamiltonian,
    compute_chi_F_gamma,
    compute_F_prime,
)
from bosonic_dla_finiteness.algebra.subspaces import (
    Subspace,
    decompose_generators,
)
from bosonic_dla_finiteness.constants import ZERO_TOL
from bosonic_dla_finiteness.operators.monomial import s_eq, s_neq
from bosonic_dla_finiteness.operators.operator import BosonicGenerator


class DimensionResult(Enum):
    FINITE = "Finite-dimensional"
    INFINITE = "Infinite"
    REMAINING = "Remaining"


class BgColor(Enum):
    RED = "red"
    ORANGE = "orange"
    GREEN = "green"
    BLUE = "blue"


class DotColor(Enum):
    GREEN = "green"
    BLUE = "blue"


@dataclass
class Cell:
    bg: BgColor = None
    dot: DotColor = None


@dataclass
class FinitenessResult:
    dimension: DimensionResult
    remaining_generators: set[BosonicGenerator] = field(default_factory=set)


def _preprocess(F: list[FreeHamiltonian], generators: list[BosonicGenerator]):
    # Step 1: reduce F to a basis for span{F}, decompose generators by subspace
    F_prime = compute_F_prime(F)
    decomposed = decompose_generators(generators)

    # A generator in G^⊥ with nonzero χ_F(γ) witnesses an infinite-dimensional DLA
    for gen in decomposed[Subspace.Gperp]:
        chiF = compute_chi_F_gamma(F_prime, gen.gamma)
        if any(abs(chi) > ZERO_TOL for chi in chiF):
            return DimensionResult.INFINITE
    decomposed[Subspace.Gperp_F] = decomposed[Subspace.Gperp]
    decomposed[Subspace.Geq_F] = decomposed[Subspace.Geq]

    # Split G2 into G2_core (χ_F ≠ 0) and G2_F (χ_F = 0)
    commuting: set[BosonicGenerator] = set()
    uncommuting: set[BosonicGenerator] = set()
    for gen in decomposed[Subspace.G2]:
        chiF = compute_chi_F_gamma(F_prime, gen.gamma)
        if all(abs(chi) <= ZERO_TOL for chi in chiF):
            commuting.add(gen)
        else:
            uncommuting.add(gen)
    decomposed[Subspace.G2_F] = commuting
    decomposed[Subspace.G2_core] = uncommuting

    return F_prime, decomposed


def _process_GperpF(
    table: dict[Subspace, list[Cell]],
    decomposed: dict[Subspace, set[BosonicGenerator]],
) -> DimensionResult:
    for gen in decomposed[Subspace.Gperp_F]:
        for j in s_neq(gen.gamma):
            if table[Subspace.Gperp_F][j].bg == BgColor.BLUE:
                return DimensionResult.INFINITE
            table[Subspace.Gperp_F][j].dot = DotColor.GREEN
            for row, color in [
                (Subspace.G0, BgColor.RED),
                (Subspace.G1, BgColor.RED),
                (Subspace.G2_core, BgColor.RED),
                (Subspace.Gom, BgColor.RED),
                (Subspace.Geq_F, BgColor.RED),
                (Subspace.G2_F, BgColor.ORANGE),
                (Subspace.Gperp_F, BgColor.GREEN),
            ]:
                if table[row][j].bg is None:
                    table[row][j].bg = color

        for j in s_eq(gen.gamma):
            if table[Subspace.Gperp_F][j].bg == BgColor.GREEN:
                return DimensionResult.INFINITE
            table[Subspace.Gperp_F][j].dot = DotColor.BLUE
            for row in [
                Subspace.G1,
                Subspace.G2_core,
                Subspace.Gom,
                Subspace.Geq_F,
                Subspace.G2_F,
                Subspace.Gperp_F,
            ]:
                if table[row][j].bg is None:
                    table[row][j].bg = BgColor.BLUE

    return DimensionResult.FINITE


def _process_G1_G2core_Gom_GeqF(
    table: dict[Subspace, list[Cell]],
    decomposed: dict[Subspace, set[BosonicGenerator]],
) -> DimensionResult:
    for row in [Subspace.G1, Subspace.G2_core, Subspace.Gom, Subspace.Geq_F]:
        for gen in decomposed[row]:
            for j in s_neq(gen.gamma):
                if table[row][j].bg in (BgColor.BLUE, BgColor.RED):
                    return DimensionResult.INFINITE
                table[row][j].dot = DotColor.GREEN
                for r in [
                    Subspace.G1,
                    Subspace.G2_core,
                    Subspace.Gom,
                    Subspace.Geq_F,
                    Subspace.G2_F,
                ]:
                    if table[r][j].bg is None:
                        table[r][j].bg = BgColor.GREEN

            for j in s_eq(gen.gamma):
                if table[row][j].bg in (BgColor.RED, BgColor.GREEN):
                    return DimensionResult.INFINITE
                table[row][j].dot = DotColor.BLUE
                for r in [
                    Subspace.G1,
                    Subspace.G2_core,
                    Subspace.Gom,
                    Subspace.Geq_F,
                    Subspace.G2_F,
                ]:
                    if table[r][j].bg is None:
                        table[r][j].bg = BgColor.BLUE
    return DimensionResult.FINITE


def _process_G0(
    table: dict[Subspace, list[Cell]],
    decomposed: dict[Subspace, set[BosonicGenerator]],
) -> DimensionResult:
    for gen in decomposed[Subspace.G0]:
        eq = s_eq(gen.gamma)
        if not eq:
            continue
        (j,) = eq
        if table[Subspace.G0][j].bg == BgColor.RED:
            return DimensionResult.INFINITE
        table[Subspace.G0][j].dot = DotColor.BLUE
        if table[Subspace.G2_F][j].bg is None:
            table[Subspace.G2_F][j].bg = BgColor.GREEN
    return DimensionResult.FINITE


def _has_orange_green_conflict(
    adj: list[list[int]],
    table: dict[Subspace, list[Cell]],
) -> bool:
    n = len(adj)
    dotted = [
        j for j in range(n) if table[Subspace.G2_F][j].dot == DotColor.GREEN
    ]
    visited: set[int] = set()
    for start in dotted:
        if start in visited:
            continue
        has_orange = False
        has_green = False
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            bg = table[Subspace.G2_F][node].bg
            has_orange |= bg == BgColor.ORANGE
            has_green |= bg == BgColor.GREEN
            if has_orange and has_green:
                return True
            for neighbor in range(n):
                if adj[node][neighbor] and neighbor not in visited:
                    queue.append(neighbor)
    return False


def _process_G2F(
    table: dict[Subspace, list[Cell]],
    decomposed: dict[Subspace, set[BosonicGenerator]],
    adj: list[list[int]],
) -> DimensionResult:
    for gen in decomposed[Subspace.G2_F]:
        p, q = s_neq(gen.gamma)
        for j in (p, q):
            if table[Subspace.G2_F][j].bg == BgColor.BLUE:
                return DimensionResult.INFINITE
            table[Subspace.G2_F][j].dot = DotColor.GREEN
        adj[p][q] = 1
        adj[q][p] = 1

    if _has_orange_green_conflict(adj, table):
        return DimensionResult.INFINITE
    return DimensionResult.FINITE


def _postprocess(
    adj: list[list[int]],
    table: dict[Subspace, list[Cell]],
    decomposed: dict[Subspace, set[BosonicGenerator]],
) -> FinitenessResult:
    n = len(adj)

    # BFS from all orange-background dotted nodes to find orange-reachable modes
    orange_reachable: set[int] = set()
    queue = [
        j
        for j in range(n)
        if table[Subspace.G2_F][j].dot == DotColor.GREEN
        and table[Subspace.G2_F][j].bg == BgColor.ORANGE
    ]
    while queue:
        node = queue.pop()
        if node in orange_reachable:
            continue
        orange_reachable.add(node)
        for neighbor in range(n):
            if adj[node][neighbor] and neighbor not in orange_reachable:
                queue.append(neighbor)

    orange_g2f = {
        gen
        for gen in decomposed[Subspace.G2_F]
        if any(j in orange_reachable for j in s_neq(gen.gamma))
    }

    if orange_g2f:
        remaining = orange_g2f | decomposed[Subspace.Gperp_F]
        return FinitenessResult(DimensionResult.REMAINING, remaining)
    return FinitenessResult(DimensionResult.FINITE)


def check_finiteness(
    n: int,
    F: list[FreeHamiltonian],
    generators: list[BosonicGenerator],
) -> FinitenessResult:
    F_prime, decomposed = _preprocess(F, generators)
    ROWS = [
        Subspace.G0,
        Subspace.G1,
        Subspace.G2_core,
        Subspace.Gom,
        Subspace.Geq_F,
        Subspace.G2_F,
        Subspace.Gperp_F,
    ]

    table = {row: [Cell() for _ in range(n)] for row in ROWS}

    # Fill in the table entries corresponding to G^⊥_F generators.
    intermediate_result = _process_GperpF(table, decomposed)
    if intermediate_result == DimensionResult.INFINITE:
        return FinitenessResult(DimensionResult.INFINITE)

    # Fill in the table entries corresponding to G^1, G^2_core, G^om, G^=F generators.
    intermediate_result = _process_G1_G2core_Gom_GeqF(table, decomposed)
    if intermediate_result == DimensionResult.INFINITE:
        return FinitenessResult(DimensionResult.INFINITE)

    # Fill in the table entries corresponding to G^0 generators.
    intermediate_result = _process_G0(table, decomposed)
    if intermediate_result == DimensionResult.INFINITE:
        return FinitenessResult(DimensionResult.INFINITE)

    # Fill in the table entries corresponding to G^2_F generators
    # and check for orange-green conflicts.
    adj = [[0] * n for _ in range(n)]
    intermediate_result = _process_G2F(table, decomposed, adj)
    if intermediate_result == DimensionResult.INFINITE:
        return FinitenessResult(DimensionResult.INFINITE)

    return _postprocess(adj, table, decomposed)
