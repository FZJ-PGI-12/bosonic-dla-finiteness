from __future__ import annotations

import logging
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

log = logging.getLogger(__name__)


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
    """One entry in the classification table: a background color and an optional dot."""

    bg: BgColor = None
    dot: DotColor = None


@dataclass
class FinitenessResult:
    """Return value of check_finiteness."""

    dimension: DimensionResult
    remaining_generators: set[BosonicGenerator] = field(default_factory=set)


def _preprocess_Gperp_G2(
    F_prime: list[FreeHamiltonian],
    decomposed: dict[Subspace, set[BosonicGenerator]],
) -> DimensionResult:
    # A generator in G^⊥ with nonzero χ_F(γ) witnesses an infinite-dimensional DLA
    for gen in decomposed[Subspace.Gperp]:
        chiF = compute_chi_F_gamma(F_prime, gen.gamma)
        if any(abs(chi) > ZERO_TOL for chi in chiF):
            return DimensionResult.INFINITE
    # All Gperp generators passed the χ_F = 0 check, so Gperp = Gperp_F here
    decomposed[Subspace.Gperp_F] = decomposed[Subspace.Gperp]
    # Diagonal generators always commute with the free Hamiltonian, so Geq = Geq_F
    decomposed[Subspace.Geq_F] = decomposed[Subspace.Geq]

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

    log.debug(
        "Subspace sizes: G2_F=%d, G2_core=%d, Gperp_F=%d, Geq_F=%d",
        len(decomposed[Subspace.G2_F]),
        len(decomposed[Subspace.G2_core]),
        len(decomposed[Subspace.Gperp_F]),
        len(decomposed[Subspace.Geq_F]),
    )
    return DimensionResult.FINITE


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
            continue  # constant (zero-gamma) operator — no diagonal index to process
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
    # Shared across all starting nodes: a node already in a checked component
    # cannot create a new conflict, so we skip it rather than re-traverse.
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
    # Seed only from dotted orange nodes: undotted orange backgrounds were set by
    # earlier steps and have no G2_F edges, so they cannot propagate connections.
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
    """
    Classify the DLA generated by `generators` as finite- or infinite-dimensional.

    Parameters
    ----------
    n : int
        Number of bosonic modes.
    F : list[FreeHamiltonian]
        Set of free Hamiltonians defining the drift F = {X^(1), ..., X^(m)}.
    generators : list[BosonicGenerator]
        Generators of the bosonic DLA to classify.

    Returns
    -------
    FinitenessResult
        .dimension == FINITE    — the DLA is finite-dimensional.
        .dimension == INFINITE  — the DLA is provably infinite-dimensional.
        .dimension == REMAINING — inconclusive; .remaining_generators contains
                                  the G2_F generators connected to an orange
                                  background and all G^⊥_F generators that
                                  require further analysis.
    """
    log.info(
        "Starting finiteness check: n=%d, |generators|=%d", n, len(generators)
    )
    log.debug("Computing F' from F")
    F_prime = compute_F_prime(F)
    log.debug("Decomposing generators into subspaces")
    decomposed = decompose_generators(generators)
    log.debug("Preprocessing G^⊥ and G^2 generators")
    intermediate_result = _preprocess_Gperp_G2(F_prime, decomposed)
    if intermediate_result == DimensionResult.INFINITE:
        log.info(
            "INFINITE: G^⊥ generator with nonzero χ_F detected in preprocessing"
        )
        return FinitenessResult(DimensionResult.INFINITE)

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

    log.debug("Processing G^⊥_F generators")
    intermediate_result = _process_GperpF(table, decomposed)
    if intermediate_result == DimensionResult.INFINITE:
        log.info("INFINITE: detected in G^⊥_F step")
        return FinitenessResult(DimensionResult.INFINITE)

    log.debug("Processing G^1, G^2_core, G^om, G^=_F generators")
    intermediate_result = _process_G1_G2core_Gom_GeqF(table, decomposed)
    if intermediate_result == DimensionResult.INFINITE:
        log.info("INFINITE: detected in G^1/G^2_core/G^om/G^=_F step")
        return FinitenessResult(DimensionResult.INFINITE)

    log.debug("Processing G^0 generators")
    intermediate_result = _process_G0(table, decomposed)
    if intermediate_result == DimensionResult.INFINITE:
        log.info("INFINITE: detected in G^0 step")
        return FinitenessResult(DimensionResult.INFINITE)

    log.debug("Processing G^2_F generators")
    adj = [[0] * n for _ in range(n)]
    intermediate_result = _process_G2F(table, decomposed, adj)
    if intermediate_result == DimensionResult.INFINITE:
        log.info("INFINITE: detected in G^2_F step")
        return FinitenessResult(DimensionResult.INFINITE)

    result = _postprocess(adj, table, decomposed)
    log.info("Result: %s", result.dimension.value)
    if result.dimension == DimensionResult.REMAINING:
        log.info("Remaining generators: %d", len(result.remaining_generators))
    return result
