from __future__ import annotations

from dataclasses import dataclass
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


# Row labels for the algorithm table
Row = Subspace


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


def _process_Gperp_F(
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


def mode_match(
    n: int,
    F: list[FreeHamiltonian],
    generators: list[BosonicGenerator],
) -> DimensionResult:
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
    intermediate_result = _process_Gperp_F(table, decomposed)
    if intermediate_result == DimensionResult.INFINITE:
        return DimensionResult.INFINITE

    # Fill in the table entries corresponding to G^1, G^2_core, G^om, G^=F generators.
    intermediate_result = _process_G1_G2core_Gom_GeqF(table, decomposed)
    if intermediate_result == DimensionResult.INFINITE:
        return DimensionResult.INFINITE

    # Fill in the table entries corresponding to G^0 generators.
    intermediate_result = _process_G0(table, decomposed)
    if intermediate_result == DimensionResult.INFINITE:
        return DimensionResult.INFINITE

    # Fill in the table entries corresponding to G^2_F generators
    # and check for orange-green conflicts.
    return DimensionResult.FINITE


def classify(
    n: int,
    F: list[FreeHamiltonian],
    generators: list[BosonicGenerator],
) -> DimensionResult:
    F_prime, decomposed = _preprocess(F, generators)

    backgrounds: dict[tuple[Row, int], BgColor] = {}
    dots: dict[tuple[Row, int], DotColor] = {}

    # ── Step 2.1: process G⊥_F ───────────────────────────────────────────────
    for gen in decomposed[Subspace.Gperp_F]:
        gamma = gen.gamma

        for j in s_neq(gamma):
            # conflict: green dot on blue background → infinite
            if backgrounds.get((Subspace.Gperp_F, j)) == BgColor.BLUE:
                return DimensionResult.INFINITE
            dots[(Subspace.Gperp_F, j)] = DotColor.GREEN
            for row, color in [
                (Subspace.G0, BgColor.RED),
                (Subspace.G1, BgColor.RED),
                (Subspace.G2_core, BgColor.RED),
                (Subspace.Gom, BgColor.RED),
                (Subspace.Geq_F, BgColor.RED),
                (Subspace.G2_F, BgColor.ORANGE),
                (Subspace.Gperp_F, BgColor.GREEN),
            ]:
                if (row, j) not in backgrounds:
                    backgrounds[(row, j)] = color

        for j in s_eq(gamma):
            # conflict: blue dot on green background → infinite
            if backgrounds.get((Subspace.Gperp_F, j)) == BgColor.GREEN:
                return DimensionResult.INFINITE
            dots[(Subspace.Gperp_F, j)] = DotColor.BLUE
            for row in [
                Subspace.G1,
                Subspace.G2_core,
                Subspace.Gom,
                Subspace.Geq_F,
                Subspace.G2_F,
                Subspace.Gperp_F,
            ]:
                if (row, j) not in backgrounds:
                    backgrounds[(row, j)] = BgColor.BLUE

    # ── Step 2.2: process G1, G2_core, Gom, Geq_F ───────────────────────────
    rows_2_2 = [Subspace.G1, Subspace.G2_core, Subspace.Gom, Subspace.Geq_F]
    propagate_2_2 = [
        Subspace.G1,
        Subspace.G2_core,
        Subspace.Gom,
        Subspace.Geq_F,
        Subspace.G2_F,
    ]

    for row in rows_2_2:
        for gen in decomposed[row]:
            gamma = gen.gamma

            for j in s_neq(gamma):
                bg = backgrounds.get((row, j))
                if bg in (BgColor.BLUE, BgColor.RED):
                    return DimensionResult.INFINITE
                dots[(row, j)] = DotColor.GREEN
                for r in propagate_2_2:
                    if (r, j) not in backgrounds:
                        backgrounds[(r, j)] = BgColor.GREEN

            for j in s_eq(gamma):
                bg = backgrounds.get((row, j))
                if bg in (BgColor.RED, BgColor.GREEN):
                    return DimensionResult.INFINITE
                dots[(row, j)] = DotColor.BLUE
                for r in propagate_2_2:
                    if (r, j) not in backgrounds:
                        backgrounds[(r, j)] = BgColor.BLUE

    # ── Step 2.3: process G0 ─────────────────────────────────────────────────
    for gen in decomposed[Subspace.G0]:
        gamma = gen.gamma
        eq = s_eq(gamma)
        if not eq:  # skip g0_+ = 2i (constant, S= is empty)
            continue
        (j,) = eq  # singleton guaranteed by definition
        if backgrounds.get((Subspace.G0, j)) == BgColor.RED:
            return DimensionResult.INFINITE
        dots[(Subspace.G0, j)] = DotColor.BLUE
        if (Subspace.G2_F, j) not in backgrounds:
            backgrounds[(Subspace.G2_F, j)] = BgColor.GREEN

    # ── Step 2.4: process G2_F ───────────────────────────────────────────────
    # Build a graph over column indices within the G2_F row.
    # Each generator g^(ιp,ιq) adds an edge {p, q}; we then check connectivity.
    g2f_adj: dict[
        int, set[int]
    ] = {}  # adjacency: j -> set of connected column indices

    for gen in decomposed[Subspace.G2_F]:
        gamma = gen.gamma
        p, q = sorted(s_neq(gamma))  # S≠(γ) = {p, q} guaranteed

        for j in (p, q):
            if backgrounds.get((Subspace.G2_F, j)) == BgColor.BLUE:
                return DimensionResult.INFINITE
            dots[(Subspace.G2_F, j)] = DotColor.GREEN
            g2f_adj.setdefault(j, set())

        g2f_adj[p].add(q)
        g2f_adj[q].add(p)

    # Check for connected component spanning orange and green backgrounds.
    visited: set[int] = set()
    for start in list(g2f_adj):
        if start in visited:
            continue
        # BFS over the connected component
        component: set[int] = set()
        queue = [start]
        while queue:
            node = queue.pop()
            if node in component:
                continue
            component.add(node)
            queue.extend(g2f_adj.get(node, set()))
        visited |= component

        bg_colors = {backgrounds.get((Subspace.G2_F, j)) for j in component}
        if BgColor.ORANGE in bg_colors and BgColor.GREEN in bg_colors:
            return DimensionResult.INFINITE

    return DimensionResult.FINITE
